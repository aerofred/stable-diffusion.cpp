#include <algorithm>
#include <exception>
#include <map>
#include <utility>

#include "core/ggml_extend.h"
#include "core/ggml_extend_backend.h"
#include "core/ggml_runner.h"
#include "core/ggml_tensor_utils.h"
#include "core/layer_split_partition.h"
#include "core/segment_graph_bindings.h"
#include "core/segment_weight_pipeline.h"

using namespace sd;

ggml_tensor* ggml_ext_attention_ext(GGMLRunnerContext* ctx,
                                    ggml_tensor* q,
                                    ggml_tensor* k,
                                    ggml_tensor* v,
                                    int64_t n_head,
                                    ggml_tensor* mask,
                                    bool skip_reshape,
                                    bool flash_attn,
                                    float kv_scale) {
    if (ctx->attn_scale > 0.f) {
        kv_scale = ctx->attn_scale;
    }
    return ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, n_head, mask, skip_reshape, flash_attn, kv_scale, ctx->sage_attn_enabled);
}

void GGMLRunner::alloc_params_ctx() {
    ggml_init_params params;
    params.mem_size   = static_cast<size_t>(MAX_PARAMS_TENSOR_NUM * ggml_tensor_overhead());
    params.mem_buffer = nullptr;
    params.no_alloc   = true;

    params_ctx = ggml_init(params);
    GGML_ASSERT(params_ctx != nullptr);
    params_tensor_set_.clear();
    params_tensor_set_dirty_ = true;
}

void GGMLRunner::free_params_ctx() {
    if (params_ctx != nullptr) {
        ggml_free(params_ctx);
        params_ctx = nullptr;
    }
    params_tensor_set_.clear();
    params_tensor_set_dirty_ = true;
}

void GGMLRunner::alloc_compute_ctx() {
    ggml_init_params params;
    params.mem_size   = static_cast<size_t>(ggml_tensor_overhead() * MAX_GRAPH_SIZE + ggml_graph_overhead());
    params.mem_buffer = nullptr;
    params.no_alloc   = true;

    compute_ctx = ggml_init(params);
    GGML_ASSERT(compute_ctx != nullptr);
}

void GGMLRunner::free_compute_ctx() {
    debug_tensors.clear();
    free_graph_inputs();
    if (compute_ctx != nullptr) {
        ggml_free(compute_ctx);
        compute_ctx = nullptr;
    }
    backend_tensor_data_map.clear();
}

void GGMLRunner::rebuild_params_tensor_set() {
    if (!params_tensor_set_dirty_) {
        return;
    }
    params_tensor_set_.clear();
    if (params_ctx == nullptr) {
        return;
    }
    for (ggml_tensor* t = ggml_get_first_tensor(params_ctx); t != nullptr; t = ggml_get_next_tensor(params_ctx, t)) {
        params_tensor_set_.insert(t);
    }
    params_tensor_set_dirty_ = false;
}

ggml_tensor* GGMLRunner::canonical_param_tensor(ggml_tensor* tensor) {
    for (auto* current = tensor; current != nullptr; current = current->view_src) {
        if (params_tensor_set_.count(current) != 0)
            return current;
    }
    return nullptr;
}

std::vector<ggml_tensor*> GGMLRunner::collect_used_param_tensors(ggml_cgraph* gf) {
    std::vector<ggml_tensor*> used_params;
    rebuild_params_tensor_set();
    if (gf == nullptr || params_tensor_set_.empty()) {
        return used_params;
    }

    std::unordered_set<const ggml_tensor*> seen_params;
    const int n_leafs = sd::ggml_graph_cut::leaf_count(gf);
    seen_params.reserve(static_cast<size_t>(n_leafs));
    for (int i = 0; i < n_leafs; ++i) {
        ggml_tensor* leaf       = sd::ggml_graph_cut::leaf_tensor(gf, i);
        ggml_tensor* param_leaf = canonical_param_tensor(leaf);
        if (param_leaf != nullptr &&
            seen_params.insert(param_leaf).second) {
            used_params.push_back(param_leaf);
        }
    }
    return used_params;
}

void GGMLRunner::evict_compute_backend_param_tensors(const std::vector<ggml_tensor*>& tensors) {
    if (tensors.empty()) {
        return;
    }
    auto manager = residency_manager.lock();
    if (manager != nullptr) {
        manager->evict_compute_backend_params(tensors);
    }
}

void GGMLRunner::prepare_build_in_tensor_before() {
    one_tensor = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_F32, 1);
    ggml_set_name(one_tensor, "ggml_runner_build_in_tensor:one");
    set_backend_tensor_data(one_tensor, one_vec.data());

    zero_int_tensor = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, 1);
    ggml_set_name(zero_int_tensor, "ggml_runner_build_in_tensor:zero_int");
    set_backend_tensor_data(zero_int_tensor, zero_int_vec.data());
}

void GGMLRunner::prepare_build_in_tensor_after(ggml_cgraph* gf) {
    ggml_build_forward_expand(gf, one_tensor);
    ggml_build_forward_expand(gf, zero_int_tensor);
}

ggml_cgraph* GGMLRunner::new_graph_custom(size_t graph_size) {
    if (weight_adapter) {
        graph_size += weight_adapter->get_extra_graph_size();
    }
    return ggml_new_graph_custom(compute_ctx, graph_size, false);
}

ggml_cgraph* GGMLRunner::get_compute_graph(get_graph_cb_t get_graph) {
    prepare_build_in_tensor_before();
    ggml_cgraph* gf = get_graph();
    if (gf == nullptr) {
        return nullptr;
    }
    if (ggml_graph_n_nodes(gf) > 0) {
        auto result = ggml_graph_node(gf, -1);
        ggml_set_name(result, final_result_name.c_str());
    }
    for (const auto& entry : debug_tensors) {
        if (entry.first != nullptr) {
            ggml_build_forward_expand(gf, entry.first);
        }
    }
    for (const auto& entry : cache_.outputs()) {
        if (entry.second != nullptr) {
            ggml_build_forward_expand(gf, entry.second);
        }
    }
    prepare_build_in_tensor_after(gf);
    return gf;
}

bool GGMLRunner::prepare_compute_graph(get_graph_cb_t get_graph,
                                       ggml_cgraph** gf_out) {
    GGML_ASSERT(gf_out != nullptr);

    reset_compute_ctx();
    ggml_cgraph* gf = get_compute_graph(get_graph);
    if (gf == nullptr) {
        free_compute_ctx();
        return false;
    }

    *gf_out = gf;
    return true;
}

ggml_backend_t GGMLRunner::backend_for_weight(const ggml_tensor* tensor) const {
    if (tensor == nullptr || tensor->buffer == nullptr) {
        return nullptr;
    }
    if (ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
        ggml_backend_buffer_is_host(tensor->buffer)) {
        return nullptr;
    }
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
    if (dev == nullptr) {
        return nullptr;
    }
    if (ggml_backend_get_device(runtime_backend) == dev) {
        return runtime_backend;
    }
    for (ggml_backend_t backend : extra_runtime_backends) {
        if (ggml_backend_get_device(backend) == dev) {
            return backend;
        }
    }
    return nullptr;
}

void GGMLRunner::pin_multi_device_nodes(ggml_backend_sched_t sched, ggml_cgraph* gf, ggml_cgraph* original_graph) {
    if (sched == nullptr || gf == nullptr) {
        return;
    }
    ggml_backend_t current = runtime_backend;
    const int n_nodes      = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor* node    = ggml_graph_node(gf, i);
        auto node_assignment = graph_cut_layer_split_node_assignments_.find(original_graph == nullptr ? node : ggml_graph_node(original_graph, i));
        if (node_assignment != graph_cut_layer_split_node_assignments_.end()) {
            current = node_assignment->second;
        }
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            ggml_backend_t weight_backend = backend_for_weight(node->src[s]);
            if (weight_backend != nullptr) {
                if (node_assignment == graph_cut_layer_split_node_assignments_.end()) {
                    current = weight_backend;
                }
            }
        }
        // In-place ops are views of their source: pinning them elsewhere would
        // run the kernel on one device over the other device's memory.
        if (node->view_src != nullptr || node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW ||
            node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) {
            continue;
        }
        if (ggml_backend_supports_op(current, node)) {
            ggml_backend_sched_set_tensor_backend(sched, node, current);
        }
    }
}

size_t GGMLRunner::retained_runtime_buffer_bytes(ggml_backend_t backend) const {
    backend                   = backend == nullptr ? runtime_backend : backend;
    size_t bytes              = workspace_.bytes(backend);
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    const size_t cache_bytes  = cache_.resident_bytes(device);
    bytes                     = cache_bytes > SIZE_MAX - bytes ? SIZE_MAX : bytes + cache_bytes;
    const size_t cut_bytes    = cut_cache_.resident_bytes(device);
    bytes                     = cut_bytes > SIZE_MAX - bytes ? SIZE_MAX : bytes + cut_bytes;
    if (backend == runtime_backend && input_buffer_ != nullptr) {
        const size_t input_bytes = ggml_backend_buffer_get_size(input_buffer_);
        bytes                    = input_bytes > SIZE_MAX - bytes ? SIZE_MAX : bytes + input_bytes;
    }
    return bytes;
}

void GGMLRunner::sync_runtime_residency() {
    if (auto manager = residency_manager.lock()) {
        manager->update_runtime_residency(reinterpret_cast<uintptr_t>(this),
                                          runtime_backend, retained_runtime_buffer_bytes());
        for (auto backend : extra_runtime_backends) {
            manager->update_runtime_residency(reinterpret_cast<uintptr_t>(this),
                                              backend, retained_runtime_buffer_bytes(backend));
        }
    }
}

std::optional<sd::Tensor<float>> GGMLRunner::read_graph_tensor(ggml_tensor* tensor, const char* label) {
    if (tensor == nullptr) {
        LOG_ERROR("%s %s tensor is null", get_desc().c_str(), label);
        return std::nullopt;
    }
    if (tensor->type != GGML_TYPE_F32) {
        LOG_ERROR("%s %s tensor type mismatch: got %s",
                  get_desc().c_str(),
                  label,
                  ggml_type_name(tensor->type));
        return std::nullopt;
    }
    ggml_backend_buffer_t buf = sd::ggml_graph_cut::tensor_buffer(tensor);
    if (buf == nullptr) {
        LOG_ERROR("%s %s tensor buffer missing: name=%s op=%s buffer=%p view_src=%p view_src_buffer=%p data=%p",
                  get_desc().c_str(),
                  label,
                  tensor->name[0] != '\0' ? tensor->name : "<unnamed>",
                  ggml_op_name(tensor->op),
                  tensor->buffer,
                  tensor->view_src,
                  tensor->view_src ? tensor->view_src->buffer : nullptr,
                  tensor->data);
        return std::nullopt;
    }

    return sd::make_sd_tensor_from_ggml<float>(tensor);
}

void GGMLRunner::copy_data_to_backend_tensor(ggml_cgraph* gf, bool clear_after_copy) {
    GGML_ASSERT(gf != nullptr);
    std::unordered_set<const ggml_tensor*> graph_tensor_set;
    const int n_leafs = sd::ggml_graph_cut::leaf_count(gf);
    const int n_nodes = ggml_graph_n_nodes(gf);
    graph_tensor_set.reserve(static_cast<size_t>(n_leafs + n_nodes));
    for (int i = 0; i < n_leafs; ++i) {
        graph_tensor_set.insert(sd::ggml_graph_cut::leaf_tensor(gf, i));
    }
    for (int i = 0; i < n_nodes; ++i) {
        graph_tensor_set.insert(ggml_graph_node(gf, i));
    }

    for (auto& kv : backend_tensor_data_map) {
        auto tensor = kv.first;
        auto data   = kv.second;
        if (tensor == nullptr || data == nullptr) {
            continue;
        }
        const char* name = ggml_get_name(tensor);
        if (graph_tensor_set.find(tensor) == graph_tensor_set.end() ||
            uploaded_inputs_.find(tensor) != uploaded_inputs_.end()) {
            continue;
        }
        if (tensor->buffer == nullptr) {
            LOG_WARN("%s skip backend tensor copy: tensor buffer not set, name='%s', ne=[%lld,%lld,%lld,%lld], type=%s",
                     get_desc().c_str(),
                     name != nullptr ? name : "",
                     (long long)tensor->ne[0],
                     (long long)tensor->ne[1],
                     (long long)tensor->ne[2],
                     (long long)tensor->ne[3],
                     ggml_type_name(tensor->type));
            continue;
        }

        ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        if (buf == nullptr) {
            LOG_WARN("%s graph exec skip tensor copy: name=%s op=%s reason=buffer_not_set data=%p view_src=%p view_src_buffer=%p",
                     get_desc().c_str(),
                     tensor && tensor->name[0] != '\0' ? tensor->name : "<unnamed>",
                     tensor ? ggml_op_name(tensor->op) : "<null>",
                     data,
                     tensor ? tensor->view_src : nullptr,
                     (tensor && tensor->view_src) ? tensor->view_src->buffer : nullptr);
            continue;
        }

        ggml_backend_tensor_set(tensor, data, 0, ggml_nbytes(tensor));
    }

    if (clear_after_copy) {
        backend_tensor_data_map.clear();
    }
}

const GGMLRunner::GraphCutPlan& GGMLRunner::resolve_graph_cut_plan(ggml_cgraph* gf) {
    GGML_ASSERT(gf != nullptr);
    return sd::ggml_graph_cut::resolve_plan(runtime_backend,
                                            gf,
                                            &graph_cut_plan_cache_,
                                            params_tensor_set_,
                                            get_desc().c_str());
}

const GGMLRunner::GraphCutPlan& GGMLRunner::resolve_graph_cut_layer_split_plan(ggml_cgraph* gf) {
    return resolve_graph_cut_plan(gf);
}

bool GGMLRunner::assign_graph_cut_layer_split_backends(ggml_cgraph* gf,
                                                       const GraphCutPlan& plan,
                                                       const std::vector<ggml_tensor*>& params) {
    graph_cut_layer_split_node_assignments_.clear();
    if (!graph_cut_layer_split_enabled) {
        return true;
    }
    if (!is_multi_device()) {
        LOG_ERROR("%s graph-cut layer split requires multiple runtime backends", get_desc().c_str());
        return false;
    }

    const int n_nodes          = ggml_graph_n_nodes(gf);
    const bool params_assigned = std::all_of(params.begin(), params.end(), [&](ggml_tensor* param) {
        return param == nullptr || graph_cut_layer_split_assignments_.count(param) != 0;
    });
    if (params_assigned && !plan.layout.empty() && plan.layout == layer_split_layout_ &&
        layer_split_node_backends_.size() == static_cast<size_t>(n_nodes)) {
        for (int i = 0; i < n_nodes; ++i) {
            if (layer_split_node_backends_[i] != nullptr) {
                graph_cut_layer_split_node_assignments_[ggml_graph_node(gf, i)] = layer_split_node_backends_[i];
            }
        }
        return true;
    }
    invalidate_layer_split_cache();
    auto cache_assignments = [&]() {
        layer_split_layout_ = plan.layout;
        layer_split_node_backends_.assign(static_cast<size_t>(std::max(n_nodes, 0)), nullptr);
        for (int i = 0; i < n_nodes; ++i) {
            auto entry = graph_cut_layer_split_node_assignments_.find(ggml_graph_node(gf, i));
            if (entry != graph_cut_layer_split_node_assignments_.end()) {
                layer_split_node_backends_[static_cast<size_t>(i)] = entry->second;
            }
        }
    };

    auto manager = residency_manager.lock();
    if (manager == nullptr) {
        LOG_ERROR("%s weight manager is not set for graph-cut layer split", get_desc().c_str());
        return false;
    }

    if (!plan.valid || !plan.has_cuts || plan.segments.size() <= 1) {
        if (!params.empty() &&
            !manager->assign_compute_backend(params, runtime_backend)) {
            LOG_ERROR("%s graph-cut layer split failed to assign unmarked graph params to %s",
                      get_desc().c_str(),
                      sd::layer_split_backend_device_display_name(runtime_backend).c_str());
            return false;
        }
        for (ggml_tensor* param : params) {
            if (param != nullptr) {
                graph_cut_layer_split_assignments_[param] = runtime_backend;
            }
        }
        for (int i = 0; i < n_nodes; i++) {
            ggml_tensor* node = ggml_graph_node(gf, i);
            if (node != nullptr) {
                graph_cut_layer_split_node_assignments_[node] = runtime_backend;
            }
        }
        if (!graph_cut_layer_split_primary_notice_logged_) {
            LOG_WARN("%s graph-cut layer split: graph has no mark_graph_cut segments; using primary backend %s for %zu graph params",
                     get_desc().c_str(),
                     sd::layer_split_backend_device_display_name(runtime_backend).c_str(),
                     params.size());
            graph_cut_layer_split_primary_notice_logged_ = true;
        } else {
            LOG_VERBOSE("%s graph-cut layer split: graph has no mark_graph_cut segments; using primary backend %s for %zu graph params",
                        get_desc().c_str(),
                        sd::layer_split_backend_device_display_name(runtime_backend).c_str(),
                        params.size());
        }
        cache_assignments();
        return true;
    }

    std::vector<ggml_backend_t> split_backends;
    split_backends.reserve(extra_runtime_backends.size() + 1);
    split_backends.push_back(runtime_backend);
    for (ggml_backend_t backend : extra_runtime_backends) {
        if (backend != nullptr) {
            split_backends.push_back(backend);
        }
    }

    sd::GraphCutLayerSplitAssignment assignment;
    auto canonicalize_param = [this](ggml_tensor* tensor) {
        return canonical_param_tensor(tensor);
    };
    if (!sd::partition_graph_cut_layer_split(get_desc().c_str(),
                                             gf,
                                             plan,
                                             split_backends,
                                             graph_cut_layer_split_backend_vram_limits_,
                                             max_graph_vram_bytes,
                                             graph_cut_layer_split_assignments_,
                                             canonicalize_param,
                                             graph_cut_layer_split_policy_,
                                             &assignment)) {
        return false;
    }

    for (size_t i = 0; i < split_backends.size(); i++) {
        if (assignment.tensors_by_backend[i].empty()) {
            continue;
        }
        if (!manager->assign_compute_backend(assignment.tensors_by_backend[i], split_backends[i])) {
            LOG_ERROR("%s graph-cut layer split failed to assign params to %s",
                      get_desc().c_str(),
                      sd::layer_split_backend_device_display_name(split_backends[i]).c_str());
            return false;
        }
    }

    graph_cut_layer_split_node_assignments_ = std::move(assignment.node_assignments);
    sd::log_graph_cut_layer_split_assignment(get_desc().c_str(), split_backends, assignment);
    cache_assignments();

    return true;
}

bool GGMLRunner::runner_start() {
    if (runner_started_) {
        return true;
    }
    cache_.clear();
    workspace_.set_extra_backends(extra_runtime_backends);
    if (auto manager = residency_manager.lock()) {
        manager->set_workspace_reclaimer(reinterpret_cast<uintptr_t>(this), [this](ggml_backend_t backend) {
            const bool uses_device = backend == nullptr || backend == runtime_backend ||
                                     std::find(extra_runtime_backends.begin(), extra_runtime_backends.end(), backend) !=
                                         extra_runtime_backends.end();
            if (!uses_device || !workspace_.release()) {
                return false;
            }
            sync_runtime_residency();
            return true;
        });
    }
    runner_started_ = true;
    return true;
}

void GGMLRunner::runner_end() {
    GGML_ASSERT(!graph_active_);
    if (!runner_started_) {
        return;
    }
    workspace_.release();
    cache_.clear();
    logged_compute_bytes_.clear();
    logged_segment_count_ = 0;
    if (auto manager = residency_manager.lock()) {
        manager->clear_prefetched_params(reinterpret_cast<uintptr_t>(this));
        std::vector<ggml_tensor*> tensors;
        for (auto tensor : params_tensor_set_) {
            auto* parameter = manager->resolve_param_tensor(const_cast<ggml_tensor*>(tensor));
            if (parameter != nullptr)
                tensors.push_back(parameter);
        }
        manager->evict_compute_backend_params(tensors);
        manager->trim_reclaimable_memory(runtime_backend);
        for (ggml_backend_t backend : extra_runtime_backends) {
            manager->trim_reclaimable_memory(backend);
        }
        manager->remove_runtime_owner(reinterpret_cast<uintptr_t>(this));
    }
    cross_step_prefetch_segment_ = SIZE_MAX;
    cross_step_prefetch_params_.clear();
    cross_step_layout_.clear();
    runner_started_ = false;
}

GGMLRunner::GGMLRunner(ggml_backend_t backend,
                       std::shared_ptr<DeviceResidencyManager> manager)
    : runtime_backend(backend),
      cache_(backend),
      cut_cache_(backend),
      workspace_(backend),
      residency_manager(manager) {
    GGML_ASSERT(runtime_backend != nullptr);
    alloc_params_ctx();
}

GGMLRunner::~GGMLRunner() {
    runner_end();
    free_compute_ctx();
    free_params_ctx();
}

GGMLRunnerContext GGMLRunner::get_context() {
    GGMLRunnerContext runner_ctx;
    runner_ctx.ggml_ctx              = compute_ctx;
    runner_ctx.backend               = runtime_backend;
    runner_ctx.flash_attn_enabled    = flash_attn_enabled;
    runner_ctx.sage_attn_enabled     = sage_attn_enabled;
    runner_ctx.linear_scale          = linear_scale;
    runner_ctx.attn_scale            = attn_scale;
    runner_ctx.conv2d_direct_enabled = conv2d_direct_enabled;
    runner_ctx.conv3d_direct_enabled = conv3d_direct_enabled;
    runner_ctx.circular_x_enabled    = circular_x_enabled;
    runner_ctx.circular_y_enabled    = circular_y_enabled;
    runner_ctx.weight_adapter        = weight_adapter;
    runner_ctx.debug_tensors         = &debug_tensors;
    runner_ctx.get_cache_tensor      = [this](const std::string& name) {
        return this->get_cache_tensor_by_name(name);
    };
    runner_ctx.cache_tensor = [this](const std::string& name, ggml_tensor* tensor) {
        this->cache(name, tensor);
    };
    runner_ctx.set_backend_tensor_data = [this](ggml_tensor* tensor, const void* data) {
        this->set_backend_tensor_data(tensor, data);
    };
    return runner_ctx;
}

void GGMLRunner::reset_compute_ctx() {
    free_compute_ctx();
    alloc_compute_ctx();
}

void GGMLRunner::free_cache_ctx_and_buffer() {
    cache_.clear();
    sync_runtime_residency();
}

void GGMLRunner::set_backend_tensor_data(ggml_tensor* tensor, const void* data) {
    // The scheduler only allocates standalone data tensors when they are
    // marked as graph inputs. The flag is harmless for single-backend graphs.
    ggml_set_input(tensor);
    backend_tensor_data_map[tensor] = data;
}

ggml_tensor* GGMLRunner::to_backend(ggml_tensor* tensor) {
    GGML_ASSERT(compute_ctx != nullptr);
    if (tensor == nullptr) {
        return nullptr;
    }
    // it's performing a compute, check if backend isn't cpu
    if (!sd_backend_is_cpu(runtime_backend) && (tensor->buffer == nullptr || ggml_backend_buffer_is_host(tensor->buffer))) {
        // pass input tensors to gpu memory
        auto backend_tensor = ggml_dup_tensor(compute_ctx, tensor);

        set_backend_tensor_data(backend_tensor, tensor->data);
        return backend_tensor;
    } else {
        return tensor;
    }
}

void GGMLRunner::cache(const std::string name, ggml_tensor* tensor) {
    if (tensor != nullptr && tensor->view_src != nullptr) {
        tensor = ggml_cont(compute_ctx, tensor);
    }
    if (tensor != nullptr) {
        ggml_set_output(tensor);
    }
    cache_.stage(name, tensor);
}

std::optional<sd::Tensor<float>> GGMLRunner::compute(get_graph_cb_t get_graph,
                                                     int n_threads,
                                                     bool auto_runner_end,
                                                     bool no_return,
                                                     const std::function<bool()>& read_outputs) {
    last_compute_status_ = GGML_STATUS_FAILED;
    if (graph_active_) {
        LOG_ERROR("%s does not support reentrant graph execution", get_desc().c_str());
        return std::nullopt;
    }
    if (!runner_start()) {
        runner_end();
        return std::nullopt;
    }
    struct RunnerEndGuard {
        GGMLRunner& runner;
        bool enabled;
        ~RunnerEndGuard() {
            if (enabled) {
                runner.runner_end();
            }
        }
    } runner_guard{*this, auto_runner_end};
    graph_active_ = true;
    bool success  = false;
    struct GraphEndGuard {
        GGMLRunner& runner;
        const bool& success;
        ~GraphEndGuard() {
            if (!runner.workspace_.segment_end()) {
                runner.last_compute_status_ = GGML_STATUS_FAILED;
            }
            runner.cache_.graph_end(false);
            runner.cut_cache_.clear();
            runner.free_compute_ctx();
            runner.graph_active_ = false;
            if (!success) {
                runner.workspace_.release();
            }
            runner.sync_runtime_residency();
        }
    } graph_guard{*this, success};

    ggml_cgraph* graph = nullptr;
    if (!prepare_compute_graph(get_graph, &graph)) {
        return std::nullopt;
    }
    params_tensor_set_dirty_ = true;
    rebuild_params_tensor_set();
    if (auto manager = residency_manager.lock()) {
        for (int i = 0; i < sd::ggml_graph_cut::leaf_count(graph); ++i) {
            auto* parameter = manager->resolve_param_tensor(sd::ggml_graph_cut::leaf_tensor(graph, i));
            if (parameter != nullptr)
                params_tensor_set_.insert(parameter);
        }
    }
    std::optional<sd::Tensor<float>> output;
    try {
        output = execute_graph(graph, n_threads, no_return, read_outputs);
    } catch (const std::exception& error) {
        last_compute_status_ = GGML_STATUS_FAILED;
        LOG_ERROR("%s graph execution failed on %s: %s", get_desc().c_str(),
                  ggml_backend_name(runtime_backend), error.what());
        return std::nullopt;
    }
    success = output.has_value();
    if (success) {
        cache_.graph_end(true);
        last_compute_status_ = GGML_STATUS_SUCCESS;
    }
    return output;
}

void GGMLRunner::set_graph_cut_layer_split_enabled(bool enabled) {
    graph_cut_layer_split_enabled = enabled;
    if (!enabled) {
        graph_cut_layer_split_assignments_.clear();
        graph_cut_layer_split_node_assignments_.clear();
        graph_cut_layer_split_primary_notice_logged_ = false;
    }
    invalidate_layer_split_cache();
}

void GGMLRunner::set_graph_cut_layer_split_backend_vram_limits(const std::vector<size_t>& limits) {
    graph_cut_layer_split_backend_vram_limits_ = limits;
    graph_cut_layer_split_assignments_.clear();
    graph_cut_layer_split_node_assignments_.clear();
    graph_cut_layer_split_primary_notice_logged_ = false;
    invalidate_layer_split_cache();
}

void GGMLRunner::set_graph_cut_layer_split_policy(const sd::LayerSplitPolicy& policy) {
    graph_cut_layer_split_policy_ = policy;
    graph_cut_layer_split_assignments_.clear();
    graph_cut_layer_split_node_assignments_.clear();
    invalidate_layer_split_cache();
}

void GGMLRunner::set_runtime_backends(const std::vector<ggml_backend_t>& backends) {
    extra_runtime_backends.clear();
    for (ggml_backend_t backend : backends) {
        if (backend == nullptr || backend == runtime_backend) {
            continue;
        }
        if (std::find(extra_runtime_backends.begin(), extra_runtime_backends.end(), backend) ==
            extra_runtime_backends.end()) {
            extra_runtime_backends.push_back(backend);
        }
    }
    workspace_.set_extra_backends(extra_runtime_backends);
    graph_cut_layer_split_assignments_.clear();
    graph_cut_layer_split_node_assignments_.clear();
    graph_cut_layer_split_primary_notice_logged_ = false;
    invalidate_layer_split_cache();
}

static size_t add_bytes(size_t a, size_t b) {
    return b > SIZE_MAX - a ? SIZE_MAX : a + b;
}

ComputeWorkspace::Measurement GGMLRunner::measure(ggml_cgraph* graph, size_t direct_bytes) {
    auto external_backend = [&](const ggml_tensor* tensor) -> ggml_backend_t {
        if (!params_tensor_set_.count(tensor)) {
            return nullptr;
        }
        auto placement = graph_cut_layer_split_assignments_.find(tensor);
        return placement == graph_cut_layer_split_assignments_.end() ? runtime_backend : placement->second;
    };
    auto assign_nodes = [&](ggml_backend_sched_t scheduler, ggml_cgraph* copy) {
        pin_multi_device_nodes(scheduler, copy, graph);
    };
    return workspace_.measure(graph, direct_bytes, external_backend, assign_nodes);
}

ComputeWorkspace::Measurement GGMLRunner::measure_cached(ggml_cgraph* graph,
                                                         size_t direct_bytes,
                                                         const std::vector<uint64_t>& layout,
                                                         size_t slot,
                                                         bool inputs_preallocated) {
    if (layout.empty()) {
        return measure(graph, direct_bytes);
    }
    if (layout != measurement_layout_) {
        invalidate_measurements();
        measurement_layout_              = layout;
        measurement_inputs_preallocated_ = inputs_preallocated;
    }
    std::optional<ComputeWorkspace::Measurement>* entry = &full_measurement_;
    if (slot != SIZE_MAX) {
        // Segment measurements exclude inputs that were uploaded up front.
        if (inputs_preallocated != measurement_inputs_preallocated_) {
            segment_measurements_.clear();
            measurement_inputs_preallocated_ = inputs_preallocated;
        }
        if (segment_measurements_.size() <= slot) {
            segment_measurements_.resize(slot + 1);
        }
        entry = &segment_measurements_[slot];
    }
    if (!entry->has_value() || (*entry)->buffers.empty()) {
        *entry = measure(graph, direct_bytes);
    }
    return **entry;
}

void GGMLRunner::invalidate_measurements() {
    measurement_layout_.clear();
    measurement_inputs_preallocated_ = false;
    segment_measurements_.clear();
    full_measurement_.reset();
}

void GGMLRunner::invalidate_layer_split_cache() {
    layer_split_layout_.clear();
    layer_split_node_backends_.clear();
    invalidate_measurements();
}

size_t GGMLRunner::vram_limit_for(ggml_backend_t backend) const {
    size_t limit = max_graph_vram_bytes;
    if (is_multi_device()) {
        size_t index = 0;
        if (backend != runtime_backend) {
            auto position = std::find(extra_runtime_backends.begin(), extra_runtime_backends.end(), backend);
            index         = static_cast<size_t>(position - extra_runtime_backends.begin()) + 1;
        }
        if (index < graph_cut_layer_split_backend_vram_limits_.size()) {
            limit = graph_cut_layer_split_backend_vram_limits_[index];
        }
    }
    return limit;
}

bool GGMLRunner::preallocate_graph_inputs(ggml_cgraph* graph) {
    free_graph_inputs();
    if (graph == nullptr || sd_backend_is_cpu(runtime_backend) || backend_tensor_data_map.empty()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(runtime_backend);
    if (buft == nullptr) {
        return false;
    }
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    const size_t max_size  = ggml_backend_buft_get_max_size(buft);
    std::vector<ggml_tensor*> inputs;
    size_t total      = 0;
    const int n_leafs = sd::ggml_graph_cut::leaf_count(graph);
    for (int i = 0; i < n_leafs; ++i) {
        ggml_tensor* leaf = sd::ggml_graph_cut::leaf_tensor(graph, i);
        if (leaf == nullptr || leaf->buffer != nullptr || leaf->data != nullptr || leaf->view_src != nullptr) {
            continue;
        }
        auto entry = backend_tensor_data_map.find(leaf);
        if (entry == backend_tensor_data_map.end() || entry->second == nullptr) {
            continue;
        }
        const size_t size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, leaf), alignment);
        if (max_size > 0 && size > max_size - std::min(total, max_size)) {
            return false;
        }
        total += size;
        inputs.push_back(leaf);
    }
    if (inputs.empty()) {
        return false;
    }
    total = add_bytes(total, alignment);
    DeviceMemoryRequest request{runtime_backend, reinterpret_cast<uintptr_t>(this), total,
                                retained_runtime_buffer_bytes(), vram_limit_for(runtime_backend)};
    if (!fits({request}, {})) {
        return false;
    }
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, total);
    if (buffer == nullptr) {
        return false;
    }
    ggml_tallocr allocator = ggml_tallocr_new(buffer);
    for (ggml_tensor* input : inputs) {
        if (ggml_tallocr_alloc(&allocator, input) != GGML_STATUS_SUCCESS) {
            for (ggml_tensor* bound : inputs) {
                bound->buffer = nullptr;
                bound->data   = nullptr;
                bound->extra  = nullptr;
            }
            ggml_backend_buffer_free(buffer);
            uploaded_inputs_.clear();
            return false;
        }
        ggml_backend_tensor_set(input, backend_tensor_data_map[input], 0, ggml_nbytes(input));
        uploaded_inputs_.insert(input);
    }
    input_buffer_ = buffer;
    sync_runtime_residency();
    LOG_DEBUG("%s uploaded %zu graph inputs (%.2f MB) once for segmented execution",
              get_desc().c_str(), inputs.size(), total / (1024.0 * 1024.0));
    return true;
}

void GGMLRunner::free_graph_inputs() {
    for (const ggml_tensor* input : uploaded_inputs_) {
        ggml_tensor* tensor = const_cast<ggml_tensor*>(input);
        tensor->buffer      = nullptr;
        tensor->data        = nullptr;
        tensor->extra       = nullptr;
    }
    uploaded_inputs_.clear();
    if (input_buffer_ != nullptr) {
        ggml_backend_buffer_free(input_buffer_);
        input_buffer_ = nullptr;
    }
}

void GGMLRunner::clear_cross_step_prefetch() {
    if (cross_step_prefetch_segment_ != SIZE_MAX) {
        if (auto manager = residency_manager.lock()) {
            manager->clear_prefetched_params(reinterpret_cast<uintptr_t>(this));
        }
    }
    cross_step_prefetch_segment_ = SIZE_MAX;
    cross_step_prefetch_params_.clear();
    cross_step_layout_.clear();
}

std::vector<DeviceMemoryRequest> GGMLRunner::memory_requests(
    const std::vector<BackendBufferSize>& sizes,
    const std::map<ggml_backend_t, size_t>& pending_cache_bytes) const {
    std::vector<DeviceMemoryRequest> requests;
    for (const auto& size : sizes) {
        const size_t retained    = retained_runtime_buffer_bytes(size.backend);
        const size_t reusable    = workspace_.bytes(size.backend);
        const auto cache_entry   = pending_cache_bytes.find(size.backend);
        const size_t cache_bytes = cache_entry == pending_cache_bytes.end() ? 0 : cache_entry->second;
        const size_t pending     = add_bytes(size.bytes > reusable ? size.bytes - reusable : 0, cache_bytes);
        requests.push_back({size.backend, reinterpret_cast<uintptr_t>(this), pending,
                            retained, vram_limit_for(size.backend)});
    }
    return requests;
}

std::vector<ggml_backend_t> GGMLRunner::segment_backends(const GraphCutPlan& plan, ggml_cgraph* gf) const {
    std::vector<ggml_backend_t> backends(plan.segments.size(), runtime_backend);
    if (!is_multi_device() || graph_cut_layer_split_node_assignments_.empty() || gf == nullptr) {
        return backends;
    }
    const int n_nodes    = ggml_graph_n_nodes(gf);
    auto find_assignment = [&](const std::vector<int>& node_indices, ggml_backend_t* backend) {
        for (int node_index : node_indices) {
            if (node_index < 0 || node_index >= n_nodes) {
                continue;
            }
            auto assignment = graph_cut_layer_split_node_assignments_.find(ggml_graph_node(gf, node_index));
            if (assignment != graph_cut_layer_split_node_assignments_.end() && assignment->second != nullptr) {
                *backend = assignment->second;
                return true;
            }
        }
        return false;
    };
    for (size_t index = 0; index < plan.segments.size(); ++index) {
        // Cut outputs belong to exactly one segment; internal nodes can be
        // shared prelude work that stays unassigned.
        if (!find_assignment(plan.segments[index].output_node_indices, &backends[index])) {
            find_assignment(plan.segments[index].internal_node_indices, &backends[index]);
        }
    }
    return backends;
}

bool GGMLRunner::fits(const std::vector<DeviceMemoryRequest>& requests,
                      const std::vector<ggml_tensor*>& params) const {
    auto manager = residency_manager.lock();
    if (manager == nullptr) {
        return params.empty();
    }
    for (const auto& request : requests) {
        if (!manager->fits_compute_backend_capacity(request, params)) {
            return false;
        }
    }
    return true;
}

bool GGMLRunner::execute_segment(ggml_cgraph* graph, int n_threads) {
    if (sd_backend_is_cpu(runtime_backend)) {
        sd_backend_cpu_set_n_threads(runtime_backend, n_threads);
    }
    if (workspace_.cpu_backend() != nullptr) {
        sd_backend_cpu_set_n_threads(workspace_.cpu_backend(), n_threads);
    }
    auto scheduler = workspace_.scheduler();
    ggml_status status;
    if (scheduler != nullptr) {
        if (sd_get_backend_eval_callback() != nullptr && !multi_device_eval_callback_warned) {
            LOG_WARN("%s: eval callback is not supported with the backend scheduler; ignoring", get_desc().c_str());
            multi_device_eval_callback_warned = true;
        }
        status = ggml_backend_sched_graph_compute(scheduler, graph);
    } else {
        status = sd_backend_graph_compute_with_eval_callback(runtime_backend, graph,
                                                             sd_get_backend_eval_callback(),
                                                             sd_get_backend_eval_callback_data());
    }
    workspace_.synchronize();
    if (status != GGML_STATUS_SUCCESS) {
        last_compute_status_ = status;
        LOG_ERROR("%s compute failed: %s", get_desc().c_str(), ggml_status_to_string(status));
        return false;
    }
    const std::string description = get_desc();
    if (!debug_tensors.empty()) {
        std::unordered_set<const ggml_tensor*> graph_tensors;
        const int leaf_count = ggml_graph_cut::leaf_count(graph);
        const int node_count = ggml_graph_n_nodes(graph);
        graph_tensors.reserve(static_cast<size_t>(leaf_count + node_count));
        for (int index = 0; index < leaf_count; ++index) {
            graph_tensors.insert(ggml_graph_cut::leaf_tensor(graph, index));
        }
        for (int index = 0; index < node_count; ++index) {
            graph_tensors.insert(ggml_graph_node(graph, index));
        }

        for (const auto& entry : debug_tensors) {
            ggml_tensor* tensor = entry.first;
            if (tensor == nullptr || graph_tensors.find(tensor) == graph_tensors.end()) {
                continue;
            }
            ggml_backend_buffer_t buffer =
                tensor->view_src != nullptr ? tensor->view_src->buffer : tensor->buffer;
            if (buffer == nullptr) {
                LOG_WARN("%s skip debug tensor '%s': tensor buffer not set",
                         description.c_str(),
                         entry.second.c_str());
                continue;
            }
            if (tensor->type != GGML_TYPE_F32) {
                LOG_WARN("%s skip debug tensor '%s': only GGML_TYPE_F32 is supported, got %s",
                         description.c_str(),
                         entry.second.c_str(),
                         ggml_type_name(tensor->type));
                continue;
            }
            auto debug_tensor = make_sd_tensor_from_ggml<float>(tensor);
            print_sd_tensor(debug_tensor, false, entry.second.c_str());
        }
    }

    return true;
}

std::optional<Tensor<float>> GGMLRunner::execute_graph(ggml_cgraph* graph, int n_threads, bool no_return, const std::function<bool()>& read_outputs) {
    struct PhaseTiming {
        int64_t measure  = 0;
        int64_t weights  = 0;
        int64_t alloc    = 0;
        int64_t upload   = 0;
        int64_t prefetch = 0;
        int64_t compute  = 0;
        int64_t capture  = 0;
    } timing;
    const int64_t graph_start = ggml_time_us();
    int64_t phase_start       = graph_start;
    auto lap                  = [&phase_start](int64_t& bucket) {
        const int64_t now = ggml_time_us();
        bucket += now - phase_start;
        phase_start = now;
    };

    const auto params       = collect_used_param_tensors(graph);
    const auto& cached_plan = resolve_graph_cut_plan(graph);
    if (!assign_graph_cut_layer_split_backends(graph, cached_plan, params)) {
        return std::nullopt;
    }
    auto full_measurement = measure_cached(graph, cached_plan.compute_buffer_size, cached_plan.layout, SIZE_MAX, false);
    if (full_measurement.buffers.empty()) {
        last_compute_status_ = GGML_STATUS_ALLOC_FAILED;
        return std::nullopt;
    }
    lap(timing.measure);
    auto manager = residency_manager.lock();
    // Multi-device graphs segment too: each segment runs on the device holding
    // its weights and the scheduler copies the residual stream at range
    // boundaries, so a split module can stream weights that do not fit resident.
    const bool segmented = !sd_backend_is_cpu(runtime_backend) &&
                           manager != nullptr && manager->segmented_compute_enabled() &&
                           cached_plan.valid && cached_plan.has_cuts && cached_plan.segments.size() > 1 &&
                           !fits(memory_requests(full_measurement.buffers, {{runtime_backend, cache_.pending_bytes(graph)}}), params);
    ggml_graph_cut::Plan monolithic_plan;
    if (!segmented) {
        monolithic_plan.segments.emplace_back();
        auto& segment               = monolithic_plan.segments.back();
        segment.group_name          = "graph";
        segment.compute_buffer_size = cached_plan.compute_buffer_size;
        segment.internal_node_indices.reserve(ggml_graph_n_nodes(graph));
        segment.input_refs.reserve(ggml_graph_cut::leaf_count(graph));
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            segment.internal_node_indices.push_back(i);
        }
        for (int i = 0; i < ggml_graph_cut::leaf_count(graph); ++i) {
            auto tensor = ggml_graph_cut::leaf_tensor(graph, i);
            ggml_graph_cut::Segment::InputRef input;
            input.leaf_index = i;
            input.type       = canonical_param_tensor(tensor) != nullptr
                                   ? ggml_graph_cut::Segment::INPUT_PARAM
                                   : ggml_graph_cut::Segment::INPUT_EXTERNAL;
            segment.input_refs.push_back(input);
        }
    }
    const auto& plan            = segmented ? cached_plan : monolithic_plan;
    const bool segments_changed = plan.segments.size() != logged_segment_count_;
    if (segments_changed && (segmented || logged_segment_count_ > 1)) {
        LOG_VERBOSE("%s using %zu segment%s", get_desc().c_str(),
                    plan.segments.size(), plan.segments.size() == 1 ? "" : "s");
    }
    const std::vector<ggml_backend_t> plan_segment_backends = segment_backends(plan, graph);
    if (segmented && segments_changed && is_multi_device()) {
        std::map<ggml_backend_t, size_t> segments_per_backend;
        for (ggml_backend_t backend : plan_segment_backends) {
            segments_per_backend[backend]++;
        }
        std::string placement;
        for (const auto& entry : segments_per_backend) {
            placement += sd_format("%s%s: %zu", placement.empty() ? "" : ", ",
                                   ggml_backend_name(entry.first), entry.second);
        }
        LOG_INFO("%s streaming %zu segments across %zu devices (%s)", get_desc().c_str(),
                 plan.segments.size(), segments_per_backend.size(), placement.c_str());
    }
    // Inputs are uploaded before the bindings capture their buffers, so every
    // segment reuses the same device copy instead of re-uploading it.
    const bool inputs_preallocated = segmented && preallocate_graph_inputs(graph);
    lap(timing.upload);
    SegmentGraphBindings bindings(cut_cache_, plan, graph);
    SegmentWeightPipeline weights(manager, runtime_backend, reinterpret_cast<uintptr_t>(this),
                                  graph, plan, params_tensor_set_,
                                  segmented && manager != nullptr && manager->prefetch_enabled(),
                                  plan_segment_backends,
                                  segmented);
    if (cross_step_prefetch_segment_ != SIZE_MAX) {
        if (segmented && cross_step_layout_ == cached_plan.layout) {
            weights.adopt_prefetch(cross_step_prefetch_segment_, std::move(cross_step_prefetch_params_));
            cross_step_prefetch_segment_ = SIZE_MAX;
            cross_step_prefetch_params_.clear();
            cross_step_layout_.clear();
        } else {
            clear_cross_step_prefetch();
        }
    }

    std::map<ggml_backend_t, size_t> peak_compute_bytes;
    auto track_compute_buffer = [&](ggml_backend_t backend) {
        if (backend != nullptr) {
            auto& peak = peak_compute_bytes[backend];
            peak       = std::max(peak, workspace_.bytes(backend));
        }
    };
    std::optional<Tensor<float>> output = Tensor<float>();
    for (size_t index = 0; index < plan.segments.size(); ++index) {
        const auto& segment = plan.segments[index];
        const bool last     = index + 1 == plan.segments.size();
        auto fail_segment   = [&](const char* phase) {
            LOG_ERROR("%s segment %zu/%zu (%s) failed during %s", get_desc().c_str(),
                        index + 1, plan.segments.size(), segment.group_name.c_str(), phase);
            return std::nullopt;
        };
        cut_cache_.prune(segment.live_cut_names);
        bindings.reset(segment);
        if (!bindings.bind_cached_inputs(segment, get_desc().c_str())) {
            return fail_segment("input binding");
        }
        ggml_context* segment_context = nullptr;
        auto segment_graph            = segmented
                                            ? ggml_graph_cut::build_segment_graph(graph, segment, &segment_context)
                                            : graph;
        struct SegmentCleanup {
            GGMLRunner& runner;
            SegmentWeightPipeline& weights;
            SegmentGraphBindings& bindings;
            ggml_context* context;
            ~SegmentCleanup() {
                if (!runner.workspace_.segment_end()) {
                    runner.last_compute_status_ = GGML_STATUS_FAILED;
                }
                bindings.restore();
                weights.segment_end();
                ggml_free(context);
                runner.sync_runtime_residency();
            }
        } segment_cleanup{*this, weights, bindings, segment_context};

        auto measurement = segmented
                               ? measure_cached(segment_graph, segment.compute_buffer_size, cached_plan.layout, index, inputs_preallocated)
                               : full_measurement;
        lap(timing.measure);
        if (!workspace_.prepare(measurement)) {
            last_compute_status_ = GGML_STATUS_ALLOC_FAILED;
            return fail_segment("workspace preparation");
        }
        ggml_backend_t segment_backend = plan_segment_backends[index];
        const size_t cut_bytes         = last ? 0 : cut_cache_.estimate_output_bytes(graph, segment, segment_backend);
        std::map<ggml_backend_t, size_t> new_cache_bytes;
        new_cache_bytes[runtime_backend] = cache_.pending_bytes(segment_graph);
        new_cache_bytes[segment_backend] = add_bytes(new_cache_bytes[segment_backend], cut_bytes);
        auto ensure_capacity             = [&]() {
            sync_runtime_residency();
            auto requests = memory_requests(measurement.buffers, new_cache_bytes);
            if (fits(requests, weights.params(index))) {
                return true;
            }
            if (workspace_.release_excess(measurement)) {
                sync_runtime_residency();
                requests = memory_requests(measurement.buffers, new_cache_bytes);
            }
            const bool ready = weights.ensure_segment_capacity(index, requests);
            if (!ready && manager != nullptr) {
                last_compute_status_ = GGML_STATUS_ALLOC_FAILED;
            }
            return ready;
        };
        std::vector<size_t> reusable_before;
        reusable_before.reserve(measurement.buffers.size());
        for (const auto& size : measurement.buffers) {
            reusable_before.push_back(workspace_.bytes(size.backend));
        }
        if (!weights.segment_start(index, ensure_capacity)) {
            return fail_segment("weight preparation");
        }
        lap(timing.weights);
        // Preparing weights can execute LoRA graphs and reclaim an idle workspace;
        // only then does the capacity check need to run again.
        bool recheck = false;
        if (!workspace_.measurement_matches(segment_graph, measurement)) {
            measurement = measure(segment_graph, segment.compute_buffer_size);
            if (segmented && index < segment_measurements_.size() && cached_plan.layout == measurement_layout_) {
                segment_measurements_[index] = measurement;
            }
            recheck = true;
        }
        for (size_t i = 0; i < measurement.buffers.size() && !recheck; ++i) {
            recheck = workspace_.bytes(measurement.buffers[i].backend) < reusable_before[i];
        }
        if (recheck) {
            if (!workspace_.prepare(measurement)) {
                last_compute_status_ = GGML_STATUS_ALLOC_FAILED;
                return fail_segment("workspace preparation");
            }
            if (!ensure_capacity()) {
                return fail_segment("workspace capacity check");
            }
        }
        lap(timing.measure);
        if (manager != nullptr) {
            for (const auto& size : measurement.buffers) {
                if (size.bytes > workspace_.bytes(size.backend)) {
                    manager->trim_reclaimable_memory(size.backend);
                }
            }
        }
        if (!workspace_.allocate(segment_graph, [&](ggml_backend_sched_t scheduler, ggml_cgraph* current) {
                pin_multi_device_nodes(scheduler, current);
            })) {
            last_compute_status_ = GGML_STATUS_ALLOC_FAILED;
            return fail_segment("workspace allocation");
        }
        for (const auto& size : measurement.buffers) {
            track_compute_buffer(size.backend);
        }
        if (workspace_.scheduler() != nullptr) {
            track_compute_buffer(workspace_.cpu_backend());
        }
        lap(timing.alloc);
        copy_data_to_backend_tensor(segment_graph, false);
        lap(timing.upload);
        weights.enqueue_next(index, memory_requests(measurement.buffers, new_cache_bytes));
        lap(timing.prefetch);
        LOG_DEBUG("%s executing segment %zu/%zu: %s on %s", get_desc().c_str(),
                  index + 1, plan.segments.size(), segment.group_name.c_str(),
                  ggml_backend_name(segment_backend));
        if (!execute_segment(segment_graph, n_threads)) {
            return fail_segment("execution");
        }
        lap(timing.compute);
        if (!cache_.capture(segment_graph) ||
            !cut_cache_.capture(graph, segment, get_desc().c_str(), segment_backend)) {
            return fail_segment("output caching");
        }
        sync_runtime_residency();
        lap(timing.capture);
        if (last) {
            if (read_outputs && !read_outputs()) {
                return fail_segment("output finalization");
            }
            if (!no_return) {
                auto result = ggml_get_tensor(compute_ctx, final_result_name.c_str());
                output      = read_graph_tensor(result, "output");
                if (!output.has_value()) {
                    return fail_segment("output readback");
                }
            }
        }
        if (!workspace_.segment_end()) {
            last_compute_status_ = GGML_STATUS_FAILED;
            return fail_segment("workspace synchronization");
        }
        // Final outputs and their callbacks may still be views of consumed cuts.
        cut_cache_.prune(segment.future_cut_names);
    }
    weights.take_cross_step_prefetch(&cross_step_prefetch_segment_, &cross_step_prefetch_params_);
    if (cross_step_prefetch_segment_ != SIZE_MAX) {
        cross_step_layout_ = cached_plan.layout;
    }
    if (segments_changed || peak_compute_bytes != logged_compute_bytes_) {
        for (const auto& entry : peak_compute_bytes) {
            LOG_VERBOSE("%s compute buffer size: %.2f MB(%s) on %s (peak across %zu segment%s)",
                        get_desc().c_str(), entry.second / (1024.0 * 1024.0),
                        sd_backend_is_cpu(entry.first) ? "RAM" : "VRAM", ggml_backend_name(entry.first),
                        plan.segments.size(), plan.segments.size() == 1 ? "" : "s");
        }
        logged_compute_bytes_ = std::move(peak_compute_bytes);
        logged_segment_count_ = plan.segments.size();
    }
    LOG_DEBUG("%s graph timing (%zu segment%s): measure %.1f ms, weights %.1f ms, alloc %.1f ms, upload %.1f ms, prefetch %.1f ms, compute %.1f ms, capture %.1f ms, total %.1f ms",
              get_desc().c_str(), plan.segments.size(), plan.segments.size() == 1 ? "" : "s",
              timing.measure / 1000.0, timing.weights / 1000.0, timing.alloc / 1000.0, timing.upload / 1000.0,
              timing.prefetch / 1000.0, timing.compute / 1000.0, timing.capture / 1000.0,
              (ggml_time_us() - graph_start) / 1000.0);
    return output;
}
