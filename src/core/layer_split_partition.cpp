#include "core/layer_split_partition.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <unordered_set>
#include <utility>

#include "core/util.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

namespace sd {

    static bool layer_split_path_segment_starts_at(const std::string& name, size_t pos) {
        return pos == 0 || name[pos - 1] == '.';
    }

    static bool layer_split_has_path_segment(const std::string& name, const char* segment) {
        size_t pos = name.find(segment);
        while (pos != std::string::npos) {
            if (layer_split_path_segment_starts_at(name, pos)) {
                return true;
            }
            pos = name.find(segment, pos + 1);
        }
        return false;
    }

    int layer_split_tensor_block_index(const std::string& name) {
        static const char* unet_block_segments[] = {"input_blocks.", "output_blocks.", "middle_block.",
                                                    "down_blocks.", "up_blocks.", "mid_block."};
        for (const char* segment : unet_block_segments) {
            if (layer_split_has_path_segment(name, segment)) {
                return -1;
            }
        }

        static const char* block_keywords[] = {"transformer_blocks.", "joint_blocks.", "double_blocks.",
                                               "single_blocks.", "blocks.", "block.", "layers."};
        for (const char* keyword : block_keywords) {
            size_t pos = name.find(keyword);
            while (pos != std::string::npos) {
                if (!layer_split_path_segment_starts_at(name, pos)) {
                    pos = name.find(keyword, pos + 1);
                    continue;
                }
                pos += std::strlen(keyword);
                size_t end = pos;
                while (end < name.size() && name[end] >= '0' && name[end] <= '9') {
                    end++;
                }
                if (end > pos && (end == name.size() || name[end] == '.')) {
                    return std::atoi(name.substr(pos, end - pos).c_str());
                }
                break;
            }
        }
        return -1;
    }

    std::string layer_split_backend_device_display_name(ggml_backend_t backend) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        const char* name       = dev != nullptr ? ggml_backend_dev_name(dev) : ggml_backend_name(backend);
        return name != nullptr ? name : "unknown";
    }

    static size_t graph_cut_layer_split_backend_vram_limit(const std::vector<size_t>& backend_vram_limits,
                                                           size_t backend_index,
                                                           size_t primary_backend_vram_limit) {
        if (backend_index < backend_vram_limits.size()) {
            return backend_vram_limits[backend_index];
        }
        return backend_index == 0 ? primary_backend_vram_limit : 0;
    }

    static std::vector<int64_t> graph_cut_layer_split_backend_capacities(const std::vector<ggml_backend_t>& backends,
                                                                         const std::vector<size_t>& backend_vram_limits,
                                                                         size_t primary_backend_vram_limit) {
        std::vector<int64_t> capacities(backends.size(), std::numeric_limits<int64_t>::max() / 4);
        constexpr int64_t compute_headroom_bytes = 2ll * 1024 * 1024 * 1024;
        for (size_t i = 0; i < backends.size(); i++) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backends[i]);
            size_t free_bytes = 0, total_bytes = 0;
            if (dev != nullptr) {
                ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
            }
            if (free_bytes > 0) {
                capacities[i] = std::max<int64_t>((int64_t)free_bytes - compute_headroom_bytes, 0);
            }
            size_t limit_bytes = graph_cut_layer_split_backend_vram_limit(backend_vram_limits,
                                                                          i,
                                                                          primary_backend_vram_limit);
            if (limit_bytes > 0) {
                capacities[i] = std::min<int64_t>(capacities[i], (int64_t)limit_bytes);
            }
        }
        return capacities;
    }

    bool parse_layer_split_policy(const std::string& spec, LayerSplitPolicy* policy, std::string* error) {
        GGML_ASSERT(policy != nullptr);
        *policy = LayerSplitPolicy{};
        std::string text;
        for (char c : spec) {
            if (!isspace(static_cast<unsigned char>(c))) {
                text += static_cast<char>(tolower(static_cast<unsigned char>(c)));
            }
        }
        if (text.empty() || text == "auto" || text == "cost") {
            policy->mode = LayerSplitPolicy::Mode::AUTO;
            return true;
        }
        if (text == "vram" || text == "memory" || text == "fill") {
            policy->mode = LayerSplitPolicy::Mode::VRAM;
            return true;
        }
        std::stringstream stream(text);
        std::string item;
        float total = 0.f;
        while (std::getline(stream, item, ',')) {
            char* end   = nullptr;
            float value = std::strtof(item.c_str(), &end);
            if (item.empty() || end == nullptr || *end != '\0' || !(value >= 0.f)) {
                if (error != nullptr) {
                    *error = "expected auto, vram, or comma-separated non-negative numbers";
                }
                return false;
            }
            policy->ratios.push_back(value);
            total += value;
        }
        if (policy->ratios.empty() || total <= 0.f) {
            if (error != nullptr) {
                *error = "block fractions must contain at least one positive value";
            }
            return false;
        }
        for (float& ratio : policy->ratios) {
            ratio /= total;
        }
        policy->mode = LayerSplitPolicy::Mode::MANUAL;
        return true;
    }

    static LayerSplitDeviceProfile benchmark_layer_split_device(ggml_backend_t backend) {
        LayerSplitDeviceProfile profile;
        ggml_backend_dev_t device = ggml_backend_get_device(backend);
        if (device == nullptr || ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            return profile;
        }
        ggml_backend_buffer_type_t buffer_type = ggml_backend_get_default_buffer_type(backend);
        if (buffer_type == nullptr) {
            return profile;
        }

        // Matmul throughput: f16 weights x f32 activations, the shape most
        // transformer projections take at inference.
        const int64_t n = 2048;
        ggml_init_params weight_params{4 * ggml_tensor_overhead(), nullptr, true};
        ggml_context* weight_ctx            = ggml_init(weight_params);
        ggml_tensor* a                      = ggml_new_tensor_2d(weight_ctx, GGML_TYPE_F16, n, n);
        ggml_tensor* b                      = ggml_new_tensor_2d(weight_ctx, GGML_TYPE_F32, n, n);
        ggml_backend_buffer_t weight_buffer = ggml_backend_alloc_ctx_tensors(weight_ctx, backend);
        if (weight_buffer == nullptr) {
            ggml_free(weight_ctx);
            return profile;
        }
        ggml_backend_buffer_clear(weight_buffer, 0);
        ggml_init_params graph_params{2 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false), nullptr, true};
        ggml_context* graph_ctx = ggml_init(graph_params);
        ggml_tensor* c          = ggml_mul_mat(graph_ctx, a, b);
        ggml_cgraph* graph      = ggml_new_graph_custom(graph_ctx, 8, false);
        ggml_build_forward_expand(graph, c);
        ggml_gallocr_t allocator = ggml_gallocr_new(buffer_type);
        bool ok                  = allocator != nullptr && ggml_gallocr_alloc_graph(allocator, graph);
        // Idle GPUs sit at low clocks; run until they have ramped up before timing.
        const int64_t warmup_start = ggml_time_us();
        while (ok && ggml_time_us() - warmup_start < 150 * 1000) {
            ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
            ggml_backend_synchronize(backend);
        }
        const int iterations = 8;
        const int64_t t0     = ggml_time_us();
        for (int i = 0; i < iterations && ok; ++i) {
            ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        }
        ggml_backend_synchronize(backend);
        const int64_t t1 = ggml_time_us();
        if (ok && t1 > t0) {
            profile.gflops = iterations * 2.0 * static_cast<double>(n) * n * n / ((t1 - t0) * 1e-6) / 1e9;
        }
        if (allocator != nullptr) {
            ggml_gallocr_free(allocator);
        }
        ggml_free(graph_ctx);
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(weight_ctx);

        // Host-to-device bandwidth from pinned memory, the path weight streaming uses.
        const size_t bytes = 64ULL * 1024ULL * 1024ULL;
        ggml_init_params transfer_params{ggml_tensor_overhead(), nullptr, true};
        ggml_context* transfer_ctx           = ggml_init(transfer_params);
        ggml_tensor* target                  = ggml_new_tensor_1d(transfer_ctx, GGML_TYPE_F32, static_cast<int64_t>(bytes / sizeof(float)));
        ggml_backend_buffer_t target_buffer  = ggml_backend_alloc_ctx_tensors(transfer_ctx, backend);
        ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(device);
        ggml_backend_buffer_t host_buffer    = host_buft != nullptr ? ggml_backend_buft_alloc_buffer(host_buft, bytes) : nullptr;
        std::vector<uint8_t> pageable;
        void* source = nullptr;
        if (host_buffer != nullptr) {
            source = ggml_backend_buffer_get_base(host_buffer);
        } else {
            pageable.resize(bytes);
            source = pageable.data();
        }
        if (target_buffer != nullptr && source != nullptr) {
            ggml_backend_tensor_set(target, source, 0, bytes);
            ggml_backend_synchronize(backend);
            const int copies = 3;
            const int64_t c0 = ggml_time_us();
            for (int i = 0; i < copies; ++i) {
                ggml_backend_tensor_set(target, source, 0, bytes);
            }
            ggml_backend_synchronize(backend);
            const int64_t c1 = ggml_time_us();
            if (c1 > c0) {
                profile.h2d_gbps = copies * static_cast<double>(bytes) / ((c1 - c0) * 1e-6) / 1e9;
            }
        }
        if (host_buffer != nullptr) {
            ggml_backend_buffer_free(host_buffer);
        }
        if (target_buffer != nullptr) {
            ggml_backend_buffer_free(target_buffer);
        }
        ggml_free(transfer_ctx);

        profile.valid = profile.gflops > 0.0 && profile.h2d_gbps > 0.0;
        return profile;
    }

    LayerSplitDeviceProfile layer_split_device_profile(ggml_backend_t backend) {
        static std::mutex mutex;
        static std::map<ggml_backend_dev_t, LayerSplitDeviceProfile> profiles;
        if (backend == nullptr) {
            return {};
        }
        ggml_backend_dev_t device = ggml_backend_get_device(backend);
        std::lock_guard<std::mutex> lock(mutex);
        auto existing = profiles.find(device);
        if (existing != profiles.end()) {
            return existing->second;
        }
        LayerSplitDeviceProfile profile = benchmark_layer_split_device(backend);
        if (profile.valid) {
            LOG_INFO("layer split device profile: %s %.1f TFLOPS (f16 matmul), %.1f GB/s host-to-device",
                     layer_split_backend_device_display_name(backend).c_str(),
                     profile.gflops / 1000.0,
                     profile.h2d_gbps);
        } else {
            LOG_WARN("layer split device profile: could not benchmark %s; falling back to memory-based placement",
                     layer_split_backend_device_display_name(backend).c_str());
        }
        profiles[device] = profile;
        return profile;
    }

    // Floating point work of a segment, counting the ops that dominate a
    // transformer block. Shared prelude nodes are counted by every segment
    // that recomputes them, which matches how they execute.
    static double layer_split_segment_flops(ggml_cgraph* gf, const ggml_graph_cut::Segment& segment) {
        double flops      = 0.0;
        const int n_nodes = ggml_graph_n_nodes(gf);
        for (int index : segment.internal_node_indices) {
            if (index < 0 || index >= n_nodes) {
                continue;
            }
            ggml_tensor* node = ggml_graph_node(gf, index);
            if (node == nullptr) {
                continue;
            }
            switch (node->op) {
                case GGML_OP_MUL_MAT:
                case GGML_OP_MUL_MAT_ID: {
                    ggml_tensor* weight = node->src[0];
                    if (weight == nullptr) {
                        break;
                    }
                    const double k      = static_cast<double>(weight->ne[0]);
                    const double m      = static_cast<double>(node->ne[0]);
                    const double tokens = static_cast<double>(node->ne[1]) * node->ne[2] * node->ne[3];
                    flops += 2.0 * k * m * tokens;
                    break;
                }
                case GGML_OP_FLASH_ATTN_EXT: {
                    ggml_tensor* q = node->src[0];
                    ggml_tensor* k = node->src[1];
                    if (q == nullptr || k == nullptr) {
                        break;
                    }
                    flops += 4.0 * static_cast<double>(q->ne[0]) * q->ne[1] * k->ne[1] * q->ne[2] * q->ne[3];
                    break;
                }
                default:
                    break;
            }
        }
        return flops;
    }

    static double layer_split_segment_output_bytes(ggml_cgraph* gf, const ggml_graph_cut::Segment& segment) {
        double bytes = 0.0;
        for (size_t i = 0; i < segment.output_node_indices.size(); ++i) {
            ggml_tensor* output = ggml_graph_cut::output_tensor(gf, segment, i);
            if (output != nullptr) {
                bytes += static_cast<double>(ggml_nbytes(output));
            }
        }
        return bytes;
    }

    // External inputs (latents, positional tables, conditioning) live on the
    // primary device; a segment on another device copies them every graph.
    static double layer_split_segment_input_bytes(ggml_cgraph* gf, const ggml_graph_cut::Segment& segment) {
        double bytes = 0.0;
        for (const auto& input : segment.input_refs) {
            if (input.type != ggml_graph_cut::Segment::INPUT_EXTERNAL) {
                continue;
            }
            ggml_tensor* tensor = ggml_graph_cut::input_tensor(gf, input);
            if (tensor != nullptr) {
                bytes += static_cast<double>(ggml_nbytes(tensor));
            }
        }
        return bytes;
    }

    static constexpr int64_t LAYER_SPLIT_UNKNOWN_CAPACITY = std::numeric_limits<int64_t>::max() / 4;

    // Legacy placement: fill devices in order by capacity; when the weights
    // exceed the combined budget, spread them proportionally and stream.
    static std::vector<size_t> layer_split_plan_vram_fill(const char* desc,
                                                          const std::vector<int64_t>& segment_param_bytes,
                                                          const std::vector<int64_t>& backend_capacities,
                                                          const std::vector<ggml_backend_t>& split_backends) {
        int64_t total_param_bytes = 0;
        for (int64_t bytes : segment_param_bytes) {
            total_param_bytes += bytes;
        }
        std::vector<int64_t> fill_targets = backend_capacities;
        int64_t total_capacity            = 0;
        bool capacities_known             = true;
        for (int64_t capacity : backend_capacities) {
            if (capacity >= LAYER_SPLIT_UNKNOWN_CAPACITY) {
                capacities_known = false;
                break;
            }
            total_capacity += capacity;
        }
        if (capacities_known && total_capacity > 0 && total_param_bytes > total_capacity) {
            for (size_t i = 0; i < fill_targets.size(); i++) {
                fill_targets[i] = (int64_t)((long double)backend_capacities[i] * (long double)total_param_bytes /
                                            (long double)total_capacity);
            }
            LOG_INFO(
                "%s graph-cut layer split: %.1f MB of weights exceed the combined %.1f MB device budget; "
                "distributing them proportionally and streaming the remainder per segment",
                desc,
                total_param_bytes / (1024.0 * 1024.0),
                total_capacity / (1024.0 * 1024.0));
        }
        std::vector<size_t> device_by_segment(segment_param_bytes.size(), 0);
        size_t current_backend = 0;
        int64_t current_used   = 0;
        bool overflow_logged   = false;
        for (size_t seg_idx = 0; seg_idx < segment_param_bytes.size(); seg_idx++) {
            int64_t bytes = segment_param_bytes[seg_idx];
            while (current_backend + 1 < split_backends.size() && bytes > 0 &&
                   current_used + bytes > fill_targets[current_backend]) {
                current_backend++;
                current_used = 0;
            }
            if (!overflow_logged && bytes > 0 && current_used + bytes > backend_capacities[current_backend]) {
                LOG_VERBOSE("%s graph-cut layer split: from segment %zu, %s holds more weights (%.1f MB) than its %.1f MB budget; the excess is streamed",
                            desc,
                            seg_idx,
                            layer_split_backend_device_display_name(split_backends[current_backend]).c_str(),
                            (current_used + bytes) / (1024.0 * 1024.0),
                            backend_capacities[current_backend] / (1024.0 * 1024.0));
                overflow_logged = true;
            }
            current_used += bytes;
            device_by_segment[seg_idx] = current_backend;
        }
        return device_by_segment;
    }

    static std::vector<size_t> layer_split_plan_manual(const std::vector<int64_t>& segment_param_bytes,
                                                       const std::vector<float>& ratios,
                                                       size_t device_count) {
        size_t param_segments = 0;
        for (int64_t bytes : segment_param_bytes) {
            param_segments += bytes > 0 ? 1 : 0;
        }
        std::vector<size_t> boundaries(device_count, param_segments);
        float cumulative = 0.f;
        for (size_t i = 0; i < device_count; ++i) {
            cumulative += i < ratios.size() ? ratios[i] : 0.f;
            boundaries[i] = std::min(param_segments, static_cast<size_t>(std::floor(cumulative * param_segments + 0.5f)));
        }
        boundaries.back() = param_segments;
        std::vector<size_t> device_by_segment(segment_param_bytes.size(), 0);
        size_t seen   = 0;
        size_t device = 0;
        for (size_t seg_idx = 0; seg_idx < segment_param_bytes.size(); ++seg_idx) {
            if (segment_param_bytes[seg_idx] > 0) {
                while (device + 1 < device_count && seen >= boundaries[device]) {
                    device++;
                }
                seen++;
            }
            device_by_segment[seg_idx] = device;
        }
        return device_by_segment;
    }

    // Cost model: each segment goes where (compute + weight transfer when not
    // resident + residual hop) is cheapest, filling residency in plan order so
    // the tail-first eviction policy keeps exactly the planned blocks resident.
    static bool layer_split_plan_auto(const char* desc,
                                      ggml_cgraph* gf,
                                      const ggml_graph_cut::Plan& plan,
                                      const std::vector<int64_t>& segment_param_bytes,
                                      const std::vector<int64_t>& backend_capacities,
                                      const std::vector<ggml_backend_t>& split_backends,
                                      std::vector<size_t>* device_by_segment_out) {
        std::vector<LayerSplitDeviceProfile> profiles;
        for (ggml_backend_t backend : split_backends) {
            profiles.push_back(layer_split_device_profile(backend));
            if (!profiles.back().valid) {
                return false;
            }
        }
        const size_t segment_count = plan.segments.size();
        std::vector<double> flops(segment_count, 0.0);
        std::vector<double> output_bytes(segment_count, 0.0);
        std::vector<double> input_bytes(segment_count, 0.0);
        double total_flops = 0.0;
        for (size_t i = 0; i < segment_count; ++i) {
            flops[i]        = layer_split_segment_flops(gf, plan.segments[i]);
            output_bytes[i] = layer_split_segment_output_bytes(gf, plan.segments[i]);
            input_bytes[i]  = layer_split_segment_input_bytes(gf, plan.segments[i]);
            total_flops += flops[i];
        }
        if (total_flops <= 0.0) {
            return false;
        }

        std::vector<int64_t> remaining = backend_capacities;
        std::vector<size_t> device_by_segment(segment_count, 0);
        std::vector<size_t> resident_count(split_backends.size(), 0);
        std::vector<size_t> streamed_count(split_backends.size(), 0);
        std::vector<double> device_seconds(split_backends.size(), 0.0);
        double hop_seconds = 0.0;
        size_t previous    = 0;
        for (size_t seg = 0; seg < segment_count; ++seg) {
            const int64_t bytes = segment_param_bytes[seg];
            if (bytes <= 0 && flops[seg] <= 0.0) {
                device_by_segment[seg] = previous;
                continue;
            }
            size_t best        = previous;
            bool best_resident = false;
            double best_cost   = std::numeric_limits<double>::infinity();
            double best_hop    = 0.0;
            for (size_t i = 0; i < split_backends.size(); ++i) {
                const bool resident = bytes <= remaining[i];
                double cost         = flops[seg] / (profiles[i].gflops * 1e9);
                if (!resident) {
                    cost += static_cast<double>(bytes) / (profiles[i].h2d_gbps * 1e9);
                }
                if (i != 0) {
                    cost += input_bytes[seg] / (std::min(profiles[i].h2d_gbps, profiles[0].h2d_gbps) * 1e9);
                }
                double hop = 0.0;
                if (seg > 0 && i != previous) {
                    // A device switch is paid once for the run of segments that
                    // can stay resident there, so amortize the hop over that run.
                    const double link = std::min(profiles[i].h2d_gbps, profiles[previous].h2d_gbps) * 1e9;
                    double run        = 1.0;
                    if (resident && bytes > 0) {
                        run = std::max(1.0, std::min(static_cast<double>(segment_count - seg),
                                                     std::floor(static_cast<double>(remaining[i]) / static_cast<double>(bytes))));
                    }
                    hop = output_bytes[seg - 1] / link / run;
                }
                if (cost + hop < best_cost) {
                    best_cost     = cost + hop;
                    best          = i;
                    best_resident = resident;
                    best_hop      = hop;
                }
            }
            if (best_resident && bytes > 0) {
                remaining[best] -= bytes;
                resident_count[best]++;
            } else if (bytes > 0) {
                streamed_count[best]++;
            }
            device_seconds[best] += best_cost - best_hop;
            hop_seconds += best_hop;
            device_by_segment[seg] = best;
            previous               = best;
        }

        std::string summary;
        double total_seconds = hop_seconds;
        for (size_t i = 0; i < split_backends.size(); ++i) {
            total_seconds += device_seconds[i];
            summary += sd_format("%s%s: %zu resident + %zu streamed segments (~%.0f ms)",
                                 summary.empty() ? "" : ", ",
                                 layer_split_backend_device_display_name(split_backends[i]).c_str(),
                                 resident_count[i],
                                 streamed_count[i],
                                 device_seconds[i] * 1000.0);
        }
        LOG_INFO("%s graph-cut layer split (auto): %s, cross-device hops ~%.0f ms, estimated %.0f ms per graph",
                 desc, summary.c_str(), hop_seconds * 1000.0, total_seconds * 1000.0);
        *device_by_segment_out = std::move(device_by_segment);
        return true;
    }

    bool partition_graph_cut_layer_split(const char* desc,
                                         ggml_cgraph* gf,
                                         const sd::ggml_graph_cut::Plan& plan,
                                         const std::vector<ggml_backend_t>& split_backends,
                                         const std::vector<size_t>& backend_vram_limits,
                                         size_t primary_backend_vram_limit,
                                         std::unordered_map<const ggml_tensor*, ggml_backend_t>& param_assignments,
                                         const std::function<ggml_tensor*(ggml_tensor*)>& canonical_param_tensor,
                                         const LayerSplitPolicy& policy,
                                         GraphCutLayerSplitAssignment* assignment_out) {
        GGML_ASSERT(gf != nullptr);
        GGML_ASSERT(assignment_out != nullptr);
        GGML_ASSERT(canonical_param_tensor != nullptr);
        GGML_ASSERT(!split_backends.empty());

        GraphCutLayerSplitAssignment assignment;
        assignment.segment_count = plan.segments.size();
        assignment.tensors_by_backend.resize(split_backends.size());
        assignment.bytes_by_backend.resize(split_backends.size(), 0);
        assignment.first_segment_by_backend.resize(split_backends.size(), plan.segments.size());
        assignment.last_segment_by_backend.resize(split_backends.size(), 0);

        std::vector<std::vector<ggml_tensor*>> segment_params(plan.segments.size());
        std::vector<int64_t> segment_param_bytes(plan.segments.size(), 0);
        std::unordered_set<ggml_tensor*> seen_params;
        for (size_t seg_idx = 0; seg_idx < plan.segments.size(); seg_idx++) {
            std::vector<ggml_tensor*> params = sd::ggml_graph_cut::param_tensors(gf, plan.segments[seg_idx]);
            for (ggml_tensor* raw_param : params) {
                ggml_tensor* param = canonical_param_tensor(raw_param);
                if (param == nullptr || !seen_params.insert(param).second) {
                    continue;
                }
                segment_params[seg_idx].push_back(param);
                segment_param_bytes[seg_idx] += (int64_t)ggml_nbytes(param);
            }
        }

        int64_t total_param_bytes = 0;
        for (int64_t bytes : segment_param_bytes) {
            total_param_bytes += bytes;
        }
        if (total_param_bytes <= 0) {
            LOG_ERROR("%s graph-cut layer split found no graph params to assign", desc);
            return false;
        }

        std::vector<int64_t> backend_capacities = graph_cut_layer_split_backend_capacities(split_backends,
                                                                                           backend_vram_limits,
                                                                                           primary_backend_vram_limit);
        // Existing placements may already occupy the reported free VRAM. Reuse
        // them; execution checks missing weights and reclaims memory as needed.
        const bool reuse_assignments = std::all_of(seen_params.begin(), seen_params.end(), [&](ggml_tensor* param) {
            return param_assignments.count(param) != 0;
        });

        std::vector<size_t> device_by_segment(plan.segments.size(), 0);
        if (!reuse_assignments) {
            bool planned = false;
            switch (policy.mode) {
                case LayerSplitPolicy::Mode::MANUAL:
                    device_by_segment = layer_split_plan_manual(segment_param_bytes, policy.ratios, split_backends.size());
                    planned           = true;
                    break;
                case LayerSplitPolicy::Mode::AUTO:
                    planned = layer_split_plan_auto(desc, gf, plan, segment_param_bytes, backend_capacities,
                                                    split_backends, &device_by_segment);
                    break;
                case LayerSplitPolicy::Mode::VRAM:
                    break;
            }
            if (!planned) {
                device_by_segment = layer_split_plan_vram_fill(desc, segment_param_bytes, backend_capacities, split_backends);
            }
        }

        std::vector<ggml_backend_t> backend_by_segment(plan.segments.size(), split_backends[0]);
        size_t current_backend = 0;
        for (size_t seg_idx = 0; seg_idx < plan.segments.size(); seg_idx++) {
            if (!reuse_assignments) {
                current_backend = std::min(device_by_segment[seg_idx], split_backends.size() - 1);
            }
            for (ggml_tensor* param : segment_params[seg_idx]) {
                ggml_backend_t target_backend = split_backends[current_backend];
                auto assigned_it              = param_assignments.find(param);
                if (assigned_it == param_assignments.end()) {
                    param_assignments[param]            = target_backend;
                    assignment.has_new_param_assignment = true;
                } else {
                    target_backend = assigned_it->second;
                }

                auto backend_it = std::find(split_backends.begin(), split_backends.end(), target_backend);
                if (backend_it == split_backends.end()) {
                    LOG_ERROR("%s graph-cut layer split tensor '%s' is assigned to an unavailable backend",
                              desc,
                              ggml_get_name(param));
                    return false;
                }
                size_t backend_idx = (size_t)std::distance(split_backends.begin(), backend_it);
                if (reuse_assignments) {
                    current_backend = backend_idx;
                }
                assignment.first_segment_by_backend[backend_idx] = std::min(assignment.first_segment_by_backend[backend_idx], seg_idx);
                assignment.last_segment_by_backend[backend_idx]  = std::max(assignment.last_segment_by_backend[backend_idx], seg_idx + 1);
                assignment.tensors_by_backend[backend_idx].push_back(param);
                assignment.bytes_by_backend[backend_idx] += (int64_t)ggml_nbytes(param);
            }
            backend_by_segment[seg_idx] = split_backends[current_backend];
        }

        // Nodes reachable from several segments (timestep embeddings, modulation
        // tables) are recomputed by each of them; leave them unpinned so the
        // scheduler places each copy next to its consumer instead of on the last
        // segment's device.
        const int n_nodes = ggml_graph_n_nodes(gf);
        std::vector<int> segment_uses(static_cast<size_t>(std::max(n_nodes, 0)), 0);
        for (const auto& segment : plan.segments) {
            for (int node_index : segment.internal_node_indices) {
                if (node_index >= 0 && node_index < n_nodes) {
                    segment_uses[node_index]++;
                }
            }
        }
        for (size_t seg_idx = 0; seg_idx < plan.segments.size(); seg_idx++) {
            ggml_backend_t backend = backend_by_segment[seg_idx];
            const auto& segment    = plan.segments[seg_idx];
            for (int node_index : segment.internal_node_indices) {
                if (node_index < 0 || node_index >= n_nodes || segment_uses[node_index] > 1) {
                    continue;
                }
                ggml_tensor* node = ggml_graph_node(gf, node_index);
                if (node != nullptr) {
                    assignment.node_assignments[node] = backend;
                }
            }
            for (int node_index : segment.output_node_indices) {
                if (node_index < 0 || node_index >= n_nodes) {
                    continue;
                }
                ggml_tensor* node = ggml_graph_node(gf, node_index);
                if (node != nullptr) {
                    assignment.node_assignments[node] = backend;
                }
            }
        }

        *assignment_out = std::move(assignment);
        return true;
    }

    void log_graph_cut_layer_split_assignment(const char* desc,
                                              const std::vector<ggml_backend_t>& split_backends,
                                              const GraphCutLayerSplitAssignment& assignment) {
        for (size_t i = 0; i < split_backends.size(); i++) {
            if (i >= assignment.tensors_by_backend.size() ||
                assignment.tensors_by_backend[i].empty()) {
                continue;
            }
            size_t first_segment = assignment.first_segment_by_backend[i] == assignment.segment_count
                                       ? 0
                                       : assignment.first_segment_by_backend[i];
            size_t last_segment  = assignment.last_segment_by_backend[i];
            if (assignment.has_new_param_assignment) {
                LOG_INFO("%s graph-cut layer split: %s <- segments [%zu, %zu), %zu tensors, %.1f MB",
                         desc,
                         layer_split_backend_device_display_name(split_backends[i]).c_str(),
                         first_segment,
                         last_segment,
                         assignment.tensors_by_backend[i].size(),
                         assignment.bytes_by_backend[i] / (1024.0 * 1024.0));
            } else {
                LOG_VERBOSE("%s graph-cut layer split: %s <- segments [%zu, %zu), %zu tensors, %.1f MB",
                            desc,
                            layer_split_backend_device_display_name(split_backends[i]).c_str(),
                            first_segment,
                            last_segment,
                            assignment.tensors_by_backend[i].size(),
                            assignment.bytes_by_backend[i] / (1024.0 * 1024.0));
            }
        }
    }

}  // namespace sd
