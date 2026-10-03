#ifndef __SD_CORE_SEGMENT_WEIGHT_PIPELINE_H__
#define __SD_CORE_SEGMENT_WEIGHT_PIPELINE_H__

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

#include "ggml-backend.h"

struct DeviceMemoryRequest;
struct DeviceResidencyManager;
struct ggml_cgraph;
struct ggml_tensor;

namespace sd::ggml_graph_cut {
    struct Plan;
}

namespace sd {
    class SegmentWeightPipeline {
    public:
        // Time spent by the last segment_start in each of its steps.
        struct StartTiming {
            int64_t activate_us = 0;
            int64_t capacity_us = 0;
            int64_t prepare_us  = 0;
        };

    private:
        std::weak_ptr<DeviceResidencyManager> residency_manager_;
        ggml_backend_t compute_backend_ = nullptr;
        uintptr_t owner_id_             = 0;
        std::vector<std::vector<ggml_tensor*>> segment_params_;
        std::vector<ggml_backend_t> segment_backends_;
        std::vector<ggml_tensor*> queued_params_;
        std::vector<ggml_tensor*> pinned_params_;
        size_t queued_segment_  = SIZE_MAX;
        bool queued_cross_step_ = false;
        bool enabled_           = true;
        bool wrap_prefetch_     = false;
        StartTiming last_start_timing_;

        size_t next_parameter_segment(size_t segment_index) const;
        ggml_backend_t segment_backend(size_t segment_index) const;
        std::vector<std::vector<ggml_tensor*>> preferred_eviction_order() const;
        void disable();
        void activate(size_t segment_index);
        void clear();

    public:
        SegmentWeightPipeline(
            const std::shared_ptr<DeviceResidencyManager>& residency_manager,
            ggml_backend_t compute_backend,
            uintptr_t owner_id,
            ggml_cgraph* graph,
            const ggml_graph_cut::Plan& plan,
            const std::unordered_set<const ggml_tensor*>& params,
            bool enabled                                        = true,
            const std::vector<ggml_backend_t>& segment_backends = {},
            bool wrap_prefetch                                  = false);
        ~SegmentWeightPipeline();

        // A prefetch queued during the last segment targets the first parameter
        // segment of the next graph with the same plan. The runner keeps it
        // alive between graphs and hands it back to the next pipeline.
        void adopt_prefetch(size_t segment_index, std::vector<ggml_tensor*> params);
        void take_cross_step_prefetch(size_t* segment_index, std::vector<ggml_tensor*>* params);

        const std::vector<ggml_tensor*>& params(size_t index) const { return segment_params_[index]; }
        bool ensure_segment_capacity(size_t segment_index,
                                     const std::vector<DeviceMemoryRequest>& requests);
        bool segment_start(size_t segment_index, const std::function<bool()>& ensure_capacity);
        const StartTiming& last_start_timing() const { return last_start_timing_; }
        void segment_end();
        // Prefetch is best effort; segment_start falls back to synchronous loading.
        // The request matching the next segment's device is used; with a single
        // device that is the runner's own request.
        void enqueue_next(size_t segment_index,
                          const std::vector<DeviceMemoryRequest>& requests);
    };
}

#endif  // __SD_CORE_SEGMENT_WEIGHT_PIPELINE_H__
