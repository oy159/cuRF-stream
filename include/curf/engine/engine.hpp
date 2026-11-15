#pragma once
// cuRF inference engine runtime.
//
//   - one device arena, laid out by a static lifetime-based memory plan: no cudaMalloc on the hot path
//   - fixed batch buckets; each bucket gets its own per-op kernel choice and its own CUDA Graph
//   - kernel registry + autotune: every conv op is timed with every supported variant (optionally
//     restricted by a name prefix), the fastest wins; results can be cached in a file
//
// Usage: write fp32 windows [batch][1024][2] (interleaved I/Q) to input(), call run(batch, stream),
// read fp32 logits [batch][classes] from logits(). Rows between batch and the bucket size are computed
// on stale data and must be ignored.

#include "curf/engine/kernels.hpp"
#include "curf/engine/model.hpp"

#include <cuda_runtime.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace curf::engine {

    struct EngineConfig {
        std::vector<int> buckets{1, 8, 32, 128, 512};
        std::string variant_prefix;       // only conv variants whose name starts with this ("" = all); for ops
                                          // nothing matches, the prefix is cut back one dotted component
        std::vector<std::string> exclude_prefixes; // conv variants never considered (e.g. "direct")
        bool fuse_blocks = true;          // BasicBlock (conv1 + conv2 + shortcut) -> one kernel, per bucket, when
                                          // a block variant fits and (with autotune) is faster than the convs
        std::string block_prefix;         // block variants considered ("" = all)
        bool fuse_gap = true;             // last conv / block -> pooled sums, head reads those
        bool autotune = true;             // false: first supported variant in registry order
        int tune_iters = 20;
        bool cuda_graphs = true;
        bool reuse_memory = true;         // false: every tensor keeps its own buffer (layer debugging)
        std::filesystem::path tune_cache; // empty: do not cache

        // INT8 inference (symmetric, per-tensor activation scales, per-output-channel weight scales). The
        // activation scales come from running the fp16 engine over `calibration` (fp32 windows [n][L][C]):
        // per tensor, the calib_percentile quantile of |v| (1 = the maximum) maps to 127. Requires the GAP
        // fusion (the head reads fp32 pooled sums). BasicBlocks fuse into INT8 block kernels (block_int8.cu).
        bool int8 = false;
        std::vector<float> calibration;
        double calib_percentile = 0.9999;
    };

    struct OpChoice {
        int bucket = 0;
        std::string op, variant; // op: step name ("block:<conv2>" for fused blocks); variant empty for heads
        float us = 0;            // measured time of the chosen variant (0 when not tuned)
        double flops = 0;
    };

    // Per bucket and BasicBlock: whole-network time with this block fused vs run as separate convs
    // (the other blocks as decided so far).
    struct FusionDecision {
        int bucket = 0;
        std::string block;          // conv2 op name
        std::string fused_variant;  // best block variant
        float fused_us = 0, unfused_us = 0; // whole network, per batch
        bool fused = false;
        // INT8 first block: the variant with the stem conv fused in front, and the network time with it
        std::string stem_variant;
        float stem_fused_us = 0;
        bool stem = false;           // chosen (implies fused)
        // INT8 block with a 1x1 shortcut: conv1, then conv2 with the shortcut fused in (a conv variant)
        std::string sc_variant;
        float sc_fused_us = 0;
        bool sc = false;             // chosen (implies fused)
    };

    class Engine {
    public:
        explicit Engine(Model model, EngineConfig cfg = {});
        ~Engine();
        Engine(const Engine &) = delete;
        Engine &operator=(const Engine &) = delete;

        const Model &model() const;
        const EngineConfig &config() const;
        int max_batch() const;
        int bucket_for(int batch) const; // smallest bucket >= batch; throws if batch > max_batch()

        float *input() const;         // device, fp32 [max_batch][L][C]
        const float *logits() const;  // device, fp32 [max_batch][classes]

        void run(int batch, cudaStream_t stream);       // CUDA Graph replay (or eager if graphs are off)
        void run_eager(int batch, cudaStream_t stream); // kernel-by-kernel launches

        // Copy any tensor of the last run to the host as fp32 [batch][length][channels] (synchronous).
        // With reuse_memory=true intermediate tensors may already have been overwritten.
        std::vector<float> read_tensor(std::uint32_t tensor, int batch) const;

        // False for tensors that only exist inside fused kernels (read_tensor throws for them).
        bool materialized(std::uint32_t tensor) const;
        std::size_t steps(int batch) const; // kernel launches per run of batch's bucket (not counting the pool memset)
        std::size_t steps() const { return steps(max_batch()); }
        const std::vector<FusionDecision> &fusion_decisions() const;

        const MemoryPlan &memory_plan() const;
        std::size_t device_bytes() const;
        const std::vector<OpChoice> &choices() const; // per bucket, per op
        const std::vector<float> &activation_scales() const; // INT8: per tensor (0 for the input / output)

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace curf::engine
