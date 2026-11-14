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
        bool autotune = true;             // false: first supported variant in registry order
        int tune_iters = 20;
        bool cuda_graphs = true;
        bool reuse_memory = true;         // false: every tensor keeps its own buffer (layer debugging)
        std::filesystem::path tune_cache; // empty: do not cache
    };

    struct OpChoice {
        int bucket = 0;
        std::string op, variant; // variant empty for non-conv ops
        float us = 0;            // measured time of the chosen variant (0 when not tuned)
        double flops = 0;
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

        const MemoryPlan &memory_plan() const;
        std::size_t device_bytes() const;
        const std::vector<OpChoice> &choices() const; // per bucket, per op

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace curf::engine
