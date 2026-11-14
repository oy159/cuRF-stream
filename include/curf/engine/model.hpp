#pragma once
// cuRF engine model: the fused graph written by model/convert.py (see the format notes there).
// CPU-only: loading, validation and static memory planning do not need CUDA.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace curf::engine {

    inline constexpr std::uint32_t kNoTensor = 0xFFFFFFFFu;

    // Activations are channels-last: [batch][length][channels].
    struct TensorDesc {
        std::uint32_t channels = 0, length = 0;
        std::string name;
        std::size_t elems_per_window() const { return std::size_t{channels} * length; }
    };

    enum class OpKind : std::uint32_t { conv = 0, gap_linear = 1 };

    // conv:       y = act(conv1d(x, w) + bias [+ res]);  w fp16 [cout][kpad], k-index = tap * cin + ci
    // gap_linear: y = w * mean_L(x) + b;                  w fp32 [cout][cin]
    struct OpDesc {
        OpKind kind = OpKind::conv;
        std::uint32_t in = kNoTensor, res = kNoTensor, out = kNoTensor;
        std::uint32_t cin = 0, cout = 0, k = 1, stride = 1, pad = 0, kpad = 0;
        bool relu = false;
        std::uint64_t w_off = 0, b_off = 0;
        std::string name;
    };

    struct Model {
        std::vector<TensorDesc> tensors;
        std::vector<OpDesc> ops;
        std::uint32_t input = 0, output = 0;
        std::vector<std::byte> blob; // weights, 64-byte aligned offsets

        static Model load(const std::filesystem::path &path);
        double flops_per_window() const;
    };

    // Reference data written by convert.py: inputs/labels/fp32 logits for n_full windows, and the
    // fp32 activation of every tensor for the first n_layer windows.
    struct Golden {
        std::uint32_t n_full = 0, n_layer = 0, classes = 0;
        std::vector<float> input;          // [n_full][L][2]
        std::vector<std::int32_t> labels;  // [n_full]
        std::vector<float> logits;         // [n_full][classes]
        std::vector<std::vector<float>> activations; // per tensor, [n_layer][length][channels]

        static Golden load(const std::filesystem::path &path, const Model &model);
    };

    // Static memory plan: every intermediate tensor (not the graph input/output) gets an offset in one
    // arena. Tensors whose lifetimes [producer, last consumer] overlap never share bytes.
    struct MemoryPlan {
        std::vector<std::size_t> offset;   // per tensor, bytes; input/output get kNoOffset
        std::size_t arena_bytes = 0;       // with reuse
        std::size_t no_reuse_bytes = 0;    // sum of all intermediates (for comparison)
        static constexpr std::size_t kNoOffset = ~std::size_t{0};
    };

    // bytes_per_elem: storage size of an activation element (2 for fp16). reuse=false gives every
    // tensor its own region (for layer-by-layer debugging).
    MemoryPlan plan_memory(const Model &model, std::size_t max_batch, std::size_t bytes_per_elem, bool reuse);

} // namespace curf::engine
