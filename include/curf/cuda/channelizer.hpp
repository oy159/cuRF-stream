#pragma once
// GPU polyphase channelizer (see curf/channelizer.hpp for the definition).
//
//   fold kernel (FIR + polyphase fold, one thread per phase m, its P taps in registers, consecutive frames)
//   -> batched inverse cuFFT of length M, in place
//
// Input per call: one block of `block_samples` samples per input stream (RX channel), either planar cf32
// ([streams][block_samples], the unpack kernel's output) or raw wire frames, decoded inside the fold kernel
// (unpack fused away: no fp32 intermediate). The last L input samples of each stream are kept on the device,
// so consecutive blocks form one continuous stream.
//
// Output: [streams][frames][M] cf32, frames = block_samples / D; out[s][n][k] = channel k of frame n.

#include "curf/channelizer.hpp"
#include "curf/format.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

#include <cstddef>
#include <cstdint>

namespace curf::cuda {

    enum class FoldVariant : std::uint8_t {
        direct, // each thread reads (and decodes) its taps straight from global memory
        smem,   // the block's input segment is decoded once into shared memory first
        best = smem
    };

    class Channelizer {
    public:
        Channelizer(const ChannelizerConfig &cfg, std::uint32_t streams, std::size_t block_samples);
        ~Channelizer();
        Channelizer(const Channelizer &) = delete;
        Channelizer &operator=(const Channelizer &) = delete;

        void run(const cf32 *d_planar, cf32 *d_out, cudaStream_t stream);
        void run_wire(const std::uint8_t *d_wire, WireFormat format, cf32 *d_out, cudaStream_t stream);
        void reset(); // zero history (stream restarts at sample 0)
        void set_variant(FoldVariant v) noexcept { variant_ = v; }

        const ChannelizerConfig &config() const noexcept { return cfg_; }
        std::uint32_t streams() const noexcept { return streams_; }
        std::size_t frames() const noexcept { return frames_; }
        std::size_t out_elems() const noexcept { return std::size_t{streams_} * frames_ * cfg_.channels; }
        // Real FLOPs per call (fold: 2 real MACs per tap per complex sample; FFT: 5 M log2 M per frame).
        double flops() const noexcept;

    private:
        void fold(const void *in, bool wire, WireFormat format, cf32 *d_out, cudaStream_t stream);

        ChannelizerConfig cfg_;
        std::uint32_t streams_;
        std::size_t block_samples_, frames_;
        float *d_h_ = nullptr;     // prototype [L]
        cf32 *d_hist_ = nullptr;   // [streams][L] last L samples of the previous block
        std::uint64_t frames_done_ = 0;
        FoldVariant variant_ = FoldVariant::best;
        cufftHandle plan_ = 0;
    };

} // namespace curf::cuda
