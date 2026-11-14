#pragma once

// Wideband power spectral density (Welch, Hann window, no overlap) of planar
// complex samples, one PSD per channel per block.
//
//   planar  [channels][frames] cf32      (frames = segments * nfft)
//   psd     [channels][nfft]   float     fft-shifted (DC at nfft/2), linear power,
//                                         normalised so a full-scale tone A*exp(jwt)
//                                         at a bin centre reads A^2.

#include "curf/format.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

#include <cstddef>
#include <cstdint>

namespace curf::cuda {

    class PsdStage {
    public:
        PsdStage(std::uint32_t channels, std::size_t frames, std::size_t nfft);
        ~PsdStage();
        PsdStage(const PsdStage &) = delete;
        PsdStage &operator=(const PsdStage &) = delete;

        // d_scratch must hold channels * frames cf32 (see scratch_elems()).
        void run(const cf32 *d_planar, cf32 *d_scratch, float *d_psd, cudaStream_t stream);

        std::size_t scratch_elems() const noexcept { return std::size_t{channels_} * frames_; }
        std::size_t psd_elems() const noexcept { return std::size_t{channels_} * nfft_; }
        std::size_t segments() const noexcept { return frames_ / nfft_; }

    private:
        std::uint32_t channels_;
        std::size_t frames_, nfft_;
        float *d_window_ = nullptr;
        float norm_ = 1.0f;
        cufftHandle plan_ = 0;
    };

} // namespace curf::cuda
