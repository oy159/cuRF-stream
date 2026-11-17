#pragma once

#include "curf/extraction.hpp"

#include <cuda_runtime.h>

#include <memory>

namespace curf::cuda {

    // Finite FIR via overlap-save. One extractor per signal, consecutive planar cf32 blocks.
    // Allocations and cuFFT plans are constructed once; run/reset only enqueue on the caller's stream.
    // Output is unnormalised cf32: gain, phase, time alignment and block seams can all be inspected.
    class SignalExtractor {
    public:
        explicit SignalExtractor(const ExtractionConfig &config);
        ~SignalExtractor();
        SignalExtractor(const SignalExtractor &) = delete;
        SignalExtractor &operator=(const SignalExtractor &) = delete;

        // Rejects discontinuities before queuing work; call reset explicitly after a gap/retune.
        // Use one CUDA stream for the lifetime of an extractor, or externally order different streams.
        void run(const cf32 *input, cf32 *output, std::uint64_t first_sample, cudaStream_t stream);
        void reset(cudaStream_t stream);
        std::size_t fft_frames() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace curf::cuda
