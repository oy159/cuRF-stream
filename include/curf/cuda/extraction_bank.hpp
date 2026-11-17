#pragma once
#include "curf/extraction.hpp"
#include <cuda_runtime.h>
#include <memory>
#include <span>

namespace curf::cuda {
    struct ExtractionRequest {
        std::uint32_t stream = 0;
        int centre_bin = 0;             // signed bin in the 4*hop FFT
        std::uint64_t first_sample = 0; // first sample of the valid hop (not of the history)
    };

    // Shared raw FFT + finite FIR filtering/rational decimation + batched small IFFTs.
    // Context is [streams][4*hop]: three hops of history followed by the current hop.
    // Rates W/hop preserve the stage-2 training grid. The first W samples in each 4W output row are valid,
    // baseband, full-scale IQ; the other 3W samples are scratch. Causal group delay: 16 output samples.
    // The response is retained through +/-8 output-rate harmonics; tests quantify truncation vs direct FIR.
    // Plans and work areas are bounded and allocated up front; use one externally ordered stream.
    class ExtractionBank {
    public:
        ExtractionBank(std::uint32_t hop, std::uint32_t streams, double fs);
        ~ExtractionBank();
        void forward(cf32 *context, cudaStream_t stream);
        void extract(const cf32 *spectrum, std::uint32_t w, std::span<const ExtractionRequest> requests,
                     cf32 *output, cudaStream_t stream);
        static std::uint32_t batch_size(std::size_t count);
        static std::uint32_t max_batch(std::uint32_t w); // rate-specific preplanned IFFT budget
    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
}
