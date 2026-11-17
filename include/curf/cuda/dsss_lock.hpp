#pragma once
#include "curf/dsss_lock.hpp"
#include "curf/format.hpp"
#include <cuda_runtime.h>
#include <memory>
#include <span>

namespace curf::cuda {
    struct DsssLockRequest { unsigned row, rate; };
    class DsssFrequencyLock {
    public:
        explicit DsssFrequencyLock(std::span<const double> sample_rates);
        ~DsssFrequencyLock();
        unsigned rate_index(double sample_rate) const;
        // Caller-owned pending-slot buffers: count*660 cf32 and count*61 float.
        // Requests refer to rows of the engine's [rows][1024][3] fp32 input.
        void run(const float *input, const DsssLockRequest *requests, unsigned count,
                 cf32 *waveform, float *scores, DsssFrequencyEstimate *output, cudaStream_t stream);
    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
}
