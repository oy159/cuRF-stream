#pragma once

// P1 streaming pipeline:
//
//   producer thread                    consumer (calling) thread
//   source.read -> HostRing slot  ==>  per slot, on one of `depth` CUDA streams:
//                                        H2D (wire) -> unpack -> PSD -> D2H (PSD)
//                                      retire in order: stats, callback, release slot
//
// Up to `depth` blocks are in flight, each with its own stream and device
// buffers, so the H2D of block i+1 overlaps the kernels of block i and the
// D2H of block i-1 (the GPU has separate H2D and D2H copy engines).

#include "curf/cuda/unpack.hpp"
#include "curf/source.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace curf::cuda {

    struct PipelineConfig {
        std::size_t block_frames = std::size_t{1} << 17; // frames per ring slot / GPU block
        std::size_t ring_slots = 32;
        std::size_t depth = 3;            // blocks in flight on the GPU
        std::size_t nfft = 4096;          // PSD size
        bool drop_when_full = false;      // true: drop like a hardware overflow; false: back-pressure the source
        bool register_ring = true;        // false: leave the ring pageable (for comparison)
        UnpackVariant unpack = UnpackVariant::best;
        double seconds = 5.0;             // stop after this long (0 = until the source ends or max_blocks)
        std::uint64_t max_blocks = 0;     // 0 = unlimited
    };

    struct Percentiles {
        double p50 = 0, p99 = 0, p999 = 0, max = 0;
    };

    struct PipelineStats {
        std::uint64_t blocks = 0;
        std::uint64_t frames = 0;
        std::uint64_t dropped_frames = 0; // discarded by the producer because the ring was full
        std::uint64_t gap_frames = 0;     // sample-time gaps seen by the consumer (includes drops)
        double wall_s = 0;
        double wire_GBps = 0;
        double msps_per_channel = 0;
        double x_realtime = 0;            // processing rate / source sample rate
        Percentiles latency_ms;           // slot committed -> results back on the host
        Percentiles h2d_ms, gpu_ms;       // GPU time: H2D copy, whole block (H2D .. D2H)
        double ring_huge_coverage = -1;   // fraction of the ring backed by huge pages (-1 unknown)
        bool ring_registered = false;

        std::string to_json() const;
    };

    // psd: [channels][nfft] on the host, valid only during the call.
    using PsdCallback =
            std::function<void(const float *psd, std::uint32_t channels, std::size_t nfft, std::uint64_t first_sample)>;

    PipelineStats run_pipeline(ISource &source, const PipelineConfig &config, const PsdCallback &on_psd = {});

} // namespace curf::cuda
