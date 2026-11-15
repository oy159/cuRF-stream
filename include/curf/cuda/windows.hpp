#pragma once
// Window stage: from one window of channelizer output to classifier input, on the device.
//
//   win    [streams][W][M] cf32   channelizer output, W frames (one classifier window) of every channel
//   power  [streams][M]           mean |y|^2 per channel over the window
//   floor  [streams]              noise floor: a low percentile of that stream's channel powers
//   active list of (stream * M + channel) with power > floor * threshold, in index order, and its count
//   out    [count][W][2] fp32     the active windows, transposed to time-major and scaled to unit RMS
//                                 (the classifier's training normalisation) - the engine's input layout
//
// Detection modes: relative to the per-stream noise floor (default), an absolute power threshold, or every
// channel (for load tests). Compaction: a hand-written single-block warp-ballot scan, or CUB DeviceSelect.

#include "curf/format.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace curf::cuda {

    enum class DetectMode : std::uint8_t { relative, absolute, all };
    enum class Compaction : std::uint8_t { ballot, cub };

    struct DetectConfig {
        DetectMode mode = DetectMode::relative;
        float threshold_db = 6.f;    // relative: power > floor * 10^(dB/10) (6 dB: full recall at 5-20 dB SNR)
        float percentile = 0.25f;    // noise floor = this quantile of the stream's channel powers
        float absolute_power = 0.f;  // absolute mode: power > this
        // Leakage rejection (relative / absolute modes): a channel whose frequency neighbour is this many dB
        // stronger is not reported. The 2x-oversampled bank puts the adjacent channel centre in the prototype's
        // transition band, so a strong signal shows up ~42 dB down in channels k +- 1. 0 disables.
        float neighbor_reject_db = 20.f;
        Compaction compaction = Compaction::ballot;
    };

    class WindowStage {
    public:
        WindowStage(std::uint32_t streams, std::uint32_t channels, std::uint32_t window);
        ~WindowStage();
        WindowStage(const WindowStage &) = delete;
        WindowStage &operator=(const WindowStage &) = delete;

        // Power, floor, active list and count (device), asynchronously.
        void detect(const cf32 *d_win, const DetectConfig &cfg, cudaStream_t stream);
        // Gather active windows [first, first + n) of the list into d_out ([n][W][2] fp32), unit RMS.
        void gather(const cf32 *d_win, std::uint32_t first, std::uint32_t n, float *d_out, cudaStream_t stream);

        const std::uint32_t *d_count() const noexcept { return d_count_; }
        const std::uint32_t *d_active() const noexcept { return d_active_; }
        const float *d_power() const noexcept { return d_power_; }
        const float *d_floor() const noexcept { return d_floor_; }
        std::uint32_t entries() const noexcept { return streams_ * channels_; }

    private:
        std::uint32_t streams_, channels_, window_;
        float *d_power_ = nullptr, *d_floor_ = nullptr;
        std::uint8_t *d_flags_ = nullptr;
        std::uint32_t *d_active_ = nullptr, *d_count_ = nullptr;
        void *d_cub_temp_ = nullptr;
        std::size_t cub_temp_bytes_ = 0;
    };

} // namespace curf::cuda
