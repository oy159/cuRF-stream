#pragma once
// P3 end-to-end pipeline: wideband samples in, per-channel modulation classes out.
//
//   producer thread: source -> HostRing (pinned, huge pages)
//   consumer thread, all asynchronous on two CUDA streams:
//     copy stream     H2D of each block (wire format)
//     compute stream  channelizer (unpack fused) -> window buffer (ping-pong, one window = W frames)
//                     per completed window: detect + compact -> active count D2H
//                     when the count has arrived (event, no blocking): gather + normalise -> engine, in
//                     chunks of <= the engine's largest bucket -> argmax / softmax -> results D2H
//   results are handed to the callback in window order; latency = last sample of the window in the ring ->
//   its results on the host.
//
// The batch size of each engine run is the number of active channels of that window: the "dynamic batch" is
// decided on the GPU by the detector and read back before the engine launch.

#include "curf/channelizer.hpp"
#include "curf/cuda/pipeline.hpp"
#include "curf/cuda/windows.hpp"
#include "curf/engine/engine.hpp"
#include "curf/source.hpp"

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace curf {

    struct ClassifierConfig {
        ChannelizerConfig channelizer;                 // 512 channels, 8 taps / phase
        std::size_t block_frames = std::size_t{1} << 17; // samples per stream per ring slot
        std::uint32_t window = 1024;                   // frames per classifier window (the model's input length)
        cuda::DetectConfig detect;
        std::vector<float> noise_shape;                // [streams][channels] or [channels]; see WindowStage
        engine::EngineConfig engine;                   // buckets: the largest one is the chunk size
        std::size_t ring_slots = 16;
        std::size_t depth = 4;                         // blocks in flight (H2D / channelizer)
        bool drop_when_full = false;
        double seconds = 5.0;                          // 0 = until the source ends or max_windows
        std::uint64_t max_windows = 0;
        // Optional, called in window order right before the results of the same window: mean power of every
        // channel [streams][channels] (linear, full scale = 1; channel k at k * fs / channels, k >= channels / 2
        // negative) and the per-stream noise floor. Setting it adds a 4 KB D2H per window (for displays).
        std::function<void(std::uint64_t window, const float *power, const float *floor)> on_power;
        // Optional first wire block of each window on the consumer thread. The span is valid during the call.
        std::function<void(std::uint64_t window, std::span<const std::uint8_t>)> on_first_block;
    };

    struct WindowResult {
        std::uint32_t stream = 0, channel = 0;
        std::uint32_t cls = 0;    // argmax class
        float prob = 0;           // softmax probability of that class
        float power = 0;          // mean channel power over the window (linear, full scale = 1)
    };

    struct ClassifierStats {
        std::uint64_t blocks = 0, frames = 0, windows = 0, classified = 0;
        std::uint64_t dropped_frames = 0, gap_frames = 0;
        std::uint32_t max_active = 0;
        double wall_s = 0, msps_per_stream = 0, x_realtime = 0, windows_per_s = 0;
        cuda::Percentiles latency_ms;       // window's last block in the ring -> results on the host
        cuda::Percentiles window_gpu_ms;    // compute-stream time per window: detect -> results D2H done
        double channelize_ms_per_block = 0; // mean, from events
        std::vector<std::uint64_t> class_histogram;
        std::string to_json() const;
    };

    // Called in window order; results valid only during the call.
    using ResultCallback =
            std::function<void(std::uint64_t window, std::uint64_t first_sample, const WindowResult *r, std::size_t n)>;

    ClassifierStats run_classifier(ISource &source, const engine::Model &model, const ClassifierConfig &config,
                                   const ResultCallback &on_results = {});

} // namespace curf
