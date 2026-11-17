#pragma once
// Signal-level stage 2: coarse channels -> persistent carriers -> rational FFT-bank extraction -> engine.
// A ~100 ms packed-wire ring retains the current hop plus three hops of FIR history. One shared 4*hop FFT
// per RX feeds noise-subtracted fine bandwidth estimates, then batched IFFTs grouped by output rate.
// Broad-band geometry uses a separate current-hop spectrum; extraction retains all FIR history.
// The {2^m, 1.5*2^m} rates preserve the run4 training grid (normally 2B..3B). Finite Blackman FIRs and valid
// output intervals remove independent-block seams; the explicit causal delay is 16 output samples.
// Narrow streams keep their centre/rate while collecting consecutive samples; carry is retained across
// 1024-point model windows. Discontinuities reset accumulation. Input is unit RMS IQ plus the BW channel.
// Real-time submission only polls; offline replay may apply bounded back-pressure. Reported processing
// latency starts when the final raw hop reaches the host ring; model observation time is separate.

#include "curf/engine/engine.hpp"
#include "curf/format.hpp"
#include "curf/runtime/classifier.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstddef>
#include <chrono>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace curf {

    struct SignalResult;
    struct SignalBandEstimate {
        std::uint64_t window;
        std::uint32_t stream, track;
        double lo_hz, hi_hz; // request edges, including the coarse guards
        double measured_fc_hz, measured_bw_hz;
        double applied_fc_hz, applied_bw_hz;
        bool current_hop;
    };

    struct SignalStageConfig {
        std::uint32_t channels = 512;          // stage-1 channel grid (M)
        std::uint32_t window_frames = 1024;    // stage-1 window, channel frames (x M/2 raw samples)
        std::uint32_t length = 1024;           // stage-2 model window
        std::uint32_t ring_blocks = 48;        // ~102 ms at the default block size (per RX)
        std::uint32_t max_signals = 64;        // per window, up to 512
        std::uint32_t max_tracks = 256;
        std::size_t scratch_complex = std::size_t{1} << 22; // total IFFT storage per pending window
        std::uint32_t pending = 3;             // windows in flight on the stage-2 stream
        std::uint32_t reclassify_windows = 6;  // a track is classified at most once per this many windows (W >= L)
        std::uint32_t track_timeout_windows = 48; // a track not detected for this long is dropped (~200 ms)
        std::uint32_t edge_guard = 4;          // channels at each band edge never used (analogue filter edge)
        std::function<void(const SignalResult &, std::span<const float>)> on_input; // optional diagnostic [L][3]
        std::function<void(const SignalBandEstimate &, std::span<const float>)> on_band; // optional existing host PSD bins
        bool current_hop_bandwidth = true; // broad requests: estimate from this hop; extraction keeps FIR history
        bool drop_when_busy = true; // offline replay can back-pressure instead of skipping a window
        engine::EngineConfig engine;           // buckets: the largest is the max windows per stage-2 run
    };

    struct SignalResult {
        std::uint64_t window = 0;     // stage-1 window it came from (the last one, for narrow signals)
        std::uint32_t stream = 0, track = 0;
        double fc_hz = 0;             // centre, relative to the tuned frequency
        double bw_hz = 0;             // 99 % occupied bandwidth estimate (the model's side input)
        double fs_hz = 0;             // rate the signal was extracted at
        std::uint32_t cls = 0;
        float prob = 0;
        std::vector<float> probs;     // softmax over the stage-2 classes
        float power_db = 0;           // mean power of the window before normalisation (dBFS)
        double t0_s = 0, t1_s = 0;   // model window in source sample time, FIR delay removed
        double source_time_s = 0;    // end of the final raw hop, including a nonzero source origin or gaps
        double processing_ms = 0;    // final raw hop ready on host -> stage-2 result
        double observation_ms = 0;   // L / extracted rate; reported separately from processing
        float activity_fraction = -1; // fraction of sampled L-point segments above 20% of peak energy; -1: span too short
        bool narrow = false;
    };

    struct SignalStageStats {
        std::uint64_t windows = 0, signals = 0, classified = 0, skipped_busy = 0, skipped_ring = 0, tracks = 0;
        std::uint64_t skipped_capacity = 0, narrow = 0, wide = 0;
        std::uint64_t skipped_signals = 0, skipped_tracks = 0, skipped_workspace = 0;
        std::uint64_t component_probes = 0; // connected wide components rechecked despite old narrow fragments
        cuda::Percentiles narrow_latency_ms, wide_latency_ms;
        std::string to_json() const;
    };

    class SignalStage final : public BlockTap {
    public:
        using Callback = std::function<void(const SignalResult &)>;
        // model: the stage-2 network (input [L][3]); fs: raw sample rate; streams / wire: the classifier's input
        SignalStage(const SignalStageConfig &config, const engine::Model &model, StreamFormat format,
                    std::size_t block_frames, Callback on_result);
        ~SignalStage() override;
        SignalStage(const SignalStage &) = delete;
        SignalStage &operator=(const SignalStage &) = delete;

        // BlockTap: keep a copy of every raw block (compute stream, block order); retire polls finished work.
        void enqueue(std::size_t slot, const std::uint8_t *d_wire, std::size_t frames, std::uint64_t first_sample,
                     cudaStream_t stream) override;
        void retire(std::size_t slot) override;

        // From the classifier's result callback (same thread): stage-1 detections of one window.
        void on_window(std::uint64_t window, std::uint64_t first_sample, const WindowResult *r, std::size_t n,
                       std::chrono::steady_clock::time_point ready_at = std::chrono::steady_clock::now());
        // Delivers finished stage-2 results through the callback; call often (retire does it too).
        void poll();
        // Waits for everything in flight and delivers it (end of run).
        void flush() override;

        SignalStageStats stats() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace curf
