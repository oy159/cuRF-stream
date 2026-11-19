#pragma once

#include "curf/engine/engine.hpp"
#include "curf/source.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace curf {

    inline constexpr unsigned burst_bins = 1024;
    inline constexpr unsigned burst_block_frames = 262144;
    inline constexpr unsigned burst_power_frames = 2 * burst_bins;
    inline constexpr unsigned burst_classes = 7;

    struct BurstRegion {
        unsigned first = 0, last = 0; // positions within the current input block
        float offset_hz = 0, bandwidth_hz = 0, power_db = 0;
        std::array<float, burst_classes> probabilities{};
    };

    struct BurstConfig {
        engine::EngineConfig engine;
        double seconds = 5;
        double tune_hz = 0;
        double band_hz = 50e6;
        float threshold_db = 12;
        float dc_guard_hz = 120e3f;
        std::vector<double> exclude_hz;
        bool drop_when_full = false;
        unsigned ring_slots = 64;
        std::uint64_t max_blocks = 0;
    };

    struct BurstStats {
        std::uint64_t frames = 0, blocks = 0, regions = 0, skipped = 0, dropped = 0, gaps = 0;
        double wall_s = 0, processing_p99_ms = 0, queue_p99_ms = 0;
    };

    struct BurstPower {
        std::span<const float> windows; // short windows, each with burst_bins FFT-shifted values
        std::span<const float> floor;
        std::span<const bool> valid;
    };

    // The peak spectrum is in FFT order. All views are valid only inside the callback.
    using BurstCallback = std::function<void(std::uint64_t, std::span<const float>, float,
        std::span<const BurstRegion>, std::span<const std::uint8_t>, BurstPower)>;

    BurstStats run_burst_classifier(ISource &, const engine::Model &, const BurstConfig &, const BurstCallback &);

} // namespace curf
