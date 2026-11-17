#pragma once

#include "curf/format.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace curf {

    // Correctness baseline for one known signal. Mix to DC, finite low-pass, then decimate.
    // Passband covers |f| <= output_rate/4 (a signal of bandwidth <= output_rate/2).
    // Blackman FIR: 32*D+1 taps, cutoff 0.375*output_rate, unit DC gain.
    // A causal implementation: output j corresponds to input first+j*D, with group delay 16*D.
    // This baseline accepts integer D; the experimental stage-2 FFT crop has a separate rate grid.
    struct ExtractionConfig {
        double sample_rate = 61.44e6;
        double centre_hz = 0;
        std::uint32_t decimation = 256;
        std::size_t block_frames = std::size_t{1} << 18;

        void validate() const;
        std::size_t history_frames() const { return std::size_t{32} * decimation; }
        std::size_t output_frames() const { return block_frames / decimation; }
        double output_rate() const { return sample_rate / decimation; }
        std::size_t delay_frames() const { return std::size_t{16} * decimation; }
    };

    std::vector<float> extraction_filter(const ExtractionConfig &cfg);
    // Centre oscillator phase (cycles, modulo one), without multiplying a large timestamp by a float.
    double extraction_phase_cycles(const ExtractionConfig &cfg, std::uint64_t sample);
    // Reduced rational rate W/N for the runtime's {2^m, 3*2^m} output sizes.
    struct ExtractionRatio { std::uint32_t up, down; };
    ExtractionRatio extraction_ratio(std::uint32_t hop, std::uint32_t output);
    std::vector<float> resampling_filter(ExtractionRatio ratio);

    namespace ref {
        // Direct double-precision convolution of the entire continuous capture. Zero history before first.
        // Independent of the GPU's FFT/blocking implementation; output.size() == input.size()/D.
        void extract_signal(const ExtractionConfig &cfg, std::span<const cf32> input,
                            std::uint64_t first_sample, std::span<cf32> output);
        void resample_signal(double fs, double centre_hz, ExtractionRatio ratio, std::span<const cf32> input,
                             std::uint64_t first_sample, std::span<cf32> output);
    }

} // namespace curf
