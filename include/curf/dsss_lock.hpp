#pragma once
#include <span>
#include <vector>

namespace curf {
    // Structural frequency evidence, not a modulation probability or packet/CRC decoder.
    struct DsssFrequencyEstimate {
        float offset_hz = 0;
        float coherence = 0;
        float contrast = 0;
        float retained_power = 0;
        bool valid() const noexcept { return coherence >= .8f && contrast >= .1f && retained_power >= .3f; }
    };
    namespace barker11 {
        inline constexpr unsigned length = 1024, waveform = 660, phases = 256, taps = 32;
        inline constexpr unsigned frequencies = 61, timings = 44, symbols = 14;
        inline constexpr double output_rate = 44e6, cutoff = 8.8e6, step_hz = 50e3;
        inline constexpr int code[11] = {1,1,1,-1,-1,-1,1,-1,-1,1,-1};
        bool supports(double sample_rate) noexcept;
        // Phase-major, unit DC gain, 32-tap Blackman interpolation / antialias filter.
        std::vector<float> interpolation_filter(double sample_rate);
    }
    namespace ref {
        // [1024][3] I/Q + side input. Uses independent double-precision interpolation.
        DsssFrequencyEstimate dsss_frequency(std::span<const float> input, double sample_rate);
    }
}
