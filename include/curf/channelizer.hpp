#pragma once
// Polyphase channelizer: parameters, prototype filter design and the CPU reference.
//
// M uniformly spaced channels over the input band, 2x oversampled (decimation D = M / 2):
//   channel k is centred on k * fs / M (k > M/2: negative frequencies), output rate fs / D.
//   With fs = 61.44 MS/s, M = 512: 120 kHz spacing, 240 kS/s per channel (= the classifier's input rate).
//
// Definition (frame n of a stream, sample index i = n * D):
//   y_k[n] = sum_{l=0}^{L-1} h[l] * x[nD - l] * exp(-j 2 pi k (nD - l) / M)
// i.e. each channel is shifted to baseband, low-pass filtered by h and decimated. Samples before the
// start of the stream count as zero. The polyphase form computes, per frame,
//   u[m]   = sum_{p=0}^{P-1} h[m + pM] * x[nD - m - pM]          (L = P * M taps, folded)
//   y_k[n] = (-1)^{kn} * sum_m u[m] * exp(+j 2 pi k m / M)      (an inverse M-point DFT)
// and the (-1)^{kn} of odd frames is a circular shift of u by M/2 before the DFT.

#include "curf/format.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace curf {

    struct ChannelizerConfig {
        std::uint32_t channels = 512;      // M (power of two)
        std::uint32_t taps_per_phase = 8;  // P; prototype length L = P * M
        double cutoff = 0.75;              // -6 dB edge in units of the channel spacing fs / M
        double kaiser_beta = 8.0;

        std::uint32_t decimation() const noexcept { return channels / 2; }
        std::size_t taps() const noexcept { return std::size_t{taps_per_phase} * channels; }
    };

    // Kaiser-windowed sinc low-pass, length L, normalised to unit DC gain (a tone at a channel centre comes
    // out of that channel with its input amplitude).
    std::vector<float> design_prototype(const ChannelizerConfig &cfg);

    namespace ref {
        // Direct evaluation of the definition above for frames [0, frames) of x (x[i], i < 0 = 0).
        // out[n * M + k]. O(frames * M * L): for tests only.
        void channelize(const ChannelizerConfig &cfg, const std::vector<float> &h, const cf32 *x, std::size_t samples,
                        std::size_t frames, cf32 *out);
    } // namespace ref

} // namespace curf
