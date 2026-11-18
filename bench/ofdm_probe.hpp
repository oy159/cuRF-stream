#pragma once

#include "curf/reference.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <optional>
#include <span>

namespace curf {

    struct OfdmEvidence {
        double offset_hz = 0;
        float contrast = 0;
    };

    // Focus the probe on the strongest wide span in the previous display row.
    inline std::optional<double> ofdm_probe_centre(std::span<const std::uint8_t> hits,
                                                  std::span<const float> power, double rate) {
        const auto channels = hits.size();
        const double step = rate / static_cast<double>(channels);
        const auto max_gap = static_cast<std::size_t>(480e3 / step);
        const auto bin = [channels](std::size_t i) { return (i + channels / 2) % channels; };
        double best_power = 0;
        std::optional<double> centre;
        for (std::size_t x = 0; x < channels;) {
            if (!hits[bin(x)]) { ++x; continue; }
            const auto left = x;
            auto right = left;
            double energy = power[bin(x)];
            for (++x; x < channels && x - right <= max_gap + 1; ++x) {
                if (hits[bin(x)]) {
                    right = x;
                    energy += power[bin(x)];
                }
            }
            if (static_cast<double>(right - left + 1) * step >= 8e6 && energy > best_power) {
                best_power = energy;
                centre = (static_cast<double>(left + right) / 2 - static_cast<double>(channels / 2)) * step;
            }
        }
        return centre;
    }

    // The cyclic prefix repeats the end of the OFDM symbol. Look for several
    // such intervals at the same symbol cadence and for a drop between them.
    // A continuous tone has similar correlation at both timings.
    class OfdmProbe {
        using Complex = std::complex<float>;
        static constexpr std::size_t kInput = 8192, kHalfTaps = 15, kSamples = (kInput - 2 * kHalfTaps) / 2;

    public:
        explicit OfdmProbe(double rate) {
            const double cutoff = 10e6 / rate;
            for (std::size_t j = 0; j < kHalfTaps; ++j) {
                const double t = static_cast<double>(kHalfTaps - j);
                const double window = 0.5 + 0.5 * std::cos(std::numbers::pi * t / kHalfTaps);
                taps_[j] = static_cast<float>(window * std::sin(2 * std::numbers::pi * cutoff * t) / (std::numbers::pi * t));
            }
            taps_.back() = static_cast<float>(2 * cutoff);
        }

        OfdmEvidence observe(std::span<const std::uint8_t> wire, StreamFormat format, std::uint32_t stream, double centre) {
            const auto stride = format.frame_bytes();
            const auto frames = wire.size() / stride;
            const auto stream_offset = stream * wire_sample_bytes(format.wire);
            OfdmEvidence best;
            if (frames < kInput) return best;
            const auto rotation = std::polar(1.0f, static_cast<float>(-2 * std::numbers::pi * centre / format.sample_rate));
            Complex phase{1, 0};
            for (std::size_t k = 0; k < kInput; ++k) {
                const auto z = ref::unpack_sample(wire.data() + k * stride + stream_offset, format.wire);
                mixed_[k] = Complex{z.re, z.im} * phase;
                phase *= rotation;
            }
            for (std::size_t k = 0; k < kSamples; ++k) {
                const auto base = 2 * k;
                auto value = mixed_[base + kHalfTaps] * taps_.back();
                for (std::size_t j = 0; j < kHalfTaps; ++j)
                    value += (mixed_[base + j] + mixed_[base + 2 * kHalfTaps - j]) * taps_[j];
                iq_[k] = value;
            }
            const double rate = format.sample_rate / 2;
            scan_lag<8>(rate, 3.2e-6, std::array{0.4e-6, 0.8e-6}, best);
            scan_lag<6>(rate, 12.8e-6, std::array{0.8e-6, 1.6e-6, 3.2e-6}, best);
            scan_lag<6>(rate, 1.0 / 60e3, std::array{1.0 / (8 * 60e3)}, best);
            best.offset_hz += centre;
            return best;
        }

    private:
        template<std::size_t Symbols, std::size_t N>
        void scan_lag(double rate, double useful, const std::array<double, N> &guards, OfdmEvidence &best) {
            const double delay = rate * useful;
            const int lag = static_cast<int>(delay);
            const auto fraction = static_cast<float>(delay - lag);
            const int count = static_cast<int>(kSamples) - lag - 1;
            if (count <= 0) return;

            for (int i = 0; i < count; ++i) {
                const auto a = iq_[i];
                const auto b = delayed(i + lag, fraction);
                corr_[i + 1] = corr_[i] + std::conj(a) * b;
                power_[i + 1] = power_[i] + std::norm(a) + std::norm(b);
            }

            for (const double guard : guards) {
                const auto width = static_cast<int>(std::lround(0.6 * guard * rate));
                if (width < 7) continue;
                const double period = (useful + guard) * rate;
                std::array<int, Symbols> prefix{}, between{};
                for (std::size_t i = 0; i < Symbols; ++i) {
                    prefix[i] = static_cast<int>(std::lround(i * period));
                    between[i] = static_cast<int>(std::lround((i + 0.5) * period));
                }
                const int limit = count - between.back() - width - 4;
                for (int start = 0; start <= limit; start += 2) {
                    Complex correlation;
                    const float peak = coherence(start, width, prefix, correlation);
                    if (peak < 0.68f) continue;
                    Complex unused;
                    const float contrast = peak - coherence(start, width, between, unused);
                    if (contrast < 0.35f || contrast <= best.contrast) continue;

                    // Phase between adjacent lags gives the centre of the
                    // correlated energy, so the UI can associate it with a band.
                    Complex adjacent, fourth;
                    for (const int shift : prefix) {
                        for (int i = start + shift; i < start + shift + width; ++i) {
                            const auto a = std::conj(iq_[i]);
                            adjacent += a * delayed(i + lag + 1, fraction);
                            fourth += a * delayed(i + lag + 4, fraction);
                        }
                    }
                    if (std::norm(fourth) > 0.56f * std::norm(correlation)) continue;
                    best.offset_hz = std::arg(adjacent * std::conj(correlation)) * rate / (2 * std::numbers::pi);
                    best.contrast = contrast;
                }
            }
        }

        Complex delayed(int i, float fraction) const {
            return iq_[i] + (iq_[i + 1] - iq_[i]) * fraction;
        }

        template<std::size_t N>
        float coherence(int start, int width, const std::array<int, N> &shifts, Complex &correlation) const {
            float energy = 0;
            correlation = {};
            for (const int shift : shifts) {
                const int a = start + shift, b = a + width;
                correlation += corr_[b] - corr_[a];
                energy += power_[b] - power_[a];
            }
            return energy > 0 ? 2 * std::abs(correlation) / energy : 0;
        }

        std::array<Complex, kSamples> iq_{};
        std::array<Complex, kInput> mixed_{};
        std::array<float, kHalfTaps + 1> taps_{};
        std::array<Complex, kSamples + 1> corr_{};
        std::array<float, kSamples + 1> power_{};
    };

} // namespace curf
