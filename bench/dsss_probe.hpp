#pragma once

#include "ofdm_probe.hpp"

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <optional>
#include <span>
#include <vector>

namespace curf {

    inline constexpr unsigned kBarkerChannels = 13;
    inline constexpr double barker_frequency(unsigned index) { return 2412e6 + index * 5e6; }

    inline std::vector<unsigned> barker_probe_channels(std::span<const std::uint8_t> hits,
                                                       std::span<const float> power, double rate, double centre) {
        std::bitset<kBarkerChannels> selected;
        for_each_probe_band(hits, power, rate, [&](double left, double right, double) {
            std::array<unsigned, kBarkerChannels> candidates{};
            std::size_t count = 0;
            for (unsigned i = 0; i < kBarkerChannels; ++i) {
                const double offset = barker_frequency(i) - centre;
                if (offset >= left + 2e6 && offset <= right - 2e6 && std::abs(offset) < rate / 2 - 9e6)
                    candidates[count++] = i;
            }
            const double middle = centre + (left + right) / 2;
            std::sort(candidates.begin(), candidates.begin() + count, [&](unsigned a, unsigned b) {
                return std::abs(barker_frequency(a) - middle) < std::abs(barker_frequency(b) - middle);
            });
            const auto limit = std::min(count, right - left >= 19e6 ? std::size_t{2} : std::size_t{1});
            for (std::size_t i = 0; i < limit; ++i) selected.set(candidates[i]);
        });
        std::vector<unsigned> result;
        for (unsigned i = 0; i < kBarkerChannels; ++i) if (selected.test(i)) result.push_back(i);
        return result;
    }

    class DsssProbe {
        using Complex = std::complex<float>;
        static constexpr unsigned kPhases = 32, kHalfTaps = 12, kTaps = 2 * kHalfTaps + 1;
        static constexpr unsigned kChipSamples = 2, kChips = 11, kSymbol = kChipSamples * kChips, kHalfSymbols = 16;
        static constexpr std::array<int, kChips> kBarker{1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1};

    public:
        static constexpr std::size_t input_frames = 8192;

        explicit DsssProbe(double rate) {
            // Resample the isolated channel to two samples per 11 Mchip/s chip.
            const double cutoff = 10e6 / rate;
            for (unsigned p = 0; p < kPhases; ++p) {
                auto &taps = filter_[p];
                double sum = 0;
                for (unsigned i = 0; i < kTaps; ++i) {
                    const double t = static_cast<double>(i) - kHalfTaps - static_cast<double>(p) / kPhases;
                    const double window = 0.5 - 0.5 * std::cos(2 * std::numbers::pi * i / (kTaps - 1));
                    const double sinc = t == 0 ? 2 * cutoff : std::sin(2 * std::numbers::pi * cutoff * t) / (std::numbers::pi * t);
                    taps[i] = static_cast<float>(window * sinc);
                    sum += taps[i];
                }
                for (float &h : taps) h = static_cast<float>(h / sum);
            }
            const double stride = rate / 22e6;
            for (unsigned j = 0; j * stride + kTaps < input_frames; ++j) {
                const double position = j * stride;
                const auto first = static_cast<std::size_t>(position);
                positions_.push_back({first, static_cast<unsigned>((position - first) * kPhases)});
            }
            waveform_.resize(positions_.size());
        }

        bool observes_barker(std::span<const std::uint8_t> wire, StreamFormat format, unsigned stream, double offset) {
            return barker_first(wire, format, stream, offset).has_value();
        }

        std::optional<unsigned> barker_first(std::span<const std::uint8_t> wire, StreamFormat format,
                                              unsigned stream, double offset) {
            const auto stride = format.frame_bytes();
            if (wire.size() / stride < input_frames) return std::nullopt;
            const auto stream_offset = stream * wire_sample_bytes(format.wire);
            const auto rotation = std::polar(1.0f, static_cast<float>(-2 * std::numbers::pi * offset / format.sample_rate));
            Complex phase{1, 0};
            for (std::size_t k = 0; k < input_frames; ++k) {
                const auto z = ref::unpack_sample(wire.data() + k * stride + stream_offset, format.wire);
                mixed_[k] = Complex{z.re, z.im} * phase;
                phase *= rotation;
            }
            return barker_first_mixed(mixed_);
        }

        std::optional<unsigned> barker_first_mixed(std::span<const Complex, input_frames> mixed) {
            for (std::size_t j = 0; j < positions_.size(); ++j) {
                const auto [first, fraction] = positions_[j];
                Complex value;
                for (unsigned i = 0; i < kTaps; ++i) value += mixed[first + i] * filter_[fraction][i];
                waveform_[j] = value;
            }
            const auto span = kHalfSymbols * kSymbol;
            for (unsigned start = 0; start < kSymbol; ++start) {
                std::array<float, 2> previous{};
                for (std::size_t begin = start; begin + span <= waveform_.size(); begin += span) {
                    std::array<float, 2> projection{};
                    float energy = 0;
                    for (std::size_t symbol = begin; symbol < begin + span; symbol += kSymbol) {
                        std::array<Complex, 2> sums{};
                        for (unsigned chip = 0; chip < kChips; ++chip) {
                            const auto value = waveform_[symbol + chip * kChipSamples];
                            sums[0] += value * static_cast<float>(kBarker[chip]);
                            sums[1] += value * static_cast<float>(kBarker[kChips - 1 - chip]);
                            energy += std::norm(value);
                        }
                        projection[0] += std::norm(sums[0]);
                        projection[1] += std::norm(sums[1]);
                    }
                    // Both consecutive halves must match the same chip order.
                    for (unsigned order = 0; order < 2; ++order) {
                        const float score = energy > 0 ? projection[order] / (kChips * energy) : 0;
                        if (score >= 0.76f && previous[order] >= 0.76f)
                            return static_cast<unsigned>(positions_[begin - span].first + kHalfTaps);
                        previous[order] = score;
                    }
                }
            }
            return std::nullopt;
        }

    private:
        struct Position { std::size_t first; unsigned fraction; };
        std::array<std::array<float, kTaps>, kPhases> filter_{};
        std::vector<Position> positions_;
        std::array<Complex, input_frames> mixed_{};
        std::vector<Complex> waveform_;
    };

} // namespace curf
