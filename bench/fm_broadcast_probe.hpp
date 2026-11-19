#pragma once

#include "curf/reference.hpp"
#include "curf/runtime/burst_classifier.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <numbers>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

namespace curf {
    inline constexpr double fm_band_low_hz = 87.5e6, fm_band_high_hz = 108e6;

    // Confirm stations after FM discrimination. An RF peak or a high model
    // score alone cannot produce an FM label on the real-time feed.
    class FmBroadcastProbe {
        using Complex = std::complex<float>;
        static constexpr unsigned taps = 65, final_decimation = 3;
        static constexpr unsigned probe_blocks = 2, probe_period = 3, power_group = 8;
        static constexpr unsigned required_evidence = 2;
        static constexpr float probe_floor_ratio = 2.25f, hold_floor_ratio = 2.f;
        static constexpr double raster_hz = 100e3, channel_hz = 180e3, tau = 2 * std::numbers::pi;
        static constexpr std::array<double, 7> pilot_hz{19000, 18300, 18700, 19300, 19700, 17900, 20100};

    public:
        struct Hit { unsigned first, last; double offset_hz; };
        static constexpr double bandwidth_hz = channel_hz;

        FmBroadcastProbe(StreamFormat fmt, double tune, double band_hz)
            : fmt_(fmt), enabled_(tune >= fm_band_low_hz && tune <= fm_band_high_hz &&
                                  fmt.sample_rate >= 40e6 && fmt.sample_rate <= 65e6) {
            if (!enabled_) return;
            const double step = fmt.sample_rate / burst_bins;
            const auto bin = [&](double hz) {
                return std::clamp(static_cast<int>(std::ceil((hz - tune) / step + burst_bins / 2)), 0, int(burst_bins));
            };
            low_ = bin(std::max(fm_band_low_hz, tune - band_hz / 2));
            high_ = bin(std::min(fm_band_high_hz, tune + band_hz / 2) + step / 2);
            decimation_ = static_cast<unsigned>(std::lround(fmt.sample_rate / 960e3));
            for (double hz = fm_band_low_hz; hz <= fm_band_high_hz; hz += raster_hz) {
                if (std::abs(hz - tune) + channel_hz / 2 > band_hz / 2) continue;
                const int lo = bin(hz - channel_hz / 2), hi = bin(hz + channel_hz / 2);
                if (hi - lo < 2) continue;
                const double offset = hz - tune, angle = -tau * offset / fmt.sample_rate;
                Station s{offset, lo, hi, std::polar(1.f, static_cast<float>(angle * decimation_)), {}};
                s.oscillator.resize(decimation_);
                for (unsigned k = 0; k < decimation_; ++k)
                    s.oscillator[k] = std::polar(1.f, static_cast<float>(angle * k));
                stations_.push_back(std::move(s));
            }
            const double intermediate_rate = fmt.sample_rate / decimation_;
            fm_rate_ = intermediate_rate / final_decimation;
            for (unsigned k = 0; k < taps; ++k) {
                const double x = static_cast<double>(k) - (taps - 1) / 2;
                const double cutoff = 101e3 / intermediate_rate;
                const auto sinc = x == 0 ? 2 * cutoff : std::sin(tau * cutoff * x) / (std::numbers::pi * x);
                filter_[k] = static_cast<float>(sinc * (.5 - .5 * std::cos(tau * k / (taps - 1))));
            }
            intermediate_.resize(burst_block_frames * probe_blocks / decimation_ - 1);
            const auto samples = (intermediate_.size() - taps) / final_decimation + 1;
            deviation_.resize(samples - 1);
            window_.resize(deviation_.size());
            for (std::size_t i = 0; i < window_.size(); ++i)
                window_[i] = static_cast<float>(.5 - .5 * std::cos(tau * static_cast<double>(i) / static_cast<double>(window_.size() - 1)));
            window_sum_ = std::accumulate(window_.begin(), window_.end(), 0.);
            for (unsigned k = 0; k < pilot_hz.size(); ++k)
                coefficient_[k] = 2 * std::cos(tau * pilot_hz[k] / fm_rate_);
        }

        bool enabled() const { return enabled_; }
        float noise() const { return noise_; }

        std::vector<Hit> observe(std::uint64_t first, BurstPower power, std::span<const std::uint8_t> wire) {
            if (stations_.empty()) return {};
            if (first != next_block_) {
                for (auto &s : stations_) s.evidence = 0;
                next_station_ = 0;
                probe_slot_ = 0;
            }
            next_block_ = first + burst_block_frames;
            const auto rows = power.windows.size() / burst_bins;
            std::array<float, burst_bins> average{};
            for (std::size_t row = 0; row < rows; ++row)
                for (int k = low_; k < high_; ++k) average[k] += power.windows[row * burst_bins + k];
            std::array<float, burst_bins> sorted = average;
            auto mid = sorted.begin() + (low_ + high_) / 2;
            std::nth_element(sorted.begin() + low_, mid, sorted.begin() + high_);
            noise_ = std::max(*mid / static_cast<float>(rows), 1e-15f);
            for (auto &s : stations_)
                s.power = mean_band(average.data(), s) / static_cast<float>(rows);

            const auto lifetime = static_cast<std::uint64_t>(1.5 * fmt_.sample_rate);
            if (probe_slot_ == 0) {
                // Double the pilot integration time across consecutive blocks.
                // Leave a block between probes for the rest of the receiver.
                probe_station_ = stations_.size();
                for (std::size_t examined = 0; examined < stations_.size(); ++examined) {
                    const auto index = next_station_;
                    next_station_ = (next_station_ + 1) % stations_.size();
                    if (stations_[index].power < probe_floor_ratio * noise_) continue;
                    probe_station_ = index;
                    phase_ = {1, 0};
                    previous_ = sum_ = weighted_ = {};
                    intermediate_count_ = group_fill_ = 0;
                    break;
                }
            }
            if (probe_slot_ < probe_blocks && probe_station_ < stations_.size()) {
                auto &s = stations_[probe_station_];
                mix(wire, s);
                const auto evidence = probe_slot_ + 1 == probe_blocks ? pilot_evidence() : 0u;
                if (evidence) {
                    if (first - s.last_pilot > lifetime) s.evidence = 0;
                    s.evidence = std::min(s.evidence + evidence, required_evidence);
                    s.last_pilot = first;
                    s.confirmed_power = s.power;
                }
            }
            probe_slot_ = (probe_slot_ + 1) % probe_period;

            std::vector<Hit> hits;
            for (const auto &s : stations_) {
                if (s.evidence < required_evidence || first - s.last_pilot > lifetime ||
                    s.power < hold_floor_ratio * noise_) continue;
                const auto threshold = std::max(hold_floor_ratio * noise_, .1f * s.confirmed_power);
                std::size_t start = rows;
                // Average FM sideband power over sub-millisecond groups.
                for (std::size_t r = 0; r < rows;) {
                    const auto end = std::min(r + power_group, rows);
                    float group_power = 0;
                    for (auto i = r; i < end; ++i)
                        group_power += mean_band(power.windows.data() + i * burst_bins, s);
                    const bool on = group_power >= threshold * static_cast<float>(end - r);
                    if (on && start == rows) start = r;
                    if (!on && start != rows) {
                        hits.push_back({static_cast<unsigned>(start * burst_power_frames),
                                       static_cast<unsigned>(r * burst_power_frames), s.offset});
                        start = rows;
                    }
                    r = end;
                }
                if (start != rows)
                    hits.push_back({static_cast<unsigned>(start * burst_power_frames),
                                    static_cast<unsigned>(rows * burst_power_frames), s.offset});
            }
            return hits;
        }

    private:
        struct Station {
            double offset;
            int lo, hi;
            Complex rotation;
            std::vector<Complex> oscillator;
            float power = 0, confirmed_power = 0;
            std::uint64_t last_pilot = 0;
            unsigned evidence = 0;
        };

        static float mean_band(const float *bins, const Station &s) {
            float sum = 0;
            int count = 0;
            for (int k = s.lo; k < s.hi; ++k) {
                if (k == burst_bins / 2) continue; // LO leakage at the exact center bin
                sum += bins[k];
                ++count;
            }
            return sum / static_cast<float>(count);
        }

        void mix(std::span<const std::uint8_t> wire, const Station &s) {
            const auto stride = fmt_.frame_bytes();
            for (unsigned n = 0; n < burst_block_frames;) {
                const auto count = std::min(decimation_ - group_fill_, burst_block_frames - n);
                auto sum = sum_, weighted = weighted_;
                for (unsigned k = 0; k < count; ++k) {
                    const auto z = ref::unpack_sample(wire.data() + (n + k) * stride, fmt_.wire);
                    const auto sample = Complex{z.re, z.im} * s.oscillator[group_fill_ + k];
                    sum += sample;
                    weighted += sample * static_cast<float>(group_fill_ + k + 1);
                }
                group_fill_ += count;
                n += count;
                if (group_fill_ != decimation_) {
                    sum_ = sum;
                    weighted_ = weighted;
                    break;
                }
                // Two adjacent boxcars form a triangular anti-alias filter
                // with state carried across the input block boundary.
                sum *= phase_;
                weighted *= phase_;
                phase_ *= s.rotation;
                if (intermediate_count_)
                    intermediate_[intermediate_count_ - 1] = previous_ + sum * static_cast<float>(decimation_) - weighted;
                ++intermediate_count_;
                previous_ = weighted;
                sum_ = weighted_ = {};
                group_fill_ = 0;
            }
        }

        unsigned pilot_evidence() {
            Complex last{};
            for (std::size_t n = 0; n <= deviation_.size(); ++n) {
                Complex z{};
                const std::size_t start = n * final_decimation;
                for (unsigned k = 0; k < taps; ++k) z += intermediate_[start + k] * filter_[k];
                if (n) deviation_[n - 1] = std::arg(z * std::conj(last)) * static_cast<float>(fm_rate_ / tau);
                last = z;
            }
            const double mean = std::accumulate(deviation_.begin(), deviation_.end(), 0.) / static_cast<double>(deviation_.size());
            // Reject alternate 100 kHz raster centers of the same transmission.
            if (std::abs(mean) >= 20e3) return 0;
            double variance = 0;
            for (std::size_t i = 0; i < deviation_.size(); ++i) {
                const double value = deviation_[i] - mean;
                variance += value * value;
                deviation_[i] = static_cast<float>(value) * window_[i];
            }
            if (variance < 4e6 * static_cast<double>(deviation_.size())) return 0;
            std::array<double, pilot_hz.size()> peaks{};
            for (unsigned k = 0; k < peaks.size(); ++k) {
                double a = 0, b = 0;
                for (float value : deviation_) {
                    const double next = value + coefficient_[k] * a - b;
                    b = a;
                    a = next;
                }
                peaks[k] = a * a + b * b - coefficient_[k] * a * b;
            }
            const auto side = std::span(peaks).subspan(1);
            const double pilot = peaks[0];
            const auto amplitude = 2 * std::sqrt(std::max(0., pilot)) / window_sum_;
            const auto strongest = *std::max_element(side.begin(), side.end());
            const auto side_mean = std::accumulate(side.begin(), side.end(), 0.) / static_cast<double>(side.size());
            // A well-isolated pilot can confirm immediately. Weaker evidence
            // needs a second observation within the tracking interval.
            if (amplitude >= 3500 && pilot >= 32 * strongest && pilot >= 64 * side_mean) return required_evidence;
            if (amplitude >= 1800 && pilot >= 3.25 * strongest && pilot >= 7 * side_mean) return 1;
            return 0;
        }

        StreamFormat fmt_;
        bool enabled_;
        int low_ = 0, high_ = 0;
        unsigned decimation_ = 1, group_fill_ = 0, probe_slot_ = 0;
        std::uint64_t next_block_ = 0;
        std::size_t next_station_ = 0, probe_station_ = 0, intermediate_count_ = 0;
        Complex phase_{}, previous_{}, sum_{}, weighted_{};
        float noise_ = 1e-15f;
        double fm_rate_ = 0, window_sum_ = 0;
        std::array<float, taps> filter_{};
        std::array<double, pilot_hz.size()> coefficient_{};
        std::vector<Station> stations_;
        std::vector<Complex> intermediate_;
        std::vector<float> deviation_, window_;
    };
} // namespace curf
