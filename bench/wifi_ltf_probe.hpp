#pragma once

#include "curf/format.hpp"
#include "curf/reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <numbers>
#include <span>
#include <vector>

namespace curf {
    class WifiLtfProbe {
        using Complex = std::complex<float>;
        static constexpr int taps = 17, phases = 16, symbol = 64;
        static constexpr double rate = 20e6;

    public:
        static constexpr unsigned input_frames = 8192;
        struct Match { unsigned sample; float score; };

        explicit WifiLtfProbe(double fs) : stride_(fs / rate) {
            constexpr std::array<int, symbol> ltf{
                0,0,0,0,0,0,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,
                0,1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,1,1,1,0,0,0,0,0};
            constexpr double tau = 2 * std::numbers::pi;
            for (int n = 0; n < symbol; ++n) {
                Complex z{};
                for (int k = 0; k < symbol; ++k)
                    z += std::polar(1.f, static_cast<float>(tau * (k - 32) * n / symbol)) * static_cast<float>(ltf[k]);
                reference_[n] = std::conj(z);
                reference_energy_ += std::norm(z);
            }
            const auto cutoff = 9.4e6 / fs;
            for (int p = 0; p < phases; ++p) {
                float sum = 0;
                for (int k = 0; k < taps; ++k) {
                    const double t = k - taps / 2 - static_cast<double>(p) / phases;
                    const double h = std::abs(t) < 1e-12 ? 2 * cutoff : std::sin(tau * cutoff * t) / (std::numbers::pi * t);
                    const double w = .5 - .5 * std::cos(tau * k / (taps - 1));
                    filter_[p][k] = static_cast<float>(w * h);
                    sum += filter_[p][k];
                }
                for (float &v : filter_[p]) v /= sum;
            }
            out_.resize(static_cast<unsigned>((input_frames - taps) / stride_));
        }

        std::vector<Match> observe(std::span<const std::uint8_t> wire, StreamFormat fmt, double offset) {
            if (wire.size() < input_frames * fmt.frame_bytes()) return {};
            Complex phase(1, 0);
            const auto rotate = std::polar(1.f, static_cast<float>(-2 * std::numbers::pi * offset / fmt.sample_rate));
            for (unsigned i = 0; i < input_frames; ++i) {
                const auto z = ref::unpack_sample(wire.data() + i * fmt.frame_bytes(), fmt.wire);
                base_[i] = Complex(z.re, z.im) * phase;
                phase *= rotate;
            }
            for (unsigned m = 0; m < out_.size(); ++m) {
                const auto t = m * stride_;
                const int i = static_cast<int>(t);
                const int p = static_cast<int>((t - i) * phases);
                Complex v{};
                for (int k = 0; k < taps; ++k) v += base_[i + k] * filter_[p][k];
                out_[m] = v;
            }
            // L-LTF has two consecutive 64-sample copies. Use the repetition
            // both to shortlist positions and to estimate the residual CFO.
            Complex repeat{};
            double e1 = 0, e2 = 0;
            for (int i = 0; i < symbol; ++i) {
                repeat += std::conj(out_[i]) * out_[i + symbol];
                e1 += std::norm(out_[i]); e2 += std::norm(out_[i + symbol]);
            }
            std::vector<Match> result;
            for (unsigned i = 0; i + 2 * symbol < out_.size(); ++i) {
                if (e1 > 1e-9 && e2 > 1e-9 && std::norm(repeat) >= .76 * e1 * e2) {
                    Complex sum{};
                    Complex correction{1, 0};
                    const auto rot = std::polar(1.f, -std::arg(repeat) / symbol);
                    for (int j = 0; j < symbol; ++j) {
                        sum += out_[i + j] * reference_[j] * correction;
                        correction *= rot;
                    }
                    const float score = std::sqrt(static_cast<float>(std::norm(sum) / (e1 * reference_energy_)));
                    if (score >= .61f) {
                        const auto at = static_cast<unsigned>(std::lround(i * stride_ + taps / 2));
                        if (result.empty() || at - result.back().sample > 9e-6 * fmt.sample_rate)
                            result.push_back({at, score});
                        else if (score > result.back().score) result.back() = {at, score};
                    }
                }
                const auto a = out_[i], b = out_[i + symbol], c = out_[i + 2 * symbol];
                repeat += std::conj(b) * c - std::conj(a) * b;
                e1 += std::norm(b) - std::norm(a);
                e2 += std::norm(c) - std::norm(b);
            }
            return result;
        }

    private:
        double stride_;
        float reference_energy_ = 0;
        std::array<Complex, input_frames> base_{};
        std::array<Complex, symbol> reference_{};
        std::array<std::array<float, taps>, phases> filter_{};
        std::vector<Complex> out_;
    };
}
