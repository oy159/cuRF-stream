#include "curf/channelizer.hpp"

#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>

namespace curf {

    namespace {

        // Modified Bessel function of the first kind, order 0 (series), for the Kaiser window.
        double bessel_i0(double x) {
            double sum = 1.0, term = 1.0;
            for (int k = 1; k < 64; ++k) {
                term *= (x / (2.0 * k)) * (x / (2.0 * k));
                sum += term;
                if (term < 1e-17 * sum) break;
            }
            return sum;
        }

    } // namespace

    std::vector<float> design_prototype(const ChannelizerConfig &cfg) {
        const auto M = cfg.channels;
        if (M < 4 || (M & (M - 1)) != 0) throw std::invalid_argument("channelizer: channels must be a power of two");
        if (cfg.taps_per_phase == 0) throw std::invalid_argument("channelizer: taps_per_phase must be > 0");
        const std::size_t L = cfg.taps();
        const double fc = cfg.cutoff / M; // cycles per input sample at -6 dB
        const double mid = 0.5 * static_cast<double>(L - 1);
        const double i0b = bessel_i0(cfg.kaiser_beta);
        std::vector<double> h(L);
        double sum = 0.0;
        for (std::size_t l = 0; l < L; ++l) {
            const double t = static_cast<double>(l) - mid;
            const double sinc = t == 0.0 ? 2.0 * fc : std::sin(2.0 * std::numbers::pi * fc * t) / (std::numbers::pi * t);
            const double r = 2.0 * t / static_cast<double>(L - 1);
            const double w = bessel_i0(cfg.kaiser_beta * std::sqrt(std::max(0.0, 1.0 - r * r))) / i0b;
            h[l] = sinc * w;
            sum += h[l];
        }
        std::vector<float> out(L);
        for (std::size_t l = 0; l < L; ++l) out[l] = static_cast<float>(h[l] / sum);
        return out;
    }

    namespace ref {

        void channelize(const ChannelizerConfig &cfg, const std::vector<float> &h, const cf32 *x, std::size_t samples,
                        std::size_t frames, cf32 *out) {
            const std::size_t M = cfg.channels, D = cfg.decimation(), L = h.size();
            if (L != cfg.taps()) throw std::invalid_argument("ref::channelize: prototype length mismatch");
            std::vector<std::complex<double>> tw(M); // exp(-j 2 pi r / M)
            for (std::size_t r = 0; r < M; ++r)
                tw[r] = std::polar(1.0, -2.0 * std::numbers::pi * static_cast<double>(r) / static_cast<double>(M));
            for (std::size_t n = 0; n < frames; ++n) {
                const long long i = static_cast<long long>(n * D);
                for (std::size_t k = 0; k < M; ++k) {
                    std::complex<double> acc = 0.0;
                    for (std::size_t l = 0; l < L; ++l) {
                        const long long s = i - static_cast<long long>(l);
                        if (s < 0 || s >= static_cast<long long>(samples)) continue;
                        const auto r = (k * static_cast<std::size_t>(s)) % M; // exp(-j 2 pi k s / M)
                        acc += static_cast<double>(h[l]) * std::complex<double>(x[s].re, x[s].im) * tw[r];
                    }
                    out[n * M + k] = {static_cast<float>(acc.real()), static_cast<float>(acc.imag())};
                }
            }
        }

    } // namespace ref

} // namespace curf
