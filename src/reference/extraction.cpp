#include "curf/extraction.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>

namespace curf {

    ExtractionRatio extraction_ratio(std::uint32_t hop, std::uint32_t output) {
        if (!hop || !output || output > hop) throw std::invalid_argument("invalid extraction ratio");
        const auto g = std::gcd(hop, output);
        return {output / g, hop / g};
    }

    std::vector<float> resampling_filter(ExtractionRatio ratio) {
        ExtractionConfig c;
        c.decimation = ratio.down;
        c.block_frames = std::size_t{ratio.down} * 32;
        auto h = extraction_filter(c);
        for (auto &v : h) v *= static_cast<float>(ratio.up);
        return h;
    }

    void ExtractionConfig::validate() const {
        if (!std::isfinite(sample_rate) || sample_rate <= 0 || !std::isfinite(centre_hz) ||
            std::abs(centre_hz) >= sample_rate / 2)
            throw std::invalid_argument("extraction: invalid sample rate or centre");
        if (decimation == 0 || block_frames == 0 || block_frames % decimation != 0 ||
            block_frames > static_cast<std::size_t>(std::numeric_limits<int>::max()) / 2 ||
            history_frames() > static_cast<std::size_t>(std::numeric_limits<int>::max()) / 2)
            throw std::invalid_argument("extraction: invalid decimation or block size");
        // Keep the desired passband inside the input's Nyquist interval.
        if (std::abs(centre_hz) + output_rate() / 4 >= sample_rate / 2)
            throw std::invalid_argument("extraction: desired passband crosses Nyquist");
    }

    std::vector<float> extraction_filter(const ExtractionConfig &cfg) {
        cfg.validate();
        const auto m = cfg.history_frames();
        const double cutoff = 0.375 / cfg.decimation;
        std::vector<float> h(m + 1);
        double sum = 0;
        for (std::size_t k = 0; k <= m; ++k) {
            const double t = static_cast<double>(k) - static_cast<double>(m) / 2;
            const double a = 2 * std::numbers::pi * static_cast<double>(k) / static_cast<double>(m);
            const double win = 0.42 - 0.5 * std::cos(a) + 0.08 * std::cos(2 * a);
            const double v = (t == 0 ? 2 * cutoff : std::sin(2 * std::numbers::pi * cutoff * t) /
                                                               (std::numbers::pi * t)) * win;
            h[k] = static_cast<float>(v);
            sum += v;
        }
        for (auto &v : h) v = static_cast<float>(v / sum);
        return h;
    }

    double extraction_phase_cycles(const ExtractionConfig &cfg, std::uint64_t sample) {
        long double phase = 0, step = static_cast<long double>(cfg.centre_hz) / cfg.sample_rate;
        while (sample) {
            if (sample & 1) phase = std::remainder(phase + step, 1.0L);
            step = std::remainder(2 * step, 1.0L);
            sample >>= 1;
        }
        return static_cast<double>(phase);
    }

    namespace ref {

        void resample_signal(double fs, double centre_hz, ExtractionRatio ratio, std::span<const cf32> input,
                             std::uint64_t first, std::span<cf32> output) {
            if (!ratio.up || !ratio.down || input.size() * ratio.up != output.size() * ratio.down)
                throw std::invalid_argument("resampling reference: invalid shape");
            const auto h = resampling_filter(ratio);
            ExtractionConfig c;
            c.sample_rate = fs; c.centre_hz = centre_hz;
            const long double start = extraction_phase_cycles(c, first), cycles = static_cast<long double>(centre_hz) / fs;
            std::vector<std::complex<double>> mixed(input.size());
            for (std::size_t n = 0; n < input.size(); ++n) {
                const auto phase = static_cast<double>(-2 * std::numbers::pi_v<long double> *
                                                       std::remainder(start + cycles * n, 1.0L));
                mixed[n] = std::complex<double>(input[n].re, input[n].im) * std::polar(1.0, phase);
            }
            for (std::size_t j = 0; j < output.size(); ++j) {
                const auto n = j * ratio.down, end = std::min(n, h.size() - 1);
                std::complex<double> v = 0;
                for (std::size_t k = n % ratio.up; k <= end; k += ratio.up)
                    v += static_cast<double>(h[k]) * mixed[(n - k) / ratio.up];
                output[j] = {static_cast<float>(v.real()), static_cast<float>(v.imag())};
            }
        }

        void extract_signal(const ExtractionConfig &cfg, std::span<const cf32> input,
                            std::uint64_t first_sample, std::span<cf32> output) {
            const auto h = extraction_filter(cfg);
            if (input.size() % cfg.decimation || output.size() != input.size() / cfg.decimation)
                throw std::invalid_argument("extraction reference: invalid capture/output size");
            // Mix each original sample once, then perform a direct real FIR. Long-double phase reduction
            // preserves the sample clock even for timestamps beyond float/double's exact integer range.
            const long double cycles = static_cast<long double>(cfg.centre_hz) / cfg.sample_rate;
            const long double start = extraction_phase_cycles(cfg, first_sample);
            std::vector<std::complex<double>> mixed(input.size());
            for (std::size_t n = 0; n < input.size(); ++n) {
                const double ph = static_cast<double>(-2 * std::numbers::pi_v<long double> *
                                                     std::remainder(start + cycles * n, 1.0L));
                mixed[n] = std::complex<double>(input[n].re, input[n].im) * std::polar(1.0, ph);
            }
            for (std::size_t j = 0; j < output.size(); ++j) {
                const std::size_t n = j * cfg.decimation, last = std::min(n, h.size() - 1);
                std::complex<double> acc = 0;
                for (std::size_t k = 0; k <= last; ++k) acc += static_cast<double>(h[k]) * mixed[n - k];
                output[j] = {static_cast<float>(acc.real()), static_cast<float>(acc.imag())};
            }
        }

    } // namespace ref
} // namespace curf
