#include "curf/reference.hpp"
#include "curf/sources.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numbers>
#include <random>
#include <stdexcept>
#include <thread>

namespace curf {

    SyntheticSource::SyntheticSource(SyntheticConfig config) : config_(std::move(config)) {
        const auto &f = config_.format;
        if (f.channels == 0 || f.sample_rate <= 0 || config_.period_frames == 0)
            throw std::invalid_argument("SyntheticSource: invalid format or period");

        const std::size_t sb = wire_sample_bytes(f.wire);
        period_.resize(config_.period_frames * f.frame_bytes());

        std::mt19937_64 rng(config_.seed);
        std::normal_distribution<double> noise(0.0, config_.noise_rms);
        const double two_pi = 2.0 * std::numbers::pi;

        for (std::uint32_t c = 0; c < f.channels; ++c) {
            const std::vector<Tone> none;
            const auto &tones = c < config_.tones.size() ? config_.tones[c] : none;
            std::vector<double> step(tones.size());
            for (std::size_t t = 0; t < tones.size(); ++t)
                step[t] = two_pi * snapped_frequency(tones[t].frequency_hz) / f.sample_rate;

            for (std::size_t k = 0; k < config_.period_frames; ++k) {
                double re = noise(rng), im = noise(rng);
                for (std::size_t t = 0; t < tones.size(); ++t) {
                    // Phase from an integer product keeps precision over long periods.
                    const double phase = std::fmod(step[t] * static_cast<double>(k), two_pi);
                    re += tones[t].amplitude * std::cos(phase);
                    im += tones[t].amplitude * std::sin(phase);
                }
                ref::pack_sample({static_cast<float>(re), static_cast<float>(im)}, f.wire,
                                 period_.data() + (k * f.channels + c) * sb);
            }
        }
    }

    double SyntheticSource::snapped_frequency(double frequency_hz) const noexcept {
        const double bin = config_.format.sample_rate / static_cast<double>(config_.period_frames);
        return std::round(frequency_hz / bin) * bin;
    }

    void SyntheticSource::start() {
        next_sample_ = 0;
        t0_ = std::chrono::steady_clock::now();
    }

    ReadResult SyntheticSource::read(std::span<std::uint8_t> dst, std::uint32_t timeout_ms) {
        const std::size_t fb = config_.format.frame_bytes();
        std::size_t want = dst.size() / fb;
        if (want == 0) return {};

        if (config_.realtime) {
            using namespace std::chrono;
            if (t0_ == steady_clock::time_point{}) start();
            const auto deadline = steady_clock::now() + milliseconds(timeout_ms);
            const double fs = config_.format.sample_rate;
            auto available = [&] {
                const double elapsed = duration<double>(steady_clock::now() - t0_).count();
                const auto produced = static_cast<std::uint64_t>(elapsed * fs);
                return produced > next_sample_ ? produced - next_sample_ : 0;
            };
            // Like hardware: return once a full request is ready, or whatever is
            // ready at the timeout.
            while (available() < want && steady_clock::now() < deadline)
                std::this_thread::sleep_for(microseconds(200));
            want = std::min<std::size_t>(want, available());
            if (want == 0) return {ReadStatus::timeout, 0, next_sample_, false};
        }

        ReadResult r{ReadStatus::ok, want, next_sample_, false};
        std::size_t pos = static_cast<std::size_t>(next_sample_ % config_.period_frames);
        std::uint8_t *out = dst.data();
        for (std::size_t left = want; left > 0;) {
            const std::size_t n = std::min(left, config_.period_frames - pos);
            std::memcpy(out, period_.data() + pos * fb, n * fb);
            out += n * fb;
            left -= n;
            pos = 0;
        }
        next_sample_ += want;
        return r;
    }

} // namespace curf
