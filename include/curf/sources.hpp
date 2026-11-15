#pragma once

#include "curf/source.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

namespace curf {

    struct Tone {
        double frequency_hz = 1e6; // baseband offset, may be negative
        double amplitude = 0.5;    // full scale = 1.0
        double phase = 0.0;        // radians at sample 0
    };

    struct SyntheticConfig {
        StreamFormat format{};
        std::vector<std::vector<Tone>> tones; // per channel; missing channels carry noise only
        double noise_rms = 0.01;              // per I/Q component, full scale = 1.0
        std::uint64_t seed = 1;
        // The source loops over a pregenerated period of this many frames. Tone
        // frequencies are snapped to multiples of sample_rate / period_frames so
        // the loop is phase-continuous.
        std::size_t period_frames = std::size_t{1} << 20;
        // true: pace output to sample_rate (like hardware). false: as fast as
        // memcpy allows, to drive the pipeline beyond USB rates.
        bool realtime = false;
    };

    // Deterministic generator of tones + Gaussian noise in wire format.
    class SyntheticSource final : public ISource {
    public:
        explicit SyntheticSource(SyntheticConfig config);

        StreamFormat format() const override { return config_.format; }
        void start() override;
        ReadResult read(std::span<std::uint8_t> dst, std::uint32_t timeout_ms) override;

        // Tone frequency actually generated after snapping.
        double snapped_frequency(double frequency_hz) const noexcept;
        const std::vector<std::uint8_t> &period() const noexcept { return period_; }

    private:
        SyntheticConfig config_;
        std::vector<std::uint8_t> period_;
        std::uint64_t next_sample_ = 0;
        std::chrono::steady_clock::time_point t0_{};
    };

    struct FileConfig {
        std::filesystem::path path; // raw wire frames, no header
        StreamFormat format{};
        bool loop = false;
    };

    // Replays a raw wire capture (e.g. `och` RX capture in packed12/signed8).
    class FileSource final : public ISource {
    public:
        explicit FileSource(FileConfig config);

        StreamFormat format() const override { return config_.format; }
        ReadResult read(std::span<std::uint8_t> dst, std::uint32_t timeout_ms) override;

    private:
        FileConfig config_;
        std::ifstream file_;
        std::uint64_t next_sample_ = 0;
        std::uint64_t file_frames_ = 0;
    };

} // namespace curf
