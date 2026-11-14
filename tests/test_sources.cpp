#include "curf/reference.hpp"
#include "curf/sources.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <vector>

using namespace curf;

namespace {

    SyntheticConfig small_config(std::uint32_t channels = 2) {
        SyntheticConfig c;
        c.format = {WireFormat::packed12, channels, 1e6};
        c.tones = {{{100e3, 0.5}}, {{-250e3, 0.25}}};
        c.noise_rms = 0.001;
        c.period_frames = 4096;
        return c;
    }

    std::vector<std::uint8_t> read_all(ISource &src, std::size_t frames, std::size_t chunk_frames) {
        const std::size_t fb = src.format().frame_bytes();
        std::vector<std::uint8_t> out(frames * fb);
        std::size_t got = 0;
        while (got < frames) {
            const std::size_t n = std::min(chunk_frames, frames - got);
            const auto r = src.read({out.data() + got * fb, n * fb}, 1000);
            REQUIRE(r.status == ReadStatus::ok);
            REQUIRE(r.first_sample == got);
            got += r.frames;
        }
        return out;
    }

} // namespace

TEST_CASE("synthetic source is deterministic and loops seamlessly", "[source]") {
    SyntheticSource a(small_config()), b(small_config());
    REQUIRE(a.period() == b.period());

    // Read 2.5 periods in an awkward chunk size; must equal the period repeated.
    const std::size_t frames = 4096 * 5 / 2;
    const auto data = read_all(a, frames, 777);
    const std::size_t fb = a.format().frame_bytes();
    for (std::size_t k = 0; k < frames; ++k)
        REQUIRE(std::equal(data.begin() + k * fb, data.begin() + (k + 1) * fb,
                           a.period().begin() + (k % 4096) * fb));
}

TEST_CASE("synthetic tones land at the snapped frequency with the right amplitude", "[source]") {
    SyntheticSource src(small_config());
    const std::size_t n = 4096;
    std::vector<cf32> planar(2 * n);
    ref::unpack_planar(src.period().data(), WireFormat::packed12, 2, n, planar.data());

    auto tone_amplitude = [&](std::uint32_t ch, double f) {
        const double w = 2.0 * std::numbers::pi * src.snapped_frequency(f) / 1e6;
        std::complex<double> acc{};
        for (std::size_t k = 0; k < n; ++k) {
            const auto &s = planar[ch * n + k];
            acc += std::complex<double>(s.re, s.im) * std::polar(1.0, -w * static_cast<double>(k));
        }
        return std::abs(acc) / static_cast<double>(n);
    };
    CHECK(tone_amplitude(0, 100e3) == Catch::Approx(0.5).margin(2e-3));
    CHECK(tone_amplitude(1, -250e3) == Catch::Approx(0.25).margin(2e-3));
    CHECK(tone_amplitude(0, -250e3) < 5e-3);
}

TEST_CASE("realtime synthetic source is paced to the sample rate", "[source]") {
    auto cfg = small_config(1);
    cfg.realtime = true;
    SyntheticSource src(cfg);
    src.start();
    const auto t0 = std::chrono::steady_clock::now();
    read_all(src, 20000, 5000); // 20 ms of samples at 1 MS/s
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CHECK(elapsed >= 0.019);
    CHECK(elapsed < 0.5);
}

TEST_CASE("file source replays a capture, optionally looping", "[source]") {
    SyntheticSource gen(small_config());
    const auto path = std::filesystem::temp_directory_path() / "curf_file_source_test.raw";
    {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char *>(gen.period().data()), static_cast<std::streamsize>(gen.period().size()));
    }
    const std::size_t fb = gen.format().frame_bytes();

    SECTION("single pass ends") {
        FileSource src({path, gen.format(), false});
        const auto data = read_all(src, 4096, 1000);
        CHECK(data == gen.period());
        std::vector<std::uint8_t> extra(fb * 10);
        CHECK(src.read(extra, 0).status == ReadStatus::end_of_stream);
    }
    SECTION("loop wraps around") {
        FileSource src({path, gen.format(), true});
        const auto data = read_all(src, 4096 * 2 + 100, 1500);
        for (std::size_t k = 0; k < 4096 * 2 + 100; ++k)
            REQUIRE(std::equal(data.begin() + k * fb, data.begin() + (k + 1) * fb,
                               gen.period().begin() + (k % 4096) * fb));
    }
    std::filesystem::remove(path);
}
