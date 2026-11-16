#include "curf/adsb.hpp"
#include "curf/cuda/adsb.hpp"
#include "curf/cuda/memory.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <numbers>
#include <random>
#include <string>
#include <vector>

using namespace curf;

namespace {

    std::vector<std::uint8_t> hex(const std::string &s) {
        std::vector<std::uint8_t> v;
        for (std::size_t i = 0; i + 1 < s.size(); i += 2) v.push_back(static_cast<std::uint8_t>(std::stoul(s.substr(i, 2), nullptr, 16)));
        return v;
    }

    struct Burst {
        std::vector<std::uint8_t> msg; // as transmitted
        std::uint64_t start;           // sample of the preamble start
        double amp;
    };

    // PPM at 1 Mbit/s on a carrier `offset` Hz from the centre, complex noise of rms `noise`, packed12 frames of
    // `streams` interleaved streams; the bursts go on stream `target`, the others carry noise only.
    std::vector<std::uint8_t> make_wire(std::size_t n, double fs, double offset, double noise, const std::vector<Burst> &bursts,
                                        std::uint32_t streams, std::uint32_t target) {
        std::vector<double> env(n, 0.0);
        for (const auto &b : bursts) {
            const int bits = static_cast<int>(b.msg.size()) * 8;
            auto pulse = [&](double us) {
                const auto a = static_cast<std::size_t>(std::llround(static_cast<double>(b.start) + us * 1e-6 * fs));
                const auto e = static_cast<std::size_t>(std::llround(static_cast<double>(b.start) + (us + 0.5) * 1e-6 * fs));
                for (std::size_t k = a; k < e && k < n; ++k) env[k] = b.amp;
            };
            for (double us : {0.0, 1.0, 3.5, 4.5}) pulse(us);
            for (int i = 0; i < bits; ++i) {
                const bool one = (b.msg[static_cast<std::size_t>(i >> 3)] >> (7 - (i & 7))) & 1;
                pulse(8.0 + i + (one ? 0.0 : 0.5));
            }
        }
        std::mt19937 rng(7);
        std::normal_distribution<double> nd(0, noise / std::sqrt(2.0));
        std::vector<std::uint8_t> w(n * streams * 3);
        auto put = [&](std::size_t k, std::uint32_t s, double i, double q) {
            auto clip = [](double v) { return static_cast<int>(std::clamp(std::lround(v * 2048), -2048L, 2047L)) & 0xFFF; };
            const int I = clip(i), Q = clip(q);
            std::uint8_t *p = &w[(k * streams + s) * 3];
            p[0] = static_cast<std::uint8_t>(I >> 4);
            p[1] = static_cast<std::uint8_t>(((I & 0xF) << 4) | (Q >> 8));
            p[2] = static_cast<std::uint8_t>(Q & 0xFF);
        };
        for (std::size_t k = 0; k < n; ++k) {
            const double ph = 2 * std::numbers::pi * offset * static_cast<double>(k) / fs + 0.3;
            for (std::uint32_t s = 0; s < streams; ++s) {
                const double a = s == target ? env[k] : 0.0;
                put(k, s, a * std::cos(ph) + nd(rng), a * std::sin(ph) + nd(rng));
            }
        }
        return w;
    }

    std::vector<std::uint8_t> df11(std::uint32_t icao) { // all-call reply with interrogator code 0
        std::vector<std::uint8_t> m = {static_cast<std::uint8_t>(11 << 3 | 5), static_cast<std::uint8_t>(icao >> 16),
                                       static_cast<std::uint8_t>(icao >> 8), static_cast<std::uint8_t>(icao), 0, 0, 0};
        const std::uint32_t crc = adsb::parity_remainder(m.data(), 56);
        m[4] = static_cast<std::uint8_t>(crc >> 16);
        m[5] = static_cast<std::uint8_t>(crc >> 8);
        m[6] = static_cast<std::uint8_t>(crc);
        return m;
    }

} // namespace

TEST_CASE("GPU ADS-B demodulator finds every frame, across block boundaries, with 1-bit correction", "[cuda][adsb]") {
    constexpr double fs = 61.44e6, offset = 10e6;
    constexpr std::size_t block = std::size_t{1} << 17, nblocks = 4;
    const auto id = hex("8D4840D6202CC371C32CE0576098"), pos = hex("8D40621D58C382D690C8AC2863A7");
    const auto vel = hex("8D485020994409940838175B284F");
    auto flipped = vel;
    flipped[9] ^= 0x10; // one bad data bit

    std::vector<Burst> bursts = {
        {id, 10'000, 0.05},
        {pos, block - 3'000, 0.05},          // ends in the second block
        {df11(0x4840D6), 200'000, 0.03},
        {flipped, 2 * block - 4'000, 0.05},  // across the second boundary, needs correction
        {pos, 3 * block + 50'000, 0.02},     // weak: ~-34 dBFS pulses on -40 dBFS wideband noise
    };
    const std::uint32_t streams = 2, target = 1;
    const auto wire = make_wire(block * nblocks, fs, offset, 0.01, bursts, streams, target);

    cuda::AdsbConfig cfg;
    cfg.sample_rate = fs;
    cfg.offset_hz = offset;
    cfg.stream = target;
    cuda::AdsbDemod demod(cfg, WireFormat::packed12, streams, block, 2);
    cuda::DeviceBuffer<std::uint8_t> d(block * streams * 3);
    std::vector<cuda::AdsbFrame> got;
    for (std::size_t b = 0; b < nblocks; ++b) {
        CURF_CUDA_CHECK(cudaMemcpy(d.get(), wire.data() + b * block * streams * 3, d.bytes(), cudaMemcpyHostToDevice));
        demod.enqueue(d.get(), block, b * block, b % 2, nullptr);
        CURF_CUDA_CHECK(cudaDeviceSynchronize());
        const auto r = demod.results(b % 2);
        got.insert(got.end(), r.begin(), r.end());
    }
    // keep one detection per (message, time). Address/parity formats (DF0/4/5/16/20/21) are reported with their
    // remainder for the host to check against known addresses: pulses inside a long frame can look like a
    // preamble and give such garbage, which no known address accepts.
    std::map<std::pair<std::string, std::uint64_t>, cuda::AdsbFrame> uniq;
    for (const auto &f : got) {
        const int df = f.msg[0] >> 3;
        if (df != 11 && df != 17 && df != 18 && f.remainder != 0x4840D6 && f.remainder != 0x40621D && f.remainder != 0x485020)
            continue;
        const std::string key(reinterpret_cast<const char *>(f.msg), f.bits / 8);
        bool dup = false;
        for (const auto &[k, v] : uniq)
            if (k.first == key && (k.second > f.sample ? k.second - f.sample : f.sample - k.second) < 300) dup = true;
        if (!dup) uniq[{key, f.sample}] = f;
    }
    REQUIRE(uniq.size() == bursts.size());
    for (const auto &b : bursts) {
        const bool corrupt = &b == &bursts[3];
        const auto &want = corrupt ? vel : b.msg;
        const std::string key(reinterpret_cast<const char *>(want.data()), want.size());
        bool found = false;
        for (const auto &[k, f] : uniq) {
            if (k.first != key) continue;
            const auto dt = f.sample > b.start ? f.sample - b.start : b.start - f.sample;
            if (dt > 20) continue; // 0.33 us
            found = true;
            CHECK(f.corrected == (corrupt ? 1 : 0));
            CHECK(f.remainder == 0);
            CHECK(std::fabs(f.signal - b.amp) < 0.3 * b.amp);
        }
        CHECK(found);
    }
}

TEST_CASE("GPU ADS-B demodulator reports nothing on noise", "[cuda][adsb]") {
    constexpr double fs = 61.44e6;
    constexpr std::size_t block = std::size_t{1} << 17;
    const auto wire = make_wire(block * 8, fs, 0, 0.02, {}, 1, 0);
    cuda::AdsbConfig cfg;
    cuda::AdsbDemod demod(cfg, WireFormat::packed12, 1, block, 1);
    cuda::DeviceBuffer<std::uint8_t> d(block * 3);
    std::size_t n = 0;
    for (std::size_t b = 0; b < 8; ++b) {
        CURF_CUDA_CHECK(cudaMemcpy(d.get(), wire.data() + b * block * 3, d.bytes(), cudaMemcpyHostToDevice));
        demod.enqueue(d.get(), block, b * block, 0, nullptr);
        CURF_CUDA_CHECK(cudaDeviceSynchronize());
        n += demod.found(0);
    }
    CHECK(n == 0);
    CHECK(demod.noise(0) > 0.f);
}
