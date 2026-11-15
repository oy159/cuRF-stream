#include "curf/cuda/memory.hpp"
#include "curf/cuda/windows.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <random>
#include <set>
#include <vector>

using namespace curf;

namespace {

    constexpr std::uint32_t S = 2, M = 512, W = 1024;

    struct Scene {
        std::vector<cf32> win;              // [S][W][M]
        std::vector<std::uint32_t> active;  // s * M + k, ascending
    };

    // Noise in every channel (power ~1e-4), a few channels 20 dB louder.
    Scene make_scene(unsigned seed, const std::set<std::uint32_t> &loud) {
        Scene sc;
        sc.win.resize(std::size_t{S} * W * M);
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (std::uint32_t s = 0; s < S; ++s)
            for (std::uint32_t t = 0; t < W; ++t)
                for (std::uint32_t k = 0; k < M; ++k) {
                    const float a = loud.count(s * M + k) ? 0.1f : 0.01f;
                    sc.win[(std::size_t{s} * W + t) * M + k] = {a * nd(rng) * 0.7071f, a * nd(rng) * 0.7071f};
                }
        sc.active.assign(loud.begin(), loud.end());
        return sc;
    }

    std::vector<std::uint32_t> read_active(const cuda::WindowStage &ws) {
        std::uint32_t n = 0;
        CURF_CUDA_CHECK(cudaMemcpy(&n, ws.d_count(), sizeof n, cudaMemcpyDeviceToHost));
        std::vector<std::uint32_t> a(n);
        if (n) CURF_CUDA_CHECK(cudaMemcpy(a.data(), ws.d_active(), n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
        return a;
    }

} // namespace

TEST_CASE("detection finds the loud channels with either compaction", "[cuda][windows]") {
    const std::set<std::uint32_t> loud{0, 5, 6, 7, 300, 511, 512, 513, 800, 1023}; // neighbours at equal level
    const auto sc = make_scene(1, loud);
    cuda::DeviceBuffer<cf32> win(sc.win.size());
    CURF_CUDA_CHECK(cudaMemcpy(win.get(), sc.win.data(), win.bytes(), cudaMemcpyHostToDevice));
    cuda::WindowStage ws(S, M, W);
    for (const auto c : {cuda::Compaction::ballot, cuda::Compaction::cub}) {
        cuda::DetectConfig cfg;
        cfg.compaction = c;
        ws.detect(win.get(), cfg, nullptr);
        CHECK(read_active(ws) == sc.active);

        cfg.mode = cuda::DetectMode::all;
        ws.detect(win.get(), cfg, nullptr);
        CHECK(read_active(ws).size() == S * M);

        cfg.mode = cuda::DetectMode::absolute;
        cfg.absolute_power = 1e-3f; // between noise (1e-4) and loud (1e-2)
        ws.detect(win.get(), cfg, nullptr);
        CHECK(read_active(ws) == sc.active);
    }
}

TEST_CASE("leakage into neighbouring channels is rejected", "[cuda][windows]") {
    // channel 100 strong (power 1e-2), 99 and 101 at -42 dBc (leakage level), 300 a weak real signal (1e-3)
    std::vector<cf32> win(std::size_t{S} * W * M);
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 0.7071f);
    for (std::uint32_t t = 0; t < W; ++t)
        for (std::uint32_t k = 0; k < M; ++k) {
            float a = 0.0001f; // noise, power 1e-8 (the leakage below sits 18 dB above it)
            if (k == 100) a = 0.1f;
            if (k == 99 || k == 101) a = 0.1f * 0.0079f; // -42 dB
            if (k == 300) a = 0.0316f;
            win[std::size_t{t} * M + k] = {a * nd(rng), a * nd(rng)};
        }
    cuda::DeviceBuffer<cf32> d(win.size());
    CURF_CUDA_CHECK(cudaMemcpy(d.get(), win.data(), d.bytes(), cudaMemcpyHostToDevice));
    cuda::WindowStage ws(S, M, W);
    cuda::DetectConfig cfg;
    ws.detect(d.get(), cfg, nullptr);
    CHECK(read_active(ws) == std::vector<std::uint32_t>{100, 300});
    cfg.neighbor_reject_db = 0;
    ws.detect(d.get(), cfg, nullptr);
    CHECK(read_active(ws) == std::vector<std::uint32_t>{99, 100, 101, 300});
}

TEST_CASE("gather transposes and normalises active windows, in chunks", "[cuda][windows]") {
    std::set<std::uint32_t> loud;
    for (std::uint32_t i = 0; i < 70; ++i) loud.insert((i * 37) % (S * M)); // 70 scattered channels
    const auto sc = make_scene(2, loud);
    cuda::DeviceBuffer<cf32> win(sc.win.size());
    CURF_CUDA_CHECK(cudaMemcpy(win.get(), sc.win.data(), win.bytes(), cudaMemcpyHostToDevice));
    cuda::WindowStage ws(S, M, W);
    ws.detect(win.get(), {}, nullptr);
    REQUIRE(read_active(ws) == sc.active);

    const std::uint32_t n = static_cast<std::uint32_t>(sc.active.size());
    cuda::DeviceBuffer<float> out(std::size_t{n} * W * 2);
    ws.gather(win.get(), 0, 40, out.get(), nullptr);                    // two chunks
    ws.gather(win.get(), 40, 64, out.get() + std::size_t{40} * W * 2, nullptr); // asks for more than remain
    std::vector<float> got(out.size());
    CURF_CUDA_CHECK(cudaMemcpy(got.data(), out.get(), out.bytes(), cudaMemcpyDeviceToHost));

    double max_err = 0;
    for (std::uint32_t e = 0; e < n; ++e) {
        const auto id = sc.active[e], s = id / M, k = id % M;
        double p = 0;
        for (std::uint32_t t = 0; t < W; ++t) {
            const auto v = sc.win[(std::size_t{s} * W + t) * M + k];
            p += double(v.re) * v.re + double(v.im) * v.im;
        }
        const double scale = 1.0 / std::sqrt(p / W);
        double rms = 0;
        for (std::uint32_t t = 0; t < W; ++t) {
            const auto v = sc.win[(std::size_t{s} * W + t) * M + k];
            const float re = got[(std::size_t{e} * W + t) * 2], im = got[(std::size_t{e} * W + t) * 2 + 1];
            max_err = std::max({max_err, std::fabs(re - v.re * scale), std::fabs(im - v.im * scale)});
            rms += double(re) * re + double(im) * im;
        }
        CHECK(std::fabs(rms / W - 1.0) < 1e-4); // unit RMS
    }
    CHECK(max_err < 1e-4);
}
