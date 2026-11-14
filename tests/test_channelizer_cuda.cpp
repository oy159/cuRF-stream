#include "curf/channelizer.hpp"
#include "curf/cuda/channelizer.hpp"
#include "curf/cuda/memory.hpp"
#include "curf/reference.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <random>
#include <vector>

using namespace curf;

namespace {

    std::vector<cf32> run_gpu(cuda::Channelizer &ch, const std::vector<cf32> &planar_block) {
        cuda::DeviceBuffer<cf32> in(planar_block.size()), out(ch.out_elems());
        CURF_CUDA_CHECK(cudaMemcpy(in.get(), planar_block.data(), in.bytes(), cudaMemcpyHostToDevice));
        ch.run(in.get(), out.get(), nullptr);
        std::vector<cf32> h(out.size());
        CURF_CUDA_CHECK(cudaMemcpy(h.data(), out.get(), out.bytes(), cudaMemcpyDeviceToHost));
        return h;
    }

    double max_err(const cf32 *a, const cf32 *b, std::size_t n, double &scale) {
        double e = 0;
        scale = 0;
        for (std::size_t i = 0; i < n; ++i) {
            e = std::max(e, static_cast<double>(std::hypot(a[i].re - b[i].re, a[i].im - b[i].im)));
            scale = std::max(scale, static_cast<double>(std::hypot(b[i].re, b[i].im)));
        }
        return e;
    }

} // namespace

TEST_CASE("prototype filter: unit DC gain, symmetric", "[channelizer]") {
    ChannelizerConfig cfg;
    const auto h = design_prototype(cfg);
    REQUIRE(h.size() == 4096);
    double sum = 0;
    for (float v : h) sum += v;
    CHECK(sum == Catch::Approx(1.0).epsilon(1e-5));
    for (std::size_t l = 0; l < h.size(); ++l) CHECK(std::fabs(h[l] - h[h.size() - 1 - l]) < 1e-7f);
}

TEST_CASE("GPU channelizer matches the direct definition across block boundaries", "[cuda][channelizer]") {
    ChannelizerConfig cfg; // 512 channels, 8 taps / phase
    const std::size_t block = 2 * cfg.taps(), frames = block / cfg.decimation();
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 0.3f);
    std::vector<cf32> x(3 * block); // one stream, three consecutive blocks
    for (auto &v : x) v = {nd(rng), nd(rng)};

    std::vector<cf32> want(3 * frames * cfg.channels);
    ref::channelize(cfg, design_prototype(cfg), x.data(), x.size(), 3 * frames, want.data());

    for (const auto variant : {cuda::FoldVariant::direct, cuda::FoldVariant::smem}) {
        cuda::Channelizer ch(cfg, 1, block);
        ch.set_variant(variant);
        for (int b = 0; b < 3; ++b) {
            const auto got = run_gpu(ch, std::vector<cf32>(x.begin() + b * block, x.begin() + (b + 1) * block));
            double scale = 0;
            const double err = max_err(got.data(), want.data() + b * frames * cfg.channels, got.size(), scale);
            INFO("variant " << static_cast<int>(variant) << " block " << b << " max err " << err << " scale " << scale);
            CHECK(err < 2e-5 * std::max(scale, 1.0));
        }
    }
}

TEST_CASE("a tone lands in its channel at full amplitude, neighbours far below", "[cuda][channelizer]") {
    ChannelizerConfig cfg;
    const std::size_t M = cfg.channels, block = std::size_t{1} << 17, frames = block / cfg.decimation();
    const double A = 0.5;
    for (const long k0 : {3L, 200L, 511L}) {
        std::vector<cf32> x(block);
        for (std::size_t i = 0; i < block; ++i) {
            const double ph = 2.0 * std::numbers::pi * static_cast<double>(k0) * static_cast<double>(i) / static_cast<double>(M);
            x[i] = {static_cast<float>(A * std::cos(ph)), static_cast<float>(A * std::sin(ph))};
        }
        cuda::Channelizer ch(cfg, 1, block);
        const auto y = run_gpu(ch, x);
        // skip the first frames (filter fill-up from the zero history)
        auto power = [&](long k) {
            double p = 0;
            for (std::size_t n = 32; n < frames; ++n) {
                const auto v = y[n * M + static_cast<std::size_t>((k + static_cast<long>(M)) % static_cast<long>(M))];
                p += double(v.re) * v.re + double(v.im) * v.im;
            }
            return p / static_cast<double>(frames - 32);
        };
        const double p0 = power(k0);
        INFO("channel " << k0);
        CHECK(std::sqrt(p0) == Catch::Approx(A).epsilon(1e-3));
        CHECK(10 * std::log10(power(k0 + 2) / p0) < -70.0);
        CHECK(10 * std::log10(power(k0 - 2) / p0) < -70.0);
        CHECK(10 * std::log10(power(k0 + 100) / p0) < -90.0);
    }
}

TEST_CASE("fused wire decode matches unpack + planar channelizer", "[cuda][channelizer]") {
    ChannelizerConfig cfg;
    const std::size_t block = std::size_t{1} << 15;
    const std::uint32_t streams = 2;
    for (const auto fmt : {WireFormat::packed12, WireFormat::signed8}) {
        std::mt19937 rng(11);
        std::normal_distribution<float> nd(0.f, 0.25f);
        const auto bytes = wire_sample_bytes(fmt);
        std::vector<std::uint8_t> wire(block * streams * bytes);
        for (std::size_t i = 0; i < block * streams; ++i) ref::pack_sample({nd(rng), nd(rng)}, fmt, &wire[i * bytes]);
        std::vector<cf32> planar(block * streams);
        ref::unpack_planar(wire.data(), fmt, streams, block, planar.data());

        cuda::Channelizer a(cfg, streams, block), b(cfg, streams, block);
        cuda::DeviceBuffer<std::uint8_t> dw(wire.size());
        cuda::DeviceBuffer<cf32> dp(planar.size()), oa(a.out_elems()), ob(b.out_elems());
        CURF_CUDA_CHECK(cudaMemcpy(dw.get(), wire.data(), wire.size(), cudaMemcpyHostToDevice));
        CURF_CUDA_CHECK(cudaMemcpy(dp.get(), planar.data(), dp.bytes(), cudaMemcpyHostToDevice));
        for (int rep = 0; rep < 2; ++rep) { // second call exercises the history of each path
            a.run_wire(dw.get(), fmt, oa.get(), nullptr);
            b.run(dp.get(), ob.get(), nullptr);
        }
        std::vector<cf32> ya(oa.size()), yb(ob.size());
        CURF_CUDA_CHECK(cudaMemcpy(ya.data(), oa.get(), oa.bytes(), cudaMemcpyDeviceToHost));
        CURF_CUDA_CHECK(cudaMemcpy(yb.data(), ob.get(), ob.bytes(), cudaMemcpyDeviceToHost));
        CHECK(std::equal(ya.begin(), ya.end(), yb.begin(),
                         [](cf32 p, cf32 q) { return p.re == q.re && p.im == q.im; }));
    }
}
