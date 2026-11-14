// Channelizer throughput: unpack + planar channelizer vs wire-fused channelizer.
//
//   curf_bench_channelizer [--streams 2] [--block 131072] [--channels 512] [--taps 8] [--format packed12] [--ms 500]
//
// One JSON line per path: per-call time, input rate per stream (MS/s), x real time at 61.44 MS/s,
// effective GFLOP/s, and the minimum DRAM traffic (wire in + channel output) as GB/s.

#include "curf/cuda/channelizer.hpp"
#include "curf/cuda/memory.hpp"
#include "curf/cuda/unpack.hpp"
#include "curf/reference.hpp"

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace curf;

namespace {

    double time_ms(const std::function<void(cudaStream_t)> &fn, cudaStream_t s, double budget_ms) {
        for (int i = 0; i < 5; ++i) fn(s);
        CURF_CUDA_CHECK(cudaStreamSynchronize(s));
        cudaEvent_t a, b;
        CURF_CUDA_CHECK(cudaEventCreate(&a));
        CURF_CUDA_CHECK(cudaEventCreate(&b));
        int iters = 10;
        float ms = 0;
        for (;;) {
            CURF_CUDA_CHECK(cudaEventRecord(a, s));
            for (int i = 0; i < iters; ++i) fn(s);
            CURF_CUDA_CHECK(cudaEventRecord(b, s));
            CURF_CUDA_CHECK(cudaEventSynchronize(b));
            CURF_CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
            if (static_cast<double>(ms) >= budget_ms || iters > (1 << 20)) break;
            iters *= 4; // NOLINT
        }
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        return static_cast<double>(ms) / iters;
    }

} // namespace

int main(int argc, char **argv) {
    std::uint32_t streams = 2;
    std::size_t block = std::size_t{1} << 17;
    ChannelizerConfig cfg;
    WireFormat fmt = WireFormat::packed12;
    double budget = 500;
    std::string only; // "fused" / "planar": run just that path (for profiling)
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--streams") streams = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--block") block = std::stoull(next());
        else if (a == "--channels") cfg.channels = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--taps") cfg.taps_per_phase = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--format") fmt = next() == "signed8" ? WireFormat::signed8 : WireFormat::packed12;
        else if (a == "--ms") budget = std::stod(next());
        else if (a == "--path") only = next();
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }

    const auto bytes = wire_sample_bytes(fmt);
    std::vector<std::uint8_t> wire(block * streams * bytes);
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 0.2f);
    for (std::size_t i = 0; i < block * streams; ++i) ref::pack_sample({nd(rng), nd(rng)}, fmt, &wire[i * bytes]);

    cuda::Channelizer ch(cfg, streams, block);
    cuda::DeviceBuffer<std::uint8_t> d_wire(wire.size());
    cuda::DeviceBuffer<cf32> d_planar(block * streams), d_out(ch.out_elems());
    CURF_CUDA_CHECK(cudaMemcpy(d_wire.get(), wire.data(), wire.size(), cudaMemcpyHostToDevice));
    cudaStream_t s;
    CURF_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    const double in_bytes = static_cast<double>(wire.size()), out_bytes = static_cast<double>(d_out.bytes());
    auto report = [&](const char *path, double ms, double extra_bytes) {
        const double msps = static_cast<double>(block) / (ms * 1e3); // per stream
        std::printf("{\"path\":\"%s\",\"streams\":%u,\"block\":%zu,\"channels\":%u,\"taps\":%u,\"ms_per_block\":%.4f,"
                    "\"msps_per_stream\":%.1f,\"x_realtime\":%.1f,\"gflops\":%.1f,\"min_dram_GBps\":%.1f}\n",
                    path, streams, block, cfg.channels, cfg.taps_per_phase, ms, msps, msps / 61.44,
                    ch.flops() / (ms * 1e6), (in_bytes + out_bytes + extra_bytes) / (ms * 1e6));
    };
    if (only == "fused") {
        report("wire_fused", time_ms([&](cudaStream_t st) { ch.run_wire(d_wire.get(), fmt, d_out.get(), st); }, s, budget), 0);
        return 0;
    }
    const double t_unpack = time_ms([&](cudaStream_t st) {
        cuda::unpack_planar(d_wire.get(), fmt, streams, block, d_planar.get(), st);
    }, s, budget);
    const double t_planar = time_ms([&](cudaStream_t st) { ch.run(d_planar.get(), d_out.get(), st); }, s, budget);
    const double t_both = time_ms([&](cudaStream_t st) {
        cuda::unpack_planar(d_wire.get(), fmt, streams, block, d_planar.get(), st);
        ch.run(d_planar.get(), d_out.get(), st);
    }, s, budget);
    const double t_fused = time_ms([&](cudaStream_t st) { ch.run_wire(d_wire.get(), fmt, d_out.get(), st); }, s, budget);
    ch.set_variant(cuda::FoldVariant::direct);
    const double t_fused_direct = time_ms([&](cudaStream_t st) { ch.run_wire(d_wire.get(), fmt, d_out.get(), st); }, s, budget);
    ch.set_variant(cuda::FoldVariant::best);
    std::printf("{\"path\":\"unpack_only\",\"ms_per_block\":%.4f}\n", t_unpack);
    report("channelizer_planar_only", t_planar, 0);
    report("unpack+channelizer", t_both, 2.0 * static_cast<double>(d_planar.bytes()));
    report("wire_fused", t_fused, 0);
    report("wire_fused_direct_fold", t_fused_direct, 0);
    cudaStreamDestroy(s);
    return 0;
}
