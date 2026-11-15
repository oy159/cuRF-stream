// Engine benchmark: the conv1d optimisation ladder across batch buckets.
//
//   curf_bench_engine [--model DIR] [--steps naive,smem,wmma.f32.unfused,wmma.f32.fused,wmma.f16.fused,all]
//                     [--buckets 1,8,32,128,512] [--ms 400] [--no-graph] [--ops] [--profile]
//
// A step is PREFIX[+gap][+stem][+block][+lib]: conv ops are autotuned among variants with that name prefix
// ("all" = whole registry); +gap fuses global pooling into the last layer, +stem allows the direct stem
// kernel, +block fuses each BasicBlock into one kernel, +lib lets cuDNN / CUTLASS variants compete too
// (a "cudnn" or "cutlass" step uses that library wherever it supports the layer), +int8 runs the INT8 engine
// (prefix "i8", calibrated on the golden windows). The bucket's CUDA Graph is replayed for --ms
// milliseconds per bucket.
// One JSON line per (step, bucket); with --ops also one per (step, op) at the largest bucket.

#include "curf/cuda/memory.hpp"
#include "curf/engine/engine.hpp"

#include <cuda_profiler_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

using namespace curf::engine;

namespace {

    std::vector<std::string> split(const std::string &s) {
        std::vector<std::string> out;
        std::stringstream ss(s);
        for (std::string x; std::getline(ss, x, ',');) out.push_back(x);
        return out;
    }

    std::vector<std::string> split_plus(const std::string &s) {
        std::vector<std::string> out;
        std::stringstream ss(s);
        for (std::string x; std::getline(ss, x, '+');) out.push_back(x);
        return out;
    }

    struct Result {
        double us_per_batch, windows_per_s;
    };

    Result measure(Engine &e, int batch, double ms, bool graph, cudaStream_t s) {
        auto run = [&] { graph ? e.run(batch, s) : e.run_eager(batch, s); };
        for (int i = 0; i < 5; ++i) run();
        CURF_CUDA_CHECK(cudaStreamSynchronize(s));
        // Calibrate the replay count, then time a single burst with events.
        cudaEvent_t a, b;
        CURF_CUDA_CHECK(cudaEventCreate(&a));
        CURF_CUDA_CHECK(cudaEventCreate(&b));
        int iters = 10;
        float elapsed = 0;
        for (;;) {
            CURF_CUDA_CHECK(cudaEventRecord(a, s));
            for (int i = 0; i < iters; ++i) run();
            CURF_CUDA_CHECK(cudaEventRecord(b, s));
            CURF_CUDA_CHECK(cudaEventSynchronize(b));
            CURF_CUDA_CHECK(cudaEventElapsedTime(&elapsed, a, b));
            if (elapsed >= ms || iters >= (1 << 22)) break;
            iters = static_cast<int>(iters * std::min(16.0, 1.2 * ms / std::max(elapsed, 0.01f)));
        }
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        const double us = 1000.0 * elapsed / iters;
        return {us, batch / (us * 1e-6)};
    }

} // namespace

int main(int argc, char **argv) {
    std::string dir = CURF_MODEL_DIR;
    std::vector<std::string> steps{"naive",       "smem",           "wmma.f32.unfused",     "wmma.f32.fused",
                                   "wmma.f16.fused", "mma.f32",     "mma.f16",              "mma.f16+gap",
                                   "mma.f16+gap+stem", "mma.f16+gap+stem+block", "all+gap+stem+block",
                                   "cudnn",       "cutlass",        "all+gap+stem+block+lib"};
    std::vector<int> buckets{1, 8, 32, 128, 512};
    double ms = 400;
    bool graph = true, ops = false, profile = false;
    std::string block_prefix; // restrict fused-block variants (e.g. "block.f16.l64p.")
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model") dir = next();
        else if (a == "--steps") steps = split(next());
        else if (a == "--buckets") { buckets.clear(); for (auto &x : split(next())) buckets.push_back(std::stoi(x)); }
        else if (a == "--ms") ms = std::stod(next());
        else if (a == "--no-graph") graph = false;
        else if (a == "--ops") ops = true;
        else if (a == "--block-prefix") block_prefix = next();
        else if (a == "--profile") profile = true; // one eager run of the largest bucket per step, for ncu --profile-from-start off
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }

    const auto model = Model::load(std::filesystem::path(dir) / "amc.curf");
    const double flops = model.flops_per_window();
    cudaStream_t s;
    CURF_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    for (const auto &step : steps) {
        EngineConfig cfg;
        cfg.buckets = buckets;
        const auto tok = split_plus(step);
        cfg.variant_prefix = tok[0] == "all" ? "" : tok[0];
        auto has = [&](const char *f) { return std::find(tok.begin() + 1, tok.end(), f) != tok.end(); };
        cfg.fuse_gap = has("gap");
        cfg.fuse_blocks = has("block");
        cfg.block_prefix = block_prefix;
        // Library kernels (cudnn, cutlass.*) only compete when the step is a library step or says +lib, so the
        // engine's own numbers never quietly include a library kernel.
        if (!has("stem")) cfg.exclude_prefixes.push_back("direct");
        if (!has("lib"))
            for (const char *lib : {"cudnn", "cutlass"})
                if (tok[0].rfind(lib, 0) != 0) cfg.exclude_prefixes.push_back(lib);
        cfg.cuda_graphs = graph;
        cfg.tune_iters = 20;
        if (has("int8")) {
            static const Golden golden = Golden::load(std::filesystem::path(dir) / "amc.golden", model);
            cfg.int8 = true;
            cfg.calibration = golden.input;
        }
        Engine e(model, cfg);
        if (profile) {
            CURF_CUDA_CHECK(cudaStreamSynchronize(s));
            CURF_CUDA_CHECK(cudaProfilerStart());
            e.run_eager(buckets.back(), s);
            CURF_CUDA_CHECK(cudaStreamSynchronize(s));
            CURF_CUDA_CHECK(cudaProfilerStop());
            std::printf("{\"step\":\"%s\",\"profiled_batch\":%d}\n", step.c_str(), buckets.back());
            continue;
        }
        for (int b : buckets) {
            const auto r = measure(e, b, ms, graph, s);
            std::printf("{\"step\":\"%s\",\"batch\":%d,\"graph\":%s,\"us_per_batch\":%.2f,\"windows_per_s\":%.0f,"
                        "\"tflops\":%.3f,\"arena_MiB\":%.2f,\"kernels\":%zu}\n",
                        step.c_str(), b, graph ? "true" : "false", r.us_per_batch, r.windows_per_s,
                        r.windows_per_s * flops / 1e12, static_cast<double>(e.memory_plan().arena_bytes) / 1048576.0, e.steps());
        }
        if (ops) {
            for (const auto &d : e.fusion_decisions())
                std::printf("  {\"step\":\"%s\",\"batch\":%d,\"fusion\":\"%s\",\"variant\":\"%s\",\"fused_us\":%.2f,"
                            "\"unfused_us\":%.2f,\"fused\":%s,\"stem_variant\":\"%s\",\"stem_fused_us\":%.2f,\"stem\":%s}\n",
                            step.c_str(), d.bucket, d.block.c_str(), d.fused_variant.c_str(), d.fused_us, d.unfused_us,
                            d.fused ? "true" : "false", d.stem_variant.c_str(), d.stem_fused_us, d.stem ? "true" : "false");
            for (const auto &c : e.choices()) {
                if (c.bucket != buckets.back()) continue;
                std::printf("  {\"step\":\"%s\",\"batch\":%d,\"op\":\"%s\",\"variant\":\"%s\",\"us\":%.2f,\"tflops\":%.3f}\n",
                            step.c_str(), c.bucket, c.op.c_str(), c.variant.c_str(), c.us,
                            c.us > 0 ? c.flops / (c.us * 1e-6) / 1e12 : 0.0);
            }
        }
        std::fflush(stdout);
    }
    cudaStreamDestroy(s);
    return 0;
}
