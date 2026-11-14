// Layer-by-layer alignment of the engine against the fp32 golden activations from convert.py.

#include "curf/cuda/memory.hpp"
#include "curf/engine/engine.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <optional>

using namespace curf::engine;

namespace {

    const std::filesystem::path kDir = CURF_MODEL_DIR;

    struct Data {
        Model model;
        Golden golden;
    };

    const Data *data() {
        static const std::optional<Data> d = []() -> std::optional<Data> {
            if (!std::filesystem::exists(kDir / "amc.curf") || !std::filesystem::exists(kDir / "amc.golden"))
                return std::nullopt;
            Data x{Model::load(kDir / "amc.curf"), {}};
            x.golden = Golden::load(kDir / "amc.golden", x.model);
            return x;
        }();
        return d ? &*d : nullptr;
    }

    void upload(const Engine &e, const Golden &g, int first, int n) {
        const auto per = e.model().tensors[e.model().input].elems_per_window();
        CURF_CUDA_CHECK(cudaMemcpy(e.input(), g.input.data() + per * first, per * n * sizeof(float), cudaMemcpyHostToDevice));
    }

    // max |got - want| relative to max |want| over the tensor
    double rel_err(const std::vector<float> &got, const float *want, std::size_t n) {
        double err = 0, scale = 0;
        for (std::size_t i = 0; i < n; ++i) {
            err = std::max(err, static_cast<double>(std::fabs(got[i] - want[i])));
            scale = std::max(scale, static_cast<double>(std::fabs(want[i])));
        }
        return err / std::max(scale, 1e-12);
    }

    int argmax(const float *p, int n) { return static_cast<int>(std::max_element(p, p + n) - p); }

} // namespace

TEST_CASE("every conv variant matches the fp32 golden layer by layer", "[cuda][engine]") {
    const auto *d = data();
    if (!d) SKIP("no converted model in " + kDir.string());
    const int n = static_cast<int>(d->golden.n_layer);
    for (const auto &v : conv_variants()) {
        EngineConfig cfg;
        cfg.buckets = {n};
        cfg.variant_prefix = v.name;
        cfg.autotune = false;
        cfg.cuda_graphs = false;
        cfg.reuse_memory = false;
        cfg.fuse_blocks = false; // every tensor materialised, so every layer can be compared
        cfg.fuse_gap = false;
        Engine e(d->model, cfg);
        upload(e, d->golden, 0, n);
        e.run(n, nullptr);
        CURF_CUDA_CHECK(cudaDeviceSynchronize());

        // fp16 storage: ~1e-3 per layer, growing with depth; fp16 accumulation adds a bit more.
        const double tol = v.name.find(".f16.") != std::string::npos ? 3e-2 : 1.5e-2;
        double worst = 0;
        for (std::uint32_t t = 0; t < d->model.tensors.size(); ++t) {
            if (t == d->model.input) continue;
            const auto got = e.read_tensor(t, n);
            const double err = rel_err(got, d->golden.activations[t].data(), got.size());
            worst = std::max(worst, err);
            INFO(v.name << " tensor " << t << " (" << d->model.tensors[t].name << ") rel err " << err);
            CHECK(err < tol);
        }
        std::printf("  %-24s worst layer rel err %.2e\n", v.name.c_str(), worst);
    }
}

TEST_CASE("fused BasicBlocks and fused global pooling match the golden", "[cuda][engine][fusion]") {
    const auto *d = data();
    if (!d) SKIP("no converted model in " + kDir.string());
    const auto &g = d->golden;
    const int n = static_cast<int>(g.n_layer), c = static_cast<int>(g.classes);
    for (const std::string block : {"block.f16.l128p.", "block.f16.l64p.", "block.f32.l128p.", "block.f32.l64p.",
                                     "block.f16.l128.", "block.f16.l64.", "block.f32.l128.", "block.f32.l64."}) {
        for (const bool gap : {false, true}) {
            EngineConfig cfg;
            cfg.buckets = {n};
            cfg.variant_prefix = "mma.f16";
            cfg.block_prefix = block;
            cfg.fuse_gap = gap;
            cfg.autotune = false;
            cfg.cuda_graphs = false;
            cfg.reuse_memory = false;
            Engine e(d->model, cfg);
            // stem + 4 BasicBlocks + head
            CHECK(e.steps() == 6);
            upload(e, g, 0, n);
            e.run(n, nullptr);
            CURF_CUDA_CHECK(cudaDeviceSynchronize());
            const double tol = 3e-2;
            int checked = 0;
            for (std::uint32_t t = 0; t < d->model.tensors.size(); ++t) {
                if (t == d->model.input || !e.materialized(t)) continue;
                const auto got = e.read_tensor(t, n);
                const double err = rel_err(got, g.activations[t].data(), got.size());
                INFO(block << (gap ? "+gap" : "") << " tensor " << t << " (" << d->model.tensors[t].name << ") rel err " << err);
                CHECK(err < tol);
                ++checked;
            }
            CHECK(checked == (gap ? 5 : 6)); // stem out, 4 block outputs (last one pooled when gap), logits
            const auto logits = e.read_tensor(d->model.output, n);
            int agree = 0;
            for (int i = 0; i < n; ++i) agree += argmax(&logits[i * c], c) == argmax(&g.logits[i * c], c);
            std::printf("  %s%s: %zu steps, logits rel err %.2e, top-1 %d/%d\n", block.c_str(), gap ? "+gap" : "",
                        e.steps(), rel_err(logits, g.activations[d->model.output].data(), logits.size()), agree, n);
            CHECK(agree >= n - 1);
        }
    }
}

TEST_CASE("tuned engine with memory reuse and CUDA graphs matches the golden logits", "[cuda][engine]") {
    const auto *d = data();
    if (!d) SKIP("no converted model in " + kDir.string());
    const auto &g = d->golden;
    const int n = static_cast<int>(g.n_full);
    EngineConfig cfg;
    cfg.buckets = {1, 8, 32, 128, n};
    cfg.tune_iters = 5;
    Engine e(d->model, cfg);
    CHECK(e.memory_plan().arena_bytes < e.memory_plan().no_reuse_bytes);

    upload(e, g, 0, n);
    e.run(n, nullptr);
    CURF_CUDA_CHECK(cudaDeviceSynchronize());
    const auto logits = e.read_tensor(d->model.output, n);
    const int c = static_cast<int>(g.classes);
    int agree = 0, correct_ref = 0, correct = 0;
    double err = 0;
    for (int i = 0; i < n; ++i) {
        const int a = argmax(&logits[i * c], c), b = argmax(&g.logits[i * c], c);
        agree += a == b;
        correct += a == g.labels[i];
        correct_ref += b == g.labels[i];
        for (int j = 0; j < c; ++j) err = std::max(err, static_cast<double>(std::fabs(logits[i * c + j] - g.logits[i * c + j])));
    }
    std::printf("  tuned engine: top-1 agreement %d/%d, accuracy %d vs fp32 %d, max |logit diff| %.4f\n", agree, n, correct,
                correct_ref, err);
    CHECK(agree >= n * 99 / 100);
    CHECK(err < 0.1);

    // Same windows through a smaller bucket (graph for 8, batch 5) give the same logits.
    upload(e, g, 0, 5);
    e.run(5, nullptr);
    CURF_CUDA_CHECK(cudaDeviceSynchronize());
    const auto small = e.read_tensor(d->model.output, 5);
    double diff = 0;
    for (int i = 0; i < 5 * c; ++i) diff = std::max(diff, static_cast<double>(std::fabs(small[i] - logits[i])));
    CHECK(diff < 0.1); // buckets may pick different kernels (accumulation order, fp16 accumulators)
}
