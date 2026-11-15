// Layer-by-layer alignment of the engine against the fp32 golden activations from convert.py.

#include "curf/cuda/memory.hpp"
#include "curf/cuda/windows.hpp"
#include "curf/engine/engine.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
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
        if (v.int8) continue; // INT8 variants: own test below
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

namespace {
    // Calibration on the first half of the golden windows, evaluation on all of them.
    EngineConfig int8_config(const Golden &g, const Model &m) {
        EngineConfig cfg;
        cfg.int8 = true;
        const auto per = m.tensors[m.input].elems_per_window();
        cfg.calibration.assign(g.input.begin(), g.input.begin() + static_cast<std::ptrdiff_t>(per * (g.n_full / 2)));
        return cfg;
    }
} // namespace

TEST_CASE("every INT8 conv variant matches the fp32 golden layer by layer", "[cuda][engine][int8]") {
    const auto *d = data();
    if (!d) SKIP("no converted model in " + kDir.string());
    const auto &g = d->golden;
    const int n = static_cast<int>(g.n_layer);
    for (const auto &v : conv_variants()) {
        if (!v.int8 || v.name.rfind("i8.mma.", 0) != 0) continue;
        EngineConfig cfg = int8_config(g, d->model);
        cfg.buckets = {n};
        cfg.variant_prefix = v.name; // the stem falls back to the INT8 direct kernel
        cfg.calib_percentile = 1.0;  // max calibration: no clipping, so the max error measures the kernels
        cfg.autotune = false;
        cfg.cuda_graphs = false;
        cfg.reuse_memory = false;
        Engine e(d->model, cfg);
        upload(e, g, 0, n);
        e.run(n, nullptr);
        CURF_CUDA_CHECK(cudaDeviceSynchronize());
        double worst = 0, worst_rms = 0;
        for (std::uint32_t t = 0; t < d->model.tensors.size(); ++t) {
            if (t == d->model.input || !e.materialized(t)) continue;
            const auto got = e.read_tensor(t, n);
            const double err = rel_err(got, g.activations[t].data(), got.size());
            double se = 0, sv = 0;
            for (std::size_t i = 0; i < got.size(); ++i) {
                const double dlt = got[i] - g.activations[t][i];
                se += dlt * dlt;
                sv += static_cast<double>(g.activations[t][i]) * g.activations[t][i];
            }
            const double rms = std::sqrt(se / std::max(sv, 1e-30));
            worst = std::max(worst, err);
            worst_rms = std::max(worst_rms, rms);
            INFO(v.name << " tensor " << t << " (" << d->model.tensors[t].name << ") rel err " << err << " rms " << rms);
            CHECK(err < 0.12); // 8-bit steps (1/127 of the range) per layer, compounding over 11 convs
            CHECK(rms < 0.12); // sparse post-ReLU tensors: many small values below one step
        }
        std::printf("  %-28s worst tensor: max rel err %.2e, rms rel err %.2e\n", v.name.c_str(), worst, worst_rms);
    }
}

TEST_CASE("fused INT8 BasicBlocks match the golden", "[cuda][engine][int8][fusion]") {
    const auto *d = data();
    if (!d) SKIP("no converted model in " + kDir.string());
    const auto &g = d->golden;
    const int n = static_cast<int>(g.n_layer), c = static_cast<int>(g.classes);
    for (const std::string block : {"i8.block.l128.", "i8.block.l64.", "i8.block.l128r.", "i8.block.l64r.", "i8.block.l64k2.",
                                     "i8.block.l128k2."}) {
        // no resident 64x128 block; k2 variants only for cin = 64 (stage2.1, stage3): two blocks run as convs
        const bool resident = block[block.size() - 2] == 'r', k2 = block.find("k2.") != std::string::npos;
        EngineConfig cfg = int8_config(g, d->model);
        cfg.buckets = {n};
        cfg.variant_prefix = "i8.mma.64x64.w32.s3.z";
        cfg.block_prefix = block;
        cfg.calib_percentile = 1.0;
        cfg.autotune = false;
        cfg.cuda_graphs = false;
        cfg.reuse_memory = false;
        Engine e(d->model, cfg);
        // stem + 4 blocks + head; unfused blocks as convs (stage3 without a resident block: conv1 + conv2 with the
        // shortcut fused in)
        CHECK(e.steps() == (resident ? 7u : k2 ? 9u : 6u));
        upload(e, g, 0, n);
        e.run(n, nullptr);
        CURF_CUDA_CHECK(cudaDeviceSynchronize());
        int checked = 0;
        for (std::uint32_t t = 0; t < d->model.tensors.size(); ++t) {
            if (t == d->model.input || t == d->model.output || !e.materialized(t)) continue;
            const auto got = e.read_tensor(t, n);
            const double err = rel_err(got, g.activations[t].data(), got.size());
            INFO(block << " tensor " << t << " (" << d->model.tensors[t].name << ") rel err " << err);
            CHECK(err < 0.12);
            ++checked;
        }
        CHECK(checked == (resident ? 5 : k2 ? 7 : 4)); // stem out + 3 block outputs + the unfused blocks' inner tensors
        const auto logits = e.read_tensor(d->model.output, n);
        int agree = 0;
        for (int i = 0; i < n; ++i) agree += argmax(&logits[i * c], c) == argmax(&g.logits[i * c], c);
        std::printf("  %s: logits rel err %.2e, top-1 %d/%d\n", block.c_str(),
                    rel_err(logits, g.activations[d->model.output].data(), logits.size()), agree, n);
        CHECK(agree >= n - 2);
    }
}

TEST_CASE("INT8 stem fused into the first block matches the separate stem", "[cuda][engine][int8][fusion]") {
    const auto *d = data();
    if (!d) SKIP("no converted model in " + kDir.string());
    const auto &g = d->golden;
    const int n = static_cast<int>(g.n_layer), c = static_cast<int>(g.classes);
    std::uint32_t stem_out = kNoTensor, block1_out = kNoTensor;
    for (const auto &op : d->model.ops) {
        if (op.in == d->model.input) stem_out = op.out;
        if (op.name.find("stage1.0/conv2") != std::string::npos) block1_out = op.out;
    }
    REQUIRE(stem_out != kNoTensor);
    REQUIRE(block1_out != kNoTensor);
    auto make = [&](const std::string &block) {
        EngineConfig cfg = int8_config(g, d->model);
        cfg.buckets = {n};
        cfg.variant_prefix = "i8.mma.64x64.w32.s3.z";
        cfg.block_prefix = block;
        cfg.calib_percentile = 1.0;
        cfg.autotune = false;
        cfg.cuda_graphs = false;
        cfg.reuse_memory = false;
        auto e = std::make_unique<Engine>(d->model, cfg);
        upload(*e, g, 0, n);
        e->run(n, nullptr);
        CURF_CUDA_CHECK(cudaDeviceSynchronize());
        return e;
    };
    for (const std::string tile : {"l128r.", "l64r.", "l128.", "l64."}) {
        const auto ref = make("i8.block." + tile);      // stem kernel + block kernel
        const auto fused = make("i8.block.stem." + tile); // one kernel; the other blocks run as convs
        CHECK(fused->materialized(block1_out));
        CHECK_FALSE(fused->materialized(stem_out));
        CHECK(ref->materialized(stem_out));
        const auto a = ref->read_tensor(block1_out, n), b = fused->read_tensor(block1_out, n);
        // The fused stem rounds its input to fp16 for the MMA: a stem value next to a rounding boundary moves by
        // one int8 step, which conv1 spreads over 3 positions x all channels. Count and size of the differences, in
        // steps of the block output's scale:
        const double step = fused->activation_scales()[block1_out];
        std::size_t diff = 0;
        double worst = 0, se = 0, sv = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            diff += a[i] != b[i];
            worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - b[i]) / step);
            se += (static_cast<double>(a[i]) - b[i]) * (a[i] - b[i]);
            sv += static_cast<double>(a[i]) * a[i];
        }
        std::printf("  i8.block.stem.%s: %zu of %zu block-1 outputs differ from stem kernel + block, max %.0f steps, "
                    "rms rel %.2e\n", tile.c_str(), diff, a.size(), worst, std::sqrt(se / sv));
        CHECK(diff <= a.size() / 20);
        CHECK(worst <= 3.0);
        CHECK(std::sqrt(se / sv) < 0.01);
        for (std::uint32_t t = 0; t < d->model.tensors.size(); ++t) {
            if (t == d->model.input || t == d->model.output || !fused->materialized(t)) continue;
            const auto got = fused->read_tensor(t, n);
            INFO(tile << " tensor " << t << " (" << d->model.tensors[t].name << ")");
            CHECK(rel_err(got, g.activations[t].data(), got.size()) < 0.12);
        }
        const auto logits = fused->read_tensor(d->model.output, n);
        int agree = 0;
        for (int i = 0; i < n; ++i) agree += argmax(&logits[i * c], c) == argmax(&g.logits[i * c], c);
        std::printf("  i8.block.stem.%s: %zu steps, top-1 %d/%d\n", tile.c_str(), fused->steps(), agree, n);
        CHECK(agree >= n - 2);
    }
}

TEST_CASE("INT8 conv2 with the shortcut fused in matches the separate shortcut", "[cuda][engine][int8][fusion]") {
    const auto *d = data();
    if (!d) SKIP("no converted model in " + kDir.string());
    const auto &g = d->golden;
    const int n = static_cast<int>(g.n_layer), c = static_cast<int>(g.classes);
    std::uint32_t sc_out = kNoTensor;
    for (const auto &op : d->model.ops)
        if (op.name.find("stage3.0/shortcut") != std::string::npos) sc_out = op.out;
    REQUIRE(sc_out != kNoTensor);
    auto make = [&](const std::string &prefix, bool sc) {
        EngineConfig cfg = int8_config(g, d->model);
        cfg.buckets = {n};
        cfg.variant_prefix = prefix;
        cfg.block_prefix = "none"; // no block kernels: every BasicBlock as convs
        if (!sc) cfg.exclude_prefixes.push_back("i8.mma.sc.");
        cfg.calib_percentile = 1.0;
        cfg.autotune = false;
        cfg.cuda_graphs = false;
        cfg.reuse_memory = false;
        auto e = std::make_unique<Engine>(d->model, cfg);
        upload(*e, g, 0, n);
        e->run(n, nullptr);
        CURF_CUDA_CHECK(cudaDeviceSynchronize());
        return e;
    };
    const auto ref = make("i8.mma.64x64.w32.s3.z", false);
    const auto ref_logits = ref->read_tensor(d->model.output, n);
    for (const std::string tile : {"64x64.w32.s3.z", "64x128.w32.s3.z", "128x64.w32.s3.z", "128x128.w64.s3.z", "64x64.w32.s3",
                                   "64x128.w32.s3"}) {
        const auto e = make("i8.mma.sc." + tile, true); // other convs fall back to the plain i8.mma tiles
        CHECK(e->steps() == ref->steps() - 1);
        CHECK_FALSE(e->materialized(sc_out));
        bool used = false;
        for (const auto &ch : e->choices()) used = used || ch.variant.rfind("i8.mma.sc." + tile, 0) == 0; // ".s3" also matches ".s3.z"
        CHECK(used);
        // The fused shortcut is added in fp32 instead of being rounded to int8 first: logits move by about one
        // int8 step of the residual, the class decisions do not.
        const auto logits = e->read_tensor(d->model.output, n);
        const double vs_ref = rel_err(logits, ref_logits.data(), logits.size());
        const double vs_gold = rel_err(logits, g.activations[d->model.output].data(), logits.size());
        int agree = 0;
        for (int i = 0; i < n; ++i) agree += argmax(&logits[i * c], c) == argmax(&g.logits[i * c], c);
        std::printf("  i8.mma.sc.%s: logits rel err vs separate shortcut %.2e, vs fp32 %.2e, top-1 %d/%d\n", tile.c_str(),
                    vs_ref, vs_gold, agree, n);
        CHECK(vs_ref < 0.05);
        CHECK(vs_gold < 0.12);
        CHECK(agree >= n - 2);
    }
}

TEST_CASE("INT8 engine with fp16 input equals the fp32-input engine", "[cuda][engine][int8][fusion]") {
    const auto *d = data();
    if (!d) SKIP("no converted model in " + kDir.string());
    const auto &g = d->golden;
    const int c = static_cast<int>(g.classes);
    constexpr std::uint32_t S = 2, M = 128, W = 1024;
    REQUIRE(g.n_full == S * M);
    REQUIRE(d->model.tensors[d->model.input].length == W);
    // channelizer-layout buffer [S][W][M]: entry e = s * M + k holds golden window e at gain 0.05 + 0.02 e
    std::vector<curf::cf32> h(std::size_t{S} * W * M);
    for (std::uint32_t s = 0; s < S; ++s)
        for (std::uint32_t t = 0; t < W; ++t)
            for (std::uint32_t k = 0; k < M; ++k) {
                const std::size_t e = s * M + k;
                const float gain = 0.05f + 0.02f * static_cast<float>(e);
                h[(std::size_t{s} * W + t) * M + k] = {gain * g.input[(e * W + t) * 2], gain * g.input[(e * W + t) * 2 + 1]};
            }
    curf::cuda::DeviceBuffer<curf::cf32> win(h.size());
    CURF_CUDA_CHECK(cudaMemcpy(win.get(), h.data(), h.size() * sizeof(h[0]), cudaMemcpyHostToDevice));
    curf::cuda::WindowStage stage(S, M, W);
    curf::cuda::DetectConfig det;
    det.mode = curf::cuda::DetectMode::all;
    stage.detect(win.get(), det, nullptr);
    CURF_CUDA_CHECK(cudaDeviceSynchronize());
    for (const std::string tile : {"l128r.", "l64."}) {
        auto config = [&](bool half) {
            EngineConfig cfg = int8_config(g, d->model);
            cfg.buckets = {160}; // chunks of 160 and 96 (a partial bucket)
            cfg.autotune = false;
            cfg.block_prefix = "i8.block.stem." + tile;
            cfg.half_input = half;
            return cfg;
        };
        Engine ref(d->model, config(false)), half(d->model, config(true));
        CHECK(half.steps() == ref.steps());
        for (std::uint32_t first = 0; first < S * M; first += 160) {
            const int n = static_cast<int>(std::min<std::uint32_t>(160, S * M - first));
            stage.gather(win.get(), first, static_cast<std::uint32_t>(n), ref.input(), nullptr);
            stage.gather_half(win.get(), first, static_cast<std::uint32_t>(n), half.input_half(), nullptr);
            ref.run(n, nullptr);
            half.run(n, nullptr);
            CURF_CUDA_CHECK(cudaDeviceSynchronize());
            const auto a = ref.read_tensor(d->model.output, n), b = half.read_tensor(d->model.output, n);
            std::size_t diff = 0;
            for (std::size_t i = 0; i < a.size(); ++i) diff += a[i] != b[i];
            int agree = 0;
            for (int i = 0; i < n; ++i) agree += argmax(&b[i * c], c) == argmax(&g.logits[(first + i) * c], c);
            std::printf("  fp16 input, %s chunk at %u (%d windows): %zu of %zu logits differ, top-1 vs fp32 %d/%d\n",
                        tile.c_str(), first, n, diff, a.size(), agree, n);
            CHECK(diff == 0); // the stem rounds to fp16 either way
            CHECK(agree >= n * 95 / 100);
        }
    }
}

TEST_CASE("tuned INT8 engine keeps the fp32 accuracy", "[cuda][engine][int8]") {
    const auto *d = data();
    if (!d) SKIP("no converted model in " + kDir.string());
    const auto &g = d->golden;
    const int n = static_cast<int>(g.n_full), c = static_cast<int>(g.classes);
    EngineConfig cfg = int8_config(g, d->model);
    cfg.buckets = {1, 8, 128, n};
    cfg.tune_iters = 5;
    Engine e(d->model, cfg);
    upload(e, g, 0, n);
    e.run(n, nullptr);
    CURF_CUDA_CHECK(cudaDeviceSynchronize());
    const auto logits = e.read_tensor(d->model.output, n);
    int agree = 0, correct = 0, correct_ref = 0, held_agree = 0, held = 0;
    for (int i = 0; i < n; ++i) {
        const int a = argmax(&logits[i * c], c), b = argmax(&g.logits[i * c], c);
        agree += a == b;
        correct += a == g.labels[i];
        correct_ref += b == g.labels[i];
        if (i >= n / 2) { held_agree += a == b; ++held; }
    }
    std::printf("  int8 engine: top-1 agreement %d/%d (held-out %d/%d), accuracy %d vs fp32 %d\n", agree, n, held_agree,
                held, correct, correct_ref);
    for (const auto &ch : e.choices())
        if (ch.bucket == n) std::printf("    %-40s %-28s %8.1f us\n", ch.op.c_str(), ch.variant.c_str(), ch.us);
    CHECK(held_agree >= held * 97 / 100);
    CHECK(correct >= correct_ref - n / 100);
}
