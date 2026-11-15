#include "curf/engine/engine.hpp"

#include "curf/cuda/memory.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>

namespace curf::engine {

    using cuda::DeviceBuffer;

    namespace {

        constexpr std::size_t kNone = ~std::size_t{0};

        bool starts_with(const std::string &s, const std::string &p) { return s.rfind(p, 0) == 0; }

        // A BasicBlock that may run as one kernel: conv1 (a), conv2 (b, with the residual), shortcut (c).
        // s: the stem conv in front of it (fp32 graph input), if a stem-fused block variant can take it.
        // sc: conv2 can run with the 1x1 shortcut fused in (INT8 conv variants "i8.mma.sc.*").
        struct Group {
            std::size_t a = kNone, b = kNone, c = kNone, s = kNone;
            bool sc = false;
        };

        // One kernel launch of a bucket's execution plan.
        struct Step {
            enum class Kind { conv, gap_linear, linear_from_pool, block } kind = Kind::conv;
            std::size_t op = 0;        // conv / head op; block: its conv2
            std::size_t group = kNone; // block only
            bool stem = false;         // block with the stem conv fused in front
            bool sc = false;           // conv (a group's conv2) with the shortcut fused in
        };

        // Per bucket and group: how the BasicBlock runs (separate convs; one block kernel; block kernel with the
        // stem in front; conv1, then conv2 with the shortcut fused in).
        enum class Fuse : unsigned char { none, block, stem, shortcut };

    } // namespace

    struct Engine::Impl {
        Model model;
        EngineConfig cfg;
        MemoryPlan plan;
        std::vector<Group> groups;
        std::vector<std::size_t> group_of;      // op -> group (kNone if none)
        std::size_t pool_op = kNone;            // conv whose output is pooled (fuse_gap), or kNone
        std::size_t head_op = kNone;            // the gap_linear op
        std::vector<std::vector<Step>> steps;   // [bucket]
        std::vector<bool> materialized;
        DeviceBuffer<std::byte> arena, weights;
        DeviceBuffer<float> input, logits, pool;
        mutable DeviceBuffer<float> scratch; // read_tensor conversions
        cudaStream_t stream = nullptr;       // private stream for tuning and capture
        std::vector<int> buckets;
        std::vector<std::vector<const ConvVariant *>> conv_choice;   // [bucket][op]
        std::vector<std::vector<float>> conv_us;                     // [bucket][op]
        std::vector<std::vector<const BlockVariant *>> block_choice; // [bucket][group]
        std::vector<std::vector<const BlockVariant *>> stem_choice;  // [bucket][group], stem-fused variant
        std::vector<std::vector<Fuse>> fuse_mode;                    // [bucket][group]
        std::vector<std::vector<const ConvVariant *>> sc_choice;     // [bucket][group], conv2 + shortcut
        std::vector<cudaGraphExec_t> graphs;                         // [bucket]
        std::vector<OpChoice> choices;
        std::vector<FusionDecision> decisions;
        std::string gpu;
        // INT8
        std::vector<float> act_scale;            // [tensor]
        std::vector<int> kpad8;                  // [op] int8 weight row length (multiple of 64)
        std::vector<std::size_t> w8_off, alpha_off; // [op] into w8 / alpha
        DeviceBuffer<std::int8_t> w8;
        DeviceBuffer<float> alpha;
        DeviceBuffer<__half> stem_w16;           // stem weights fp16 [cout][16] (k padded) for stem-fused blocks

        const TensorDesc &tensor(std::uint32_t t) const { return model.tensors[t]; }

        void *ptr(std::uint32_t t) const {
            if (t == model.input) return input.get();
            if (t == model.output) return logits.get();
            if (!materialized[t] || plan.offset.empty()) return nullptr; // fusion pass runs before planning
            return arena.get() + plan.offset[t];
        }
        template <class T> const T *wptr(std::uint64_t off) const { return reinterpret_cast<const T *>(weights.get() + off); }
        std::size_t pool_channels() const { return head_op == kNone || pool_op == kNone ? 0 : model.ops[head_op].cin; }

        ConvParams conv_params(std::size_t i, int batch, bool with_sc = false) const {
            const auto &op = model.ops[i];
            ConvParams p;
            p.x = ptr(op.in);
            p.x_f32 = op.in == model.input;
            p.w = wptr<__half>(op.w_off);
            p.bias = wptr<float>(op.b_off);
            p.res = op.res == kNoTensor ? nullptr : static_cast<const __half *>(ptr(op.res));
            p.y = static_cast<__half *>(ptr(op.out));
            p.pool = i == pool_op ? (pool.get() ? pool.get() : reinterpret_cast<float *>(16)) : nullptr; // non-null marks "pooled"
            p.batch = batch;
            p.lin = static_cast<int>(tensor(op.in).length);
            p.lout = static_cast<int>(tensor(op.out).length);
            p.cin = static_cast<int>(op.cin);
            p.cout = static_cast<int>(op.cout);
            p.k = static_cast<int>(op.k);
            p.stride = static_cast<int>(op.stride);
            p.pad = static_cast<int>(op.pad);
            p.kpad = static_cast<int>(op.kpad);
            p.relu = op.relu;
            if (cfg.int8 && !act_scale.empty()) {
                p.int8 = true;
                p.kpad = kpad8[i];
                p.w8 = w8.get() + w8_off[i];
                p.alpha = alpha.get() + alpha_off[i];
                p.res_scale = op.res == kNoTensor ? 0.f : act_scale[op.res];
                p.y_inv_scale = i == pool_op ? 0.f : 1.f / act_scale[op.out];
                if (with_sc && group_of[i] != kNone && groups[group_of[i]].c != kNone) {
                    const auto sc = groups[group_of[i]].c;
                    const auto &c = model.ops[sc];
                    p.res = nullptr;
                    p.res_scale = 0.f;
                    p.sc_x = static_cast<const std::int8_t *>(ptr(c.in));
                    if (!p.sc_x) p.sc_x = reinterpret_cast<const std::int8_t *>(16); // shape check before allocation
                    p.sc_w8 = w8.get() + w8_off[sc];
                    p.sc_alpha = alpha.get() + alpha_off[sc];
                    p.sc_bias = wptr<float>(c.b_off);
                    p.sc_lin = static_cast<int>(tensor(c.in).length);
                    p.sc_cin = static_cast<int>(c.cin);
                    p.sc_stride = static_cast<int>(c.stride);
                    p.sc_kpad = kpad8[sc];
                }
            }
            return p;
        }

        BlockParams block_params(std::size_t g, int batch, bool with_stem = false) const {
            const auto &grp = groups[g];
            const auto &a = model.ops[grp.a], &b = model.ops[grp.b];
            BlockParams p;
            p.x = static_cast<const __half *>(ptr(a.in));
            p.y = static_cast<__half *>(ptr(b.out));
            p.pool = grp.b == pool_op ? pool.get() : nullptr;
            p.w1 = wptr<__half>(a.w_off);
            p.b1 = wptr<float>(a.b_off);
            p.w2 = wptr<__half>(b.w_off);
            p.b2 = wptr<float>(b.b_off);
            if (grp.c != kNone) {
                const auto &c = model.ops[grp.c];
                p.wsc = wptr<__half>(c.w_off);
                p.bsc = wptr<float>(c.b_off);
                p.kpadsc = static_cast<int>(c.kpad);
            }
            p.batch = batch;
            p.lin = static_cast<int>(tensor(a.in).length);
            p.lout = static_cast<int>(tensor(b.out).length);
            p.cin = static_cast<int>(a.cin);
            p.cout = static_cast<int>(b.cout);
            p.stride = static_cast<int>(a.stride);
            p.kpad1 = static_cast<int>(a.kpad);
            p.kpad2 = static_cast<int>(b.kpad);
            if (cfg.int8 && !act_scale.empty()) {
                p.int8 = true;
                p.w1_8 = w8.get() + w8_off[grp.a];
                p.w2_8 = w8.get() + w8_off[grp.b];
                p.alpha1 = alpha.get() + alpha_off[grp.a];
                p.alpha2 = alpha.get() + alpha_off[grp.b];
                p.kpad1 = kpad8[grp.a];
                p.kpad2 = kpad8[grp.b];
                if (grp.c != kNone) {
                    p.wsc_8 = w8.get() + w8_off[grp.c];
                    p.alphasc = alpha.get() + alpha_off[grp.c];
                    p.kpadsc = kpad8[grp.c];
                }
                p.x_scale = act_scale[a.in];
                p.y1_inv_scale = 1.f / act_scale[a.out];
                p.y_inv_scale = grp.b == pool_op ? 0.f : 1.f / act_scale[b.out];
                if (with_stem && grp.s != kNone) {
                    const auto &st = model.ops[grp.s];
                    p.stem_x = static_cast<const float *>(ptr(st.in));
                    p.stem_w16 = stem_w16.get();
                    p.stem_b = wptr<float>(st.b_off);
                    p.stem_lin = static_cast<int>(tensor(st.in).length);
                    p.stem_cin = static_cast<int>(st.cin);
                    p.stem_k = static_cast<int>(st.k);
                    p.stem_stride = static_cast<int>(st.stride);
                    p.stem_pad = static_cast<int>(st.pad);
                    p.x_inv_scale = 1.f / act_scale[st.out]; // as the unfused stem's y_inv_scale
                    if (!p.stem_x) p.stem_x = reinterpret_cast<const float *>(16); // shape check before allocation
                }
            }
            return p;
        }
        bool block_variant_ok(const BlockVariant &v, bool with_stem) const {
            return v.int8 == cfg.int8 && v.stem == with_stem && starts_with(v.name, cfg.block_prefix);
        }

        bool allowed(const ConvVariant &v) const {
            if (v.int8 != cfg.int8) return false;
            for (const auto &x : cfg.exclude_prefixes)
                if (starts_with(v.name, x)) return false;
            return true;
        }

        // ------------------------------------------------------------ fusion candidates (shape-only)
        void find_fusions() {
            const auto nt = model.tensors.size(), no = model.ops.size();
            std::vector<int> consumers(nt, 0);
            std::vector<std::size_t> producer(nt, kNone);
            for (std::size_t i = 0; i < no; ++i) {
                const auto &op = model.ops[i];
                ++consumers[op.in];
                if (op.res != kNoTensor) ++consumers[op.res];
                producer[op.out] = i;
            }
            group_of.assign(no, kNone);
            auto is_conv = [&](std::size_t i, std::uint32_t k, std::uint32_t pad) {
                return i != kNone && model.ops[i].kind == OpKind::conv && model.ops[i].k == k && model.ops[i].pad == pad;
            };
            // GAP: the head's input is produced by a conv that nothing else reads
            for (std::size_t i = 0; i < no; ++i) {
                if (model.ops[i].kind != OpKind::gap_linear) continue;
                head_op = i;
                const auto t = model.ops[i].in;
                if (!cfg.fuse_gap || consumers[t] != 1 || producer[t] == kNone) break;
                const std::size_t pi = producer[t];
                pool_op = pi; // tentatively; must be poolable as a conv (blocks always are)
                const auto p = conv_params(pi, 1);
                bool any = false;
                for (const auto &v : conv_variants()) any = any || (v.pool && allowed(v) && v.supported(p));
                if (!any) pool_op = kNone;
                break;
            }
            if (!cfg.fuse_blocks) return;
            for (std::size_t ib = 0; ib < no; ++ib) {
                const auto &b = model.ops[ib];
                if (!is_conv(ib, 3, 1) || b.stride != 1 || !b.relu || b.res == kNoTensor) continue;
                const std::size_t ia = producer[b.in];
                if (!is_conv(ia, 3, 1) || consumers[b.in] != 1) continue;
                const auto &a = model.ops[ia];
                if (!a.relu || a.res != kNoTensor || a.in == model.input || a.cout != b.cin) continue;
                Group g{ia, ib, kNone};
                if (b.res != a.in) {
                    const std::size_t ic = producer[b.res];
                    if (!is_conv(ic, 1, 0) || consumers[b.res] != 1) continue;
                    const auto &c = model.ops[ic];
                    if (c.in != a.in || c.stride != a.stride || c.relu || c.res != kNoTensor) continue;
                    g.c = ic;
                } else if (a.stride != 1) {
                    continue;
                }
                groups.push_back(g);
                auto any_variant = [&](bool with_stem) {
                    const auto bp = block_params(groups.size() - 1, 1, with_stem);
                    for (const auto &v : block_variants())
                        if (block_variant_ok(v, with_stem) && v.supported(bp)) return true;
                    return false;
                };
                // stem in front: a conv on the graph input whose output only this block reads
                const std::size_t is = producer[a.in];
                if (cfg.int8 && is != kNone && model.ops[is].kind == OpKind::conv && model.ops[is].in == model.input &&
                    model.ops[is].relu && model.ops[is].res == kNoTensor && is != pool_op &&
                    consumers[a.in] == 1 + (b.res == a.in) + (g.c != kNone)) {
                    groups.back().s = is;
                    if (!any_variant(true)) groups.back().s = kNone;
                }
                group_of[ia] = group_of[ib] = groups.size() - 1;
                if (g.c != kNone) group_of[g.c] = groups.size() - 1;
                if (cfg.int8 && g.c != kNone) { // conv2 with the shortcut fused in
                    const auto cp = conv_params(ib, 1, true);
                    for (const auto &v : conv_variants()) groups.back().sc = groups.back().sc || (allowed(v) && v.supported(cp));
                }
                if (groups.back().s == kNone && !groups.back().sc && !any_variant(false)) {
                    group_of[ia] = group_of[ib] = kNone;
                    if (g.c != kNone) group_of[g.c] = kNone;
                    groups.pop_back();
                    continue;
                }
            }
        }

        // ------------------------------------------------------------ execution
        void launch_step(const Step &st, std::size_t bi, int batch, cudaStream_t s) const {
            const auto &op = model.ops[st.op];
            switch (st.kind) {
                case Step::Kind::conv:
                    if (st.sc) launch_conv(*sc_choice[bi][st.group], conv_params(st.op, batch, true), s);
                    else launch_conv(*conv_choice[bi][st.op], conv_params(st.op, batch), s);
                    break;
                case Step::Kind::block:
                    (st.stem ? stem_choice : block_choice)[bi][st.group]->launch(block_params(st.group, batch, st.stem), s);
                    break;
                case Step::Kind::gap_linear:
                    launch_gap_linear(static_cast<const __half *>(ptr(op.in)), wptr<float>(op.w_off), wptr<float>(op.b_off),
                                      static_cast<float *>(ptr(op.out)), batch, static_cast<int>(tensor(op.in).length),
                                      static_cast<int>(op.cin), static_cast<int>(op.cout), s);
                    break;
                case Step::Kind::linear_from_pool:
                    launch_linear_from_pool(pool.get(), wptr<float>(op.w_off), wptr<float>(op.b_off),
                                            static_cast<float *>(ptr(op.out)), batch, static_cast<int>(op.cin),
                                            static_cast<int>(op.cout), 1.f / static_cast<float>(tensor(op.in).length), s);
                    break;
            }
        }

        void run_eager(std::size_t bi, int batch, cudaStream_t s) const {
            if (const auto pc = pool_channels())
                CURF_CUDA_CHECK(cudaMemsetAsync(pool.get(), 0, static_cast<std::size_t>(batch) * pc * sizeof(float), s));
            for (const auto &st : steps[bi]) launch_step(st, bi, batch, s);
        }

        // ------------------------------------------------------------ kernel selection
        std::string conv_key(std::size_t i, int batch, bool with_sc = false) const {
            const auto &op = model.ops[i];
            std::ostringstream k;
            k << gpu << '|' << (cfg.int8 ? "I8" : "") << (with_sc ? "SC" : "") << cfg.variant_prefix << '|';
            for (const auto &x : cfg.exclude_prefixes) k << '-' << x;
            k << '|' << op.cin << ',' << op.cout << ',' << op.k << ',' << op.stride << ',' << op.pad << ','
              << tensor(op.in).length << ',' << (op.res != kNoTensor) << op.relu << (op.in == model.input)
              << (i == pool_op) << '|' << batch;
            auto s = k.str();
            std::replace(s.begin(), s.end(), ' ', '_');
            return s;
        }
        std::string block_key(std::size_t g, int batch, bool with_stem = false) const {
            const auto p = block_params(g, batch, with_stem);
            std::ostringstream k;
            k << gpu << "|B" << (cfg.int8 ? "I8" : "") << (with_stem ? "S" : "") << cfg.block_prefix << '|' << p.cin << ','
              << p.cout << ',' << p.stride << ',' << (p.wsc != nullptr) << ',' << p.lin << ',' << (groups[g].b == pool_op)
              << '|' << batch;
            auto s = k.str();
            std::replace(s.begin(), s.end(), ' ', '_');
            return s;
        }

        float time_us(const std::function<void()> &launch) const {
            cudaEvent_t a, b;
            CURF_CUDA_CHECK(cudaEventCreate(&a));
            CURF_CUDA_CHECK(cudaEventCreate(&b));
            for (int i = 0; i < 3; ++i) launch();
            CURF_CUDA_CHECK(cudaEventRecord(a, stream));
            for (int i = 0; i < cfg.tune_iters; ++i) launch();
            CURF_CUDA_CHECK(cudaEventRecord(b, stream));
            CURF_CUDA_CHECK(cudaEventSynchronize(b));
            float ms = 0;
            CURF_CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
            cudaEventDestroy(a);
            cudaEventDestroy(b);
            return 1000.f * ms / static_cast<float>(cfg.tune_iters);
        }

        std::map<std::string, std::pair<std::string, float>> load_cache() const {
            std::map<std::string, std::pair<std::string, float>> c;
            if (cfg.tune_cache.empty()) return c;
            std::ifstream f(cfg.tune_cache);
            std::string key, var;
            float us;
            while (f >> key >> var >> us) c[key] = {var, us};
            return c;
        }

        const ConvVariant *pick_conv(const ConvParams &p, float &best_us) const {
            const ConvVariant *best = nullptr;
            // Most specific prefix first: "wmma.f32.fused.64x64" falls back to "wmma.f32.fused." ... "" for ops
            // that tile cannot handle (e.g. cout = 32).
            for (std::string prefix = cfg.variant_prefix;;) {
                for (const auto &v : conv_variants()) {
                    if (!starts_with(v.name, prefix) || !allowed(v) || !v.supported(p) || (p.pool && !v.pool)) continue;
                    if (!cfg.autotune) { best = &v; break; }
                    const float us = time_us([&] { launch_conv(v, p, stream); });
                    if (!best || us < best_us) { best = &v; best_us = us; }
                }
                if (best || prefix.empty()) return best;
                const auto dot = prefix.size() < 2 ? std::string::npos : prefix.find_last_of('.', prefix.size() - 2);
                prefix.resize(dot == std::string::npos ? 0 : dot + 1); // ... -> "mma." -> "" (everything)
            }
        }

        const BlockVariant *pick_block(const BlockParams &p, float &best_us) const {
            const BlockVariant *best = nullptr;
            for (const auto &v : block_variants()) {
                if (!block_variant_ok(v, p.stem_x != nullptr) || !v.supported(p)) continue;
                if (!cfg.autotune) { best = &v; break; }
                const float us = time_us([&] { v.launch(p, stream); });
                if (!best || us < best_us) { best = &v; best_us = us; }
            }
            return best;
        }

        std::vector<Step> make_steps(const std::vector<Fuse> &mode) const {
            std::vector<Step> out;
            std::vector<bool> stem_fused(model.ops.size(), false);
            for (std::size_t g = 0; g < groups.size(); ++g)
                if (mode[g] == Fuse::stem) stem_fused[groups[g].s] = true;
            for (std::size_t i = 0; i < model.ops.size(); ++i) {
                const auto g = group_of[i];
                if (stem_fused[i]) continue; // inside its block
                if (g != kNone && mode[g] == Fuse::shortcut) {
                    if (i == groups[g].a) out.push_back({Step::Kind::conv, i});
                    else if (i == groups[g].b) out.push_back({Step::Kind::conv, i, g, false, true});
                } else if (g != kNone && mode[g] != Fuse::none) {
                    if (i == groups[g].b) out.push_back({Step::Kind::block, i, g, mode[g] == Fuse::stem}); // where conv2 was
                } else if (model.ops[i].kind == OpKind::conv) {
                    out.push_back({Step::Kind::conv, i});
                } else {
                    out.push_back({pool_op != kNone ? Step::Kind::linear_from_pool : Step::Kind::gap_linear, i});
                }
            }
            return out;
        }

        // Tune every conv (as if unfused) and every block candidate per bucket, then decide fusion per bucket.
        // Needs memory planned with every candidate tensor materialised.
        void select_kernels() {
            auto cache = load_cache();
            bool dirty = false;
            const auto no = model.ops.size();
            conv_choice.assign(buckets.size(), std::vector<const ConvVariant *>(no, nullptr));
            conv_us.assign(buckets.size(), std::vector<float>(no, 0.f));
            block_choice.assign(buckets.size(), std::vector<const BlockVariant *>(groups.size(), nullptr));
            stem_choice = block_choice;
            sc_choice.assign(buckets.size(), std::vector<const ConvVariant *>(groups.size(), nullptr));
            fuse_mode.assign(buckets.size(), std::vector<Fuse>(groups.size(), Fuse::none));
            steps.assign(buckets.size(), {});
            auto cached = [&](const std::string &key, auto find, auto pick, float &us) {
                using V = decltype(pick(us));
                V v = nullptr;
                if (const auto hit = cache.find(key); cfg.autotune && hit != cache.end()) {
                    v = find(hit->second.first);
                    us = hit->second.second;
                }
                if (!v && (v = pick(us)) && cfg.autotune) { cache[key] = {v->name, us}; dirty = true; }
                return v;
            };
            for (std::size_t bi = 0; bi < buckets.size(); ++bi) {
                const int batch = buckets[bi];
                for (std::size_t i = 0; i < no; ++i) {
                    if (model.ops[i].kind != OpKind::conv) continue;
                    const auto p = conv_params(i, batch);
                    float us = 0;
                    const ConvVariant *v = cached(conv_key(i, batch), [](const std::string &n) { return find_conv_variant(n); },
                                                  [&](float &u) { return pick_conv(p, u); }, us);
                    if (!v) throw std::runtime_error("engine: no conv variant for " + model.ops[i].name);
                    conv_choice[bi][i] = v;
                    conv_us[bi][i] = us;
                }
                auto &mode = fuse_mode[bi];
                std::vector<float> block_us(groups.size(), 0.f), stem_us(groups.size(), 0.f), sc_us(groups.size(), 0.f);
                auto find_block = [](const std::string &n) -> const BlockVariant * {
                    for (const auto &x : block_variants())
                        if (x.name == n) return &x;
                    return nullptr;
                };
                for (std::size_t g = 0; g < groups.size(); ++g) {
                    const auto p = block_params(g, batch);
                    const BlockVariant *v = cached(block_key(g, batch), find_block, [&](float &u) { return pick_block(p, u); },
                                                   block_us[g]);
                    block_choice[bi][g] = v;
                    mode[g] = v ? Fuse::block : Fuse::none;
                    if (groups[g].s != kNone) {
                        const auto ps = block_params(g, batch, true);
                        stem_choice[bi][g] = cached(block_key(g, batch, true), find_block,
                                                    [&](float &u) { return pick_block(ps, u); }, stem_us[g]);
                        if (stem_choice[bi][g]) mode[g] = Fuse::stem;
                    }
                    if (groups[g].sc) {
                        const auto pc = conv_params(groups[g].b, batch, true);
                        sc_choice[bi][g] = cached(conv_key(groups[g].b, batch, true),
                                                  [](const std::string &n) { return find_conv_variant(n); },
                                                  [&](float &u) { return pick_conv(pc, u); }, sc_us[g]);
                        if (sc_choice[bi][g] && mode[g] == Fuse::none) mode[g] = Fuse::shortcut;
                    }
                }
                // Fusion decided in context: toggle one block at a time and time the whole network, alternating
                // the two plans for three rounds (best of each). Per-kernel timings in isolation mislead here: an
                // unfused conv re-launched on the same buffers finds its input in L2, which the real sequence
                // does not give it, and the balance shifts with the laptop GPU's clocks as it heats up.
                // With a stem candidate there are three plans (separate convs, block, stem + block), timed the same way.
                for (std::size_t g = 0; g < groups.size(); ++g) {
                    const auto &grp = groups[g];
                    FusionDecision d;
                    d.bucket = batch;
                    d.block = model.ops[grp.b].name;
                    if (block_choice[bi][g]) d.fused_variant = block_choice[bi][g]->name;
                    if (stem_choice[bi][g]) d.stem_variant = stem_choice[bi][g]->name;
                    std::vector<Fuse> options{Fuse::none};
                    if (block_choice[bi][g]) options.push_back(Fuse::block);
                    if (stem_choice[bi][g]) options.push_back(Fuse::stem);
                    if (sc_choice[bi][g]) {
                        options.push_back(Fuse::shortcut);
                        d.sc_variant = sc_choice[bi][g]->name;
                    }
                    if (options.size() > 1 && cfg.autotune) {
                        float best[4] = {1e30f, 1e30f, 1e30f, 1e30f};
                        for (int r = 0; r < 3; ++r)
                            for (const Fuse o : options) {
                                mode[g] = o;
                                steps[bi] = make_steps(mode);
                                const float t = time_us([&] { run_eager(bi, batch, stream); });
                                auto &b = best[static_cast<int>(o)];
                                b = std::min(b, t);
                            }
                        mode[g] = Fuse::none;
                        for (const Fuse o : options)
                            if (best[static_cast<int>(o)] < best[static_cast<int>(mode[g])]) mode[g] = o;
                        d.unfused_us = best[0];
                        d.fused_us = best[1] < 1e29f ? best[1] : 0.f;
                        d.stem_fused_us = best[2] < 1e29f ? best[2] : 0.f;
                        d.sc_fused_us = best[3] < 1e29f ? best[3] : 0.f;
                    }
                    d.fused = mode[g] != Fuse::none;
                    d.stem = mode[g] == Fuse::stem;
                    d.sc = mode[g] == Fuse::shortcut;
                    decisions.push_back(d);
                }
                steps[bi] = make_steps(mode);
                for (const auto &st : steps[bi]) {
                    const auto &op = model.ops[st.op];
                    OpChoice oc{batch, op.name + (st.op == pool_op ? "+gap" : ""), "", 0.f, 0.0};
                    if (st.kind == Step::Kind::block) {
                        oc.op = (st.stem ? "block:" + model.ops[groups[st.group].s].name + "+" : "block:") + oc.op;
                        oc.variant = (st.stem ? stem_choice : block_choice)[bi][st.group]->name;
                        oc.us = (st.stem ? stem_us : block_us)[st.group];
                        oc.flops = block_params(st.group, batch, st.stem).flops();
                    } else if (st.kind == Step::Kind::conv && st.sc) {
                        oc.op = model.ops[groups[st.group].c].name + "+" + oc.op;
                        oc.variant = sc_choice[bi][st.group]->name;
                        oc.us = sc_us[st.group];
                        oc.flops = conv_params(st.op, batch, true).flops();
                    } else if (st.kind == Step::Kind::conv) {
                        oc.variant = conv_choice[bi][st.op]->name;
                        oc.us = conv_us[bi][st.op];
                        oc.flops = conv_params(st.op, batch).flops();
                    } else {
                        oc.flops = 2.0 * batch * op.cin * op.cout;
                    }
                    choices.push_back(oc);
                }
            }
            if (dirty && !cfg.tune_cache.empty()) {
                std::ofstream f(cfg.tune_cache);
                for (const auto &[k, v] : cache) f << k << ' ' << v.first << ' ' << v.second << '\n';
            }
        }

        // Tensors some bucket's plan reads or writes in memory.
        std::vector<bool> needed_tensors() const {
            std::vector<bool> need(model.tensors.size(), false);
            auto mark_op = [&](std::size_t i) {
                const auto &op = model.ops[i];
                need[op.in] = true;
                if (op.res != kNoTensor) need[op.res] = true;
                if (i != pool_op) need[op.out] = true;
            };
            for (const auto &plan_steps : steps)
                for (const auto &st : plan_steps) {
                    if (st.kind == Step::Kind::block) {
                        const auto &g = groups[st.group];
                        if (!st.stem) need[model.ops[g.a].in] = true;
                        if (g.b != pool_op) need[model.ops[g.b].out] = true;
                    } else if (st.kind == Step::Kind::linear_from_pool) {
                        need[model.ops[st.op].out] = true;
                    } else if (st.sc) { // reads conv1's output and the block input, not the shortcut output
                        const auto &op = model.ops[st.op];
                        need[op.in] = true;
                        need[model.ops[groups[st.group].c].in] = true;
                        if (st.op != pool_op) need[op.out] = true;
                    } else {
                        mark_op(st.op);
                    }
                }
            need[model.input] = need[model.output] = true;
            return need;
        }

        void allocate_arena() {
            const auto maxb = static_cast<std::size_t>(buckets.back());
            plan = plan_memory(model, maxb, cfg.int8 ? 1 : sizeof(__half), cfg.reuse_memory, &materialized);
            arena = DeviceBuffer<std::byte>(plan.arena_bytes);
            if (arena.bytes()) CURF_CUDA_CHECK(cudaMemset(arena.get(), 0, arena.bytes()));
        }

        void capture_graphs() {
            graphs.assign(buckets.size(), nullptr);
            for (std::size_t bi = 0; bi < buckets.size(); ++bi) {
                // Warm up outside capture (module loading, smem attributes, static launch config).
                run_eager(bi, buckets[bi], stream);
                CURF_CUDA_CHECK(cudaStreamSynchronize(stream));
                if (!cfg.cuda_graphs) continue;
                cudaGraph_t g = nullptr;
                CURF_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                run_eager(bi, buckets[bi], stream);
                CURF_CUDA_CHECK(cudaStreamEndCapture(stream, &g));
                CURF_CUDA_CHECK(cudaGraphInstantiate(&graphs[bi], g, 0));
                CURF_CUDA_CHECK(cudaGraphDestroy(g));
            }
        }

        // ------------------------------------------------------------ INT8
        // Activation scales: the fp16 engine (no fusion, every tensor kept, fp32-accumulating kernels) runs over
        // the calibration windows; pass 1 finds max |v| per tensor, pass 2 a 4096-bin histogram of |v| / max,
        // from which the calib_percentile quantile is read.
        void calibrate() {
            const auto &in = model.tensors[model.input];
            const std::size_t per = in.elems_per_window();
            if (cfg.calibration.empty() || cfg.calibration.size() % per != 0)
                throw std::invalid_argument("engine: int8 needs calibration windows [n][L][C]");
            const int n = static_cast<int>(cfg.calibration.size() / per);
            const int B = std::min(n, 256);
            EngineConfig c;
            c.buckets = {B};
            c.variant_prefix = "mma.f32.";
            c.fuse_blocks = c.fuse_gap = c.autotune = c.cuda_graphs = c.reuse_memory = false;
            Engine fp(model, c);
            const auto nt = model.tensors.size();
            std::vector<double> mx(nt, 0.0);
            constexpr int BINS = 4096;
            std::vector<std::vector<std::uint64_t>> hist(nt, std::vector<std::uint64_t>(BINS, 0));
            const bool use_hist = cfg.calib_percentile < 1.0;
            for (int pass = 0; pass < (use_hist ? 2 : 1); ++pass)
                for (int b0 = 0; b0 < n; b0 += B) {
                    const int nb = std::min(B, n - b0);
                    CURF_CUDA_CHECK(cudaMemcpy(fp.input(), cfg.calibration.data() + static_cast<std::size_t>(b0) * per,
                                               static_cast<std::size_t>(nb) * per * sizeof(float), cudaMemcpyHostToDevice));
                    fp.run(B, nullptr);
                    for (std::uint32_t t = 0; t < nt; ++t) {
                        if (t == model.input || t == model.output) continue;
                        const auto v = fp.read_tensor(t, nb);
                        if (pass == 0) {
                            for (const float x : v) mx[t] = std::max(mx[t], static_cast<double>(std::fabs(x)));
                        } else if (mx[t] > 0) {
                            const double k = BINS / mx[t];
                            for (const float x : v) ++hist[t][std::min<std::size_t>(BINS - 1, static_cast<std::size_t>(std::fabs(x) * k))];
                        }
                    }
                }
            act_scale.assign(nt, 0.f);
            for (std::uint32_t t = 0; t < nt; ++t) {
                if (t == model.input || t == model.output) continue;
                double a = mx[t];
                if (use_hist && a > 0) {
                    std::uint64_t total = 0, run = 0;
                    for (auto h : hist[t]) total += h;
                    const auto target = static_cast<std::uint64_t>(std::ceil(cfg.calib_percentile * static_cast<double>(total)));
                    for (int i = 0; i < BINS; ++i)
                        if ((run += hist[t][i]) >= target) { a = mx[t] * (i + 1) / BINS; break; }
                }
                act_scale[t] = static_cast<float>(std::max(a, 1e-8) / 127.0);
            }
        }

        // Weights: per output channel s_w = max |w| / 127, rows re-padded to a multiple of 64 bytes;
        // alpha[n] = s_w[n] * s_x (s_x: the conv input's scale; 1 for the fp32 stem, which keeps fp16 weights).
        void quantize_weights() {
            const auto no = model.ops.size();
            kpad8.assign(no, 0);
            w8_off.assign(no, 0);
            alpha_off.assign(no, 0);
            std::vector<std::int8_t> hw;
            std::vector<float> ha;
            for (std::size_t i = 0; i < no; ++i) {
                const auto &op = model.ops[i];
                if (op.kind != OpKind::conv) continue;
                const int K = static_cast<int>(op.k * op.cin), kp = (K + 63) / 64 * 64;
                kpad8[i] = op.in == model.input ? static_cast<int>(op.kpad) : kp; // stem: fp16 weights, kpad unchanged
                w8_off[i] = hw.size();
                alpha_off[i] = ha.size();
                const auto *w = reinterpret_cast<const __half *>(model.blob.data() + op.w_off);
                const float sx = op.in == model.input ? 1.f : act_scale[op.in];
                for (std::uint32_t nn = 0; nn < op.cout; ++nn) {
                    float a = 0.f;
                    for (int k = 0; k < K; ++k) a = std::max(a, std::fabs(__half2float(w[nn * op.kpad + k])));
                    const float sw = std::max(a, 1e-12f) / 127.f;
                    ha.push_back(sw * sx);
                    for (int k = 0; k < kp; ++k) {
                        const float v = k < K ? __half2float(w[nn * op.kpad + k]) / sw : 0.f;
                        hw.push_back(static_cast<std::int8_t>(std::lround(std::clamp(v, -127.f, 127.f))));
                    }
                }
                while (hw.size() % 64) hw.push_back(0); // 64-byte aligned rows for the next op
                while (ha.size() % 4) ha.push_back(0.f);
                if (op.in == model.input && K <= 16) { // stem: fp16 rows of 16 for the stem-fused block's MMA
                    std::vector<__half> t(std::size_t{16} * op.cout, __float2half(0.f));
                    for (std::uint32_t nn = 0; nn < op.cout; ++nn)
                        for (int k = 0; k < K; ++k) t[nn * 16 + k] = w[nn * op.kpad + k];
                    stem_w16 = DeviceBuffer<__half>(t.size());
                    CURF_CUDA_CHECK(cudaMemcpy(stem_w16.get(), t.data(), t.size() * sizeof(__half), cudaMemcpyHostToDevice));
                }
            }
            w8 = DeviceBuffer<std::int8_t>(hw.size());
            alpha = DeviceBuffer<float>(ha.size());
            CURF_CUDA_CHECK(cudaMemcpy(w8.get(), hw.data(), hw.size(), cudaMemcpyHostToDevice));
            CURF_CUDA_CHECK(cudaMemcpy(alpha.get(), ha.data(), ha.size() * sizeof(float), cudaMemcpyHostToDevice));
        }

        std::size_t bucket_index(int batch) const {
            if (batch <= 0 || batch > buckets.back())
                throw std::out_of_range("engine: batch " + std::to_string(batch) + " outside 1.." +
                                        std::to_string(buckets.back()));
            return static_cast<std::size_t>(std::lower_bound(buckets.begin(), buckets.end(), batch) - buckets.begin());
        }
    };

    Engine::Engine(Model model, EngineConfig cfg) : impl_(std::make_unique<Impl>()) {
        auto &m = *impl_;
        m.model = std::move(model);
        m.cfg = std::move(cfg);
        m.buckets = m.cfg.buckets;
        std::sort(m.buckets.begin(), m.buckets.end());
        m.buckets.erase(std::unique(m.buckets.begin(), m.buckets.end()), m.buckets.end());
        if (m.buckets.empty() || m.buckets.front() <= 0) throw std::invalid_argument("engine: bad batch buckets");

        for (const auto &op : m.model.ops) {
            if (op.kind == OpKind::conv && op.out == m.model.output)
                throw std::runtime_error("engine: a conv cannot produce the graph output (fp32 logits expected)");
            if (op.kind == OpKind::gap_linear && op.in == m.model.input)
                throw std::runtime_error("engine: gap_linear on the raw input is not supported");
            if (op.res == m.model.input) throw std::runtime_error("engine: residual from the raw input not supported");
        }

        int dev = 0;
        cudaDeviceProp prop{};
        CURF_CUDA_CHECK(cudaGetDevice(&dev));
        CURF_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        m.gpu = prop.name;

        m.weights = DeviceBuffer<std::byte>(m.model.blob.size());
        CURF_CUDA_CHECK(cudaMemcpy(m.weights.get(), m.model.blob.data(), m.model.blob.size(), cudaMemcpyHostToDevice));

        if (m.cfg.int8) {
            m.calibrate();
            m.quantize_weights();
        }

        // Fusion candidates only depend on shapes, so they are found before activations exist (null pointers).
        m.materialized.assign(m.model.tensors.size(), true);
        m.find_fusions();
        if (m.cfg.int8 && m.pool_op == kNone) throw std::runtime_error("engine: int8 needs the GAP fusion (fuse_gap)");

        const auto maxb = static_cast<std::size_t>(m.buckets.back());
        m.input = DeviceBuffer<float>(maxb * m.model.tensors[m.model.input].elems_per_window());
        m.logits = DeviceBuffer<float>(maxb * m.model.tensors[m.model.output].elems_per_window());
        if (const auto pc = m.pool_channels()) m.pool = DeviceBuffer<float>(maxb * pc);
        CURF_CUDA_CHECK(cudaMemset(m.input.get(), 0, m.input.bytes()));
        CURF_CUDA_CHECK(cudaStreamCreateWithFlags(&m.stream, cudaStreamNonBlocking));

        // Tune with everything a plan might need in memory (pooled tensor excluded), decide fusion per bucket,
        // then re-plan the arena for what the chosen plans actually use.
        if (m.pool_op != kNone) m.materialized[m.model.ops[m.pool_op].out] = false;
        m.allocate_arena();
        m.select_kernels();
        m.materialized = m.needed_tensors();
        m.allocate_arena();
        m.capture_graphs();
    }

    Engine::~Engine() {
        if (!impl_) return;
        for (auto g : impl_->graphs)
            if (g) cudaGraphExecDestroy(g);
        if (impl_->stream) cudaStreamDestroy(impl_->stream);
    }

    const Model &Engine::model() const { return impl_->model; }
    const EngineConfig &Engine::config() const { return impl_->cfg; }
    int Engine::max_batch() const { return impl_->buckets.back(); }
    float *Engine::input() const { return impl_->input.get(); }
    const float *Engine::logits() const { return impl_->logits.get(); }
    const MemoryPlan &Engine::memory_plan() const { return impl_->plan; }
    const std::vector<OpChoice> &Engine::choices() const { return impl_->choices; }
    const std::vector<float> &Engine::activation_scales() const { return impl_->act_scale; }
    const std::vector<FusionDecision> &Engine::fusion_decisions() const { return impl_->decisions; }
    bool Engine::materialized(std::uint32_t t) const { return impl_->materialized.at(t); }
    std::size_t Engine::steps(int batch) const { return impl_->steps[impl_->bucket_index(batch)].size(); }

    std::size_t Engine::device_bytes() const {
        return impl_->arena.bytes() + impl_->weights.bytes() + impl_->input.bytes() + impl_->logits.bytes() +
               impl_->pool.bytes();
    }

    int Engine::bucket_for(int batch) const { return impl_->buckets[impl_->bucket_index(batch)]; }

    void Engine::run(int batch, cudaStream_t stream) {
        const auto bi = impl_->bucket_index(batch);
        if (impl_->cfg.cuda_graphs) CURF_CUDA_CHECK(cudaGraphLaunch(impl_->graphs[bi], stream));
        else impl_->run_eager(bi, impl_->buckets[bi], stream);
    }

    void Engine::run_eager(int batch, cudaStream_t stream) {
        const auto bi = impl_->bucket_index(batch);
        impl_->run_eager(bi, impl_->buckets[bi], stream);
    }

    std::vector<float> Engine::read_tensor(std::uint32_t t, int batch) const {
        const auto &m = *impl_;
        if (t >= m.model.tensors.size()) throw std::out_of_range("engine: tensor index");
        if (!m.materialized[t]) throw std::logic_error("engine: tensor " + std::to_string(t) + " only exists inside a fused kernel");
        if (batch <= 0 || batch > max_batch()) throw std::out_of_range("engine: batch");
        const std::size_t n = static_cast<std::size_t>(batch) * m.model.tensors[t].elems_per_window();
        std::vector<float> out(n);
        CURF_CUDA_CHECK(cudaStreamSynchronize(m.stream));
        CURF_CUDA_CHECK(cudaDeviceSynchronize());
        if (t == m.model.input || t == m.model.output) {
            CURF_CUDA_CHECK(cudaMemcpy(out.data(), m.ptr(t), n * sizeof(float), cudaMemcpyDeviceToHost));
            return out;
        }
        if (m.scratch.size() < n) m.scratch = DeviceBuffer<float>(n);
        if (m.cfg.int8) launch_int8_to_float(static_cast<const std::int8_t *>(m.ptr(t)), m.scratch.get(), n, m.act_scale[t], m.stream);
        else launch_half_to_float(static_cast<const __half *>(m.ptr(t)), m.scratch.get(), n, m.stream);
        CURF_CUDA_CHECK(cudaStreamSynchronize(m.stream));
        CURF_CUDA_CHECK(cudaMemcpy(out.data(), m.scratch.get(), n * sizeof(float), cudaMemcpyDeviceToHost));
        return out;
    }

} // namespace curf::engine
