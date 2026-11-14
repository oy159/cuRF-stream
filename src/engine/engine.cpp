#include "curf/engine/engine.hpp"

#include "curf/cuda/memory.hpp"

#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace curf::engine {

    using cuda::DeviceBuffer;

    struct Engine::Impl {
        Model model;
        EngineConfig cfg;
        MemoryPlan plan;
        DeviceBuffer<std::byte> arena, weights;
        DeviceBuffer<float> input, logits;
        mutable DeviceBuffer<float> scratch; // read_tensor conversions
        cudaStream_t stream = nullptr;       // private stream for tuning and capture
        std::vector<int> buckets;
        std::vector<std::vector<const ConvVariant *>> chosen; // [bucket][op]
        std::vector<cudaGraphExec_t> graphs;                   // [bucket]
        std::vector<OpChoice> choices;
        std::string gpu;

        const TensorDesc &tensor(std::uint32_t t) const { return model.tensors[t]; }

        void *ptr(std::uint32_t t) const {
            if (t == model.input) return input.get();
            if (t == model.output) return logits.get();
            return arena.get() + plan.offset[t];
        }

        ConvParams conv_params(const OpDesc &op, int batch) const {
            ConvParams p;
            p.x = ptr(op.in);
            p.x_f32 = op.in == model.input;
            p.w = reinterpret_cast<const __half *>(weights.get() + op.w_off);
            p.bias = reinterpret_cast<const float *>(weights.get() + op.b_off);
            p.res = op.res == kNoTensor ? nullptr : static_cast<const __half *>(ptr(op.res));
            p.y = static_cast<__half *>(ptr(op.out));
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
            return p;
        }

        void launch_op(std::size_t i, int bucket_index, int batch, cudaStream_t s) const {
            const auto &op = model.ops[i];
            if (op.kind == OpKind::conv) {
                launch_conv(*chosen[bucket_index][i], conv_params(op, batch), s);
            } else {
                launch_gap_linear(static_cast<const __half *>(ptr(op.in)),
                                  reinterpret_cast<const float *>(weights.get() + op.w_off),
                                  reinterpret_cast<const float *>(weights.get() + op.b_off),
                                  static_cast<float *>(ptr(op.out)), batch, static_cast<int>(tensor(op.in).length),
                                  static_cast<int>(op.cin), static_cast<int>(op.cout), s);
            }
        }

        void run_eager(int bucket_index, int batch, cudaStream_t s) const {
            for (std::size_t i = 0; i < model.ops.size(); ++i) launch_op(i, bucket_index, batch, s);
        }

        std::string op_key(const OpDesc &op, int batch) const {
            std::ostringstream k;
            k << gpu << '|' << (cfg.variant_prefix.empty() ? "*" : cfg.variant_prefix) << '|' << op.cin << ','
              << op.cout << ',' << op.k << ',' << op.stride << ',' << op.pad << ',' << tensor(op.in).length << ','
              << (op.res != kNoTensor) << op.relu << (op.in == model.input) << '|' << batch;
            return k.str();
        }

        float time_variant(const ConvVariant &v, const ConvParams &p) const {
            cudaEvent_t a, b;
            CURF_CUDA_CHECK(cudaEventCreate(&a));
            CURF_CUDA_CHECK(cudaEventCreate(&b));
            for (int i = 0; i < 3; ++i) launch_conv(v, p, stream);
            CURF_CUDA_CHECK(cudaEventRecord(a, stream));
            for (int i = 0; i < cfg.tune_iters; ++i) launch_conv(v, p, stream);
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

        void select_kernels() {
            auto cache = load_cache();
            bool cache_dirty = false;
            chosen.assign(buckets.size(), std::vector<const ConvVariant *>(model.ops.size(), nullptr));
            for (std::size_t bi = 0; bi < buckets.size(); ++bi) {
                const int batch = buckets[bi];
                for (std::size_t i = 0; i < model.ops.size(); ++i) {
                    const auto &op = model.ops[i];
                    OpChoice oc{batch, op.name, "", 0.f, 0.0};
                    if (op.kind != OpKind::conv) {
                        oc.flops = 2.0 * batch * op.cin * op.cout;
                        choices.push_back(oc);
                        continue;
                    }
                    const auto p = conv_params(op, batch);
                    oc.flops = p.flops();
                    const auto key = op_key(op, batch);
                    const ConvVariant *best = nullptr;
                    float best_us = 0;
                    if (auto it = cache.find(key); cfg.autotune && it != cache.end()) {
                        best = find_conv_variant(it->second.first);
                        best_us = it->second.second;
                    }
                    // Most specific prefix first: "wmma.f32.fused.64x64" falls back to "wmma.f32.fused."
                    // for ops that tile cannot handle (e.g. cout = 32).
                    for (std::string prefix = cfg.variant_prefix; !best;) {
                        for (const auto &v : conv_variants()) {
                            if (v.name.rfind(prefix, 0) != 0 || !v.supported(p)) continue;
                            if (!cfg.autotune) { best = &v; break; }
                            const float us = time_variant(v, p);
                            if (!best || us < best_us) { best = &v; best_us = us; }
                        }
                        if (best && cfg.autotune) { cache[key] = {best->name, best_us}; cache_dirty = true; }
                        if (best || prefix.empty()) break;
                        const auto dot = prefix.size() < 2 ? std::string::npos : prefix.find_last_of('.', prefix.size() - 2);
                        prefix.resize(dot == std::string::npos ? 0 : dot + 1); // ... -> "mma." -> "" (everything)
                    }
                    if (!best) throw std::runtime_error("engine: no conv variant matching '" + cfg.variant_prefix +
                                                        "' supports " + op.name);
                    chosen[bi][i] = best;
                    oc.variant = best->name;
                    oc.us = best_us;
                    choices.push_back(oc);
                }
            }
            if (cache_dirty && !cfg.tune_cache.empty()) {
                std::ofstream f(cfg.tune_cache);
                for (const auto &[k, v] : cache) f << k << ' ' << v.first << ' ' << v.second << '\n';
            }
        }

        void capture_graphs() {
            graphs.assign(buckets.size(), nullptr);
            for (std::size_t bi = 0; bi < buckets.size(); ++bi) {
                // Warm up outside capture (module loading, static launch config).
                run_eager(static_cast<int>(bi), buckets[bi], stream);
                CURF_CUDA_CHECK(cudaStreamSynchronize(stream));
                if (!cfg.cuda_graphs) continue;
                cudaGraph_t g = nullptr;
                CURF_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                run_eager(static_cast<int>(bi), buckets[bi], stream);
                CURF_CUDA_CHECK(cudaStreamEndCapture(stream, &g));
                CURF_CUDA_CHECK(cudaGraphInstantiate(&graphs[bi], g, 0));
                CURF_CUDA_CHECK(cudaGraphDestroy(g));
            }
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
        std::replace(m.gpu.begin(), m.gpu.end(), ' ', '_');

        const auto maxb = static_cast<std::size_t>(m.buckets.back());
        m.plan = plan_memory(m.model, maxb, sizeof(__half), m.cfg.reuse_memory);
        m.arena = DeviceBuffer<std::byte>(m.plan.arena_bytes);
        m.weights = DeviceBuffer<std::byte>(m.model.blob.size());
        CURF_CUDA_CHECK(cudaMemcpy(m.weights.get(), m.model.blob.data(), m.model.blob.size(), cudaMemcpyHostToDevice));
        m.input = DeviceBuffer<float>(maxb * m.model.tensors[m.model.input].elems_per_window());
        m.logits = DeviceBuffer<float>(maxb * m.model.tensors[m.model.output].elems_per_window());
        CURF_CUDA_CHECK(cudaMemset(m.input.get(), 0, m.input.bytes()));
        CURF_CUDA_CHECK(cudaMemset(m.arena.get(), 0, m.arena.bytes()));
        CURF_CUDA_CHECK(cudaStreamCreateWithFlags(&m.stream, cudaStreamNonBlocking));

        m.select_kernels();
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

    std::size_t Engine::device_bytes() const {
        return impl_->arena.bytes() + impl_->weights.bytes() + impl_->input.bytes() + impl_->logits.bytes();
    }

    int Engine::bucket_for(int batch) const {
        const auto &b = impl_->buckets;
        if (batch <= 0 || batch > b.back())
            throw std::out_of_range("engine: batch " + std::to_string(batch) + " outside 1.." + std::to_string(b.back()));
        return *std::lower_bound(b.begin(), b.end(), batch);
    }

    void Engine::run(int batch, cudaStream_t stream) {
        const int bucket = bucket_for(batch);
        const auto bi = static_cast<std::size_t>(
            std::lower_bound(impl_->buckets.begin(), impl_->buckets.end(), bucket) - impl_->buckets.begin());
        if (impl_->cfg.cuda_graphs) CURF_CUDA_CHECK(cudaGraphLaunch(impl_->graphs[bi], stream));
        else impl_->run_eager(static_cast<int>(bi), bucket, stream);
    }

    void Engine::run_eager(int batch, cudaStream_t stream) {
        const int bucket = bucket_for(batch);
        const auto bi = static_cast<std::size_t>(
            std::lower_bound(impl_->buckets.begin(), impl_->buckets.end(), bucket) - impl_->buckets.begin());
        impl_->run_eager(static_cast<int>(bi), bucket, stream);
    }

    std::vector<float> Engine::read_tensor(std::uint32_t t, int batch) const {
        const auto &m = *impl_;
        if (t >= m.model.tensors.size()) throw std::out_of_range("engine: tensor index");
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
        launch_half_to_float(static_cast<const __half *>(m.ptr(t)), m.scratch.get(), n, m.stream);
        CURF_CUDA_CHECK(cudaStreamSynchronize(m.stream));
        CURF_CUDA_CHECK(cudaMemcpy(out.data(), m.scratch.get(), n * sizeof(float), cudaMemcpyDeviceToHost));
        return out;
    }

} // namespace curf::engine
