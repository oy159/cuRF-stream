#include "curf/engine/model.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace curf::engine {

    namespace {

        class Reader {
        public:
            explicit Reader(const std::filesystem::path &p) : f_(p, std::ios::binary), path_(p.string()) {
                if (!f_) throw std::runtime_error("cannot open " + path_);
            }
            template <class T> T get() {
                T v{};
                raw(&v, sizeof v);
                return v;
            }
            void raw(void *dst, std::size_t n) {
                f_.read(static_cast<char *>(dst), static_cast<std::streamsize>(n));
                if (!f_) throw std::runtime_error(path_ + ": truncated");
            }
            std::string name(std::size_t n) {
                std::string s(n, '\0');
                raw(s.data(), n);
                s.resize(std::strlen(s.c_str()));
                return s;
            }
            void magic(const char (&m)[9]) {
                char got[8];
                raw(got, 8);
                if (std::memcmp(got, m, 8) != 0) throw std::runtime_error(path_ + ": bad magic");
            }
            template <class T> void vec(std::vector<T> &v, std::size_t n) {
                v.resize(n);
                raw(v.data(), n * sizeof(T));
            }

        private:
            std::ifstream f_;
            std::string path_;
        };

        [[noreturn]] void bad(const std::string &what) { throw std::runtime_error("model: " + what); }

        void validate(const Model &m) {
            const auto nt = m.tensors.size();
            if (m.input >= nt || m.output >= nt) bad("input/output index out of range");
            std::vector<bool> ready(nt, false);
            ready[m.input] = true;
            for (const auto &op : m.ops) {
                if (op.in >= nt || op.out >= nt || (op.res != kNoTensor && op.res >= nt)) bad(op.name + ": tensor index");
                if (!ready[op.in] || (op.res != kNoTensor && !ready[op.res])) bad(op.name + ": not topologically ordered");
                if (ready[op.out]) bad(op.name + ": output written twice");
                const auto &x = m.tensors[op.in], &y = m.tensors[op.out];
                if (x.channels != op.cin || y.channels != op.cout) bad(op.name + ": channel mismatch");
                if (op.kind == OpKind::conv) {
                    if (op.stride == 0 || op.k == 0 || x.length + 2 * op.pad < op.k) bad(op.name + ": bad geometry");
                    if ((x.length + 2 * op.pad - op.k) / op.stride + 1 != y.length) bad(op.name + ": output length");
                    if (op.kpad < op.k * op.cin || op.kpad % 32 != 0) bad(op.name + ": kpad");
                    if (op.res != kNoTensor && (m.tensors[op.res].channels != y.channels ||
                                                m.tensors[op.res].length != y.length))
                        bad(op.name + ": residual shape");
                    if (op.w_off % 64 || op.w_off + std::size_t{op.cout} * op.kpad * 2 > m.blob.size()) bad(op.name + ": weights");
                } else if (op.kind == OpKind::gap_linear) {
                    if (y.length != 1) bad(op.name + ": gap_linear output must have length 1");
                    if (op.w_off % 64 || op.w_off + std::size_t{op.cout} * op.cin * 4 > m.blob.size()) bad(op.name + ": weights");
                } else {
                    bad(op.name + ": unknown op kind");
                }
                if (op.b_off % 64 || op.b_off + std::size_t{op.cout} * 4 > m.blob.size()) bad(op.name + ": bias");
                ready[op.out] = true;
            }
            if (!ready[m.output]) bad("output never produced");
        }

    } // namespace

    Model Model::load(const std::filesystem::path &path) {
        Reader r(path);
        r.magic("CURFNET1");
        const auto version = r.get<std::uint32_t>();
        if (version != 1) bad("unsupported version " + std::to_string(version));
        const auto nt = r.get<std::uint32_t>(), no = r.get<std::uint32_t>();
        Model m;
        m.input = r.get<std::uint32_t>();
        m.output = r.get<std::uint32_t>();
        for (int i = 0; i < 3; ++i) (void)r.get<std::uint32_t>();
        m.tensors.resize(nt);
        for (auto &t : m.tensors) {
            t.channels = r.get<std::uint32_t>();
            t.length = r.get<std::uint32_t>();
            t.name = r.name(56);
        }
        m.ops.resize(no);
        for (auto &op : m.ops) {
            std::uint32_t u[12];
            r.raw(u, sizeof u);
            op.kind = static_cast<OpKind>(u[0]);
            op.in = u[1]; op.res = u[2]; op.out = u[3];
            op.cin = u[4]; op.cout = u[5]; op.k = u[6]; op.stride = u[7]; op.pad = u[8];
            op.relu = u[9] != 0;
            op.kpad = u[10];
            op.w_off = r.get<std::uint64_t>();
            op.b_off = r.get<std::uint64_t>();
            op.name = r.name(48);
        }
        const auto blob = r.get<std::uint64_t>();
        r.vec(m.blob, blob);
        validate(m);
        return m;
    }

    double Model::flops_per_window() const {
        double f = 0;
        for (const auto &op : ops)
            f += op.kind == OpKind::conv ? 2.0 * tensors[op.out].length * op.cout * op.cin * op.k : 2.0 * op.cin * op.cout;
        return f;
    }

    Golden Golden::load(const std::filesystem::path &path, const Model &model) {
        Reader r(path);
        r.magic("CURFGLD1");
        Golden g;
        g.n_full = r.get<std::uint32_t>();
        g.n_layer = r.get<std::uint32_t>();
        const auto nt = r.get<std::uint32_t>();
        (void)r.get<std::uint32_t>();
        if (nt != model.tensors.size()) throw std::runtime_error("golden: tensor count does not match the model");
        const auto &in = model.tensors[model.input];
        g.classes = model.tensors[model.output].channels;
        r.vec(g.input, std::size_t{g.n_full} * in.elems_per_window());
        r.vec(g.labels, g.n_full);
        r.vec(g.logits, std::size_t{g.n_full} * g.classes);
        g.activations.resize(nt);
        for (std::uint32_t t = 0; t < nt; ++t) {
            const auto c = r.get<std::uint32_t>(), l = r.get<std::uint32_t>();
            if (c != model.tensors[t].channels || l != model.tensors[t].length)
                throw std::runtime_error("golden: shape of tensor " + std::to_string(t) + " does not match");
            r.vec(g.activations[t], std::size_t{g.n_layer} * c * l);
        }
        return g;
    }

    MemoryPlan plan_memory(const Model &m, std::size_t max_batch, std::size_t bytes_per_elem, bool reuse) {
        constexpr std::size_t kAlign = 256;
        const auto nt = m.tensors.size();
        MemoryPlan plan;
        plan.offset.assign(nt, MemoryPlan::kNoOffset);

        // Lifetime of each tensor in op indices: [def, last_use].
        std::vector<long> def(nt, -1), last(nt, -1);
        for (std::size_t i = 0; i < m.ops.size(); ++i) {
            const auto &op = m.ops[i];
            def[op.out] = static_cast<long>(i);
            last[op.in] = std::max(last[op.in], static_cast<long>(i));
            if (op.res != kNoTensor) last[op.res] = std::max(last[op.res], static_cast<long>(i));
        }
        struct Item { std::size_t t, bytes; };
        std::vector<Item> items;
        for (std::size_t t = 0; t < nt; ++t) {
            if (t == m.input || t == m.output) continue;
            const auto b = (m.tensors[t].elems_per_window() * max_batch * bytes_per_elem + kAlign - 1) / kAlign * kAlign;
            items.push_back({t, b});
            plan.no_reuse_bytes += b;
            if (last[t] < def[t]) last[t] = def[t]; // dead tensor: still needs room while written
        }
        if (!reuse) {
            std::size_t off = 0;
            for (const auto &it : items) { plan.offset[it.t] = off; off += it.bytes; }
            plan.arena_bytes = off;
            return plan;
        }
        // Greedy by size: place each tensor at the lowest offset that does not collide with an
        // already-placed tensor whose lifetime overlaps.
        std::stable_sort(items.begin(), items.end(), [](const Item &a, const Item &b) { return a.bytes > b.bytes; });
        std::vector<Item> placed;
        for (const auto &it : items) {
            std::vector<std::pair<std::size_t, std::size_t>> busy;
            for (const auto &p : placed)
                if (def[p.t] <= last[it.t] && def[it.t] <= last[p.t])
                    busy.emplace_back(plan.offset[p.t], plan.offset[p.t] + p.bytes);
            std::sort(busy.begin(), busy.end());
            std::size_t off = 0;
            for (const auto &[s, e] : busy) {
                if (off + it.bytes <= s) break;
                off = std::max(off, e);
            }
            plan.offset[it.t] = off;
            plan.arena_bytes = std::max(plan.arena_bytes, off + it.bytes);
            placed.push_back(it);
        }
        return plan;
    }

} // namespace curf::engine
