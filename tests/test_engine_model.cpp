#include "curf/engine/model.hpp"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>

using namespace curf::engine;

namespace {

    // in(0) -> a(1) -> b(2) -> c(3, res=a) -> d(4) -> out(5)
    Model chain() {
        Model m;
        m.tensors = {{2, 64, "in"}, {8, 32, "a"}, {8, 32, "b"}, {8, 32, "c"}, {16, 16, "d"}, {4, 1, "out"}};
        auto conv = [](std::uint32_t in, std::uint32_t out, std::uint32_t res, std::uint32_t cin, std::uint32_t cout,
                       std::uint32_t stride) {
            OpDesc op;
            op.in = in; op.out = out; op.res = res; op.cin = cin; op.cout = cout;
            op.k = 3; op.stride = stride; op.pad = 1; op.kpad = 32;
            return op;
        };
        m.ops = {conv(0, 1, kNoTensor, 2, 8, 2), conv(1, 2, kNoTensor, 8, 8, 1), conv(2, 3, 1, 8, 8, 1),
                 conv(3, 4, kNoTensor, 8, 16, 2)};
        OpDesc head;
        head.kind = OpKind::gap_linear;
        head.in = 4; head.out = 5; head.cin = 16; head.cout = 4;
        m.ops.push_back(head);
        m.input = 0;
        m.output = 5;
        return m;
    }

    bool overlap(std::size_t a, std::size_t an, std::size_t b, std::size_t bn) { return a < b + bn && b < a + an; }

} // namespace

TEST_CASE("memory plan never aliases tensors that are live together", "[engine][plan]") {
    const auto m = chain();
    const auto plan = plan_memory(m, 4, 2, true);
    const auto no = plan_memory(m, 4, 2, false);
    CHECK(plan.offset[0] == MemoryPlan::kNoOffset);
    CHECK(plan.offset[5] == MemoryPlan::kNoOffset);
    CHECK(no.arena_bytes == no.no_reuse_bytes);
    CHECK(plan.arena_bytes < no.arena_bytes);

    auto bytes = [&](std::uint32_t t) { return (m.tensors[t].elems_per_window() * 4 * 2 + 255) / 256 * 256; };
    // live together: (a, b) at op1, (a, b, c) at op2, (c, d) at op3
    for (auto [x, y] : {std::pair{1u, 2u}, {1u, 3u}, {2u, 3u}, {3u, 4u}})
        CHECK_FALSE(overlap(plan.offset[x], bytes(x), plan.offset[y], bytes(y)));
    for (std::uint32_t t = 1; t <= 4; ++t) {
        CHECK(plan.offset[t] % 256 == 0);
        CHECK(plan.offset[t] + bytes(t) <= plan.arena_bytes);
    }
}

TEST_CASE("model file from convert.py loads and validates", "[engine][model]") {
    const std::filesystem::path dir = CURF_MODEL_DIR;
    if (!std::filesystem::exists(dir / "amc.curf")) SKIP("no converted model in " + dir.string());
    const auto m = Model::load(dir / "amc.curf");
    CHECK(m.ops.size() == 12);
    CHECK(m.tensors[m.input].channels == 2);
    CHECK(m.tensors[m.input].length == 1024);
    CHECK(m.tensors[m.output].channels == 10);
    CHECK(m.flops_per_window() / 1e6 > 50.7);
    CHECK(m.flops_per_window() / 1e6 < 50.9);
    const auto plan = plan_memory(m, 512, 2, true);
    INFO("arena " << plan.arena_bytes << " vs " << plan.no_reuse_bytes);
    CHECK(plan.arena_bytes * 3 < plan.no_reuse_bytes);
}
