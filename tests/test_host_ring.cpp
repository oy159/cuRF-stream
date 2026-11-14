#include "curf/runtime/host_ring.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <deque>
#include <thread>

using namespace curf;
using namespace std::chrono_literals;

TEST_CASE("host ring is 2 MiB aligned and sized in whole slots", "[ring]") {
    HostRing ring(5, 1000);
    CHECK(reinterpret_cast<std::uintptr_t>(ring.base()) % (std::size_t{2} << 20) == 0);
    CHECK(ring.slot_bytes() == 4096);
    CHECK(ring.bytes() >= 5 * 4096);
}

TEST_CASE("host ring enforces capacity and in-order release", "[ring]") {
    HostRing ring(3, 4096);
    for (int i = 0; i < 3; ++i) {
        HostSlot *s = ring.try_begin_write();
        REQUIRE(s);
        s->frames = static_cast<std::size_t>(i);
        ring.commit_write();
    }
    CHECK(ring.try_begin_write() == nullptr); // full

    // Submit two, release one: exactly one slot frees up.
    for (int i = 0; i < 2; ++i) {
        HostSlot *s = ring.try_next_submit();
        REQUIRE(s);
        CHECK(s->frames == static_cast<std::size_t>(i));
        ring.mark_submitted();
    }
    CHECK(ring.in_flight() == 2);
    CHECK(ring.oldest_in_flight().frames == 0);
    ring.release_oldest();
    CHECK(ring.in_flight() == 1);
    CHECK(ring.oldest_in_flight().frames == 1);
    CHECK(ring.try_begin_write() != nullptr);
    CHECK(ring.begin_write(1ms) != nullptr);
}

TEST_CASE("host ring passes data intact between threads", "[ring]") {
    constexpr std::uint64_t n = 20000;
    constexpr std::size_t depth = 3;
    HostRing ring(8, 4096);

    std::thread producer([&] {
        for (std::uint64_t i = 0; i < n; ++i) {
            HostSlot *s = nullptr;
            while (!(s = ring.begin_write(10ms))) {}
            std::memcpy(s->data, &i, sizeof i);
            std::memset(s->data + sizeof i, static_cast<int>(i & 0xff), 64);
            s->first_sample = i * 10;
            ring.commit_write();
        }
    });

    // Consumer keeps up to `depth` slots "in flight", like the GPU pipeline.
    std::uint64_t next_submit = 0, next_release = 0;
    bool ok = true;
    while (next_release < n) {
        if (ring.in_flight() < depth) {
            if (HostSlot *s = ring.next_submit(1ms)) {
                std::uint64_t v = 0;
                std::memcpy(&v, s->data, sizeof v);
                ok = ok && v == next_submit && s->first_sample == next_submit * 10 && s->sequence == next_submit &&
                     s->data[sizeof v + 63] == static_cast<std::uint8_t>(next_submit & 0xff);
                ring.mark_submitted();
                ++next_submit;
                continue;
            }
        }
        if (ring.in_flight() > 0) {
            ring.release_oldest();
            ++next_release;
        }
    }
    producer.join();
    CHECK(ok);
    CHECK(ring.in_flight() == 0);
}
