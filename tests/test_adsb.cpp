#include "curf/adsb.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

using namespace curf;
using Catch::Approx;

namespace {
    // Test frames from J. Sun, "The 1090 Megahertz Riddle".
    std::vector<std::uint8_t> hex(const std::string &s) {
        std::vector<std::uint8_t> v;
        for (std::size_t i = 0; i + 1 < s.size(); i += 2) v.push_back(static_cast<std::uint8_t>(std::stoul(s.substr(i, 2), nullptr, 16)));
        return v;
    }
} // namespace

TEST_CASE("Mode S parity: valid DF17 frames, single-bit syndromes", "[adsb]") {
    for (const char *f : {"8D4840D6202CC371C32CE0576098", "8D40621D58C382D690C8AC2863A7", "8D40621D58C386435CC412692AD6",
                          "8D485020994409940838175B284F", "8DA05F219B06B6AF189400CBC33F"}) {
        auto m = hex(f);
        CHECK(adsb::parity_remainder(m.data(), 112) == 0);
        for (int bit : {0, 7, 40, 87, 88, 111}) {
            auto b = m;
            b[static_cast<std::size_t>(bit >> 3)] ^= static_cast<std::uint8_t>(0x80u >> (bit & 7));
            CHECK(adsb::parity_remainder(b.data(), 112) == adsb::single_bit_syndrome(bit, 112));
        }
    }
}

TEST_CASE("ADS-B identification, altitude, velocity", "[adsb]") {
    auto id = hex("8D4840D6202CC371C32CE0576098");
    const auto a = adsb::decode(id.data(), 112);
    CHECK(a.df == 17);
    CHECK(a.icao == 0x4840D6);
    CHECK(a.tc == 4);
    REQUIRE(a.callsign);
    CHECK(*a.callsign == "KLM1023");

    auto pos = hex("8D40621D58C382D690C8AC2863A7");
    const auto p = adsb::decode(pos.data(), 112);
    REQUIRE(p.altitude_ft);
    CHECK(*p.altitude_ft == 38000);
    CHECK(p.has_cpr);
    CHECK_FALSE(p.cpr_odd);

    auto vel = hex("8D485020994409940838175B284F");
    const auto v = adsb::decode(vel.data(), 112);
    REQUIRE(v.ground_speed_kt);
    CHECK(*v.ground_speed_kt == Approx(159.20).margin(0.01));
    CHECK(*v.track_deg == Approx(182.88).margin(0.01));
    CHECK(*v.vertical_rate_fpm == -832);

    auto air = hex("8DA05F219B06B6AF189400CBC33F"); // subtype 3: heading + true airspeed
    const auto w = adsb::decode(air.data(), 112);
    REQUIRE(w.track_deg);
    CHECK(*w.track_deg == Approx(243.98).margin(0.01));
    CHECK(*w.ground_speed_kt == Approx(375));
    CHECK(*w.vertical_rate_fpm == -2304);
}

TEST_CASE("CPR global and local position decoding", "[adsb]") {
    auto e = hex("8D40621D58C382D690C8AC2863A7"), o = hex("8D40621D58C386435CC412692AD6");
    const auto me = adsb::decode(e.data(), 112), mo = adsb::decode(o.data(), 112);
    const auto g = adsb::cpr_global(me.cpr_lat, me.cpr_lon, mo.cpr_lat, mo.cpr_lon, false); // even is newer
    REQUIRE(g);
    CHECK((*g)[0] == Approx(52.2572).margin(1e-4));
    CHECK((*g)[1] == Approx(3.91937).margin(1e-4));
    const auto l = adsb::cpr_local(me.cpr_lat, me.cpr_lon, false, 52.258, 3.918);
    CHECK(l[0] == Approx(52.2572).margin(1e-4));
    CHECK(l[1] == Approx(3.91937).margin(1e-4));

    adsb::Tracker tr;
    CHECK_FALSE(tr.position(mo, 0.0));       // one frame: no position yet
    const auto p = tr.position(me, 2.0);     // pair within 10 s
    REQUIRE(p);
    CHECK((*p)[0] == Approx(52.2572).margin(1e-4));
    CHECK(tr.position(mo, 3.0));             // later frames decode locally
    CHECK(tr.known(0x40621D, 3.0));
    CHECK_FALSE(tr.known(0x123456, 3.0));
}
