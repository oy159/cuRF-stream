#include "curf/components.hpp"
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

namespace {
    constexpr std::uint64_t burst=0x0000ffff0000ffffull,other=0xff000000ff000000ull;
    void add(std::vector<curf::ActiveChannel> &v,int a,int b,std::uint64_t mask) {
        for (int k=a;k<=b;++k) v.push_back({k,1.f,mask});
    }
}
TEST_CASE("activity closes one carrier notch and preserves independent neighbours", "[components]") {
    std::vector<curf::ActiveChannel> v;add(v,-68,-3,burst);add(v,3,68,burst);
    CHECK(curf::group_components(v,false).bands.size()==2);
    auto r=curf::group_components(v,true);
    REQUIRE(r.bands.size()==1);CHECK(r.joins==1);CHECK(r.bands[0].coherent);
    for (auto &x:v) if (x.bin>0) x.activity=other;
    r=curf::group_components(v,true);
    REQUIRE(r.bands.size()==2);CHECK(r.joins==0);
    for (auto &x:v) x.activity=~std::uint64_t{0};
    CHECK(curf::group_components(v,true).bands.size()==2); // continuous carriers cannot establish a join
}
TEST_CASE("independent temporal envelopes split a connected wide component", "[components]") {
    std::vector<curf::ActiveChannel> v;add(v,-128,-1,burst);add(v,0,127,other);
    auto r=curf::group_components(v,true);
    REQUIRE(r.bands.size()==2);CHECK(r.splits==1);
    CHECK(r.bands[0].lo==-128);CHECK(r.bands[0].hi==-1);CHECK(r.bands[1].lo==0);CHECK(r.bands[1].hi==127);
    for (auto &x:v) x.activity=burst;
    r=curf::group_components(v,true);REQUIRE(r.bands.size()==1);CHECK(r.splits==0);CHECK(r.bands[0].coherent);
    for (auto &x:v) x.activity=~std::uint64_t{0};
    r=curf::group_components(v,true);REQUIRE(r.bands.size()==1);CHECK_FALSE(r.bands[0].coherent);CHECK(r.splits==0);
}
TEST_CASE("narrow neighbours warm-up and small-grid holes do not create burst evidence", "[components]") {
    std::vector<curf::ActiveChannel> v;add(v,0,132,burst);add(v,137,144,burst);
    auto r=curf::group_components(v,true);REQUIRE(r.bands.size()==2);CHECK(r.joins==0);
    for (auto &x:v) x.activity=0;
    r=curf::group_components(v,true);REQUIRE(r.bands.size()==2);CHECK_FALSE(r.bands[0].coherent);
    CHECK(curf::group_components({},true).bands.empty());
    v[1].bin=v[0].bin;CHECK_THROWS_AS(curf::group_components(v,true),std::invalid_argument);
}
