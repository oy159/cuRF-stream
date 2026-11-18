#include "curf/components.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace curf {
    namespace {
        constexpr int wide = 64, max_join_gap = 8, min_half = 16, max_join_width = 200;
        struct Evidence { std::uint64_t mask = 0; bool coherent = false; };
        bool informative(std::uint64_t m) { const int n=std::popcount(m); return n>=2 && n<=56; }
        double similarity(std::uint64_t a,std::uint64_t b) {
            return double(std::popcount(a&b))/std::max(1,std::popcount(a|b));
        }
        class Activity {
            int origin_, size_;
            std::vector<std::array<unsigned short,64>> votes_;
            std::vector<unsigned> count_;
            std::vector<float> power_;
            int index(int bin) const { return std::clamp(bin-origin_,0,size_); }
        public:
            explicit Activity(std::span<const ActiveChannel> v)
                : origin_(v.front().bin), size_(v.back().bin-origin_+1), votes_(size_+1), count_(size_+1), power_(size_) {
                std::size_t p=0;
                for (int i=0;i<size_;++i) {
                    votes_[i+1]=votes_[i];count_[i+1]=count_[i];
                    if (p<v.size() && v[p].bin==origin_+i) {
                        const auto m=v[p].activity;power_[i]=v[p].power;++p;
                        if (!m) continue;
                        ++count_[i+1];
                        for (unsigned t=0;t<64;++t) votes_[i+1][t]+=static_cast<unsigned short>((m>>t)&1);
                    }
                }
            }
            std::uint64_t mask(int lo,int hi) const {
                const int a=index(lo),b=index(hi+1);const auto n=count_[b]-count_[a];
                if (n<8) return 0;
                std::uint64_t m=0;
                for (unsigned t=0;t<64;++t) if ((votes_[b][t]-votes_[a][t])*5>=n*3) m|=std::uint64_t{1}<<t;
                return m;
            }
            Evidence evidence(int lo,int hi) const {
                if (hi-lo+1<wide) return {};
                std::uint64_t intersection=~std::uint64_t{0},united=0;
                for (int q=0;q<4;++q) {
                    const int a=lo+(hi-lo+1)*q/4,b=lo+(hi-lo+1)*(q+1)/4-1;
                    const auto m=mask(a,b);
                    if (!informative(m)) return {};
                    intersection&=m;united|=m;
                }
                return {mask(lo,hi),double(std::popcount(intersection))/std::max(1,std::popcount(united))>=.85};
            }
            unsigned mismatch(int lo,int hi,std::uint64_t m) const {
                const int a=index(lo),b=index(hi+1);const auto n=count_[b]-count_[a];unsigned cost=0;
                for (unsigned t=0;t<64;++t) {
                    const unsigned ones=votes_[b][t]-votes_[a][t];
                    cost+=m&(std::uint64_t{1}<<t)?n-ones:ones;
                }
                return cost;
            }
            float boundary_power(int b) const {
                const int i=index(b);return power_[i]+power_[std::min(i+1,size_-1)];
            }
        };
    }
    ComponentGroups group_components(std::span<const ActiveChannel> v,bool temporal) {
        ComponentGroups out;
        if (v.empty()) return out;
        if (v.size()>4096 || std::int64_t{v.back().bin}-v.front().bin>=4096)
            throw std::invalid_argument("component grid exceeds 4096 channels");
        for (std::size_t i=1;i<v.size();++i) if (v[i].bin<=v[i-1].bin)
            throw std::invalid_argument("component channels must be unique and sorted");
        for (const auto &x:v) {
            if (out.bands.empty() || x.bin>out.bands.back().hi+2) out.bands.push_back({x.bin,x.bin});
            else out.bands.back().hi=x.bin;
        }
        if (!temporal) return out;
        const Activity activity(v);
        // Close only a small internal notch between two broad pieces sharing a real burst.
        for (std::size_t i=0;i+1<out.bands.size();) {
            auto &a=out.bands[i];const auto b=out.bands[i+1];
            if (b.lo-a.hi-1<=max_join_gap && a.hi-a.lo+1>=min_half && b.hi-b.lo+1>=min_half &&
                b.hi-a.lo+1<=max_join_width && activity.evidence(a.lo,b.hi).coherent) {
                a.hi=b.hi;a.coherent=true;out.bands.erase(out.bands.begin()+static_cast<std::ptrdiff_t>(i+1));++out.joins;
            } else ++i;
        }
        // Independent burst timing can split a connected component even on its first observation.
        // Constant/unknown activity supplies no split evidence; historical neighbours remain available.
        const auto components=std::move(out.bands);out.bands.clear();
        auto split=[&](auto &&self,int lo,int hi,unsigned depth)->void {
            const auto full=activity.evidence(lo,hi);
            if (full.coherent || hi-lo+1<2*wide || depth==3) {
                out.bands.push_back({lo,hi,full.coherent});return;
            }
            int best=0;bool found=false;unsigned cost=std::numeric_limits<unsigned>::max();float power=INFINITY;
            for (int b=lo+wide-1;b<=hi-wide;++b) {
                const auto a=activity.evidence(lo,b),c=activity.evidence(b+1,hi);
                if (!a.coherent || !c.coherent || similarity(a.mask,c.mask)>.5) continue;
                const unsigned q=activity.mismatch(lo,b,a.mask)+activity.mismatch(b+1,hi,c.mask);
                const float p=activity.boundary_power(b);
                if (q<cost || (q==cost && p<power)) { best=b;cost=q;power=p;found=true; }
            }
            if (!found) out.bands.push_back({lo,hi,false});
            else { ++out.splits;self(self,lo,best,depth+1);self(self,best+1,hi,depth+1); }
        };
        for (const auto &b:components) split(split,b.lo,b.hi,0);
        return out;
    }
}
