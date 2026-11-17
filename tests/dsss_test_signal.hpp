#pragma once
#include "curf/dsss_lock.hpp"
#include <cmath>
#include <complex>
#include <numbers>
#include <random>
#include <vector>

namespace dsss_test {
    inline std::vector<float> signal(double fs,double offset,unsigned seed=74) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> noise(0,.03f);
        std::vector<int> bits(2048);
        for (auto &b:bits) b=(rng()&1)?1:-1;
        std::vector<float> x(curf::barker11::length*3);
        for (unsigned j=0;j<curf::barker11::length;++j) {
            const auto chip=static_cast<unsigned>(std::floor(j*11e6/fs+.3));
            const double a=2*std::numbers::pi*offset*j/fs+.7;
            const auto value=double(bits[chip/11]*curf::barker11::code[chip%11])*std::polar(1.,a);
            x[3*j]=static_cast<float>(value.real())+noise(rng);
            x[3*j+1]=static_cast<float>(value.imag())+noise(rng);
            x[3*j+2]=1.15f;
        }
        return x;
    }
}
