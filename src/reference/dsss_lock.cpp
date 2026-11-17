#include "curf/dsss_lock.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>

namespace curf {
    namespace {
        double coefficient(double fs, unsigned tap, double fraction) {
            const double t = static_cast<double>(tap)-15-fraction;
            const double f = barker11::cutoff/fs;
            const double a = 2*std::numbers::pi*tap/(barker11::taps-1);
            const double window = .42-.5*std::cos(a)+.08*std::cos(2*a);
            return (std::abs(t)<1e-12 ? 2*f : std::sin(2*std::numbers::pi*f*t)/(std::numbers::pi*t))*window;
        }
    }
    bool barker11::supports(double fs) noexcept { return std::isfinite(fs) && fs>=22e6 && fs<=64e6; }
    std::vector<float> barker11::interpolation_filter(double fs) {
        if (!supports(fs)) throw std::invalid_argument("Barker frequency lock: unsupported sample rate");
        std::vector<float> h(phases*taps);
        for (unsigned p=0;p<phases;++p) {
            double sum=0;
            for (unsigned j=0;j<taps;++j) { h[p*taps+j]=static_cast<float>(coefficient(fs,j,double(p)/phases)); sum+=h[p*taps+j]; }
            for (unsigned j=0;j<taps;++j) h[p*taps+j]=static_cast<float>(h[p*taps+j]/sum);
        }
        return h;
    }
    DsssFrequencyEstimate ref::dsss_frequency(std::span<const float> input,double fs) {
        using C=std::complex<double>;
        if (input.size()!=barker11::length*3 || !barker11::supports(fs) ||
            !std::all_of(input.begin(),input.end(),[](float x) { return std::isfinite(x); }))
            throw std::invalid_argument("Barker frequency lock: invalid model input");
        std::array<C,barker11::waveform> wave{};
        double original_energy=0,filtered_energy=0;
        for (unsigned i=0;i<barker11::length;++i) original_energy+=std::norm(C(input[3*i],input[3*i+1]));
        for (unsigned j=0;j<wave.size();++j) {
            const double pos=16+j*fs/barker11::output_rate;
            const int i=static_cast<int>(std::floor(pos)); const double fraction=pos-i;
            double sum=0;
            for (unsigned k=0;k<barker11::taps;++k) {
                const double h=coefficient(fs,k,fraction); const auto index=static_cast<unsigned>(i+static_cast<int>(k)-15);
                wave[j]+=h*C(input[3*index],input[3*index+1]); sum+=h;
            }
            wave[j]/=sum; filtered_energy+=std::norm(wave[j]);
        }
        std::array<double,barker11::frequencies> scores{};
        for (unsigned f=0;f<scores.size();++f) {
            const double offset=(static_cast<int>(f)-30)*barker11::step_hz;
            std::array<C,22> oscillator{};
            for (unsigned k=0;k<22;++k) oscillator[k]=std::polar(1.,-2*std::numbers::pi*offset*(2*k)/barker11::output_rate);
            for (unsigned reverse=0;reverse<2;++reverse) for (unsigned phase=0;phase<barker11::timings;++phase) {
                double projected[2]{},energy[2]{};
                for (unsigned s=0;s<barker11::symbols;++s) {
                    C projection{};
                    for (unsigned k=0;k<22;++k) {
                        const auto v=wave[phase+s*44+k*2]; const int c=barker11::code[reverse?10-k/2:k/2];
                        projection+=double(c)*v*oscillator[k]; energy[s/7]+=std::norm(v);
                    }
                    projected[s/7]+=std::norm(projection)/22;
                }
                const double score=std::min(projected[0]/std::max(energy[0],1e-20),projected[1]/std::max(energy[1],1e-20));
                scores[f]=std::max(scores[f],score);
            }
        }
        unsigned best=30;
        for (unsigned f=0;f<scores.size();++f) if (scores[f]>scores[best]) best=f;
        double far=0;
        for (unsigned f=0;f<scores.size();++f) if (std::abs(int(f)-int(best))>=6) far=std::max(far,scores[f]);
        return {static_cast<float>((int(best)-30)*barker11::step_hz),static_cast<float>(std::clamp(scores[best],0.,1.)),
                static_cast<float>(scores[best]-far),static_cast<float>((filtered_energy/wave.size())/std::max(original_energy/barker11::length,1e-20))};
    }
}
