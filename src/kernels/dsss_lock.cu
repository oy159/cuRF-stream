#include "curf/cuda/dsss_lock.hpp"
#include "curf/cuda/memory.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace curf::cuda {
    namespace {
        __device__ constexpr int code[11]={1,1,1,-1,-1,-1,1,-1,-1,1,-1};
        __global__ void interpolate(const float *input,const DsssLockRequest *jobs,unsigned count,
                                    const float *ratio,const float *filters,cf32 *out) {
            const unsigned j=blockIdx.x*blockDim.x+threadIdx.x,row=blockIdx.y;
            if (j>=barker11::waveform || row>=count) return;
            const auto job=jobs[row]; const float pos=16.f+j*ratio[job.rate];
            const int i=static_cast<int>(floorf(pos));
            const unsigned phase=min(255u,static_cast<unsigned>((pos-i)*256.f));
            const float *h=filters+(job.rate*256+phase)*32;
            const float *x=input+std::size_t{job.row}*1024*3;
            float re=0,im=0;
            for (unsigned k=0;k<32;++k) {
                const unsigned n=static_cast<unsigned>(i+int(k)-15);
                re=fmaf(x[3*n],h[k],re); im=fmaf(x[3*n+1],h[k],im);
            }
            out[std::size_t{row}*barker11::waveform+j]={re,im};
        }
        __global__ void match(const cf32 *wave,const cf32 *osc,float *scores) {
            const unsigned row=blockIdx.x,f=blockIdx.y,t=threadIdx.x;
            __shared__ float score[128];
            float value=0;
            if (t<88) {
                const unsigned phase=t%44,reverse=t/44;
                const auto *x=wave+std::size_t{row}*barker11::waveform;
                float projected[2]{},energy[2]{};
                for (unsigned s=0;s<14;++s) {
                    float re=0,im=0;
                    for (unsigned k=0;k<22;++k) {
                        const auto v=x[phase+s*44+k*2]; const auto o=osc[f*22+k];
                        const float c=static_cast<float>(code[reverse?10-k/2:k/2]);
                        re+=c*(v.re*o.re-v.im*o.im); im+=c*(v.re*o.im+v.im*o.re);
                        energy[s/7]+=v.re*v.re+v.im*v.im;
                    }
                    projected[s/7]+=(re*re+im*im)/22.f;
                }
                value=fminf(projected[0]/fmaxf(energy[0],1e-20f),projected[1]/fmaxf(energy[1],1e-20f));
            }
            score[t]=value; __syncthreads();
            for (unsigned n=64;n;n>>=1) { if (t<n) score[t]=fmaxf(score[t],score[t+n]); __syncthreads(); }
            if (!t) scores[std::size_t{row}*barker11::frequencies+f]=score[0];
        }
        __global__ void finish(const float *input,const DsssLockRequest *jobs,const cf32 *wave,const float *scores,
                              DsssFrequencyEstimate *out) {
            const unsigned row=blockIdx.x,t=threadIdx.x;
            __shared__ float candidate[64],original[64],filtered[64];
            const auto *x=input+std::size_t{jobs[row].row}*1024*3;
            float a=0,b=0;
            for (unsigned j=t;j<1024;j+=64) a+=x[3*j]*x[3*j]+x[3*j+1]*x[3*j+1];
            for (unsigned j=t;j<barker11::waveform;j+=64) { const auto v=wave[std::size_t{row}*barker11::waveform+j]; b+=v.re*v.re+v.im*v.im; }
            original[t]=a;filtered[t]=b;candidate[t]=t<61?scores[std::size_t{row}*61+t]:0; __syncthreads();
            for (unsigned n=32;n;n>>=1) { if (t<n) { original[t]+=original[t+n];filtered[t]+=filtered[t+n]; } __syncthreads(); }
            if (!t) {
                unsigned best=30;
                for (unsigned f=0;f<61;++f) if (candidate[f]>candidate[best]) best=f;
                float far=0;
                for (unsigned f=0;f<61;++f) if (abs(int(f)-int(best))>=6) far=fmaxf(far,candidate[f]);
                out[row]={float((int(best)-30)*50000),fminf(1.f,fmaxf(0.f,candidate[best])),candidate[best]-far,
                          (filtered[0]/barker11::waveform)/fmaxf(original[0]/1024.f,1e-20f)};
            }
        }
    }
    struct DsssFrequencyLock::Impl {
        DeviceBuffer<float> filters,ratios;
        DeviceBuffer<cf32> oscillator;
        std::map<double,unsigned> rates;
        explicit Impl(std::span<const double> sample_rates) {
            std::vector<float> h,r;
            for (double fs:sample_rates) {
                if (!barker11::supports(fs)) throw std::invalid_argument("Barker frequency lock: unsupported rate");
                if (rates.count(fs)) continue;
                rates[fs]=static_cast<unsigned>(r.size()); r.push_back(static_cast<float>(fs/barker11::output_rate));
                const auto v=barker11::interpolation_filter(fs); h.insert(h.end(),v.begin(),v.end());
            }
            if (r.empty()) throw std::invalid_argument("Barker frequency lock requires sample rates");
            filters=DeviceBuffer<float>(h.size());ratios=DeviceBuffer<float>(r.size());
            CURF_CUDA_CHECK(cudaMemcpy(filters.get(),h.data(),filters.bytes(),cudaMemcpyHostToDevice));
            CURF_CUDA_CHECK(cudaMemcpy(ratios.get(),r.data(),ratios.bytes(),cudaMemcpyHostToDevice));
            std::vector<cf32> o(61*22);
            for (unsigned f=0;f<61;++f) for (unsigned k=0;k<22;++k) {
                const double angle=-2*std::numbers::pi*((int(f)-30)*barker11::step_hz)*(2*k)/barker11::output_rate;
                o[f*22+k]={static_cast<float>(std::cos(angle)),static_cast<float>(std::sin(angle))};
            }
            oscillator=DeviceBuffer<cf32>(o.size());CURF_CUDA_CHECK(cudaMemcpy(oscillator.get(),o.data(),oscillator.bytes(),cudaMemcpyHostToDevice));
        }
    };
    DsssFrequencyLock::DsssFrequencyLock(std::span<const double> rates):impl_(std::make_unique<Impl>(rates)) {}
    DsssFrequencyLock::~DsssFrequencyLock()=default;
    unsigned DsssFrequencyLock::rate_index(double fs) const {
        const auto i=impl_->rates.find(fs);
        if (i==impl_->rates.end()) throw std::invalid_argument("Barker frequency lock: unregistered rate");
        return i->second;
    }
    void DsssFrequencyLock::run(const float *input,const DsssLockRequest *jobs,unsigned count,cf32 *wave,float *scores,
                                DsssFrequencyEstimate *output,cudaStream_t stream) {
        if (!count) return;
        if (count>512) throw std::invalid_argument("Barker frequency lock batch exceeds 512");
        interpolate<<<dim3((barker11::waveform+127)/128,count),128,0,stream>>>(input,jobs,count,impl_->ratios.get(),impl_->filters.get(),wave);
        CURF_CUDA_CHECK(cudaGetLastError());
        match<<<dim3(count,barker11::frequencies),128,0,stream>>>(wave,impl_->oscillator.get(),scores);
        CURF_CUDA_CHECK(cudaGetLastError());
        finish<<<count,64,0,stream>>>(input,jobs,wave,scores,output);
        CURF_CUDA_CHECK(cudaGetLastError());
    }
}
