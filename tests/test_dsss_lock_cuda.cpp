#include "curf/cuda/dsss_lock.hpp"
#include "curf/cuda/memory.hpp"
#include "dsss_test_signal.hpp"
#include <catch2/catch_test_macros.hpp>
#include <array>
#include <stdexcept>

TEST_CASE("Barker GPU batches reordered rows and agrees with the independent reference", "[cuda][dsss]") {
    using namespace curf;
    const std::array<double,3> rates={61.44e6,23.04e6,46.08e6};
    cuda::DsssFrequencyLock lock(rates);
    std::vector<float> input(7*barker11::length*3,-.2f);
    auto first=dsss_test::signal(46.08e6,.9e6),last=dsss_test::signal(61.44e6,-1.25e6);
    std::vector<float> noise(barker11::length*3);
    std::mt19937 rng(76);std::normal_distribution<float> n;
    for (unsigned j=0;j<barker11::length;++j) { noise[3*j]=n(rng);noise[3*j+1]=n(rng); }
    std::copy(first.begin(),first.end(),input.begin()+3*barker11::length*3);
    std::copy(noise.begin(),noise.end(),input.begin()+barker11::length*3);
    std::copy(last.begin(),last.end(),input.begin()+5*barker11::length*3);
    const std::array<cuda::DsssLockRequest,3> jobs={{{3,lock.rate_index(46.08e6)},{1,lock.rate_index(23.04e6)},{5,lock.rate_index(61.44e6)}}};
    cuda::DeviceBuffer<float> d_input(input.size()),scores(3*barker11::frequencies);
    cuda::DeviceBuffer<cf32> wave(3*barker11::waveform);
    cuda::DeviceBuffer<cuda::DsssLockRequest> d_jobs(jobs.size());
    cuda::DeviceBuffer<DsssFrequencyEstimate> out(4);
    std::array<DsssFrequencyEstimate,4> results{};results[3].offset_hz=12345;
    CURF_CUDA_CHECK(cudaMemcpy(d_input.get(),input.data(),d_input.bytes(),cudaMemcpyHostToDevice));
    CURF_CUDA_CHECK(cudaMemcpy(d_jobs.get(),jobs.data(),d_jobs.bytes(),cudaMemcpyHostToDevice));
    CURF_CUDA_CHECK(cudaMemcpy(out.get(),results.data(),out.bytes(),cudaMemcpyHostToDevice));
    lock.run(d_input.get(),d_jobs.get(),3,wave.get(),scores.get(),out.get(),nullptr);
    CURF_CUDA_CHECK(cudaDeviceSynchronize());
    CURF_CUDA_CHECK(cudaMemcpy(results.data(),out.get(),out.bytes(),cudaMemcpyDeviceToHost));
    const std::array<DsssFrequencyEstimate,3> ref={ref::dsss_frequency(first,46.08e6),ref::dsss_frequency(noise,23.04e6),ref::dsss_frequency(last,61.44e6)};
    for (unsigned i=0;i<3;++i) {
        INFO("row "<<i<<" gpu "<<results[i].coherence<<" reference "<<ref[i].coherence);
        CHECK(results[i].valid()==ref[i].valid());
        CHECK(std::abs(results[i].offset_hz-ref[i].offset_hz)<=50e3);
        CHECK(std::abs(results[i].coherence-ref[i].coherence)<.005);
        CHECK(std::abs(results[i].contrast-ref[i].contrast)<.01);
        CHECK(std::abs(results[i].retained_power-ref[i].retained_power)<.01);
    }
    CHECK(results[3].offset_hz==12345.f);
    std::vector<float> after(input.size());
    CURF_CUDA_CHECK(cudaMemcpy(after.data(),d_input.get(),d_input.bytes(),cudaMemcpyDeviceToHost));
    CHECK(after==input);
    lock.run(nullptr,nullptr,0,nullptr,nullptr,nullptr,nullptr);
    CHECK_THROWS_AS(lock.rate_index(10e6),std::invalid_argument);
    CHECK_THROWS_AS(lock.run(nullptr,nullptr,513,nullptr,nullptr,nullptr,nullptr),std::invalid_argument);
}

TEST_CASE("Barker GPU handles the full mixed-rate batch on a nondefault stream", "[cuda][dsss]") {
    using namespace curf;
    constexpr unsigned count=512;
    const std::array<double,4> rates={23.04e6,30.72e6,46.08e6,61.44e6};
    cuda::DsssFrequencyLock lock(rates);
    std::vector<float> input(count*barker11::length*3);
    std::vector<cuda::DsssLockRequest> jobs(count);
    std::vector<DsssFrequencyEstimate> result(count+1);result.back().offset_hz=12345.f;
    std::mt19937 rng(104);std::normal_distribution<float> noise;
    for (unsigned i=0;i<count;++i) {
        auto *x=input.data()+std::size_t{i}*barker11::length*3;
        const double fs=rates[i%rates.size()];
        if (i%3==0) {
            const auto positive=dsss_test::signal(fs,(i%2?-.9e6:1.25e6),i+200);
            std::copy(positive.begin(),positive.end(),x);
        } else for (unsigned j=0;j<barker11::length;++j) { x[3*j]=noise(rng);x[3*j+1]=noise(rng); }
        // Reverse requests so result ownership cannot accidentally follow input order.
        jobs[count-1-i]={i,lock.rate_index(fs)};
    }
    cuda::DeviceBuffer<float> d_input(input.size()),scores(count*barker11::frequencies);
    cuda::DeviceBuffer<cf32> wave(count*barker11::waveform);
    cuda::DeviceBuffer<cuda::DsssLockRequest> d_jobs(count);
    cuda::DeviceBuffer<DsssFrequencyEstimate> out(count+1);
    cudaStream_t stream;
    CURF_CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    CURF_CUDA_CHECK(cudaMemcpyAsync(d_input.get(),input.data(),d_input.bytes(),cudaMemcpyHostToDevice,stream));
    CURF_CUDA_CHECK(cudaMemcpyAsync(d_jobs.get(),jobs.data(),d_jobs.bytes(),cudaMemcpyHostToDevice,stream));
    CURF_CUDA_CHECK(cudaMemcpyAsync(out.get(),result.data(),out.bytes(),cudaMemcpyHostToDevice,stream));
    lock.run(d_input.get(),d_jobs.get(),count,wave.get(),scores.get(),out.get(),stream);
    CURF_CUDA_CHECK(cudaMemcpyAsync(result.data(),out.get(),out.bytes(),cudaMemcpyDeviceToHost,stream));
    CURF_CUDA_CHECK(cudaStreamSynchronize(stream));
    CURF_CUDA_CHECK(cudaStreamDestroy(stream));
    CHECK(result.back().offset_hz==12345.f);
    for (unsigned j=0;j<count;++j) {
        const auto i=count-1-j;
        INFO("input row "<<i);
        CHECK(result[j].valid()==(i%3==0));
        if (i%3==0) CHECK(std::abs(result[j].offset_hz-(i%2?-.9e6:1.25e6))<=100e3);
    }
}
