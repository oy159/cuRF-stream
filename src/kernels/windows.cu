#include "curf/cuda/memory.hpp"
#include "curf/cuda/windows.hpp"

#include <cub/device/device_select.cuh>
#include <cub/iterator/counting_input_iterator.cuh>

#include <cmath>
#include <stdexcept>

namespace curf::cuda {

    namespace {

        // Partial channel energies: part[z][s][k] = sum_{t in split z} |win[s][t][k]|^2; threads over k
        // (coalesced rows), frames split over grid z. (One block per 256 channels and stream looping over all
        // W frames ran 4 blocks on a 16-SM GPU: 62 us for the 8 MB window at full load.) floor_flag sums the
        // splits in a fixed order, so the power stays deterministic.
        constexpr unsigned kPowerSplits = 16;
        __global__ void power_kernel(const float2 *__restrict__ win, float *__restrict__ part, unsigned M, unsigned W) {
            const unsigned s = blockIdx.y, k = blockIdx.x * blockDim.x + threadIdx.x, z = blockIdx.z;
            if (k >= M) return;
            const unsigned per = (W + kPowerSplits - 1) / kPowerSplits, t0 = z * per, t1 = min(W, t0 + per);
            const float2 *p = win + static_cast<std::size_t>(s) * W * M + k;
            float acc = 0.f;
#pragma unroll 8
            for (unsigned t = t0; t < t1; ++t) {
                const float2 v = p[static_cast<std::size_t>(t) * M];
                acc = fmaf(v.x, v.x, fmaf(v.y, v.y, acc));
            }
            part[(static_cast<std::size_t>(z) * gridDim.y + s) * M + k] = acc;
        }

        // One block per stream: bitonic sort of the M channel powers in shared memory, take the percentile,
        // then flag channels above the threshold.
        __global__ void floor_flag_kernel(const float *__restrict__ part, float *power, float *__restrict__ floor_out,
                                          std::uint8_t *__restrict__ flags, unsigned M, unsigned W, DetectConfig cfg) {
            extern __shared__ float v[];
            const unsigned s = blockIdx.x, S = gridDim.x;
            float *p = power + static_cast<std::size_t>(s) * M;
            for (unsigned i = threadIdx.x; i < M; i += blockDim.x) { // power = mean over the window
                float e = 0.f;
                for (unsigned z = 0; z < kPowerSplits; ++z) e += part[(static_cast<std::size_t>(z) * S + s) * M + i];
                p[i] = v[i] = e / static_cast<float>(W);
            }
            __syncthreads();
            float thr = 0.f;
            if (cfg.mode == DetectMode::relative) {
                for (unsigned k = 2; k <= M; k <<= 1)
                    for (unsigned j = k >> 1; j > 0; j >>= 1) {
                        for (unsigned i = threadIdx.x; i < M; i += blockDim.x) {
                            const unsigned ij = i ^ j;
                            if (ij > i) {
                                const bool up = (i & k) == 0;
                                if ((v[i] > v[ij]) == up) {
                                    const float t = v[i];
                                    v[i] = v[ij];
                                    v[ij] = t;
                                }
                            }
                        }
                        __syncthreads();
                    }
                const unsigned q = min(M - 1, static_cast<unsigned>(cfg.percentile * static_cast<float>(M)));
                const float fl = v[q];
                if (threadIdx.x == 0) floor_out[s] = fl;
                thr = fl * exp10f(cfg.threshold_db / 10.f);
            } else {
                if (threadIdx.x == 0) floor_out[s] = 0.f;
                thr = cfg.mode == DetectMode::absolute ? cfg.absolute_power : -1.f;
            }
            const float rej = cfg.neighbor_reject_db > 0.f ? exp10f(cfg.neighbor_reject_db / 10.f) : 0.f;
            for (unsigned i = threadIdx.x; i < M; i += blockDim.x) {
                bool f = cfg.mode == DetectMode::all || p[i] > thr;
                if (f && cfg.mode != DetectMode::all && rej > 0.f) {
                    // channels k - 1 and k + 1 are frequency neighbours, with wrap-around (0 is next to M - 1)
                    const float lo = p[(i + M - 1) & (M - 1)], hi = p[(i + 1) & (M - 1)];
                    f = fmaxf(lo, hi) <= p[i] * rej;
                }
                flags[static_cast<std::size_t>(s) * M + i] = f ? 1 : 0;
            }
        }

        // Hand-written compaction: one block of 1024 threads walks the flags in chunks of 1024; per chunk a
        // warp ballot + popc gives each thread its rank inside the warp, a shared scan over the 32 warp
        // counts gives the warp offsets. Output order = index order (deterministic).
        __global__ void __launch_bounds__(1024) compact_ballot_kernel(const std::uint8_t *__restrict__ flags, unsigned n,
                                                                       std::uint32_t *__restrict__ out,
                                                                       std::uint32_t *__restrict__ count) {
            __shared__ unsigned warp_sum[32];
            __shared__ unsigned base;
            const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
            if (threadIdx.x == 0) base = 0;
            __syncthreads();
            for (unsigned c0 = 0; c0 < n; c0 += 1024) {
                const unsigned i = c0 + threadIdx.x;
                const bool f = i < n && flags[i];
                const unsigned m = __ballot_sync(0xffffffffu, f);
                if (lane == 0) warp_sum[warp] = __popc(m);
                __syncthreads();
                if (warp == 0) { // inclusive scan of the 32 warp counts
                    unsigned x = warp_sum[lane];
                    for (int o = 1; o < 32; o <<= 1) {
                        const unsigned y = __shfl_up_sync(0xffffffffu, x, o);
                        if (lane >= static_cast<unsigned>(o)) x += y;
                    }
                    warp_sum[lane] = x;
                }
                __syncthreads();
                const unsigned warp_off = warp ? warp_sum[warp - 1] : 0;
                if (f) out[base + warp_off + __popc(m & ((1u << lane) - 1u))] = i;
                __syncthreads();
                if (threadIdx.x == 0) base += warp_sum[31];
                __syncthreads();
            }
            if (threadIdx.x == 0) *count = base;
        }

        // out[e][t] = win[s][t][k] / sqrt(power[s][k]) for active entry e = (s, k), frames tiled by 32 and
        // entries by 32, transposed through shared memory: reads along k, writes along t (both coalesced
        // when active channels are adjacent).
        // OutT = __half2: the same values rounded to fp16 (for an engine with an fp16 input).
        __device__ __forceinline__ void store_out(float2 *p, float2 v) { *p = v; }
        __device__ __forceinline__ void store_out(__half2 *p, float2 v) { *p = __floats2half2_rn(v.x, v.y); }
        template <class OutT>
        __global__ void gather_kernel(const float2 *__restrict__ win, const float *__restrict__ power,
                                      const std::uint32_t *__restrict__ active, const std::uint32_t *__restrict__ count,
                                      unsigned first, unsigned n, unsigned M, unsigned W, OutT *__restrict__ out) {
            __shared__ float2 tile[32][33];
            const unsigned e0 = blockIdx.x * 32, t0 = blockIdx.y * 32;
            const unsigned avail = min(n, *count > first ? *count - first : 0u);
            if (e0 >= avail) return;
            {
                const unsigned e = e0 + threadIdx.x;
                float scale = 0.f;
                std::size_t base = 0;
                if (e < avail) {
                    const unsigned id = active[first + e], s = id / M, k = id % M;
                    scale = rsqrtf(fmaxf(power[id], 1e-30f));
                    base = static_cast<std::size_t>(s) * W * M + k;
                }
                for (unsigned ty = threadIdx.y; ty < 32; ty += blockDim.y) {
                    float2 v = make_float2(0.f, 0.f);
                    if (e < avail) {
                        v = win[base + static_cast<std::size_t>(t0 + ty) * M];
                        v.x *= scale;
                        v.y *= scale;
                    }
                    tile[ty][threadIdx.x] = v;
                }
            }
            __syncthreads();
            for (unsigned ey = threadIdx.y; ey < 32; ey += blockDim.y) {
                const unsigned e = e0 + ey;
                if (e < avail) store_out(out + static_cast<std::size_t>(e) * W + t0 + threadIdx.x, tile[threadIdx.x][ey]);
            }
        }

    } // namespace

    WindowStage::WindowStage(std::uint32_t streams, std::uint32_t channels, std::uint32_t window)
        : streams_(streams), channels_(channels), window_(window) {
        if (channels == 0 || (channels & (channels - 1)) != 0 || channels > 4096)
            throw std::invalid_argument("WindowStage: channels must be a power of two <= 4096");
        if (window % 32 != 0) throw std::invalid_argument("WindowStage: window must be a multiple of 32");
        const std::size_t n = entries();
        CURF_CUDA_CHECK(cudaMalloc(&d_power_, n * sizeof(float)));
        CURF_CUDA_CHECK(cudaMalloc(&d_part_, n * kPowerSplits * sizeof(float)));
        CURF_CUDA_CHECK(cudaMalloc(&d_floor_, streams * sizeof(float)));
        CURF_CUDA_CHECK(cudaMalloc(&d_flags_, n));
        CURF_CUDA_CHECK(cudaMalloc(&d_active_, n * sizeof(std::uint32_t)));
        CURF_CUDA_CHECK(cudaMalloc(&d_count_, sizeof(std::uint32_t)));
        CURF_CUDA_CHECK(cudaMemset(d_count_, 0, sizeof(std::uint32_t)));
        CURF_CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr, cub_temp_bytes_, cub::CountingInputIterator<std::uint32_t>(0),
                                                   d_flags_, d_active_, d_count_, static_cast<int>(n)));
        CURF_CUDA_CHECK(cudaMalloc(&d_cub_temp_, cub_temp_bytes_));
    }

    WindowStage::~WindowStage() {
        cudaFree(d_power_);
        cudaFree(d_part_);
        cudaFree(d_floor_);
        cudaFree(d_flags_);
        cudaFree(d_active_);
        cudaFree(d_count_);
        cudaFree(d_cub_temp_);
    }

    void WindowStage::detect(const cf32 *d_win, const DetectConfig &cfg, cudaStream_t stream) {
        const auto *win = reinterpret_cast<const float2 *>(d_win);
        power_kernel<<<dim3((channels_ + 255) / 256, streams_, kPowerSplits), 256, 0, stream>>>(win, d_part_, channels_, window_);
        floor_flag_kernel<<<streams_, 512, channels_ * sizeof(float), stream>>>(d_part_, d_power_, d_floor_, d_flags_, channels_,
                                                                                window_,
                                                                                cfg);
        CURF_CUDA_CHECK(cudaGetLastError());
        if (cfg.compaction == Compaction::ballot) {
            compact_ballot_kernel<<<1, 1024, 0, stream>>>(d_flags_, entries(), d_active_, d_count_);
            CURF_CUDA_CHECK(cudaGetLastError());
        } else {
            std::size_t bytes = cub_temp_bytes_;
            CURF_CUDA_CHECK(cub::DeviceSelect::Flagged(d_cub_temp_, bytes, cub::CountingInputIterator<std::uint32_t>(0),
                                                       d_flags_, d_active_, d_count_, static_cast<int>(entries()), stream));
        }
    }

    void WindowStage::gather(const cf32 *d_win, std::uint32_t first, std::uint32_t n, float *d_out, cudaStream_t stream) {
        if (n == 0) return;
        const dim3 grid((n + 31) / 32, window_ / 32), block(32, 8);
        gather_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<const float2 *>(d_win), d_power_, d_active_, d_count_,
                                                  first, n, channels_, window_, reinterpret_cast<float2 *>(d_out));
        CURF_CUDA_CHECK(cudaGetLastError());
    }

    void WindowStage::gather_half(const cf32 *d_win, std::uint32_t first, std::uint32_t n, __half *d_out, cudaStream_t stream) {
        if (n == 0) return;
        const dim3 grid((n + 31) / 32, window_ / 32), block(32, 8);
        gather_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<const float2 *>(d_win), d_power_, d_active_, d_count_,
                                                  first, n, channels_, window_, reinterpret_cast<__half2 *>(d_out));
        CURF_CUDA_CHECK(cudaGetLastError());
    }

} // namespace curf::cuda
