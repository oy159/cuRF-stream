// Baseline: cuDNN 9 (legacy convolution API) as an ordinary conv variant, "cudnn".
//
// The engine's channels-last activations are NHWC with H = 1 and its weights [cout][k][cin] are KRSC,
// so cuDNN runs on the engine's own buffers. Each layer is one fused call:
//   y = act(conv(x, w) + z + bias),   z = residual (alpha2 = 1) or nothing (alpha2 = 0)
// via cudnnConvolutionBiasActivationForward (fp16 tensors, fp32 compute, tensor-op math). The algorithm is
// chosen once per shape with cudnnFindConvolutionForwardAlgorithmEx (cuDNN's own autotuner). cuDNN only
// allows CUDNN_ACTIVATION_IDENTITY with IMPLICIT_PRECOMP_GEMM, so layers without ReLU use that algorithm.
//
// Per-shape state (descriptors, algorithm, workspace, an fp16 copy of the bias, repacked weights when
// kpad != k * cin, an fp16 copy of an fp32 input) is created on the first launch, which the engine does
// while tuning, before any CUDA Graph capture. The fp32-input stem pays for a cast kernel.

#include "conv1d_internal.hpp"

#include "curf/cuda/memory.hpp"

#include <cudnn.h>

#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>

namespace curf::engine::detail {

    namespace {

        void check(cudnnStatus_t s, const char *what) {
            if (s != CUDNN_STATUS_SUCCESS) throw std::runtime_error(std::string("cuDNN: ") + what + ": " + cudnnGetErrorString(s));
        }
#define CURF_CUDNN_CHECK(expr) check((expr), #expr)

        cudnnHandle_t handle() {
            static cudnnHandle_t h = [] {
                cudnnHandle_t x;
                CURF_CUDNN_CHECK(cudnnCreate(&x));
                return x;
            }();
            return h;
        }

        struct State {
            cudnnTensorDescriptor_t x = nullptr, y = nullptr, b = nullptr;
            cudnnFilterDescriptor_t w = nullptr;
            cudnnConvolutionDescriptor_t conv = nullptr;
            cudnnActivationDescriptor_t act = nullptr;
            cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
            cuda::DeviceBuffer<std::byte> ws;
            cuda::DeviceBuffer<__half> bias16, wpack, xcast;
            const __half *wptr = nullptr;
            std::string algo_name;
            ~State() {
                if (x) cudnnDestroyTensorDescriptor(x);
                if (y) cudnnDestroyTensorDescriptor(y);
                if (b) cudnnDestroyTensorDescriptor(b);
                if (w) cudnnDestroyFilterDescriptor(w);
                if (conv) cudnnDestroyConvolutionDescriptor(conv);
                if (act) cudnnDestroyActivationDescriptor(act);
            }
        };

        using Key = std::tuple<const void *, const void *, const void *, const void *, const void *, int, int, int, int,
                               int, int, int, int, bool, bool>;

        State &state(const ConvParams &p) {
            static std::map<Key, std::unique_ptr<State>> states;
            static std::mutex mu;
            std::lock_guard<std::mutex> lock(mu);
            const Key key{p.w, p.bias, p.x, p.y, p.res, p.batch, p.lin, p.lout, p.cin, p.cout, p.k, p.stride, p.pad,
                          p.relu, p.x_f32};
            auto &slot = states[key];
            if (slot) return *slot;
            auto st = std::make_unique<State>();
            CURF_CUDNN_CHECK(cudnnCreateTensorDescriptor(&st->x));
            CURF_CUDNN_CHECK(cudnnCreateTensorDescriptor(&st->y));
            CURF_CUDNN_CHECK(cudnnCreateTensorDescriptor(&st->b));
            CURF_CUDNN_CHECK(cudnnCreateFilterDescriptor(&st->w));
            CURF_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&st->conv));
            CURF_CUDNN_CHECK(cudnnCreateActivationDescriptor(&st->act));
            CURF_CUDNN_CHECK(cudnnSetTensor4dDescriptor(st->x, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, p.batch, p.cin, 1, p.lin));
            CURF_CUDNN_CHECK(cudnnSetTensor4dDescriptor(st->y, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, p.batch, p.cout, 1, p.lout));
            CURF_CUDNN_CHECK(cudnnSetTensor4dDescriptor(st->b, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, 1, p.cout, 1, 1));
            CURF_CUDNN_CHECK(cudnnSetFilter4dDescriptor(st->w, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC, p.cout, p.cin, 1, p.k));
            CURF_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(st->conv, 0, p.pad, 1, p.stride, 1, 1,
                                                             CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
            CURF_CUDNN_CHECK(cudnnSetConvolutionMathType(st->conv, CUDNN_TENSOR_OP_MATH));
            CURF_CUDNN_CHECK(cudnnSetActivationDescriptor(
                st->act, p.relu ? CUDNN_ACTIVATION_RELU : CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0.0));

            // fp16 bias; dense KRSC weights; fp16 staging for an fp32 input
            st->bias16 = cuda::DeviceBuffer<__half>(static_cast<std::size_t>(p.cout));
            launch_float_to_half(p.bias, st->bias16.get(), st->bias16.size(), nullptr);
            st->wptr = p.w;
            if (p.kpad != p.k * p.cin) {
                st->wpack = cuda::DeviceBuffer<__half>(static_cast<std::size_t>(p.cout) * p.k * p.cin);
                CURF_CUDA_CHECK(cudaMemcpy2D(st->wpack.get(), sizeof(__half) * p.k * p.cin, p.w, sizeof(__half) * p.kpad,
                                             sizeof(__half) * p.k * p.cin, p.cout, cudaMemcpyDeviceToDevice));
                st->wptr = st->wpack.get();
            }
            if (p.x_f32) st->xcast = cuda::DeviceBuffer<__half>(static_cast<std::size_t>(p.batch) * p.lin * p.cin);
            const void *xin = p.x_f32 ? static_cast<const void *>(st->xcast.get()) : p.x;
            CURF_CUDA_CHECK(cudaDeviceSynchronize());

            if (p.relu) {
                // cuDNN's own benchmark over all algorithms, on the real buffers
                constexpr std::size_t kMaxWs = std::size_t{64} << 20;
                cuda::DeviceBuffer<std::byte> tmp(kMaxWs);
                cudnnConvolutionFwdAlgoPerf_t perf[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
                int n = 0;
                CURF_CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithmEx(
                    handle(), st->x, xin, st->w, st->wptr, st->conv, st->y, p.y, CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &n, perf,
                    tmp.get(), kMaxWs));
                bool found = false;
                for (int i = 0; i < n && !found; ++i)
                    if (perf[i].status == CUDNN_STATUS_SUCCESS && perf[i].memory <= kMaxWs) {
                        st->algo = perf[i].algo;
                        CURF_CUDNN_CHECK(cudnnSetConvolutionMathType(st->conv, perf[i].mathType));
                        found = true;
                    }
                if (!found) {
                    // Seen for some large-batch shapes: every candidate failed or wanted > 64 MB. The fused API's
                    // reference algorithm always works.
                    std::fprintf(stderr, "cuDNN: Find gave no usable algorithm (batch %d, %dx%d k%d s%d); using IMPLICIT_PRECOMP_GEMM\n",
                                 p.batch, p.cin, p.cout, p.k, p.stride);
                    for (int i = 0; i < n; ++i)
                        std::fprintf(stderr, "  algo %d status %s memory %zu\n", static_cast<int>(perf[i].algo),
                                     cudnnGetErrorString(perf[i].status), perf[i].memory);
                    st->algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
                    CURF_CUDNN_CHECK(cudnnSetConvolutionMathType(st->conv, CUDNN_TENSOR_OP_MATH));
                }
            }
            std::size_t ws = 0;
            CURF_CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle(), st->x, st->w, st->conv, st->y, st->algo, &ws));
            st->ws = cuda::DeviceBuffer<std::byte>(ws);
            slot = std::move(st);
            return *slot;
        }

        void launch_cudnn(const ConvParams &p, cudaStream_t s) {
            State &st = state(p);
            CURF_CUDNN_CHECK(cudnnSetStream(handle(), s));
            const void *xin = p.x;
            if (p.x_f32) {
                launch_float_to_half(static_cast<const float *>(p.x), st.xcast.get(), st.xcast.size(), s);
                xin = st.xcast.get();
            }
            const float alpha1 = 1.f, alpha2 = p.res ? 1.f : 0.f;
            CURF_CUDNN_CHECK(cudnnConvolutionBiasActivationForward(
                handle(), &alpha1, st.x, xin, st.w, st.wptr, st.conv, st.algo, st.ws.get(), st.ws.bytes(), &alpha2, st.y,
                p.res ? static_cast<const void *>(p.res) : p.y, st.b, st.bias16.get(), st.act, st.y, p.y));
        }

        bool sup_cudnn(const ConvParams &p) { return p.pool == nullptr; }

    } // namespace

    void add_cudnn_variants(std::vector<ConvVariant> &v) {
        v.push_back({"cudnn", true, &sup_cudnn, &launch_cudnn});
    }

} // namespace curf::engine::detail
