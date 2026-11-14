// Baseline: CUTLASS 2.x-style implicit-GEMM Conv2dFprop (SM80 tensor ops, multistage cp.async mainloop) as
// ordinary conv variants "cutlass.{f32,f16}.{TBM}x{TBN}".
//
// Same NHWC (H = 1) activations and KRSC weights as the engine. Epilogues, all fused:
//   no residual:  D = act(acc + C), C = the bias as a stride-0 broadcast tensor (LinearCombination[Relu])
//   residual:     D = relu((acc + bias) + residual), bias as the broadcast vector of the
//                 Conv2dFpropWithBroadcast kernel (LinearCombinationResidualBlock<Identity, plus, ReLu>)
// Not supported: the stem (cin = 2 breaks the 8-element access alignment, and its input is fp32), the fused
// global pool. Per-shape operator objects are initialised on the first launch (before graph capture).

#include "conv1d_internal.hpp"

#include "curf/cuda/memory.hpp"

#include <cutlass/conv/conv2d_problem_size.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/conv/kernel/default_conv2d_fprop_with_broadcast.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/epilogue/thread/linear_combination_residual_block.h>
#include <cutlass/functional.h>

#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>

namespace curf::engine::detail {

    namespace {

        using cutlass::half_t;
        using Layout = cutlass::layout::TensorNHWC;
        using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
        using Inst = cutlass::gemm::GemmShape<16, 8, 16>;

        void check(cutlass::Status s, const char *what) {
            if (s != cutlass::Status::kSuccess)
                throw std::runtime_error(std::string("CUTLASS: ") + what + ": " + cutlassGetStatusString(s));
        }

        // Plain fprop: D = act(acc + C), C = bias broadcast (stride 0).
        template <class Acc, int TBM, int TBN, int WM, int WN, bool Relu>
        struct PlainConv {
            using Epi = std::conditional_t<Relu, cutlass::epilogue::thread::LinearCombinationRelu<half_t, 8, Acc, float>,
                                           cutlass::epilogue::thread::LinearCombination<half_t, 8, Acc, float>>;
            using Kernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
                half_t, Layout, half_t, Layout, half_t, Layout, Acc, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                cutlass::gemm::GemmShape<TBM, TBN, 32>, cutlass::gemm::GemmShape<WM, WN, 32>, Inst, Epi, Swizzle, 3,
                cutlass::arch::OpMultiplyAdd, cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
            using Op = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;
        };

        // Residual: D = relu((acc + bias) + residual).
        template <class Acc, int TBM, int TBN, int WM, int WN>
        struct ResidualConv {
            using Epi = cutlass::epilogue::thread::LinearCombinationResidualBlock<
                half_t, Acc, float, half_t, 8, cutlass::epilogue::thread::Identity, cutlass::plus,
                cutlass::epilogue::thread::ReLu, cutlass::epilogue::thread::detail::NoOp, false, float>;
            using Kernel = typename cutlass::conv::kernel::DefaultConv2dFpropWithBroadcast<
                half_t, Layout, half_t, Layout, half_t, Layout, Acc, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                cutlass::gemm::GemmShape<TBM, TBN, 32>, cutlass::gemm::GemmShape<WM, WN, 32>, Inst, Epi, Swizzle, 3,
                cutlass::arch::OpMultiplyAdd, cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
            using Op = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;
        };

        struct Runner {
            virtual ~Runner() = default;
            virtual void run(cudaStream_t s) = 0;
        };

        cutlass::conv::Conv2dProblemSize problem(const ConvParams &p) {
            return {p.batch, 1, p.lin, p.cin, p.cout, 1, p.k, 1, p.lout, 0, p.pad, 1, p.stride, 1, 1,
                    cutlass::conv::Mode::kCrossCorrelation, 1};
        }

        template <class Op> struct OpRunner : Runner {
            Op op;
            cuda::DeviceBuffer<std::byte> ws;
            cuda::DeviceBuffer<half_t> bias16;
            void init(const typename Op::Arguments &args) {
                check(op.can_implement(args), "can_implement");
                ws = cuda::DeviceBuffer<std::byte>(Op::get_workspace_size(args));
                check(op.initialize(args, ws.get()), "initialize");
            }
            void run(cudaStream_t s) override { check(op.run(s), "run"); }
        };

        template <class Acc, int TBM, int TBN, int WM, int WN>
        std::unique_ptr<Runner> make_runner(const ConvParams &p) {
            const auto ps = problem(p);
            const auto x = const_cast<half_t *>(static_cast<const half_t *>(p.x));
            const auto w = const_cast<half_t *>(reinterpret_cast<const half_t *>(p.w));
            auto *y = reinterpret_cast<half_t *>(p.y);
            const Layout lx = Layout::packed({p.batch, 1, p.lin, p.cin});
            const Layout lw = Layout::packed({p.cout, 1, p.k, p.cin});
            const Layout ly = Layout::packed({p.batch, 1, p.lout, p.cout});
            if (p.res) {
                using Op = typename ResidualConv<Acc, TBM, TBN, WM, WN>::Op;
                auto r = std::make_unique<OpRunner<Op>>();
                typename Op::Arguments args(ps, {x, lx}, {w, lw},
                                            {const_cast<half_t *>(reinterpret_cast<const half_t *>(p.res)), ly}, {y, ly},
                                            {1.f, 1.f}, cutlass::conv::SplitKMode::kSerial,
                                            const_cast<float *>(p.bias), nullptr, 0, 0);
                r->init(args);
                return r;
            }
            auto make = [&](auto relu_c) -> std::unique_ptr<Runner> {
                using Op = typename PlainConv<Acc, TBM, TBN, WM, WN, decltype(relu_c)::value>::Op;
                auto r = std::make_unique<OpRunner<Op>>();
                r->bias16 = cuda::DeviceBuffer<half_t>(static_cast<std::size_t>(p.cout));
                launch_float_to_half(p.bias, reinterpret_cast<__half *>(r->bias16.get()), r->bias16.size(), nullptr);
                CURF_CUDA_CHECK(cudaDeviceSynchronize());
                const Layout lb(cutlass::make_Coord(0, 0, 0)); // bias[c] for every (n, h, w)
                typename Op::Arguments args(ps, {x, lx}, {w, lw}, {r->bias16.get(), lb}, {y, ly}, {1.f, 1.f});
                r->init(args);
                return r;
            };
            return p.relu ? make(std::true_type{}) : make(std::false_type{});
        }

        using Key = std::tuple<const void *, const void *, const void *, const void *, const void *, int, int, int, int,
                               int, int, int, int, bool>;

        template <class Acc, int TBM, int TBN, int WM, int WN>
        void launch_cutlass(const ConvParams &p, cudaStream_t s) {
            static std::map<Key, std::unique_ptr<Runner>> runners;
            static std::mutex mu;
            Runner *r;
            {
                std::lock_guard<std::mutex> lock(mu);
                const Key key{p.w, p.bias, p.x, p.y, p.res, p.batch, p.lin, p.lout, p.cin, p.cout, p.k, p.stride, p.pad, p.relu};
                auto &slot = runners[key];
                if (!slot) slot = make_runner<Acc, TBM, TBN, WM, WN>(p);
                r = slot.get();
            }
            r->run(s);
        }

        template <int TBN> bool sup_cutlass(const ConvParams &p) {
            return !p.x_f32 && p.pool == nullptr && p.cin % 8 == 0 && p.cout % TBN == 0 && p.kpad == p.k * p.cin &&
                   (!p.res || p.relu);
        }

        template <int TBM, int TBN, int WM, int WN> void add(std::vector<ConvVariant> &v) {
            const std::string tile = std::to_string(TBM) + "x" + std::to_string(TBN);
            v.push_back({"cutlass.f32." + tile, true, &sup_cutlass<TBN>, &launch_cutlass<float, TBM, TBN, WM, WN>});
            v.push_back({"cutlass.f16." + tile, true, &sup_cutlass<TBN>, &launch_cutlass<half_t, TBM, TBN, WM, WN>});
        }

    } // namespace

    void add_cutlass_variants(std::vector<ConvVariant> &v) {
        add<128, 32, 64, 32>(v);
        add<64, 64, 32, 32>(v);
        add<128, 64, 64, 32>(v);
        add<128, 128, 64, 64>(v);
    }

} // namespace curf::engine::detail
