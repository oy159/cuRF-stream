#pragma once
// Registry hooks shared between the conv1d kernel translation units.

#include "curf/engine/kernels.hpp"

#include <vector>

namespace curf::engine::detail {

    // Ladder step 5: cp.async multi-stage pipeline + ldmatrix + mma.sync (conv1d_mma.cu).
    void add_mma_variants(std::vector<ConvVariant> &v);

    // INT8: mma.sync.m16n8k32.s8 implicit GEMM (conv1d_int8.cu).
    void add_int8_variants(std::vector<ConvVariant> &v);
    void add_int8_block_variants(std::vector<BlockVariant> &v); // block_int8.cu

    // Library baselines registered as ordinary conv variants (only when built with them).
    void add_cudnn_variants(std::vector<ConvVariant> &v);   // baseline_cudnn.cpp, CURF_HAVE_CUDNN
    void add_cutlass_variants(std::vector<ConvVariant> &v); // baseline_cutlass.cu, CURF_HAVE_CUTLASS

} // namespace curf::engine::detail
