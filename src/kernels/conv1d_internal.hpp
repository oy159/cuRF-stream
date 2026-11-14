#pragma once
// Registry hooks shared between the conv1d kernel translation units.

#include "curf/engine/kernels.hpp"

#include <vector>

namespace curf::engine::detail {

    // Ladder step 5: cp.async multi-stage pipeline + ldmatrix + mma.sync (conv1d_mma.cu).
    void add_mma_variants(std::vector<ConvVariant> &v);

} // namespace curf::engine::detail
