#pragma once

#include "curf/runtime/burst_classifier.hpp"

#include <cstdio>
#include <string>
#include <string_view>

namespace curf {
    std::string run_burst_demo(ISource &, const engine::Model &, const BurstConfig &, std::FILE *feed,
                               std::FILE *trace, std::string_view source, unsigned rx, unsigned every);
}
