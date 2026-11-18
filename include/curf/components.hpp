#pragma once
#include <cstdint>
#include <span>
#include <vector>

namespace curf {
    struct ActiveChannel {
        int bin;
        float power;
        std::uint64_t activity;
    };
    struct SignalComponent {
        int lo, hi;
        bool coherent = false; // four frequency quarters share an informative burst envelope
    };
    struct ComponentGroups {
        std::vector<SignalComponent> bands;
        unsigned joins = 0, splits = 0;
    };
    // One RX, signed-frequency sorted unique channels. A zero mask means unavailable.
    // Keeps legacy connected components unless current, nonconstant time evidence supports a change.
    ComponentGroups group_components(std::span<const ActiveChannel> channels, bool temporal);
}
