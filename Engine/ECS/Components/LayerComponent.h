#pragma once

#include <cstdint>

namespace Eden
{
    // Bitmask for coarse categorization - which layer(s) an entity
    // belongs to. Covers both "LayerComponent" and "GroupComponent" from
    // the TODO (same idea, listed as alternate names); a bitmask was
    // picked over a single ID so an entity can belong to more than one
    // group at once (e.g. "Renderable | Interactable") and callers can
    // filter with plain bitwise ops.
    //
    // Not read by anything yet - RenderSystem doesn't filter on this,
    // there's no collision system to filter on this either. It exists so
    // future systems (culling, collision, debug-view filtering) have a
    // category to check without inventing their own per-system tagging
    // scheme.
    struct LayerComponent
    {
        using Mask = uint32_t;

        static constexpr Mask Default = 1u << 0;

        Mask mask = Default;
    };
}
