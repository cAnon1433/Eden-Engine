#pragma once

#include "../Entity.h"

namespace Eden
{
    // Reference to a parent entity, for relative transforms (e.g. a
    // turret entity positioned relative to the tank entity it's mounted
    // on). Read by Systems/HierarchySystem.h to compute a world-space
    // transform from the parent chain.
    //
    // `parent` is a generational Entity handle (see Entity.h), so if the
    // parent is destroyed, this reference becomes detectably stale rather
    // than silently resolving to whatever new entity later reuses that
    // index - HierarchySystem checks this and falls back to treating the
    // child as a root instead of parenting it to the wrong thing. Nothing
    // clears this automatically when the parent dies, though - see
    // HierarchyUtils.h's SetParent for why that's a deliberate choice.
    struct ParentComponent
    {
        Entity parent = NullEntity;
    };
}
