#pragma once

#include "../Entity.h"

#include <vector>

namespace Eden
{
    // Inverse of ParentComponent: the set of entities that consider this
    // entity their parent. Nothing currently keeps this in sync with
    // ParentComponent automatically - adding/removing a ParentComponent
    // elsewhere does NOT update the corresponding ChildrenComponent here.
    // That synchronization (and what happens to children when a parent
    // is destroyed) is exactly the kind of behavior a future
    // HierarchySystem should own, not something to silently bake into
    // Registry::AddComponent/DestroyEntity right now.
    struct ChildrenComponent
    {
        std::vector<Entity> children;
    };
}
