#pragma once

#include "Registry.h"
#include "Components/ParentComponent.h"
#include "Components/ChildrenComponent.h"

#include <algorithm>

namespace Eden::Hierarchy
{
    // ParentComponent and ChildrenComponent each describe half of the
    // same relationship, and nothing enforces they agree with each other
    // if you set them by hand with AddComponent directly - it's
    // perfectly possible to give an entity a ParentComponent without the
    // parent's ChildrenComponent ever finding out. SetParent is the one
    // correct way to change a parent-child link: it updates both sides
    // atomically, in one call.
    //
    // Pass NullEntity as `parent` to unparent `child` (removes its
    // ParentComponent entirely and detaches it from its old parent's
    // ChildrenComponent).
    //
    // What this does NOT do: cascade destruction. If a parent entity is
    // destroyed via Registry::DestroyEntity directly (not through here),
    // its children keep a ParentComponent pointing at a now-stale Entity
    // handle - detectably stale rather than silently wrong, thanks to
    // generational handles (see Entity.h, Registry::IsAlive), but still
    // just left dangling rather than cleaned up. Deciding what SHOULD
    // happen to children when a parent dies - destroy them too? re-parent
    // to the grandparent? orphan them in place? - is a real design
    // choice, not something to bake in as a side effect here.
    inline void SetParent(Registry& registry, Entity child, Entity parent)
    {
        // Detach from whatever it was previously parented to, if anything.
        if (registry.HasComponent<ParentComponent>(child))
        {
            Entity oldParent = registry.GetComponent<ParentComponent>(child).parent;

            if (oldParent != NullEntity && registry.HasComponent<ChildrenComponent>(oldParent))
            {
                auto& siblings = registry.GetComponent<ChildrenComponent>(oldParent).children;
                siblings.erase(std::remove(siblings.begin(), siblings.end(), child), siblings.end());
            }
        }

        if (parent == NullEntity)
        {
            registry.RemoveComponent<ParentComponent>(child);
            return;
        }

        registry.AddComponent(child, ParentComponent{ parent });

        if (!registry.HasComponent<ChildrenComponent>(parent))
        {
            registry.AddComponent(parent, ChildrenComponent{});
        }

        auto& kids = registry.GetComponent<ChildrenComponent>(parent).children;
        if (std::find(kids.begin(), kids.end(), child) == kids.end())
        {
            kids.push_back(child);
        }
    }
}
