#pragma once

#include "../Registry.h"
#include "../Components/TransformComponent.h"
#include "../Components/WorldTransformComponent.h"
#include "../Components/ParentComponent.h"

#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace Eden::HierarchySystem
{
    // Deliberately a free function, not an Eden::System - same reasoning
    // as Systems/RenderSystem.h: this has to run at a specific point in
    // the frame (after every system that might mutate TransformComponent
    // - SpinSystem, gameplay code, etc. - and before RenderSystem::
    // BuildDrawList consumes the result), and System registration order
    // is an easy way to get that one frame stale. Call this directly from
    // the main loop, right after Registry::UpdateSystems() and before
    // RenderSystem::BuildDrawList().
    //
    // For every entity with a TransformComponent, writes a
    // WorldTransformComponent equal to:
    //   - its own local matrix, if it has no ParentComponent (or the
    //     parent reference is missing/invalid/would form a cycle)
    //   - parent's resolved world matrix * its own local matrix, otherwise
    //
    // Resolves parents before children automatically (memoized recursion)
    // regardless of what order View() happens to return entities in.
    inline void ResolveWorldTransforms(Registry& registry)
    {
        std::unordered_map<Entity, glm::mat4> resolved;
        std::unordered_set<Entity> inProgress;

        // See Registry::GetStorage's comment - resolved once here instead
        // of on every recursive call. Safe to hold these across the
        // registry.AddComponent(entity, WorldTransformComponent{...}) call
        // in the loop below: that touches WorldTransformComponent's
        // storage, a different object entirely from these two.
        auto& transforms = registry.GetStorage<TransformComponent>();
        auto& parents = registry.GetStorage<ParentComponent>();

        // Recursive helper. Takes `resolve` by reference to itself so it
        // can call itself - a plain lambda can't recurse without this.
        std::function<glm::mat4(Entity)> resolve = [&](Entity entity) -> glm::mat4
        {
            auto it = resolved.find(entity);
            if (it != resolved.end())
            {
                return it->second;
            }

            glm::mat4 local = transforms.Get(entity).GetModelMatrix();
            glm::mat4 world = local;

            if (parents.Has(entity))
            {
                Entity parent = parents.Get(entity).parent;

                // Cycle/invalid-reference guard: only recurse into the
                // parent if it's a real, resolvable, non-cyclic entity.
                // transforms.Has(parent) doing the heavy lifting here is
                // worth calling out: ComponentStorage's Has() checks the
                // entity's GENERATION, not just its index (see
                // ComponentStorage.h), so a stale `parent` left over from
                // a destroyed entity whose index got recycled for
                // something else correctly returns false here instead of
                // silently resolving against that new, unrelated entity.
                // Self-parenting and parenting loops still fall back to
                // treating this entity as a root rather than
                // stack-overflowing or infinite-looping.
                if (parent != NullEntity &&
                    parent != entity &&
                    transforms.Has(parent) &&
                    inProgress.find(parent) == inProgress.end())
                {
                    inProgress.insert(entity);
                    glm::mat4 parentWorld = resolve(parent);
                    inProgress.erase(entity);

                    world = parentWorld * local;
                }
            }

            resolved[entity] = world;
            return world;
        };

        for (Entity entity : registry.View<TransformComponent>())
        {
            glm::mat4 world = resolve(entity);
            registry.AddComponent(entity, WorldTransformComponent{ world });
        }
    }
}
