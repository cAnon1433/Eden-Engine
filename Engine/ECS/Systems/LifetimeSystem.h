#pragma once

#include "../Registry.h"
#include "../Components/LifetimeComponent.h"

namespace Eden
{
    // Ticks every LifetimeComponent down by deltaTime; destroys the
    // entity outright once its timer reaches zero. Register alongside
    // SpinSystem via Registry::RegisterSystem<LifetimeSystem>() - it's a
    // real System (mutates ECS state via DestroyEntity), unlike
    // RenderSystem which is a read-only free function.
    class LifetimeSystem : public System
    {
    public:
        void Update(Registry& registry, float deltaTime) override
        {
            // See Registry::GetStorage's comment. Safe to hold this
            // reference across the DestroyEntity() call below - it's a
            // reference to the ComponentStorage<T> wrapper itself (which
            // Registry keeps alive in a stable unique_ptr slot), not to a
            // specific element inside it, so swap-and-pop removals during
            // the loop don't invalidate it.
            auto& lifetimes = registry.GetStorage<LifetimeComponent>();

            for (Entity entity : registry.View<LifetimeComponent>())
            {
                auto& lifetime = lifetimes.Get(entity);
                lifetime.remainingSeconds -= deltaTime;

                if (lifetime.remainingSeconds <= 0.0f)
                {
                    // Safe to destroy mid-loop: View() returned a
                    // snapshot vector of entity IDs up front, not a live
                    // iterator into storage, so removing this entity
                    // doesn't disturb iteration over the rest.
                    registry.DestroyEntity(entity);
                }
            }
        }
    };
}
