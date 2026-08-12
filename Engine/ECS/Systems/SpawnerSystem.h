#pragma once

#include "../Registry.h"
#include "../Components/SpawnerComponent.h"
#include "../Components/TransformComponent.h"
#include "../Components/MeshComponent.h"
#include "../Components/LifetimeComponent.h"

namespace Eden
{
    // Advances every SpawnerComponent's timer and creates a new entity
    // each time it crosses `interval`. Register via
    // Registry::RegisterSystem<SpawnerSystem>().
    //
    // Spawned entities get a TransformComponent (spawner's own position,
    // if it has a TransformComponent, plus spawnOffset) and a
    // MeshComponent if meshToSpawn is valid. That's it for now - no
    // AI/behavior components, since those are still deferred per the
    // TODO. This is meant to be the plumbing those attach to later, not
    // a finished spawner.
    class SpawnerSystem : public System
    {
    public:
        void Update(Registry& registry, float deltaTime) override
        {
            // See Registry::GetStorage's comment. Safe to hold `transforms`
            // across registry.AddComponent(spawned, transform) below - that
            // call may grow TransformComponent's internal dense/sparse
            // vectors, but it can't invalidate this reference to the
            // ComponentStorage<T> wrapper itself, and nothing here holds a
            // T& across that call (spawnPosition is copied out by value
            // first).
            auto& spawners = registry.GetStorage<SpawnerComponent>();
            auto& transforms = registry.GetStorage<TransformComponent>();

            for (Entity entity : registry.View<SpawnerComponent>())
            {
                auto& spawner = spawners.Get(entity);

                if (spawner.maxSpawns >= 0 && spawner.spawnCount >= spawner.maxSpawns)
                {
                    continue;
                }

                spawner.timer += deltaTime;
                if (spawner.timer < spawner.interval)
                {
                    continue;
                }
                spawner.timer -= spawner.interval;

                glm::vec3 spawnPosition = spawner.spawnOffset;
                if (transforms.Has(entity))
                {
                    spawnPosition += transforms.Get(entity).position;
                }

                Entity spawned = registry.CreateEntity();

                TransformComponent transform;
                transform.position = spawnPosition;
                registry.AddComponent(spawned, transform);

                if (spawner.meshToSpawn != InvalidMeshHandle)
                {
                    registry.AddComponent(spawned, MeshComponent{ spawner.meshToSpawn });
                }

                if (spawner.spawnedLifetimeSeconds > 0.0f)
                {
                    registry.AddComponent(spawned, LifetimeComponent{ spawner.spawnedLifetimeSeconds });
                }

                spawner.spawnCount++;
            }
        }
    };
}
