#pragma once

#include "../../Renderer/Vulkan/RendererTypes.h"

#include <glm/glm.hpp>

namespace Eden
{
    // Periodically creates new entities. SpawnerSystem (see
    // Systems/SpawnerSystem.h) owns the actual entity-creation logic;
    // this component is just the configuration + running state for one
    // spawn point.
    //
    // Deliberately minimal about what gets spawned: a mesh handle and a
    // position offset, nothing about behavior/AI components yet, since
    // those are still listed as "deferred, needs behavior design first"
    // in the TODO. Extend this (or give it a richer "entity template"
    // concept) once that design exists.
    struct SpawnerComponent
    {
        float interval = 1.0f; // seconds between spawns
        float timer = 0.0f;    // internal: counts up toward interval

        MeshHandle meshToSpawn = InvalidMeshHandle;
        glm::vec3 spawnOffset{ 0.0f }; // relative to spawner's own TransformComponent, if it has one

        // If > 0, spawned entities get a LifetimeComponent set to this
        // many seconds (self-destruct after that long). 0 = permanent
        // until something else destroys them.
        float spawnedLifetimeSeconds = 0.0f;

        int maxSpawns = -1; // -1 = unlimited
        int spawnCount = 0; // internal: how many this spawner has produced so far
    };
}
