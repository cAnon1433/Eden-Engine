#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>

namespace Eden
{
    // Struct-of-arrays storage for every particle in the simulation.
    // Deliberately NOT ECS entities/components - an SPH fluid is
    // realistically thousands of samples, and each one becoming an
    // Entity would mean paying Registry's per-entity bookkeeping
    // (generational handles, sparse-set storage lookups) for something
    // that's much closer to "one big array" than "a game object". This
    // is the same reasoning as RenderSystem::BuildDrawList bypassing
    // Registry::View<>() at the top end of its entity-count range, just
    // applied one level earlier - particles never become entities at
    // all. ParticleSystem is the only thing that owns or touches this;
    // nothing else should reach into it directly.
    //
    // All four arrays are always kept the same length and in the same
    // particle order - index i in one is the same particle as index i in
    // every other. AddParticle/RemoveParticleSwapBack keep this invariant;
    // nothing else should resize these directly.
    struct ParticleData
    {
        std::vector<glm::vec3> positions;
        std::vector<glm::vec3> velocities;

        // Per-step scratch, not persistent state - ParticleSystem::Step
        // clears and rewrites these every substep (density/pressure from
        // neighbor positions, force from density/pressure/viscosity/
        // gravity). Kept here rather than as locals inside Step() so the
        // neighbor-loop and integration-loop can be separate passes over
        // the same indices without extra allocation each substep.
        std::vector<float> densities;
        std::vector<float> pressures;
        std::vector<glm::vec3> forces;

        size_t Count() const
        {
            return positions.size();
        }

        void Reserve(size_t n)
        {
            positions.reserve(n);
            velocities.reserve(n);
            densities.reserve(n);
            pressures.reserve(n);
            forces.reserve(n);
        }

        // Appends one particle at the given world position with an
        // optional initial velocity. Returns its index - stable only
        // until the next RemoveParticleSwapBack call (see below), same
        // caveat as any swap-back removal scheme.
        size_t AddParticle(const glm::vec3& position, const glm::vec3& velocity = glm::vec3(0.0f))
        {
            positions.push_back(position);
            velocities.push_back(velocity);
            densities.push_back(0.0f);
            pressures.push_back(0.0f);
            forces.push_back(glm::vec3(0.0f));
            return positions.size() - 1;
        }

        // Swap-back removal: O(1), but reorders the array (the last
        // particle takes index `i`'s place). Nothing outside
        // ParticleSystem currently holds onto particle indices across
        // frames, so this is safe today - if something ever needs stable
        // particle identity across removals (e.g. a future "this specific
        // particle is a tracked droplet"), that's a real design change,
        // not a small tweak to this function.
        void RemoveParticleSwapBack(size_t i)
        {
            size_t last = positions.size() - 1;
            if (i != last)
            {
                positions[i] = positions[last];
                velocities[i] = velocities[last];
                densities[i] = densities[last];
                pressures[i] = pressures[last];
                forces[i] = forces[last];
            }
            positions.pop_back();
            velocities.pop_back();
            densities.pop_back();
            pressures.pop_back();
            forces.pop_back();
        }

        void Clear()
        {
            positions.clear();
            velocities.clear();
            densities.clear();
            pressures.clear();
            forces.clear();
        }
    };
}
