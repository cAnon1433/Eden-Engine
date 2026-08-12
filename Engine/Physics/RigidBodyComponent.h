#pragma once

#include <glm/glm.hpp>

namespace Eden
{
    // Static: never moves, infinite mass (inverseMass == 0), PhysicsSystem
    //   never integrates it. Immovable geometry (floor, walls).
    // Kinematic: infinite mass w.r.t. collision response (inverseMass == 0
    //   when computing impulses), but its TransformComponent IS advanced
    //   from linearVelocity/angularVelocity every physics step - driven
    //   externally (animation, AI, a moving platform script) rather than by
    //   forces. Never gets gravity or accumulated force/torque applied.
    // Dynamic: normal simulated body - integrates under gravity/forces,
    //   participates fully in collision response.
    enum class BodyType
    {
        Static,
        Kinematic,
        Dynamic
    };

    struct RigidBodyComponent
    {
        BodyType type = BodyType::Dynamic;

        // 0 for Static/Kinematic (infinite mass - PhysicsSystem enforces
        // this regardless of what's stored here, see PhysicsSystem::Step).
        // Stored as inverse mass rather than mass because every place this
        // gets used (integration, impulse resolution) wants 1/m, and
        // 1/infinity falling naturally out as 0 avoids a mass-is-zero
        // special case scattered through the resolution code later.
        float inverseMass = 1.0f;

        glm::vec3 linearVelocity{ 0.0f };
        glm::vec3 angularVelocity{ 0.0f };

        // Cleared to zero at the end of every PhysicsSystem::Step - these
        // are a per-step accumulator (ApplyForce/ApplyImpulse would push
        // into these), not persistent state. Nothing calls ApplyForce yet
        // as of this component's introduction (gravity is applied
        // directly as an acceleration, not routed through this) - it
        // exists now so collision resolution (impulse-based) has
        // somewhere to write without a second component being added
        // later.
        glm::vec3 accumulatedForce{ 0.0f };
        glm::vec3 accumulatedTorque{ 0.0f };

        bool useGravity = true;

        // Sleeping: once a Dynamic body's velocity has stayed below the
        // sleep thresholds for long enough (see PhysicsSystem::Step),
        // it's frozen - no gravity, no force accumulation, no
        // integration - until something wakes it (a meaningful new
        // collision, see CollisionSystem::ResolveContact). This is what
        // keeps a big pile of resting objects from burning CPU on tiny
        // residual-velocity jitter forever. Static/Kinematic bodies never
        // use this - PhysicsSystem never lets them accumulate a sleep
        // timer in the first place.
        bool isSleeping = false;
        float sleepTimer = 0.0f; // seconds spent under the sleep thresholds; reset to 0 the moment velocity exceeds them

        // Fraction of velocity removed per second, applied as
        // v *= (1 - damping * dt) each step - crude but stable, and cheap.
        // Not physically "air resistance", just a stabilizer so nothing
        // drifts or spins forever from small numerical error.
        float linearDamping = 0.01f;
        float angularDamping = 0.05f;
    };
}
