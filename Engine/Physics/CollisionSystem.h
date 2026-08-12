#pragma once

#include "../ECS/Registry.h"

#include <glm/glm.hpp>
#include <unordered_map>

namespace Eden
{
    // Detects and resolves collisions between every entity that has both
    // TransformComponent and ColliderComponent. Call directly after
    // PhysicsSystem::Step() inside main.cpp's fixed-timestep accumulator -
    // same reasoning as PhysicsSystem: this must run at a fixed rate, not
    // the frame's variable deltaTime, or resolution quality (and how much
    // penetration gets corrected per step) becomes framerate-dependent.
    //
    // BROAD PHASE: a uniform spatial hash grid (see BuildCandidatePairs in
    // CollisionSystem.cpp), sized adaptively from actual collider sizes
    // by default - see adaptiveBroadPhaseCellSize below.
    //
    // NARROW PHASE: fully rotation-aware. Sphere-vs-anything and
    // Capsule-vs-Plane/Capsule are exact; Box-vs-Box uses a real SAT
    // (Separating Axis Theorem) oriented-box test; Box-vs-Plane uses the
    // box's actual rotated support point. Box-vs-Capsule remains a
    // deliberate approximation (closest point on the capsule's ACTUAL
    // rotated axis to the box's CENTER, not its surface) - see
    // TestBoxCapsule in the .cpp. Plane-vs-Plane is unhandled
    // (degenerate - two infinite planes don't have a single meaningful
    // contact).
    //
    // SLEEP: wakes propagate through a whole connected contact "island"
    // in one step (see the union-find pass in Step()), not just to the
    // one body directly touching whatever moved/hit something - a tall
    // resting stack wakes together instead of one hop per step.
    class CollisionSystem
    {
    public:
        // Coefficient of restitution (0 = fully inelastic/no bounce, 1 =
        // perfectly elastic) and Coulomb friction coefficient, applied
        // uniformly to every contact for now - per-pair/per-material
        // values are a real future improvement, not done here.
        float restitution = 0.2f;
        float friction = 0.4f;

        // Positional correction (resolving interpenetration, separate
        // from velocity resolution - conflating the two is a common
        // source of jitter/sinking, see Planning Notes step 4):
        // correction = max(penetration - slop, 0) * percent, split
        // between the two bodies by inverse mass. `slop` leaves a tiny
        // sliver of allowed overlap so resting contacts don't jitter as
        // correction and re-penetration fight each other every step.
        // `percent` < 1 intentionally under-corrects each step rather
        // than fully resolving penetration in one step, which is more
        // stable for stacked/resting contacts.
        float positionalCorrectionPercent = 0.8f;
        float positionalCorrectionSlop = 0.01f;

        // Number of times the SAME set of contacts gets resolved per
        // step. Discovered as a real gap while testing sleep on a 3-box
        // stack: a single sequential pass over (floor-A, A-B, B-C)
        // resolves each contact using velocities that haven't yet been
        // corrected by the OTHER contacts in the chain, so a stack
        // settles into a self-consistent but non-zero residual velocity
        // loop instead of true rest - gravity adds a fixed amount, one
        // partial pass removes a fixed (but not matching) amount, every
        // step, forever. Re-running the same contact list multiple times
        // per step lets corrections propagate through the whole chain
        // before moving on - the standard fix for this class of problem
        // (sequential impulse solvers use exactly this "velocity
        // iterations" idea). Positional correction re-runs too, each
        // pass working off whatever penetration remains from the
        // previous one.
        int resolutionIterations = 4;

        // Caps the "implied velocity" used when a Static/Kinematic mover
        // that overlaps something at its final position gets resolved via
        // the normal friction-aware impulse math (see
        // SweepMoversAgainstDynamics's small-movement branch). That
        // implied velocity is displacement / fixedDeltaTime - technically
        // the correct average velocity for the step, but dividing a small
        // spatial nudge by the small fixed timestep (1/60s ~ 0.0167)
        // produces a deceptively large number: even a 0.05-unit drag
        // implies ~3 m/s. A real editor drag is meant to nudge something
        // gently, not simulate a supersonic impact - this caps how hard
        // that borrowed velocity can push, regardless of how it got
        // there.
        float maxMoverBorrowedSpeed = 2.0f;

        // Safety clamp on angular speed (radians/sec) applied at the end
        // of every contact resolution. Naive iterative Coulomb-friction
        // solvers (this one included - proper warm-starting/LCP solving
        // is a much bigger undertaking) can develop a genuine positive-
        // feedback loop: friction at an off-center contact point spins
        // the body, and that spin changes the contact point's own
        // velocity in a way that can reinforce rather than damp further
        // friction impulses, especially across multiple resolution
        // iterations and a contact point that's only recomputed once per
        // step. Caught directly: a box being dragged by a slowly-moving
        // platform spun up to physically absurd speed within a single
        // step. This doesn't fix the underlying solver limitation, it
        // bounds how bad it can look - tumbling from a real impact stays
        // well under this; runaway spin from the feedback case gets
        // capped instead of diverging.
        float maxAngularSpeed = 6.0f;

        // SDF-based continuous collision (conservative advancement /
        // sphere tracing - see ConservativeAdvanceDynamicBodies in
        // CollisionSystem.cpp). For any Dynamic body that moved
        // further this step than roughly its own size, this walks it
        // forward using the actual SDF distance to nearby colliders to
        // know exactly how far it's safe to move before checking again -
        // not a fixed number of samples, which is what an earlier
        // version of this did and could still skip a thin-enough
        // obstacle at high-enough speed. The moving body itself is still
        // approximated as a bounding sphere for this pass (its true Box/
        // Capsule shape isn't swept) - the one remaining approximation,
        // corrected for by the normal discrete resolution pass that runs
        // immediately after.
        bool enableContinuousCollisionSweep = true;
        int maxSweepSubsteps = 24; // safety cap on marching iterations per fast-moving body per step - the march is self-limiting (big steps in open space, tiny steps near a surface), this just bounds worst case

        // Uniform-grid broad phase cell size. Two colliders are only
        // narrow-phase tested if their AABBs share at least one grid
        // cell - too small and one collider spans many cells (more
        // insertion work); too large and cells hold many unrelated
        // entities (back toward brute force).
        //
        // When adaptiveBroadPhaseCellSize is true (the default), this is
        // NOT a manual setting - Step() recomputes and overwrites it
        // every call from the actual average collider size present that
        // step (smoothed over time so the grid doesn't reshuffle every
        // frame from noise), and this field just reflects the live
        // computed value for the UI to show. Set
        // adaptiveBroadPhaseCellSize to false to take manual control
        // instead - useful if your scene has a genuinely bimodal size
        // distribution (a few huge static platforms plus many small
        // debris pieces) where a single average stops being a good fit.
        bool adaptiveBroadPhaseCellSize = true;
        float broadPhaseCellSize = 4.0f;

        void Step(Registry& registry, float fixedDeltaTime);

    private:
        // EMA-smoothed adaptive cell size, separate from the public
        // broadPhaseCellSize field above so smoothing has continuous
        // state to work from even while broadPhaseCellSize itself gets
        // overwritten each step to mirror it for the UI.
        float m_adaptiveCellSize = 4.0f;

        // Last known position per entity, tracked purely so Step() can
        // notice when something moved WITHOUT going through velocity -
        // an inspector drag, a script teleporting an object, anything
        // that isn't PhysicsSystem integrating a velocity. This matters
        // specifically for sleeping bodies: a sleeping body's own
        // isSleeping flag doesn't clear itself just because its
        // TransformComponent changed underneath it, and a moved Static/
        // Kinematic body has no velocity at all for the normal
        // velocity-based wake check to see. See CollisionSystem.cpp for
        // where this gets used.
        std::unordered_map<Entity, glm::vec3> m_lastKnownPositions;

        // Same idea as m_lastKnownPositions, for rotation. An earlier
        // version of this only tracked position - a pure rotation
        // (nothing else about the entity moving) was completely
        // invisible to external-move detection, so rotating a
        // Static/Kinematic body did nothing: nothing woke, nothing
        // re-resolved, and anything resting on it could clip straight
        // through the newly-rotated shape.
        std::unordered_map<Entity, glm::vec3> m_lastKnownRotations;
    };
}
