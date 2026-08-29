#pragma once

#include "ParticleData.h"
#include "ParticleSpatialHash.h"
#include "ParticleThreadPool.h"
#include "../ECS/Registry.h"
#include "../Renderer/Vulkan/RendererTypes.h"

#include <glm/glm.hpp>
#include <vector>

namespace Eden
{
    // Weakly-compressible SPH fluid solver (Muller/Charypar/Gross 2003
    // for the base method, Tait-equation pressure per Becker & Teschner
    // 2007 - see comment on `stiffness`/`gamma` below for why Tait rather
    // than the simpler linear equation of state from the original paper).
    // CPU, single-threaded, first-pass implementation - the stated plan
    // is a future compute-shader port once this is proven correct, so
    // this is written to be a correct reference implementation first,
    // not a pre-optimized one. Don't be surprised if it's the slowest
    // code in the engine per-particle; that's expected at this stage.
    //
    // Deliberately NOT a System (see Physics/PhysicsSystem.h's identical
    // reasoning) - Step() must run at a small, fixed, substepped
    // timestep, not whatever variable dt the render frame produced.
    // Call Step() directly from main.cpp's physics accumulator loop,
    // alongside PhysicsSystem::Step/CollisionSystem::Step.
    //
    // Coupling with Physics (Engine/Physics/): ONE-WAY only right now.
    // Particles are deflected by anything with a TransformComponent +
    // ColliderComponent (static or dynamic, doesn't matter - velocity is
    // read for nothing, only current shape/position), using the exact
    // same SDF distance/gradient math CollisionSystem uses for rigid
    // bodies (Physics/SDF.h). Particles do NOT push back on rigid bodies
    // yet - that's real two-way coupling (accumulate an opposing impulse
    // into the collider's RigidBodyComponent) and is intentionally
    // deferred, not an oversight.
    class ParticleSystem
    {
    public:
        // --- Tunables -------------------------------------------------
        // All of these interact with each other - changing one without
        // the others is a common way to make a first SPH attempt go
        // unstable (particles either barely interacting, or exploding
        // apart). Defaults below are a reasonable water-like starting
        // point, not a tuned result; expect to adjust smoothingRadius/
        // stiffness/viscosityCoefficient together once real behavior is
        // visible.

        // Smoothing radius (h). Every kernel in SPHKernels.h treats this
        // as the cutoff beyond which two particles don't interact at
        // all. Also drives the spatial hash's cell size (see Step()) and
        // the default particle spacing used by EmitBox.
        float smoothingRadius = 0.3f;

        // Mass assigned to every particle - SPH mass is typically
        // uniform across a simulation rather than per-particle, so this
        // is one shared constant, not stored per-particle in
        // ParticleData.
        //
        // NOT a free parameter you can pick independently of restDensity/
        // smoothingRadius/spacing - this was a real, shipped bug: the
        // previous default (0.02) made it MATHEMATICALLY IMPOSSIBLE for
        // density to ever approach restDensity, at any packing, which
        // means pressure (clamped to >= 0, see `stiffness` below) was
        // permanently 0 regardless of how compressed the fluid got -
        // nothing was ever pushing particles apart. Verified by direct
        // calculation: at the densest possible neighbor packing under
        // the old defaults, density peaked at ~3.3 against a restDensity
        // of 1000 - not "a bit low", off by two and a half orders of
        // magnitude. This is why particles could collapse into a single
        // point instead of behaving like a fluid.
        //
        // This default (restDensity * spacing^3) is the standard SPH
        // mass-initialization formula - each particle represents roughly
        // a spacing^3 chunk of fluid volume, so mass = density * volume.
        // `spacing` here matches EmitBox's own default fill spacing
        // (smoothingRadius * 0.6) - if you change smoothingRadius,
        // restDensity, or EmitBox's spacing argument independently,
        // recompute this the same way, or density will silently stop
        // being able to reach restDensity again.
        float particleMass = 5.832f; // = 1000.0f * (0.3f * 0.6f)^3 -> restDensity * spacing^3

        // Target rest density (roughly water-like in arbitrary
        // simulation units, not real kg/m^3 - nothing else in Eden is
        // physically scaled to real-world units either, see
        // PhysicsSystem::gravity's plain -9.81).
        float restDensity = 1000.0f;

        // Tait equation of state: pressure = stiffness * ((density /
        // restDensity)^gamma - 1). Used instead of Muller 2003's simpler
        // linear EOS (pressure = k * (density - restDensity)) because
        // Tait's is stiffer near the rest density, which resists
        // compression harder and reads as more "incompressible" fluid
        // behavior - the tradeoff is it needs a smaller timestep to stay
        // stable, which is exactly why Step() substeps internally rather
        // than integrating once at the full physics tick. Negative
        // pressure (density below rest - tension) is clamped to zero
        // below rather than left negative: negative pressure in WCSPH
        // is a known source of unphysical particle clumping (the
        // "tensile instability" problem), and clamping it out is the
        // standard first-pass fix.
        float stiffness = 3000.0f;
        float gamma = 7.0f;

        // Viscosity (velocity diffusion) coefficient - higher smooths
        // velocity differences between neighbors faster, which reads as
        // a thicker fluid. Zero would mean no viscosity force at all
        // (technically an inviscid fluid, which in practice usually
        // looks noisy/chaotic rather than "thin" - some small viscosity
        // is normally kept even for water-like behavior).
        float viscosityCoefficient = 0.5f;

        glm::vec3 gravity{ 0.0f, -9.81f, 0.0f };

        // Safety cap on per-particle acceleration magnitude (world units
        // per second squared) - a defensive stability valve, not a
        // tuning knob you should normally need to touch. Found necessary
        // by direct stress-testing: a whole EmitBox lattice landing on a
        // rigid floor simultaneously is a genuinely extreme initial
        // condition for an explicit WCSPH solver - many particles in
        // sudden, correlated contact at once, which the very-stiff Tait
        // equation (gamma=7 by default) can respond to with a transient
        // force spike large enough to fling individual particles out at
        // high velocity even though the bulk density/pressure field
        // stays reasonable throughout (confirmed by direct
        // instrumentation - this isn't an unbounded feedback runaway,
        // it's a bounded-but-large single-substep impulse). Increasing
        // substeps alone does NOT fix this (also confirmed empirically -
        // the resulting ejection velocity converged to the same value
        // across substep counts from 4 to 64), which is what makes this
        // a force-magnitude problem needing a hard cap, not a resolution
        // problem substepping already solves elsewhere in this class.
        float maxAcceleration = 500.0f;

        // How many sub-steps Step() divides its incoming fixedDeltaTime
        // into. WCSPH with Tait's equation needs a substantially smaller
        // timestep than the rigid-body solver's 1/60s to stay stable
        // (see `stiffness` comment above) - this decouples the SPH
        // timestep from the fixed physics tick without changing the
        // physics accumulator in main.cpp.
        int substeps = 4;

        // Treated as a small solid radius for boundary collision only -
        // NOT used by the SPH forces themselves (SPH particles are point
        // samples with no inherent size to the density/pressure/
        // viscosity math). Without some notion of particle size here,
        // particles would only get pushed out once already exactly on a
        // collider's surface, which reads as fluid partially sinking
        // into geometry before responding.
        float boundaryRadius = 0.05f;

        // Damping applied to the velocity component along a boundary's
        // normal on contact (0 = fully absorbed/no bounce, 1 = perfectly
        // elastic). Low by default - fluid hitting a surface should
        // mostly lose momentum into it, not bounce, or the boundary
        // response reads as the fluid being made of rubber balls rather
        // than liquid.
        float boundaryRestitution = 0.0f;

        // Safety cap on conservative-advancement marching iterations per
        // particle per substep, used only for the fast-particle sweep
        // path in ResolveBoundaries (see its .cpp comment) - mirrors
        // Physics/CollisionSystem.h's maxSweepSubsteps for rigid bodies:
        // same underlying technique (SDF-based conservative advancement/
        // sphere tracing), same reasoning that the march is self-
        // limiting (big steps in open space, tiny steps near a surface)
        // so this just bounds worst case rather than being a tuning
        // knob anyone should normally need to touch.
        int maxSweepIterations = 8;

        // --- API --------------------------------------------------------

        // Adds one particle at `position` with an optional initial
        // velocity.
        void Emit(const glm::vec3& position, const glm::vec3& velocity = glm::vec3(0.0f));

        // Fills an axis-aligned box [min, max] with a regular grid of
        // particles, spaced at `spacing` (defaults to smoothingRadius *
        // 0.6 if left at 0 - dense enough that the kernels above have
        // several neighbors per particle immediately, which matters:
        // starting particles too far apart means density comes back
        // near zero for the first several steps regardless of what
        // restDensity is set to). Convenience for testing the solver
        // without hand-placing individual Emit() calls - not a
        // production-shaped emitter API (no rate-over-time spawning,
        // no nozzle/cone shape) since neither of those is needed to
        // prove the solver works yet.
        void EmitBox(const glm::vec3& min, const glm::vec3& max, float spacing = 0.0f, const glm::vec3& initialVelocity = glm::vec3(0.0f));

        void Clear();

        size_t ParticleCount() const { return m_Particles.Count(); }

        // Advances the simulation by exactly fixedDeltaTime, internally
        // divided into `substeps` equal sub-steps (see comment above).
        // `registry` is read-only from this function's point of view
        // right now (queried for TransformComponent + ColliderComponent
        // pairs to collide particles against) - nothing here writes to
        // the ECS. Call from the same fixed-timestep accumulator loop
        // in main.cpp that already drives PhysicsSystem::Step/
        // CollisionSystem::Step, in whichever order boundary geometry
        // should be considered "settled" for this tick (after
        // PhysicsSystem/CollisionSystem have moved rigid bodies is the
        // natural choice, so particles collide against where solids
        // ended up this tick, not where they started it).
        void Step(Registry& registry, float fixedDeltaTime);

        // Builds one DrawCommand per particle using `pointMesh` - expected
        // to be Renderer::GetParticlePointMesh(), the dedicated single-
        // vertex mesh that only renders correctly through Renderer's
        // point-topology pipeline (see Renderer.h's comment on
        // GetParticlePointMesh - passing any other mesh here would still
        // "work" mechanically but render as whatever triangle geometry
        // that mesh actually is, not a point). Apparent on-screen size is
        // controlled globally by Renderer::ParticlePointSize (screen-space
        // pixels, via gl_PointSize - see particle_point.vert), NOT by
        // anything in this function - unlike the old cube-mesh path,
        // there's no per-call visualScale here, because scale has no
        // effect on a mesh whose only vertex sits exactly at the origin
        // (see particle_point.vert's comment on why). Append the result
        // onto the draw list built by RenderSystem::BuildDrawList before
        // passing it to Renderer::DrawFrame - this function does NOT
        // touch the ECS draw list itself, callers combine the two.
        std::vector<DrawCommand> BuildDrawList(MeshHandle pointMesh, const glm::vec4& colorOverride = glm::vec4(0.2f, 0.5f, 1.0f, 1.0f)) const;

        unsigned int ThreadCount() const { return m_ThreadPool.ThreadCount(); }

    private:
        void ComputeDensityPressure();
        void ComputeForces();
        void Integrate(float dt);

        // Takes the collider entity list rather than querying the
        // registry itself - resolved once per Step() call via
        // Registry::View (see Step()'s definition) and reused across
        // every substep, instead of re-walking the registry's storages
        // substeps times for a set of colliders that isn't expected to
        // change mid-Step. `dt` is this substep's timestep - needed to
        // reconstruct each particle's start-of-substep position
        // (currentPosition - velocity * dt) for the fast-particle sweep
        // path (see .cpp).
        void ResolveBoundaries(Registry& registry, const std::vector<Entity>& colliderEntities, float dt);

        // Walks the grid ONCE per particle per substep and caches the
        // result as a flat CSR-style neighbor list (m_NeighborOffsets[i]
        // .. m_NeighborOffsets[i+1] into m_NeighborIndices, for particle
        // i), reused by both ComputeDensityPressure and ComputeForces.
        // Parallelized across m_ThreadPool's workers - each worker walks
        // its own contiguous particle range into a private local buffer
        // (no shared-vector contention while particles are being
        // processed), then a short single-threaded merge step stitches
        // the per-worker buffers into the final flat arrays in
        // particle-index order. This keeps the "one traversal per
        // particle" property from the earlier optimization pass instead
        // of trading it for a second parallel traversal.
        void BuildNeighborLists(const ParticleSpatialHash& grid);

        ParticleData m_Particles;
        ParticleThreadPool m_ThreadPool;

        std::vector<uint32_t> m_NeighborOffsets; // size Count()+1
        std::vector<uint32_t> m_NeighborIndices; // flat, reused across substeps (see BuildNeighborLists)
    };
}
