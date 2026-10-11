#pragma once

#include <glm/glm.hpp>
#include <cstdint>

namespace Eden
{
    // --- Fixed capacities -------------------------------------------------
    // Unlike the CPU ParticleSystem (std::vector, grows freely), the GPU
    // path allocates its storage buffers ONCE at fixed capacity - resizing
    // a VkBuffer means recreating it and every descriptor set that points
    // at it, which isn't worth supporting until a real scene actually
    // needs more than these ceilings. Raise them here if it does; nothing
    // else needs to change.
    //
    // 131072 (2^17) gives headroom past a 100K target with room to spare.
    // Cost is small and linear in particle count - Positions/Velocities/
    // Forces (vec4) + Densities/Pressures/Heats (float) is 60 bytes/
    // particle, so 131072 particles is ~7.9MB across all six buffers,
    // trivial next to CellBuckets below. Was 16384 - raised alongside
    // GPU_HASH_TABLE_SIZE (see that constant's comment for why they move
    // together) when 16384 turned out to be silently capping runs that
    // were trying for 32K-100K (Emit() truncates/drops past capacity
    // with a console warning - EASY to miss if nothing's watching the
    // console, and looks exactly like "particles collide with geometry
    // slowly" if you assume the higher count you requested is the count
    // you got).
    constexpr uint32_t GPU_MAX_PARTICLES = 131072;

    // Hash-table-based uniform grid (Teschner et al. 2003 hash), NOT the
    // CPU path's exact counting-sort grid (ParticleSpatialHash.h) - see
    // ParticleSystemGPU.h's class comment for why that's a deliberate,
    // documented divergence rather than an oversight. Must be a power of
    // two - HashCell() in the compute shaders masks with (size - 1)
    // instead of using a modulo.
    //
    // Kept at the same ~2:1 particles:buckets ratio the original 16384/
    // 8192 pairing used - scale this WITH GPU_MAX_PARTICLES, not
    // independently of it. Undersizing this relative to particle count
    // is a real, separate slowdown from anything collision-related: a
    // too-small table means more distinct 3D cells collide onto the same
    // bucket, which densifies buckets, which means
    // particle_density.comp/particle_force.comp's per-particle neighbor
    // scan (27 cells x up to GPU_MAX_PER_CELL entries each) does
    // increasingly wasted work rejecting far/hash-collided particles
    // that don't share a real cell. This is a genuinely different cost
    // than the collider broad-phase fix (particle_integrate.comp) - it
    // affects the density/force passes, not the collision pass.
    constexpr uint32_t GPU_HASH_TABLE_SIZE = 65536;

    // Fixed capacity per hash bucket - a cell (or, under hash collision,
    // several distinct cells sharing one bucket) can hold at most this
    // many particles before BuildGrid.comp silently starts dropping the
    // overflow (same "known limitation, not a crash" tradeoff
    // Renderer::DrawFrame uses for MAX_INSTANCES_PER_FRAME). At
    // GPU_MAX_PARTICLES=131072 and GPU_HASH_TABLE_SIZE=65536, this is
    // generous for the smoothing-radius-scale packing WCSPH expects -
    // unchanged from the original 16384/8192 pairing since it scales
    // with the particles:buckets RATIO (kept constant above), not with
    // absolute particle count. If dense settled puddles start silently
    // losing neighbors at the new higher counts, this is the first
    // thing to raise - there's currently no overflow counter/warning
    // surfaced anywhere (BuildGrid.comp's own comment notes the drop is
    // silent), so that's a real blind spot worth a diagnostic counter
    // if this ever becomes a suspect.
    constexpr uint32_t GPU_MAX_PER_CELL = 32;

    // CPU-side mirror of particle_build_grid.comp's HashCell - bit-for-
    // bit identical formula (Teschner et al. 2003 spatial hash),
    // required for ParticleSystemGPU's collider cell mask (see that
    // class's own comment) to rasterize a collider's world AABB into
    // the SAME cells the GPU will look it up from. If this ever drifts
    // from the shader's copy, the cell mask silently stops matching
    // reality - same "kept in sync by hand" risk this project already
    // accepts for the CPU/GPU SDF math duplication (see
    // particle_integrate.comp's own top comment), not a new category of
    // fragility.
    inline uint32_t HashCellCPU(int32_t cellX, int32_t cellY, int32_t cellZ, uint32_t hashTableSize)
    {
        uint32_t h = (static_cast<uint32_t>(cellX) * 73856093u)
                   ^ (static_cast<uint32_t>(cellY) * 19349663u)
                   ^ (static_cast<uint32_t>(cellZ) * 83492791u);
        return h & (hashTableSize - 1u); // hashTableSize is power-of-two
    }

    constexpr uint32_t GPU_MAX_COLLIDERS = 32;

    // Heat/hardening for the melt/reform pipeline (see MeltSystem.h and
    // ClusterSystem.h). Previously, only particles that came from
    // MeltSystem::Melt were ever reform-eligible - ordinary EmitBox/Emit
    // particles were given GPU_HEAT_NOT_MELTED and permanently excluded
    // from heat decay/conduction/reform entirely (a per-PARTICLE origin
    // distinction). That's gone: ANY particle can now harden, regardless
    // of whether it came from a melted solid or was poured as ordinary
    // liquid - Emit/EmitBox/EmitPoints all default to a normal, decaying
    // starting heat now (see ParticleSystemGPU::Emit's comment), and
    // heat decay/conduction/color diffusion in particle_force.comp/
    // particle_integrate.comp run unconditionally for every particle,
    // no "is this particle part of the system" gate anymore.
    //
    // What replaced the per-particle exemption is a per-CHEMICAL one:
    // ChemicalPropertiesGPU::hardeningThreshold < 0 means "this chemical
    // never hardens at all" (matching the Design Document's own
    // examples - Aqua/Ignis explicitly "do not harden"), independent of
    // how any given particle of that chemical was created. See
    // ClusterSystem::FindReformClusters for where this is actually
    // checked.
    //
    // GPU_HEAT_NOT_MELTED is kept only as an available explicit sentinel
    // value (e.g. for a future chemical-agnostic "permanently exempt"
    // use case) - nothing in this codebase sets it as a default anymore.
    constexpr float GPU_HEAT_NOT_MELTED = -1.0f;
    constexpr float GPU_HEAT_REFORM_THRESHOLD = 0.05f;

    // --- Chemical ID system (base build) -----------------------------------
    // Per the Design Document's Matter system: every particle carries a
    // Chemical ID indexing into a small, fixed table of material
    // properties. This base build intentionally implements a SLICE of
    // that spec, not the whole thing - see ChemicalPropertiesGPU's own
    // comment for exactly which fields are load-bearing right now versus
    // present as inert data for a later pass (reactions, conductivity/
    // energy propagation) to consume. No reaction table exists yet -
    // chemicalIds are set once at emit time and never mutated.
    //
    // Fixed, small capacity (like GPU_MAX_COLLIDERS) - this is a lookup
    // table indexed by a per-particle ID, not a per-particle array, so
    // its size has nothing to do with GPU_MAX_PARTICLES. 8 is generous
    // for a base build (the Design Document's own illustrative example
    // only names 4: Ignis/Aqua/Ferrum-melt/Viridis) with room to grow
    // before this needs raising.
    constexpr uint32_t GPU_MAX_CHEMICALS = 8;

    // std430 layout, byte-identical to the GLSL ChemicalProperties struct
    // declared in particle_density.comp/particle_force.comp/fluid_depth.vert -
    // same "kept in sync by hand" convention SimParamsGPU already
    // established, see that struct's own comments for prior art.
    //
    // Load-bearing (actually read by the SPH compute passes this build):
    // mass, restDensity, stiffness, viscosity, cohesion - see
    // particle_density.comp/particle_force.comp for exactly how. This is
    // what makes "differing properties, not all shared globally" real
    // rather than cosmetic: two different chemicals in the same
    // simulation now genuinely push on each other with their own mass/
    // pressure/viscosity/cohesion response, not a shared global
    // multiplied by a color.
    //
    // cohesion replaces what used to be a plain alignment-padding float
    // (`_pad0`) - same 48-byte layout, no size change, no re-sync risk
    // across the struct's byte-identical GLSL mirrors (particle_density.comp/
    // particle_force.comp/fluid_depth.vert all had an unused padding
    // float in this exact slot already). Originally cohesion was kept
    // GLOBAL (SimParamsGPU::cohesion) on the reasoning that the Design
    // Document's Chemical ID field list doesn't name it explicitly - but
    // "differing properties per chemical" is the actual ask this base
    // build serves, and cohesion is exactly the kind of property that
    // should differ (a thick honey-like chemical cohering strongly,
    // water barely at all), so it moved here. particle_force.comp
    // averages both particles' own cohesion per pair, same pattern
    // viscosity already established - see that shader's cohesionIJ.
    // SimParamsGPU::cohesion (the old global) still exists as an inert
    // leftover field, same fate as the other now-superseded globals
    // (particleMass/stiffness) - harmless, just unused.
    //
    // Data-only for now (present in the table, not yet consumed by
    // anything):
    // - hardeningThreshold IS consumed, but only by ClusterSystem's CPU-
    //   side reform-eligibility check (see ClusterSystem::FindReformClusters),
    //   not by any compute shader - kept here rather than duplicated
    //   elsewhere since it's still fundamentally a per-chemical constant.
    //   A NEGATIVE value means "this chemical never hardens at all" -
    //   the Design Document's own examples include chemicals that
    //   explicitly don't (Aqua, Ignis) - independent of which particle
    //   of that chemical we're looking at or how it was created. Every
    //   particle can now harden if and only if ITS chemical allows it -
    //   see GPU_HEAT_NOT_MELTED's comment for what this replaced.
    // - hardenedDensity is NOT yet consumed anywhere - ReformSystem's
    //   blobby-SDF seeding still produces one hardcoded solid density
    //   regardless of chemical. Present because the Design Document
    //   names it as a Chemical ID field; wiring it into ReformSystem is
    //   future work, not part of this base build.
    // - conductivity is explicitly inert, matching the Design Document's
    //   own "Explicitly Deferred" section (energy propagation is
    //   deferred until voxel physics/SDF collision are solid) - present
    //   so the data model matches the doc, not because anything reads it.
    struct ChemicalPropertiesGPU
    {
        float mass = 1.728f;
        float restDensity = 1000.0f;
        float viscosity = 1.0f;
        float stiffness = 3000.0f;
        float hardeningThreshold = GPU_HEAT_REFORM_THRESHOLD;
        float hardenedDensity = 1000.0f; // reserved - see struct comment above
        float conductivity = 0.0f;       // inert - see struct comment above
        float cohesion = 0.0f;           // per-pair-averaged in particle_force.comp - see struct comment above
        glm::vec4 baseColor{ 1.0f, 1.0f, 1.0f, 1.0f }; // .a unused
    };
    static_assert(sizeof(ChemicalPropertiesGPU) == 48,
                  "ChemicalPropertiesGPU must match the GLSL ChemicalProperties struct byte-for-byte");

    // How many Step() calls' worth of compute dispatches can be queued
    // before RecordPendingSteps() flushes them into a frame's command
    // buffer - see ParticleSystemGPU.h's comment on why recording is
    // deferred to draw time. Sized above main.cpp's
    // MAX_PHYSICS_STEPS_PER_FRAME (5) with headroom, not tied to it
    // directly (Engine/Particles shouldn't need to #include Source/main.cpp
    // to know that constant) - raise both together if the physics
    // accumulator's cap ever changes.
    constexpr uint32_t GPU_MAX_PENDING_STEPS = 8;

    // GPU-side mirror of ColliderComponent + its resolved world
    // transform, one entry per collider entity, uploaded fresh by
    // ParticleSystemGPU::Step() every call (see its .cpp). Every field is
    // a vec4 on purpose - std430 storage-buffer layout, keeping every
    // member 16-byte aligned with no implicit padding to get wrong.
    //
    // Rotation is baked into three basis columns rather than uploading
    // rotationDegrees and reconstructing glm::rotate in GLSL - the CPU
    // already computed this matrix once via
    // SDF::RotationMatrixFromDegrees for the entity's OTHER physics this
    // same tick (CollisionSystem, ResolveBoundaries), so redoing the
    // trig per-collider-per-particle on the GPU would just be repeating
    // work already paid for on the CPU, for a quantity (collider count)
    // that's typically single digits - this is precomputed-few-things-
    // once, not a case that benefits from GPU parallelism.
    //
    // Voxel (shape type 4, added alongside particle-vs-voxel collision -
    // see voxelParams below) does NOT use rotRight/rotUp/rotForward at
    // all - VoxelSystemGPU::SampleSignedDistance's own convention is a
    // flat `worldPos - volume.desc.origin` translation with no rotation
    // applied (voxel volumes are always Static, axis-aligned - see
    // ColliderComponent.h's Voxel shape comment), so a Voxel collider's
    // rotation columns are left at their harmless identity defaults and
    // simply never read by ShapeDistance/ShapeNormal's Voxel branch.
    struct ColliderGPU
    {
        glm::vec4 rotRight{ 1.0f, 0.0f, 0.0f, 0.0f };   // rotation matrix column 0 - unused for Voxel, see struct comment
        glm::vec4 rotUp{ 0.0f, 1.0f, 0.0f, 0.0f };      // column 1 - unused for Voxel
        glm::vec4 rotForward{ 0.0f, 0.0f, 1.0f, 0.0f }; // column 2 - unused for Voxel

        // xyz = world-space collider center (transform.position + rotation
        // * localOffset, same as SDF::WorldToLocal computes) for shapes
        // 0-3. For Voxel (shape 4), xyz instead holds the volume's
        // desc.origin (its world-space MIN corner, not a center -
        // matches VoxelSystemGPU::SampleSignedDistance's own
        // `worldPos - volume.desc.origin` convention exactly, see
        // ShapeDistance's Voxel branch) - reusing this field rather than
        // adding a separate one, since Voxel and the analytic shapes
        // never need worldCenter and origin at the same time. w = shape
        // type as a float (0=Sphere, 1=Box, 2=Capsule, 3=Plane,
        // 4=Voxel), avoiding a 5th vec4 for one int's worth of data.
        glm::vec4 worldCenterType{ 0.0f };

        // Meaning depends on shape type, exactly mirroring
        // ColliderComponent's own per-shape field reuse (Physics/
        // ColliderComponent.h): Sphere -> x=radius. Box -> xyz=halfExtents
        // (already scaled, see SDF::ScaledCollider - this struct always
        // carries POST-scale shape parameters, never raw
        // ColliderComponent values). Capsule -> x=halfHeight, y=radius.
        // Plane -> xyz=planeNormal (local space, unaffected by scale).
        // Voxel -> unused (all voxel-specific data lives in voxelParams
        // below instead, since a density-field query needs more than 4
        // scalars).
        glm::vec4 shapeParams{ 0.0f };

        // Voxel-only (shape type 4) - meaningless/zeroed for every other
        // shape. Added alongside particle-vs-voxel collision so
        // particle_integrate.comp's ShapeDistance/ShapeNormal can sample
        // VoxelSystemGPU's shared density buffer directly, the same way
        // raymarch.frag's RaymarchObjectGPU::densityOffset does (see
        // that struct's comment) - deliberately mirrors that field's
        // meaning/units exactly (float ELEMENTS into the one shared
        // DensityBuffer, not bytes) so both consumers agree on
        // convention.
        //   x = densityOffset (as a float - see ShapeDistance's Voxel
        //       branch for the uint reinterpretation; GLSL structs here
        //       are declared as float-only per this struct's std430
        //       convention, matching worldCenterType.w's shape-type
        //       encoding above)
        //   y = voxelSize (VoxelVolumeDesc::voxelSize)
        //   zw = unused, reserved (kept at 0 - NOT sampleDims; that
        //       needs 3 components and doesn't fit here, see
        //       voxelSampleDims below instead)
        glm::vec4 voxelParams{ 0.0f };

        // Voxel-only - VoxelVolumeDesc::SampleDims(), needed by
        // ShapeDistance's Voxel branch to bounds-check/trilinear-sample
        // the density field correctly (same role RaymarchObjectGPU::
        // sampleDims plays). ivec4 rather than ivec3 purely for std430
        // 16-byte alignment - w is unused padding, always 0.
        glm::ivec4 voxelSampleDims{ 0 };
    };
    static_assert(sizeof(ColliderGPU) == 112, "ColliderGPU must match the GLSL struct byte-for-byte (std430)");

    // Push-constant block shared by all four compute shaders (BuildGrid,
    // Density, Force, Integrate) - see particle_*.comp's identical
    // `SimParams` block, which MUST be kept field-for-field identical to
    // this. 96 bytes total, comfortably under Vulkan's guaranteed-minimum
    // 128-byte push constant budget.
    //
    // alignas(16) on the struct (not just gravity) mirrors
    // UniformBufferObject's own convention in RendererTypes.h - this
    // struct's own trailing padding already gets it to a 16-byte
    // multiple, but being explicit costs nothing and matches house style.
    struct alignas(16) SimParamsGPU
    {
        uint32_t particleCount = 0;
        uint32_t numColliders = 0;
        uint32_t hashTableSize = GPU_HASH_TABLE_SIZE;
        uint32_t maxPerCell = GPU_MAX_PER_CELL;

        float h = 0.3f;             // smoothingRadius
        float h2 = 0.09f;           // h*h, precomputed for the (unused today, kept for parity/future kernels) convenience of not recomputing per-invocation
        float mass = 5.832f;        // particleMass
        float restDensity = 1000.0f;

        float stiffness = 3000.0f;  // Tait EOS
        float gamma = 7.0f;
        float viscosity = 0.5f;     // viscosityCoefficient
        float dt = 0.0f;            // this SUBSTEP's timestep (fixedDeltaTime / substeps), not the full tick

        float cellSize = 0.3f;      // matches smoothingRadius, same as the CPU grid (ParticleSpatialHash)
        float particleRadius = 0.05f; // boundaryRadius
        float boundaryRestitution = 0.0f;
        float maxAcceleration = 500.0f;

        uint32_t maxSweepIterations = 8;

        // Heat/cooldown tunables (Eden Particle State Transitions -
        // reused _pad0/_pad1 rather than growing the struct, keeping
        // SimParamsGPU at its existing 96 bytes / GLSL block layout).
        float heatDecayRate = 0.15f;      // heat units/sec lost passively (was _pad0)
        float heatColdMultiplier = 1.5f;  // extra decay factor within kColdSurfaceBand of a collider (was _pad1) - see ParticleSystemGPU.h's copy of this comment for why this was lowered from 3.0

        // Fraction of tangential (along-surface) velocity removed per
        // substep while a particle is in contact with a collider (was
        // _pad2). Boundary response above only ever zeroes the NORMAL
        // velocity component (correct - that's restitution's job); with
        // nothing damping the tangential component, particles resting
        // against a rigid boundary can skid/jitter indefinitely from
        // repeated small WCSPH pressure kicks as they pack against it,
        // never actually coming to rest. 0 = off (old behavior), 1 =
        // stops dead on contact.
        float boundaryFriction = 0.35f;

        glm::vec4 gravity{ 0.0f, -9.81f, 0.0f, 0.0f };

        // Appended at the end deliberately - see this project's own
        // convention (heatDecayRate/heatColdMultiplier/boundaryFriction's
        // comment above) of not disturbing existing offsets that all
        // four .comp shaders already rely on. Only particle_integrate.comp
        // reads this; the other three still just need it present (same
        // byte layout) so this one shared push-constant range stays
        // valid for all four pipelines.
        //
        // A particle whose Y position drops below this is considered to
        // have left the simulated world entirely - see
        // particle_integrate.comp's use of it for what "left" means
        // here and why this exists (particles falling forever into an
        // open void below the world still cost full SPH + collision +
        // render work every frame with nothing checking this, which is
        // exactly the case Hawkins found: ~88K of 90K particles fully
        // off-screen, still costing real frame time). Defaults far below
        // any real scene so it's a no-op until something sets it -
        // ParticleSystemGPU has no notion of "where the world's floor
        // is," only main.cpp (or whatever owns scene layout) does.
        float voidKillY = -1.0e9f;

        // Akinci et al. 2013 cohesion coefficient - see particle_force.comp's
        // CohesionKernel/main() for the actual force. 0 = off (identical to
        // pre-cohesion behavior). This is the knob for "bouncy separate
        // droplets" vs "one cohesive body" - pressure+viscosity alone (the
        // only forces that existed before this field) have nothing pulling
        // free-surface particles toward each other, which is exactly what
        // reads as "bouncy balls" instead of liquid.
        //
        // Appended last, same "don't disturb existing offsets" convention
        // as heatDecayRate/boundaryFriction above. Struct size stays 112
        // bytes, unchanged - voidKillY (a lone float right after gravity's
        // vec4) already left 12 bytes of alignas(16) trailing padding that
        // the compiler was silently reserving; this field just claims 4 of
        // those already-spare bytes instead of growing the struct. Still
        // 8 bytes of padding left after this for one more float/similar
        // before the next one would actually push past 112.
        float cohesion = 0.0f;
    };
    static_assert(sizeof(SimParamsGPU) == 112, "SimParamsGPU must match the GLSL push_constant block byte-for-byte");
}
