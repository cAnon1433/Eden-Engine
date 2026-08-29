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

    // Heat sentinel/threshold for the melt/reform pipeline (see
    // MeltSystem.h and the upcoming ClusterSystem). Particles emitted by
    // ordinary EmitBox/Emit calls (not MeltSystem::Melt) are given
    // GPU_HEAT_NOT_MELTED and never decay or become reform-eligible -
    // only particles that came from a melted solid should ever solidify
    // back. GPU_HEAT_REFORM_THRESHOLD is how cool (post-decay) a
    // melted particle must be before it's a reform candidate.
    constexpr float GPU_HEAT_NOT_MELTED = -1.0f;
    constexpr float GPU_HEAT_REFORM_THRESHOLD = 0.05f;

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
    };
    static_assert(sizeof(SimParamsGPU) == 112, "SimParamsGPU must match the GLSL push_constant block byte-for-byte");
}
