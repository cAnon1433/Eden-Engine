#pragma once

#include <glm/glm.hpp>
#include <cstdint>

namespace Eden
{
    // --- Fixed capacities --------------------------------------------------
    // Voxels per axis within one chunk. 8^3 = 512 cells. Worst case is
    // 12 triangles/cell, not 5 - raised when voxel_march.comp switched
    // from the plain Lorensen table (5 triangles/case max) to the full
    // Lewiner MC33 disambiguated algorithm, whose case 13.4 subconfigs
    // can legitimately produce 12 triangles for a single cell (see
    // MarchingCubesTables33's TILING13_4 / voxel_march.comp's
    // BuildCaseTriangulation). 12 * 512 cells = 6144 triangles = 18432
    // vertices - the per-chunk vertex slot size every chunk reserves in
    // VoxelSystemGPU's shared vertex buffer, whether or not it actually
    // uses that much. This is a real, meaningful memory increase (roughly
    // 2.4x this constant's old value) - worth watching total VRAM if
    // chunk counts grow much further, but the crack fix isn't optional
    // and there's no correct way to reserve less than the true worst
    // case without risking dropped triangles (see voxel_march.comp's
    // overflow guard - it discards rather than overwrites a neighbor's
    // slots if this is ever undersized). Smaller than the particle
    // system's "one big buffer" approach because this is a per-chunk
    // worst-case reservation - 16 wastes 8x the memory for the same
    // reasoning; revisit only if chunks start feeling too coarse-grained
    // for how localized carving needs to be (see class comment on
    // chunking in VoxelSystemGPU.h).
    constexpr uint32_t kVoxelChunkSize = 8;
    constexpr uint32_t kVoxelCellsPerChunk = kVoxelChunkSize * kVoxelChunkSize * kVoxelChunkSize;
    constexpr uint32_t kVoxelMaxTrianglesPerChunk = 12 * kVoxelCellsPerChunk;
    constexpr uint32_t kVoxelMaxVerticesPerChunk = kVoxelMaxTrianglesPerChunk * 3;

    // Hard ceiling on how many volumes VoxelSystemGPU will register -
    // same "documented, raisable ceiling" pattern as GPU_MAX_PARTICLES.
    // Each volume owns its own density/vertex/indirect buffers, so this
    // isn't free - kept small until there's a real reason to raise it.
    //
    // Raised from the original 16 once the raymarch conversion pass
    // (see main.cpp's SpawnRaymarchBox call sites) started registering
    // one VoxelSystemGPU volume PER converted object instead of the
    // handful of volumes the old mesh-instanced scene needed - the
    // startup scene alone now registers ~37 (27-box stress grid + a
    // handful of individually-converted spawn sites), before
    // ReformSystem::Reform (the H-key handler) adds any more at
    // runtime. Kept equal to Raymarch/RaymarchTypes.h's
    // kRaymarchMaxObjects deliberately - the two caps aren't the same
    // mechanism (this one is VoxelSystemGPU's own per-volume GPU buffer
    // ceiling, that one is the raymarch object-array SSBO size), but a
    // raymarch object always implies a registered volume behind it
    // (SpawnRaymarchBox does both together every time), so keeping them
    // in lockstep avoids one being the silent bottleneck under the
    // other. Raise both together if either needs to grow further - see
    // that constant's own comment for the same warning.
    constexpr uint32_t kVoxelMaxVolumes = 64;

    // Total capacity (in float ELEMENTS, not bytes) of
    // VoxelSystemGPU::m_SharedDensityBuffer - see that member's own
    // comment for why this is a fixed up-front size rather than a
    // dynamically-grown buffer. Sized for roughly 12 volumes' worth of
    // dense 64^3 samples (~12.6M floats, ~50MB) - deliberately NOT
    // kVoxelMaxVolumes * a worst-case-96^3 budget (that came out to
    // ~906MB, which is a lot of VRAM to commit up front on a machine
    // also running SPH particles, voxel terrain, etc. for volumes that
    // mostly won't need it). 64^3 covers the original test sphere and
    // typical reform blobs comfortably; only unusually large or
    // high-resolution volumes would need more than their share of this.
    //
    // This ceiling gets real headroom "for free," without touching this
    // constant, whenever the brickmap/sparse-storage work in the
    // planning-notes addendum ("Storage split for everything renders
    // via SDF") actually lands - a dense volume today pays for its
    // FULL bounding-box sample count even though most real shapes
    // (a sphere, a blob) only occupy a fraction of that box; sparse
    // per-brick storage would let this same byte budget hold
    // meaningfully more simultaneous volumes, since empty regions of a
    // volume's box would stop costing anything. Until then, raise this
    // constant directly if a session needs more capacity than it
    // provides - RegisterVolume throws rather than silently corrupting
    // adjacent volumes' data if this is exceeded.
    constexpr uint64_t kSharedDensityBufferCapacityElements =
        12ull * 64ull * 64ull * 64ull;

    // Isosurface threshold - density values below this are "solid",
    // above are "empty". 0.0 with signed-distance seeding (negative =
    // inside, matching Physics/SDF.h's own convention) is the natural
    // choice; kept as a named constant rather than a magic 0.0f scattered
    // across seed/march shaders and BuildSeedParams/BuildMarchParams.
    constexpr float kVoxelIsoLevel = 0.0f;

    // Which primitive shape SeedSphere/etc conceptually represent - not
    // read by any shader anymore (see VoxelParamsGPU's comment on why
    // seeding moved to plain CPU code), kept as documentation of what
    // VoxelSystemGPU's CPU-side seeding actually implements today
    // (Sphere only - Box/Capsule are reserved for whenever seeding grows
    // beyond a sphere).
    enum class VoxelSeedShape : int32_t
    {
        Sphere = 0,
        Box = 1,
        Capsule = 2,
    };

    // Push constants for voxel_march.comp - the only remaining GPU
    // compute shader in this system (seeding and carving write
    // VoxelSystemGPU::Volume::densityCPU directly from plain C++ loops
    // now - see VoxelSystemGPU.h's class comment on CPU-authoritative
    // density ownership; this struct used to also serve voxel_seed.comp/
    // voxel_carve.comp before that change, hence why it's still named
    // generically rather than "MarchParams"). Field order/sizes matter -
    // GLSL's std430 push_constant layout must match this exactly (see
    // voxel_march.comp's `layout(push_constant)` block) - reorder both
    // sides together if this ever changes. Kept within the 128-byte
    // minimum guaranteed push-constant size (Vulkan spec floor, not just
    // a MoltenVK thing) rather than assuming more is available.
    struct VoxelParamsGPU
    {
        glm::vec3 origin{ 0.0f };       // world-space min corner of the volume
        float voxelSize = 0.1f;         // world units per voxel edge

        glm::ivec3 sampleDims{ 0 };     // density sample points per axis (voxelDims + 1)
        int32_t chunkVoxelSize = static_cast<int32_t>(kVoxelChunkSize);

        glm::ivec3 chunkDims{ 0 };      // chunks per axis
        int32_t maxVerticesPerChunk = static_cast<int32_t>(kVoxelMaxVerticesPerChunk);

        float isoLevel = kVoxelIsoLevel;
        // This volume's own starting element offset within the shared
        // density buffer - binding 0 is now the FULL shared buffer, not
        // a per-volume sub-range (see RegisterVolume's comment on why),
        // so every density read needs this added explicitly, including
        // in-range ones.
        int32_t ownDensityElementOffset = 0;
        // Ghost-sample neighbor lookup, horizontal axes only (terrain
        // tiles don't neighbor each other vertically) - each is the
        // *neighboring* volume's own ownDensityElementOffset, or -1 if
        // this volume has no neighbor in that direction (every non-
        // terrain volume, and terrain tiles on the grid's outer edge).
        // See voxel_march.comp's DensityAt for how these get used - this
        // is what lets a boundary cell's gradient read one real sample
        // past its own edge instead of clamping/duplicating its last
        // sample, which was the confirmed cause of mismatched normals
        // (and, compounded with any density disagreement right at a
        // seam, visible cracks) between adjacent terrain tiles.
        int32_t neighborDensityOffsetNegX = -1;
        int32_t neighborDensityOffsetPosX = -1;
        int32_t neighborDensityOffsetNegZ = -1;

        int32_t neighborDensityOffsetPosZ = -1;
        int32_t _pad0 = 0;
        int32_t _pad1 = 0;
    };

    static_assert(sizeof(VoxelParamsGPU) <= 128, "VoxelParamsGPU exceeds the guaranteed minimum Vulkan push constant size");
}
