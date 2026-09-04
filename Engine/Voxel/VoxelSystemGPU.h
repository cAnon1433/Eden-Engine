#pragma once

#include "VoxelTypes.h"
#include "VoxelField.h"
#include "../Renderer/Vulkan/Resources/VulkanBuffer.h"
#include "../Renderer/Vulkan/Resources/VulkanMemoryAllocator.h"
#include "../Renderer/Vulkan/Pipeline/VulkanPipelineLayout.h"
#include "../Renderer/Vulkan/Pipeline/VulkanComputePipeline.h"
#include "../Renderer/Vulkan/Pipeline/VulkanDescriptorPool.h"

#include <glm/glm.hpp>
#include <vector>
#include <array>
#include <utility>

namespace Eden
{
    // Voxel/marching-cubes deformable geometry - the "SDF-as-visual-
    // geometry" fork the physics planning notes left open. A registered
    // volume's actual visible surface is extracted from a density field
    // by marching cubes (MarchingCubesTables33.h), not authored/loaded
    // triangle data - carving a region of the field and re-marching is
    // how corrosion works, without ever touching a CPU-side mesh asset.
    //
    // --- Ownership: CPU is authoritative, GPU is a one-way consumer ----
    // The density field lives on the CPU (Volume::densityCPU) as the
    // single source of truth. Seeding and carving are plain CPU loops
    // that write directly into it - there's no GPU compute dispatch for
    // either anymore (there used to be; see git history / prior session
    // notes if curious why that changed). After every CPU-side edit, the
    // whole field is pushed to the GPU's density buffer via UploadRange
    // (CPU->GPU only, one-directional) so voxel_march.comp - the ONLY
    // remaining GPU compute shader in this system - has something
    // current to triangulate. Nothing ever reads density back from the
    // GPU. This is the same shape Mike Turitzin's SDF engine uses (CPU-
    // resident "brick" data as ground truth, GPU used only for the
    // parallelizable, one-way work): a gameplay query like "what's
    // actually at this point right now" (RaycastSurfaceSegments) reads
    // volume.densityCPU directly - instant, always current, no
    // GPU->CPU stall. The earlier version of this class had carving
    // write density on the GPU and raycasting need a blocking readback
    // to see it - that round trip is exactly what this ownership model
    // avoids.
    //
    // Turitzin's engine additionally streams a sparse dictionary of
    // fixed-size "bricks" so a huge world's SDF data doesn't all have to
    // live in memory at once, and only touched bricks get re-uploaded.
    // This class's chunks (VoxelVolumeDesc::chunkDims) are that same
    // idea in spirit - independently re-triangulated units - but
    // densityCPU is still ONE dense array for the whole volume (not a
    // sparse per-chunk dictionary), just with GPU uploads scoped to only
    // the touched sample region (see UploadDensityRegion). Genuinely
    // revisit the dense-single-array part before this is asked to hold
    // a large or streamed world - a huge volume still costs one big
    // upfront allocation and CPU-side march dispatch cost even if only
    // a corner of it is ever touched.
    //
    // --- Why one shared density field per volume, not one per chunk ----
    // Chunks exist purely so a carve event only re-triangulates the
    // chunks it actually touched, not the whole volume. But the density
    // SAMPLES themselves are one contiguous array sized VoxelVolumeDesc::
    // SampleDims() - adjacent chunks read overlapping boundary samples
    // from that same array (chunk N's last sample row IS chunk N+1's
    // first sample row), which is what makes marched surfaces line up
    // seamlessly at chunk borders with no extra ghost-cell bookkeeping.
    //
    // --- Why marching cubes output reuses Renderer's existing pipeline -
    // voxel_march.comp writes RendererTypes::Vertex-formatted data
    // directly into a storage buffer that's ALSO usable as a vertex
    // buffer (STORAGE_BUFFER_BIT | VERTEX_BUFFER_BIT on the same
    // VkBuffer) - so the existing m_GraphicsPipeline (triangle.vert/
    // .frag, Blinn-Phong, textures) draws it with zero new pipeline/
    // shader work. Drawn via vkCmdDrawIndirect against a per-chunk
    // VkDrawIndirectCommand this class also maintains as a dual-usage
    // (STORAGE_BUFFER_BIT | INDIRECT_BUFFER_BIT) buffer - the compute
    // shader's per-chunk atomic triangle counter writes directly into
    // that command's vertexCount field. See Renderer::DrawFrame's
    // voxelSources parameter for the draw side.
    class VoxelSystemGPU
    {
    public:
        void Init(VkDevice device, VkPhysicalDevice physicalDevice, VmaAllocator allocator,
                  VkCommandPool commandPool, VkQueue graphicsQueue);
        void Shutdown();

        // Allocates a new volume's GPU resources (vertex/indirect
        // buffers sized per `desc`, plus a density buffer that's now
        // purely an upload target - see the class comment) and the
        // CPU-side densityCPU array + chunk bookkeeping. Uploads initial
        // per-chunk indirect-draw commands (vertexCount=0 until the
        // first March call). Does NOT seed or march - call SeedSphere
        // and then MarchDirtyChunks separately, same "construct empty,
        // fill explicitly" shape as ParticleSystemGPU's Init-then-Emit.
        VoxelVolumeHandle RegisterVolume(const VoxelVolumeDesc& desc);

        // Writes a sphere SDF (Physics/SDF.h's own convention - negative
        // = inside) into densityCPU with a plain CPU loop, uploads the
        // whole field to the GPU density buffer, and marks every chunk
        // dirty. No GPU dispatch involved - see the class comment on
        // why seeding doesn't need one. Registration-time/infrequent,
        // not a per-frame call.
        void SeedSphere(VoxelVolumeHandle handle, const glm::vec3& localCenter, float radius);

        // Same shape as SeedSphere, box SDF instead of sphere - reuses
        // Physics/SDF.h's Box() formula directly rather than
        // reimplementing it, so a voxel-box's surface matches the
        // analytic Box collider's surface exactly at registration time.
        // Added for the mesh-to-SDF conversion pass (Planning Notes'
        // "everything renders via SDF" work) - every box-shaped
        // mesh-entity (floor, walls, static/physics-test cubes) needs
        // this; SeedSphere alone only covered the original test sphere.
        void SeedBox(VoxelVolumeHandle handle, const glm::vec3& localCenter, const glm::vec3& halfExtents);

        // Blobby-SDF seeding mode for reform (Eden Particle State
        // Transitions planning notes' "Architecture decision" - reform
        // target is the melted blob's actual current shape, not the
        // original mesh). Each entry in localPositions is treated as a
        // small sphere of `particleRadius`; per-sample density is the
        // MIN over every particle of (distance to that particle -
        // particleRadius) - a hard union, not a smooth-min blend (the
        // simpler of the two options the notes called out). Same "full
        // CPU loop + full upload + mark everything dirty" shape as
        // SeedSphere, does NOT march - caller marches after, same
        // contract as SeedSphere.
        //
        // O(sample count * particles found in each sample's local 3x3x3
        // cell neighborhood) - a spatial hash over localPositions, not
        // the brute-force "check every particle for every sample" this
        // started as (that was fine at melt-test-cluster scale, but
        // ReformSystem merging every cluster in a call into one combined
        // volume needed this to actually scale - see ReformSystem.cpp).
        //
        // smoothRadius controls the blend between overlapping particle
        // spheres (see SmoothMin in the .cpp) - 0.0 gives the old hard-
        // union look (individual spheres stay visible even where they
        // overlap); a value around particleRadius rounds the seams into
        // one continuous blob instead. Defaults to 0.0 (opt-in) so
        // existing callers aren't silently changed.
        void SeedFromParticles(VoxelVolumeHandle handle, const std::vector<glm::vec3>& localPositions, float particleRadius,
                                float smoothRadius = 0.0f);

        // Terrain seeding (Terrain Gen Phase 1 - see planning notes
        // addendum's "scenes" section, this is the non-streaming proof-
        // of-pipeline step before chunk loading/LOD gets built on top).
        // Writes a pseudo-SDF heightfield: density = worldY - terrainHeight(worldX,worldZ),
        // where terrainHeight is baseHeight plus a multi-octave (fBm)
        // glm::perlin sum. NOT an exact signed distance off-flat (a true
        // SDF to a sloped/noisy surface needs distance-to-nearest-point,
        // not a vertical offset) - the vertical-offset shortcut is only
        // accurate for surfaces close to horizontal, same "good enough,
        // not exact" tradeoff SeedFromParticles' hard-union documents
        // for its own approximation. Fine for the current "flat-ish"
        // terrain target; revisit with a real distance estimate (e.g.
        // gradient-corrected offset) if amplitude/frequency ever produce
        // steep enough slopes for the approximation to visibly distort
        // collision or the marched surface.
        //
        // frequency is in world-units^-1 (matches glm::perlin's own
        // input scale - smaller = broader features). octaves > 1 layers
        // progressively higher-frequency, lower-amplitude noise on top
        // (each octave halves amplitude, doubles frequency - standard
        // fBm). seed offsets the sample domain so different seeds don't
        // just look like a panned copy of the same noise.
        //
        // Same "full CPU loop + full upload + mark everything dirty,
        // caller marches after" contract as SeedSphere/SeedBox.
        void SeedHeightfieldNoise(VoxelVolumeHandle handle, float baseHeight, float amplitude,
                                   float frequency, int octaves, uint32_t seed);

        // Re-triangulates every chunk currently flagged dirty (every
        // chunk, right after a Seed* call; a small touched subset after
        // a Carve() call) via voxel_march.comp - the one remaining GPU
        // compute step in this system - then clears their dirty flags.
        // Blocks on vkQueueWaitIdle (see .cpp); infrequent/event-driven,
        // not meant to run every frame.
        void MarchDirtyChunks(VoxelVolumeHandle handle);

        // CSG-subtracts a sphere from densityCPU (density = max(density,
        // -carveSDF), the standard SDF boolean-subtraction identity)
        // with a plain CPU loop over just the affected local region,
        // uploads ONLY that touched sample region to the GPU (see
        // UploadDensityRegion - not the whole field), and marks every
        // chunk whose region could contain the affected samples dirty.
        // Does NOT re-march on its own - call MarchDirtyChunks after
        // (same two-step shape as Seed+March, so a caller doing several
        // small carves in a row can batch them into one march instead of
        // paying the blocking-submit cost per carve).
        //
        // Also updates the cheap per-chunk collision flag this class
        // tracks (see IsWorldPointSolid) - a touched chunk is marked
        // not-solid permanently, regardless of how much of it was
        // actually removed. That's the deliberately approximate "cheap
        // fix" agreed on over full voxel-field collision.
        void Carve(VoxelVolumeHandle handle, const glm::vec3& worldPos, float radius);

        // The real fix for stale-margin false collisions after carving
        // (see this function's own .cpp comment for the full mechanism)
        // - Carve() only updates samples in its own small local region,
        // so every OTHER sample keeps whatever value it had before,
        // usually the original SeedSphere value. That's not "close
        // enough" - it's a systematic UNDERESTIMATE of the true current
        // distance (removing material can only push the nearest solid
        // further away, never closer), which is exactly what produces a
        // small enough leftover value to falsely collide with an
        // ordinary-sized collider, nowhere near anything actually solid.
        // This recomputes EVERY non-solid sample's distance to the
        // nearest CURRENTLY-solid sample from scratch (multi-source
        // Dijkstra, not an approximation tied to carve radius), so
        // correctness stops depending on carve strikes happening to
        // cover every sample. Solid samples keep their existing,
        // already-exact CSG values - untouched by this. CPU-only, no
        // GPU re-upload needed (only touches values that stay on the
        // same side of the iso level, so the marched mesh is unaffected)
        // and not free (see .cpp) - meant to run once per completed
        // carve gesture, not per carve ball or per frame.
        void RecomputeExactDistances(VoxelVolumeHandle handle);

        // DEBUG ONLY - scans every cell in the volume and prints any
        // whose 8 corners form a "face-ambiguous" configuration (a face
        // where the two corners on one diagonal are both solid, the two
        // on the other diagonal are both empty - the textbook marching-
        // cubes ambiguous case that can make two cells sharing that face
        // disagree on how to triangulate it, leaving a crack). Detected
        // structurally (checking each of the 6 faces' corner sign
        // pattern directly) rather than via a hardcoded cube-index list,
        // so it doesn't depend on this table's specific case numbering
        // matching some reference list exactly. Pure CPU, reads
        // densityCPU only - added to test whether ambiguous cells are
        // actually present at a carve site before committing to
        // rewriting the march table over it.
        void DebugScanAmbiguousCells(VoxelVolumeHandle handle) const;

        // The cheap per-chunk collision flag itself - true until a
        // Carve() call touches that point's chunk, false (permanently)
        // after. Out-of-volume-bounds points return false (safe default:
        // nothing outside the volume should read as solid because of
        // it).
        //
        // NOT currently consulted by Engine/Physics/PhysicsSystem.cpp -
        // this volume isn't a physics entity at all yet (no
        // RigidBodyComponent/ColliderComponent, not even an ECS Entity -
        // see main.cpp, it's a raw VoxelSystemGPU handle). This query
        // exists so that wiring is a matter of PhysicsSystem's narrow
        // phase calling it later, not redesigning how carving tracks
        // solidity - real, separate work, not done here.
        bool IsWorldPointSolid(VoxelVolumeHandle handle, const glm::vec3& worldPos) const;

        // Exact trilinear-sampled solidity (density < kVoxelIsoLevel) at
        // an arbitrary world point, reading densityCPU directly - unlike
        // IsWorldPointSolid (a cheap per-chunk flag meant for physics-
        // approximation use, permanently false after ANY carve touches
        // that chunk), this reflects the actual current field. What a
        // solid->particle conversion needs: "is this exact candidate
        // spawn point still inside the shape", not "has this chunk been
        // carved at all".
        bool IsWorldPointSolidExact(VoxelVolumeHandle handle, const glm::vec3& worldPos) const;

        // Trilinear-sampled signed distance at an arbitrary world point -
        // a thin public wrapper around the same SampleDensityTrilinear
        // used internally by IsWorldPointSolidExact/RaycastSurfaceSegments.
        // This is a genuine world-unit SDF value, not just a boolean:
        // SeedSphere writes an exact sphere SDF, Carve writes a true CSG
        // subtraction, and SeedFromParticles' union-of-spheres/SmoothMin
        // are themselves real (approximate) SDF blends - so densityCPU
        // already IS a signed distance field, this just exposes reading
        // it at points other than SampleIndex's grid corners. What
        // CollisionSystem's TestSphereVsVoxel (Physics/CollisionSystem.cpp)
        // needs for real penetration depth, not just in/out.
        float SampleSignedDistance(VoxelVolumeHandle handle, const glm::vec3& worldPos) const;

        // Gradient of the same field SampleSignedDistance reads, via the
        // tetrahedron technique (4 taps, not a naive 6-tap central
        // difference - see the .cpp for why). Points away from solid
        // material (world space, no rotation involved - voxel volumes
        // never rotate, see ColliderComponent::voxelVolume).
        glm::vec3 SampleGradient(VoxelVolumeHandle handle, const glm::vec3& worldPos) const;

        // Minimal, deliberately narrow world-space description of a
        // registered volume - just enough for an external caller (e.g.
        // a melt/solid-to-particle conversion) to walk its bounds,
        // without exposing the private Volume struct itself.
        struct VolumeBounds
        {
            glm::vec3 worldMin{ 0.0f };
            glm::vec3 worldMax{ 0.0f };
            float voxelSize = 0.1f;
        };
        VolumeBounds GetVolumeBounds(VoxelVolumeHandle handle) const;

        // Ghost-sample neighbor wiring for terrain tiles - see
        // Volume::neighborNegX etc.'s comment and voxel_march.comp's
        // DensityAt. Pass InvalidVoxelVolumeHandle for any direction
        // with no neighbor (a grid edge tile, or any non-terrain
        // volume - which simply never calls this at all). One-time
        // setup call, meant to run once after every terrain tile is
        // registered, not per-frame; takes effect on that volume's next
        // MarchDirtyChunks call.
        void SetVolumeNeighbors(VoxelVolumeHandle handle, VoxelVolumeHandle negX, VoxelVolumeHandle posX,
                                 VoxelVolumeHandle negZ, VoxelVolumeHandle posZ);

        // Frees a volume's per-volume GPU resources (vertex/indirect/
        // dirtyFlags/instance buffers), reclaims its region of the
        // shared density buffer via a real free-list allocator (see
        // m_FreeDensityRegions), returns its descriptor set to a reuse
        // pool (see m_FreeComputeSets - VulkanDescriptorPool has no
        // per-set free, so this is a soft-recycle: the set itself stays
        // allocated, RegisterVolume just rewrites its bindings for the
        // next volume that claims it instead of allocating a new one),
        // and recycles its slot in m_Volumes (bumps that slot's
        // generation, pushes the index to m_FreeSlotIndices - see
        // VoxelVolumeHandle's own comment on why the generation makes
        // this safe). Net effect: kVoxelMaxVolumes (64) is now a
        // concurrent-volume cap (the real VRAM-budget constraint this
        // project already documents in VoxelTypes.h), NOT a lifetime
        // cap on how many volumes can ever be registered in one run -
        // register/unregister/register cycles reuse everything.
        //
        // Safe to call twice on the same handle, or on a handle whose
        // slot has since been reused by an unrelated volume - both are
        // silent no-ops (see .cpp), same guarantee as
        // Registry::DestroyEntity.
        void UnregisterVolume(VoxelVolumeHandle handle);

        // Cheap, non-throwing companion to UnregisterVolume - true if
        // handle's slot is in range AND its generation still matches
        // (i.e. hasn't been freed-and-possibly-reused since this handle
        // was obtained). Every other public call on an invalid handle
        // throws (via GetVolume, see its comment) because a stale
        // handle reaching Carve/March/etc. is a real bug to surface
        // loudly - but a caller holding a handle that was legitimately
        // destroyed by SOMEONE ELSE since it last checked (e.g.
        // main.cpp's reformedVolumes list, populated over many frames,
        // versus an editor "Destroy" click on one of those entities)
        // needs a way to prune stale entries before touching them, not
        // a thrown exception every frame.
        bool IsValid(VoxelVolumeHandle handle) const;

        // Empties a volume's entire density field (every sample set to
        // definitively "outside") and re-marches every chunk, leaving it
        // with zero triangles while keeping its GPU resources and dirty/
        // chunkSolid bookkeeping intact for later reuse (e.g. a future
        // reform seed). Same "full CPU loop + full upload + mark all
        // dirty" shape as SeedSphere - not a giant-radius Carve() call,
        // which would leave a box volume's corners (outside the carve
        // sphere) with stale solid density.
        void ClearVolume(VoxelVolumeHandle handle);

        // Walks a world-space ray against the volume's LIVE densityCPU
        // (trilinear-sampled, so it sees the actual current carved
        // shape, not the original seed shape) and returns every solid
        // segment - (entryPoint, exitPoint) world-space pairs - the ray
        // crosses within maxDistance, in the order encountered. A ray
        // that punches through an outer shell into a previously-carved
        // hollow and hits a far wall returns TWO segments, not one -
        // this is what makes drilling all the way through, or through
        // several walls left by earlier carves, actually work, unlike
        // the old analytic-sphere-vs-ray approach it replaces (which
        // only ever knew about the original, uncarved shape and could
        // only ever find one, static, never-moving crossing point).
        //
        // Pure CPU, reads only densityCPU - no GPU interaction, no
        // stall, always reflects whatever the most recent Seed/Carve
        // call wrote, per the class comment on CPU-authoritative
        // ownership. Returns false (outSegments left empty) if the ray
        // never enters solid material within maxDistance.
        bool RaycastSurfaceSegments(VoxelVolumeHandle handle, const glm::vec3& rayOrigin, const glm::vec3& rayDir,
                                     float maxDistance, std::vector<std::pair<glm::vec3, glm::vec3>>& outSegments) const;

        // --- Renderer-facing accessors -----------------------------------
        // All three buffers are fixed-size for a volume's lifetime
        // (allocated once in RegisterVolume, never reallocated) - safe
        // to read these once when registering a draw source with
        // Renderer, same stability guarantee ParticleSystemGPU::
        // GetPositionBuffer documents.
        VkBuffer GetVertexBuffer(VoxelVolumeHandle handle) const;
        VkBuffer GetIndirectBuffer(VoxelVolumeHandle handle) const;
        VkBuffer GetInstanceBuffer(VoxelVolumeHandle handle) const;
        uint32_t GetChunkCount(VoxelVolumeHandle handle) const;

        // Raw density storage buffer - now the ONE buffer shared across
        // every registered volume (see m_SharedDensityBuffer's comment
        // on why this changed from one-buffer-per-volume). Still the
        // same GPU resource voxel_march.comp reads via UploadDensity/
        // UploadDensityRegion's writes, still exposed here so
        // RaymarchSystem can bind it directly for per-pixel
        // sphere-tracing instead of consuming marching-cubes output.
        // Same "fixed for the volume's lifetime" stability guarantee as
        // GetVertexBuffer et al - callers that cached this VkBuffer
        // handle before this change do NOT need to re-fetch it now
        // (it's the same handle for every volume), but per-volume
        // sampling now also needs GetDensityBufferOffset (below) to
        // find the right region within it.
        //
        // handle is accepted but genuinely unused (the return value is
        // the SAME m_SharedDensityBuffer regardless of which volume, or
        // even whether one exists yet - m_SharedDensityBuffer is
        // created unconditionally in Init(), before any RegisterVolume
        // call). Defaults to InvalidVoxelVolumeHandle so callers that
        // only want "the shared buffer" (e.g. ParticleSystemGPU::Init
        // wiring up particle-vs-voxel collision before any volume
        // necessarily exists yet) can call GetDensityBuffer() with no
        // argument rather than needing to construct/hold a real handle
        // just to satisfy this parameter.
        VkBuffer GetDensityBuffer(VoxelVolumeHandle handle = InvalidVoxelVolumeHandle) const;

        // This volume's starting offset (in float ELEMENTS, NOT bytes -
        // multiply by sizeof(float) if a byte offset is needed, e.g. for
        // a VkDescriptorBufferInfo::offset) within GetDensityBuffer()'s
        // shared buffer. RaymarchObjectGPU::densityOffset is populated
        // directly from this (see RaymarchSystem::BuildObjectList) so
        // raymarch.frag's single DensityBuffer binding can find each
        // object's own samples within the one shared array via
        // density[densityOffset + SampleIndexClamped(...)].
        VkDeviceSize GetDensityBufferOffset(VoxelVolumeHandle handle) const;

        // Local-space (desc.origin already subtracted) sample-grid
        // dimensions and voxel size - everything RaymarchObjectGPU needs
        // to reconstruct the same trilinear sampling raymarch.frag does,
        // without exposing VoxelVolumeDesc's full private layout.
        glm::ivec3 GetSampleDims(VoxelVolumeHandle handle) const;
        float GetVoxelSize(VoxelVolumeHandle handle) const;

        // Updates the 1-element instance buffer (model matrix + color
        // override) Renderer's indirect draw binds at binding 1 - call
        // when a volume's TransformComponent changes, NOT every frame
        // unconditionally (see VoxelVolumeComponent's comment on why
        // volume placement isn't re-read from ECS every frame yet).
        void SetTransform(VoxelVolumeHandle handle, const glm::mat4& model, const glm::vec4& colorOverride = glm::vec4(0.0f));

    private:
        struct Volume
        {
            VoxelVolumeDesc desc;

            // The authoritative SDF data - see the class comment on
            // CPU ownership. Indexed identically to how voxel_march.comp
            // indexes the GPU copy this gets uploaded into (x + y*
            // sampleDims.x + z*sampleDims.x*sampleDims.y) - SampleIndex()
            // in the .cpp is the one place that indexing is defined,
            // used by both SeedSphere/Carve (writing) and
            // RaycastSurfaceSegments (reading).
            std::vector<float> densityCPU;

            // No longer owns its own density buffer - see
            // VoxelSystemGPU::m_SharedDensityBuffer's comment for why
            // this moved to one buffer shared across every volume.
            // densityOffsetElements is this volume's starting index
            // (in float ELEMENTS, not bytes) within that shared buffer.
            // Still allocated aligned to m_StorageBufferOffsetAlignment
            // (queried once at Init time) even though binding 0 no
            // longer uses this as a VkDescriptorBufferInfo::offset (see
            // RegisterVolume's comment on why binding 0 is now the full
            // buffer) - harmless to keep, and this value is also what
            // gets handed to other volumes as a ghost-sample neighbor
            // offset (see SetVolumeNeighbors/MarchDirtyChunks), so
            // nothing here actually needed to change.
            VkDeviceSize densityOffsetElements = 0;
            VulkanBuffer vertices;   // Vertex, desc.NumChunks() * kVoxelMaxVerticesPerChunk entries
            VulkanBuffer indirect;   // VkDrawIndirectCommand, desc.NumChunks() entries
            VulkanBuffer dirtyFlags; // uint32, desc.NumChunks() entries (1 = needs re-march)
            VulkanBuffer instance;   // InstanceData, 1 entry

            // Captured from VulkanBuffer::Init's out-param when
            // `instance` above is created (VMA_ALLOCATION_CREATE_MAPPED_BIT
            // - see RegisterVolume) - VulkanBuffer itself doesn't expose
            // the mapped pointer after Init returns, so this is the only
            // place it survives past that call. SetTransform writes
            // through this directly (plain memcpy, no GPU command buffer
            // involved) rather than through UploadRange, which is the
            // whole reason `instance` was created host-visible+mapped in
            // the first place instead of device-local. This field not
            // existing was a real, shipped bug: SetTransform previously
            // fell back to UploadRange (a staged vkCmdCopyBuffer) since
            // it had no other way to reach the mapped memory, but
            // `instance` was never given VK_BUFFER_USAGE_TRANSFER_DST_BIT
            // (no reason to - it wasn't supposed to need copying into at
            // all), so that call was invalid Vulkan every single time it
            // ran - "dstBuffer was created with VERTEX_BUFFER_BIT but
            // requires TRANSFER_DST_BIT", corrupting GPU state on the
            // very first SetTransform call of a session, not something
            // introduced by volume register/unregister reuse.
            void* instanceMapped = nullptr;

            VkDescriptorSet computeSet = VK_NULL_HANDLE;

            // CPU-side source of truth for "which chunks need
            // re-marching" - the GPU dirtyFlags buffer above is written
            // FROM this list (set to 1) right before MarchDirtyChunks
            // dispatches and cleared (back to 0) right after, rather
            // than SeedSphere/Carve writing dirtyFlags themselves. That
            // keeps "what's dirty" in exactly one place regardless of
            // whether the whole volume became dirty (Seed) or a handful
            // of chunks did (Carve).
            std::vector<uint32_t> dirtyChunkIndices;

            // The cheap per-chunk collision flag - see Carve()'s comment
            // in the header for what "cheap" means here.
            std::vector<bool> chunkSolid;

            // Ghost-sample neighbors for gradient continuity across
            // terrain tile boundaries - see SetVolumeNeighbors and
            // voxel_march.comp's DensityAt. InvalidVoxelVolumeHandle
            // (the default) means "no neighbor in that direction," which
            // is every volume except terrain tiles that aren't on the
            // grid's outer edge - a small test shape or a reform blob
            // simply never has these set, so it keeps the exact old
            // clamp-at-edge behavior with zero extra cost.
            VoxelVolumeHandle neighborNegX = InvalidVoxelVolumeHandle;
            VoxelVolumeHandle neighborPosX = InvalidVoxelVolumeHandle;
            VoxelVolumeHandle neighborNegZ = InvalidVoxelVolumeHandle;
            VoxelVolumeHandle neighborPosZ = InvalidVoxelVolumeHandle;
        };

        void CreateComputeLayout();
        void CreateComputePipelines();

        // Cell coordinate -> owning chunk's flat index, using the SAME
        // formula voxel_march.comp uses internally (x + y*chunkDims.x +
        // z*chunkDims.x*chunkDims.y) - kept as one function so Carve(),
        // IsWorldPointSolid, and RaycastSurfaceSegments can't drift from
        // what the shader assumes.
        static int32_t ChunkIndexForCell(const VoxelVolumeDesc& desc, glm::ivec3 cell);

        // Flat sample index for a (x,y,z) grid coordinate - see
        // Volume::densityCPU's comment; this ordering must match
        // voxel_march.comp's SampleIndex() exactly.
        static uint32_t SampleIndex(const VoxelVolumeDesc& desc, glm::ivec3 sample);

        // Trilinear-sampled density at an arbitrary LOCAL-space point
        // (not just at grid points) - what makes RaycastSurfaceSegments
        // able to find a sub-voxel-accurate crossing instead of only
        // ever landing exactly on a grid sample.
        static float SampleDensityTrilinear(const Volume& volume, glm::vec3 localPos);

        // Same "existing buffer, partial range, blocking staged upload"
        // helper as ParticleSystemGPU::UploadRange - identical shape,
        // duplicated rather than shared because pulling it into a common
        // Vulkan-utility header is a bigger refactor than this milestone
        // calls for.
        void UploadRange(VkBuffer dst, VkDeviceSize dstOffset, const void* data, VkDeviceSize size);

        // Pushes volume.densityCPU to the GPU density buffer in full -
        // used by SeedSphere, where every sample genuinely changes.
        void UploadDensity(Volume& volume);

        // Pushes only the samples within [sampleMin, sampleMax] - what
        // Carve() actually uses, since a carve only ever touches a small
        // local region. Batches every touched row into ONE staging
        // buffer and ONE vkCmdCopyBuffer call with multiple regions
        // (not one UploadRange call per row - that would mean one
        // blocking submit per row, which for a several-dozen-row region
        // would cost far MORE than the whole-buffer upload this
        // replaces, not less).
        void UploadDensityRegion(Volume& volume, glm::ivec3 sampleMin, glm::ivec3 sampleMax);

        Volume& GetVolume(VoxelVolumeHandle handle);
        const Volume& GetVolume(VoxelVolumeHandle handle) const;

        // Real free-list allocator for m_SharedDensityBuffer, replacing
        // the old bump-only scheme (see that member's comment for why
        // bump-only used to be fine and why it stopped being enough).
        // First-fit over m_FreeDensityRegions; falls back to bumping
        // m_SharedDensityBufferUsedElements (the original path) when
        // nothing free is big enough - so a freshly-Init'd system with
        // an empty free list behaves identically to before. Splits a
        // free region that's larger than needed, keeping the leftover
        // (aligned) as a smaller free region rather than discarding it.
        VkDeviceSize AllocateDensityRegion(VkDeviceSize numSamples);

        // Returns a freed volume's region to m_FreeDensityRegions and
        // coalesces it with any now-adjacent free region (offset
        // arithmetic only - two free regions are merged when one's end
        // exactly equals the other's start), so repeated register/
        // unregister cycles don't fragment the buffer into permanently
        // small, unusable slivers.
        void ReleaseDensityRegion(VkDeviceSize offsetElements, VkDeviceSize numSamples);

        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;
        VkCommandPool m_CommandPool = VK_NULL_HANDLE;
        VkQueue m_GraphicsQueue = VK_NULL_HANDLE;

        // --- Shared density buffer -----------------------------------
        // ONE storage buffer holding every registered volume's density
        // samples back-to-back, replacing the earlier design (one
        // VulkanBuffer per volume, bound as a separate descriptor per
        // object in RaymarchSystem). That earlier design put N volumes'
        // worth of raymarch density buffers into N separate descriptor
        // slots, which hit MoltenVK's maxPerStageDescriptorStorageBuffers
        // ceiling (31 on Apple Silicon) at a low object count, and a
        // subsequent attempt at true bindless/descriptor-indexing to
        // work around that ceiling turned out to be unsupported outright
        // by MoltenVK on this project's actual dev hardware
        // (vkGetDescriptorSetLayoutSupport returned supported=VK_FALSE
        // for the required flag combination, independent of count) - see
        // git history/RaymarchTypes.h's kRaymarchMaxObjects comment for
        // that dead end. One shared buffer sidesteps the problem
        // entirely: however many volumes exist, RaymarchSystem only ever
        // needs ONE density-buffer descriptor (see raymarch.frag's
        // DensityBuffer binding), with each volume/object instead
        // carrying its own OFFSET into that one buffer
        // (Volume::densityOffsetElements, RaymarchObjectGPU::
        // densityOffset) - a technique that works identically on every
        // platform (no descriptor-indexing feature dependency at all),
        // not just a MoltenVK-specific fix.
        //
        // voxel_march.comp (the OTHER consumer of density data, separate
        // from raymarch) needs ZERO changes for this - its per-volume
        // compute descriptor set already binds density as
        // VkDescriptorBufferInfo{buffer, offset, range} (see
        // RegisterVolume), and Vulkan already remaps `density[0]` inside
        // the shader to `buffer[offset]` transparently via that offset;
        // only RaymarchSystem's single fullscreen-pass binding (which
        // must see every volume through ONE descriptor, unlike
        // voxel_march.comp's one-dispatch-per-volume shape) needed the
        // explicit offset field added to its per-object data instead.
        //
        // Bump-allocated for the FAST path (append past every region
        // ever handed out), with a real free-list (m_FreeDensityRegions,
        // AllocateDensityRegion/ReleaseDensityRegion) checked FIRST so a
        // freed volume's region gets reused by a later RegisterVolume
        // instead of always eating fresh capacity - this is what makes
        // UnregisterVolume an actual fix for the "64 volumes forever"
        // problem rather than only reclaiming CPU/dedicated-buffer
        // memory. m_SharedDensityBufferUsedElements is now a highwater
        // mark (capacity ever bumped into), not "capacity in use" -
        // don't read it as a live usage figure.
        //
        // Sized generously up front (kSharedDensityBufferCapacityElements)
        // rather than dynamically grown - growing this buffer means every
        // existing volume's density data must be copied into a new,
        // larger buffer (their absolute offsets don't change, but the
        // VkBuffer handle itself does, so every consumer holding that
        // handle - RaymarchSystem's per-frame BuildObjectList output,
        // every volume's own computeSet descriptor - would need
        // re-pointing). Not implemented; if this capacity is ever
        // exceeded (free list included), RegisterVolume throws rather
        // than silently corrupting adjacent volumes' data. Raise the
        // constant if that happens - see its own comment for sizing
        // guidance.
        VulkanBuffer m_SharedDensityBuffer;
        VkDeviceSize m_SharedDensityBufferCapacityElements = 0;
        VkDeviceSize m_SharedDensityBufferUsedElements = 0;

        // Free regions available for AllocateDensityRegion to reuse
        // before falling back to the bump path - see that function.
        // Kept sorted by offsetElements so ReleaseDensityRegion's
        // coalescing pass (merge two entries when one's end equals the
        // other's start) is a simple adjacent-pair scan, not a search.
        // Not indexed/spatially-accelerated - fine at this project's
        // scale (kVoxelMaxVolumes = 64 concurrent volumes, so at most
        // 64 free regions to scan/sort), revisit only if that constant
        // grows by an order of magnitude.
        struct FreeDensityRegion
        {
            VkDeviceSize offsetElements = 0;
            VkDeviceSize sizeElements = 0;
        };
        std::vector<FreeDensityRegion> m_FreeDensityRegions;

        // Queried once at Init via vkGetPhysicalDeviceProperties - the
        // minimum alignment (in bytes) a VkDescriptorBufferInfo::offset
        // into a storage buffer must respect
        // (VkPhysicalDeviceLimits::minStorageBufferOffsetAlignment).
        // Every volume's densityOffsetElements is rounded up to a
        // multiple of this (converted to float-element units) before
        // being handed out - required for voxel_march.comp's per-volume
        // sub-range bind (RegisterVolume) to be spec-legal; an
        // unaligned offset there is undefined behavior/a validation
        // error, not just a performance concern.
        VkDeviceSize m_StorageBufferOffsetAlignmentBytes = 1;

        // Shared, uploaded once at Init - every volume's compute
        // descriptor set binds the SAME table buffer (read-only,
        // never changes), only their density/vertex/indirect/dirty
        // buffers differ per volume. Single buffer (was two - edge
        // table + tri table) since the MC33 rewrite packs every table
        // Lewiner's algorithm needs into one blob - see
        // MarchingCubesTables33.h's provenance comment for why.
        VulkanBuffer m_MC33Table;

        VkDescriptorSetLayout m_ComputeSetLayout = VK_NULL_HANDLE;
        VulkanDescriptorPool m_ComputeDescriptorPool;
        VulkanPipelineLayout m_ComputePipelineLayout;

        // Descriptor sets returned by UnregisterVolume, reused by
        // RegisterVolume before allocating a new one from
        // m_ComputeDescriptorPool. This is a SOFT recycle, not a real
        // Vulkan free - VulkanDescriptorPool was never created with
        // VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT, so
        // individual sets can't be returned to the pool itself, only to
        // this application-side reuse list. RegisterVolume's existing
        // vkUpdateDescriptorSets call (it already rewrites every
        // binding unconditionally for whichever volume is registering)
        // is what makes reuse safe - a recycled set gets its bindings
        // pointed at the NEW volume's buffers before anything reads
        // from it again.
        std::vector<VkDescriptorSet> m_FreeComputeSets;

        // The ONLY GPU compute pipeline left in this system - see the
        // class comment on why seeding/carving no longer need their own
        // (they used to; voxel_seed.comp/voxel_carve.comp are gone).
        VulkanComputePipeline m_MarchPipeline;

        // m_Volumes is slot storage, not a 1:1 "handle == index" table
        // anymore (see VoxelVolumeHandle's own comment) - a slot's
        // occupant can change across its lifetime as UnregisterVolume/
        // RegisterVolume recycle it. m_Generations is parallel to
        // m_Volumes (same size, same indexing); m_FreeSlotIndices holds
        // indices UnregisterVolume has released, checked FIRST by
        // RegisterVolume before growing m_Volumes - the exact shape
        // Registry::CreateEntity/DestroyEntity already use for ECS
        // entities (see Registry.h), mirrored here rather than
        // reinvented.
        std::vector<Volume> m_Volumes;
        std::vector<VoxelVolumeGeneration> m_Generations;
        std::vector<VoxelVolumeIndex> m_FreeSlotIndices;
    };
}
