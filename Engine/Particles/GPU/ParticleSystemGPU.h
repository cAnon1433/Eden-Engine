#pragma once

#include "ParticleGPUTypes.h"
#include "../../ECS/Registry.h"
#include "../../Renderer/Vulkan/Resources/VulkanBuffer.h"
#include "../../Renderer/Vulkan/Resources/VulkanMemoryAllocator.h"
#include "../../Renderer/Vulkan/Pipeline/VulkanPipelineLayout.h"
#include "../../Renderer/Vulkan/Pipeline/VulkanComputePipeline.h"
#include "../../Renderer/Vulkan/Pipeline/VulkanDescriptorPool.h"

#include <glm/glm.hpp>
#include <vector>
#include <array>
#include <string>

namespace Eden
{
    class VoxelSystemGPU; // forward-declared - Step() only needs a pointer, see its own comment for why

    // Full GPU-resident port of ParticleSystem (Engine/Particles/
    // ParticleSystem.h) - same WCSPH solver (Poly6/Spiky/Viscosity
    // kernels, Tait EOS, SDF boundary collision with conservative-
    // advancement sweep), same tunables and defaults, but neighbor
    // search, density/pressure, forces, integration, AND boundary
    // collision all run as compute shaders with no CPU fallback -
    // positions/velocities never leave the GPU between Step() and
    // rendering (see Renderer::RegisterParticleGPUSource,
    // particle_point_gpu.vert).
    //
    // --- Neighbor search: hash grid, not counting sort -----------------
    // The CPU version's ParticleSpatialHash does an exact counting sort
    // (every particle lands in exactly the right cell, no collisions).
    // A GPU counting sort needs a parallel prefix sum over cell counts -
    // real, separate work. This class instead uses a fixed-size hash
    // table with atomic bucket append (particle_build_grid.comp) - two
    // distinct 3D cells can collide onto the same bucket, which
    // occasionally merges or drops a neighbor contribution near that
    // collision. A deliberate, documented tradeoff scaled by keeping
    // GPU_HASH_TABLE_SIZE proportional to GPU_MAX_PARTICLES (see that
    // constant's comment in ParticleGPUTypes.h for the ratio), not an
    // oversight. Revisit with a real GPU radix/bitonic sort if particle
    // counts grow enough, or get dense enough in one region, for
    // collisions to become visually significant.
    //
    // --- Why Step() doesn't submit anything itself ----------------------
    // main.cpp's physics accumulator can call Step() zero to
    // MAX_PHYSICS_STEPS_PER_FRAME times before a single Renderer::
    // DrawFrame() call. Submitting (and fencing on) a compute command
    // buffer per Step() call would mean a CPU/GPU sync stall per physics
    // substep - up to 5 stalls before a frame ever reaches the render
    // pass. Instead, Step() just snapshots colliders and queues the
    // dispatch parameters (see PendingStep); RecordPendingSteps() -
    // called once by Renderer::DrawFrame, INSIDE that frame's own
    // command buffer, before the render pass begins - flushes every
    // queued Step() into real vkCmdDispatch calls with the barriers
    // between them, then a single memory barrier before the render pass
    // starts covers compute-write -> vertex-shader-read. One submission,
    // one queue, no semaphores, no extra CPU waits beyond what already
    // existed (the frame's own in-flight fence).
    class ParticleSystemGPU
    {
    public:
        // --- Tunables - same defaults/meaning as ParticleSystem.h -------
        float smoothingRadius = 0.2f;
        float particleMass = 1.728f; // = 1000.0f * (0.2f * 0.6f)^3, see ParticleSystem.h's comment on why this isn't a free parameter
        float restDensity = 1000.0f;
        float stiffness = 3000.0f;
        float gamma = 7.0f;

        // Bumped from 0.5 - see boundaryFriction's comment below.
        // Boundary friction only ever damps the layer actually touching
        // a collider; without enough physical (inter-particle) damping,
        // everything stacked above that bottom layer just keeps
        // jittering indefinitely off the pressure term. This is a real
        // tuning knob, not a fixed constant - lower it if particles ever
        // read as too "syrupy"/slow to respond.
        float viscosityCoefficient = 1.2f;

        glm::vec3 gravity{ 0.0f, -9.81f, 0.0f };
        float maxAcceleration = 500.0f;
        int substeps = 4;

        // Scaled down to stay proportional with smoothingRadius's 0.3 ->
        // 0.2 change above (same ratio to the natural rest-spacing,
        // smoothingRadius * 0.6).
        float boundaryRadius = 0.0333f;
        float boundaryRestitution = 0.0f;

        // See SimParamsGPU::boundaryFriction's comment - damps
        // along-surface velocity on contact so resting particles
        // actually stop instead of skidding/bouncing indefinitely.
        // Added when the container-catch test (main.cpp) turned out to
        // be this engine's first real resting-contact case - open-space
        // SPH motion never exercised this before. Bumped from 0.35 -
        // still not, alone, enough for particles resting on TOP of
        // other particles (not directly touching a collider) to settle;
        // see viscosityCoefficient above for that half of the fix.
        float boundaryFriction = 0.5f;

        int maxSweepIterations = 8;

        // Akinci cohesion coefficient - see particle_force.comp's
        // CohesionKernel/main() for the force this drives, and
        // SimParamsGPU::cohesion's comment (ParticleGPUTypes.h) for why
        // it exists. 0 = off, identical to pre-cohesion behavior.
        // Un-tuned starting point - dial this from the "Particles (GPU)"
        // ImGui panel while a resting pool/splash is on screen; a few
        // thousand is a reasonable starting order of magnitude given
        // this project's mass/restDensity/h, same scale as `stiffness`.
        // Too high pulls the whole body into a single shrinking clump
        // instead of holding a free surface together - if that happens,
        // come back down.
        float cohesion = 0.0f;

        // A particle whose position.y drops below this is considered to
        // have left the simulated world - see SimParamsGPU::voidKillY's
        // own comment (ParticleGPUTypes.h) and particle_integrate.comp's
        // use of it for the full reasoning. This engine subsystem has no
        // notion of "where the world's floor is" on its own; whatever
        // owns scene layout (main.cpp) is expected to set this once it
        // knows the floor's Y, with a comfortable margin below it -
        // defaults far below any real scene so it's a no-op otherwise.
        float voidKillY = -1.0e9f;

        // --- Chemical ID table (base build) --------------------------------
        // Public, directly editable - same "live-tunable field, no
        // getter/setter ceremony" pattern `cohesion` above already uses.
        // Index 0 always exists (every Emit/EmitBox/EmitPoints call
        // defaults to chemicalId=0) and is seeded here to reproduce the
        // exact single-material behavior this class had before this
        // field existed (mass/restDensity/stiffness/viscosity matching
        // the old global SimParamsGPU defaults, hardeningThreshold ==
        // GPU_HEAT_REFORM_THRESHOLD). Index 1 is seeded as a second,
        // visibly different example chemical (a dark tint, lower
        // viscosity) purely so a fresh build has something to test
        // "differing properties + color mixing" against immediately via
        // EmitBox(..., chemicalId=1) without hand-authoring one first.
        // Indices 2+ start as copies of index 0 - edit them (directly, or
        // via ImGui) and call UploadChemicalTable() to push changes to
        // the GPU; entries are otherwise inert (never emitted, never
        // referenced) until something actually uses that ID.
        std::array<ChemicalPropertiesGPU, GPU_MAX_CHEMICALS> ChemicalTable = []
        {
            std::array<ChemicalPropertiesGPU, GPU_MAX_CHEMICALS> table;
            table[1].viscosity = 0.4f;
            table[1].stiffness = 2200.0f;
            table[1].baseColor = glm::vec4(0.05f, 0.35f, 0.1f, 1.0f); // dark green, per the "drop dye into a pool" test case
            return table;
        }();

        // Display-only labels for the ImGui chemical editor - purely a
        // UI convenience, never read by any shader or by anything on the
        // GPU side (a chemical is fully identified by its integer ID
        // everywhere else in the engine; this exists only so a human
        // editing/spawning chemicals doesn't have to remember "ID 4 is
        // the honey one"). Indices 0/1 match ChemicalTable's own seeded
        // defaults; everything else starts as a generic placeholder
        // until someone names it.
        std::array<std::string, GPU_MAX_CHEMICALS> ChemicalNames = []
        {
            std::array<std::string, GPU_MAX_CHEMICALS> names;
            for (uint32_t i = 0; i < GPU_MAX_CHEMICALS; ++i)
            {
                names[i] = "Chemical " + std::to_string(i);
            }
            names[0] = "Default";
            names[1] = "Dye";
            return names;
        }();

        // Re-uploads the whole ChemicalTable (a handful of chemicals,
        // 48 bytes each - trivial) to the GPU. Cheap enough to call
        // unconditionally once a frame after ImGui edits ChemicalTable
        // directly, same reasoning BuildSimParams() re-sending every
        // tunable every Step() already relies on - no dirty-tracking
        // needed for something this small and infrequent.
        void UploadChemicalTable();

        // Heat/cooldown tunables (Eden Particle State Transitions) - see
        // SimParamsGPU's own comment for why these ride along in the
        // formerly-padding push-constant slots. Mirrored into
        // SimParamsGPU every Step(), same as every other tunable above.
        float heatDecayRate = 0.15f;
        // Extra decay factor within kColdSurfaceBand of a collider.
        // Lowered from 3.0 - that big a gap between "touching the floor"
        // and "resting on top of other particles" meant only the
        // bottom-most layer ever cooled fast enough to be reform-
        // eligible while everything stacked on top of it was still
        // warm; ClusterSystem only connects particles that are BOTH
        // eligible, so an isolated cold layer forms its own tiny
        // cluster instead of joining the (still-warm) body above it -
        // this is the actual cause of "hardens into individual balls"
        // and, indirectly, of most of the reform freeze (many tiny
        // clusters = many separate GPU round trips - see
        // ReformSystem::Reform's maxClustersPerCall for the hard cap on
        // that). A real fix would have cold particles conduct some of
        // their coolness to warm neighbors they're resting against, not
        // just colliders - not implemented; this is a tuning mitigation,
        // not that fix. If it's still too uneven, waiting longer before
        // reforming (the whole pool decays on the same clock regardless
        // of this multiplier) is the other lever.
        float heatColdMultiplier = 1.5f;

        // --- Lifecycle ---------------------------------------------------
        // physicalDevice is used for exactly one check: that the queue
        // family behind graphicsQueue actually supports compute (see
        // .cpp) - everything here dispatches on the SAME queue/command
        // buffer as graphics, deliberately not a second queue family
        // (see class comment), which only works if that family really
        // is compute-capable. True on effectively all real hardware and
        // MoltenVK, but checked rather than assumed.
        //
        // voxelDensityBuffer: VoxelSystemGPU's single shared density
        // buffer (VoxelSystemGPU::GetDensityBuffer - any handle works,
        // every volume shares the same underlying VkBuffer, see that
        // system's own comment), bound once here at compute-descriptor-
        // set creation time as the fixed density-sampling source for
        // particle-vs-voxel collision (see Step()'s ColliderShape::Voxel
        // handling and particle_integrate.comp's DensityBuffer binding).
        // Passed as a raw VkBuffer rather than a VoxelSystemGPU&
        // reference - this module doesn't otherwise depend on
        // Engine/Voxel at all (see Step()'s forward-declared pointer
        // param for the one place it does, for a different reason), and
        // a single buffer handle is all descriptor binding actually
        // needs. Caller contract: voxelSystem.Init() must run before
        // this Init() call (main.cpp already orders it that way, since
        // RegisterVolume/Init are what actually create
        // m_SharedDensityBuffer - see VoxelSystemGPU::Init).
        void Init(VkDevice device, VkPhysicalDevice physicalDevice, VmaAllocator allocator,
                  VkCommandPool commandPool, VkQueue graphicsQueue, VkBuffer voxelDensityBuffer);
        void Shutdown();

        // Same shape as ParticleSystem's - one-time/infrequent, does a
        // real staged GPU upload + queue wait (see .cpp), not meant for
        // per-frame use.
        //
        // Both set the new particles' heat to a normal starting value
        // (1.0 - "freshly poured, hasn't cooled yet") rather than any
        // kind of "exempt from the melt system" sentinel - see
        // GPU_HEAT_NOT_MELTED's comment for what changed and why. Any
        // particle can now decay/conduct heat and become reform-eligible
        // over time, gated only by its OWN chemical's hardeningThreshold
        // (see ChemicalPropertiesGPU), not by how it was created.
        //
        // chemicalId indexes ChemicalTable (see that field's comment) -
        // defaults to 0 so every existing call site (nothing in this
        // codebase passed a chemical before this base build existed)
        // keeps behaving exactly as before. The new particle's color is
        // initialized from ChemicalTable[chemicalId].baseColor at emit
        // time, then left free to diffuse independently afterward (see
        // m_Colors' comment) - color is a per-PARTICLE value from this
        // point on, not a live lookup into the table.
        void Emit(const glm::vec3& position, const glm::vec3& velocity = glm::vec3(0.0f), uint32_t chemicalId = 0);
        void EmitBox(const glm::vec3& min, const glm::vec3& max, float spacing = 0.0f,
                     const glm::vec3& initialVelocity = glm::vec3(0.0f), uint32_t chemicalId = 0);

        // Batch-emits an arbitrary, pre-computed scatter of positions
        // (same single-upload-per-call shape as EmitBox, which fills an
        // AABB uniformly - this is for callers, like MeltSystem, whose
        // spawn set isn't a filled box, e.g. only the solid samples of a
        // voxel density field). All particles share one initial
        // velocity/heat/chemical; per-particle values aren't needed for
        // the callers this exists for today. initialHeat defaults to
        // 1.0, same "starts warm, needs to cool before it can harden"
        // convention Emit/EmitBox now use - MeltSystem also passes 1.0
        // explicitly (freshly melted particles start just as warm as
        // freshly poured ones; there's no meaningful difference anymore,
        // see GPU_HEAT_NOT_MELTED's comment).
        //
        // colorOverride defaults to glm::vec4(0.0f) meaning "use
        // ChemicalTable[chemicalId].baseColor" - the SAME "transparent
        // black = no explicit color set" convention
        // VoxelSystemGPU::SetTransform already established (see that
        // function's own default parameter), reused here rather than
        // inventing a second, different sentinel value for the same
        // "nothing was actually set" meaning. This matters concretely:
        // Volume::lastColor defaults to exactly this value for any
        // volume that's never had an explicit color set (e.g. the
        // original hardcoded test sphere, which predates the Chemical ID
        // system) - MeltSystem passes that default straight through, and
        // it needs to resolve to "use the chemical's real color", not to
        // literal black.
        //
        // MeltSystem passes an explicit non-zero color here whenever one
        // exists - a volume's actual hardened color (see
        // VoxelSystemGPU::GetVolumeColor) may have drifted from its
        // chemical's flat default through diffusion before it ever
        // reformed (see ReformSystem/ReformCluster::averageColor), and
        // melting it back should hand out particles of THAT color, not
        // reset everything to the chemical's out-of-the-box default.
        void EmitPoints(const std::vector<glm::vec3>& positions, const glm::vec3& initialVelocity = glm::vec3(0.0f),
                         float initialHeat = 1.0f, uint32_t chemicalId = 0,
                         const glm::vec4& colorOverride = glm::vec4(0.0f));

        void Clear();

        uint32_t ParticleCount() const { return m_ParticleCount; }
        static constexpr uint32_t Capacity() { return GPU_MAX_PARTICLES; }

        // Snapshots colliders (TransformComponent + ColliderComponent
        // pairs) and queues this call's substeps for the next
        // RecordPendingSteps() flush - does NOT touch the GPU directly.
        // Same call-site contract as ParticleSystem::Step: call from the
        // fixed-timestep accumulator, after rigid bodies have moved for
        // this tick. voxelSystem is optional (nullptr default), same
        // "no voxel system, no voxel collision, not a crash" contract
        // CollisionSystem::Step's own VoxelSystemGPU* parameter has -
        // needed here because a Voxel-shaped ColliderComponent (see
        // ColliderShape::Voxel) can't be turned into a ColliderGPU
        // entry without querying that volume's density-buffer offset/
        // sample dims/voxel size (see ParticleSystemGPU.cpp's Step
        // body). If nullptr, any Voxel colliders in the scene are
        // silently skipped for GPU particle collision (logged once,
        // same pattern as the GPU_MAX_COLLIDERS overflow warning) rather
        // than uploading a garbage/zeroed entry.
        void Step(Registry& registry, float fixedDeltaTime, const VoxelSystemGPU* voxelSystem = nullptr);

        // Records every pending Step() (in the order they were queued)
        // into `cmd` as compute dispatches with barriers between passes,
        // then clears the pending queue. Must be called with `cmd` in
        // the recording state, OUTSIDE any render pass (compute
        // dispatches can't happen inside one) - Renderer::DrawFrame is
        // the one real caller, right before vkCmdBeginRenderPass.
        void RecordPendingSteps(VkCommandBuffer cmd);

        // Stable for the object's lifetime (fixed-capacity buffer,
        // never reallocated) - safe to build a descriptor set against
        // once and keep it, see Renderer::RegisterParticleGPUSource.
        VkBuffer GetPositionBuffer() const { return m_Positions.Get(); }
        // See m_Colors' comment - per-particle, diffusing color, added
        // for the Chemical ID base build. Same "stable for the object's
        // lifetime" guarantee as GetPositionBuffer.
        VkBuffer GetColorBuffer() const { return m_Colors.Get(); }
        // World-space units/second, vec4 (w unused, same layout as
        // GetPositionBuffer). General-purpose accessor, same "stable for
        // the object's lifetime" guarantee as GetPositionBuffer. Not
        // currently bound by Renderer - an earlier version of the fluid
        // temporal reprojection pass used this to approximate each
        // particle's previous position, but that was replaced by an
        // exact previous-position buffer snapshot the renderer maintains
        // itself (see Renderer.h's m_ParticlePreviousPositions comment
        // for why the velocity-based estimate was wrong).
        VkBuffer GetVelocityBuffer() const { return m_Velocities.Get(); }

        // Blocking GPU->CPU readback of every active particle's
        // position, velocity, and heat - a deliberate, narrow,
        // event-driven exception to this class's usual "zero CPU
        // readback" design rule (see the class comment above): the
        // reform half of the melt/reform pipeline has to hand
        // VoxelSystemGPU real CPU-side positions to build its
        // CPU-authoritative density field from (see VoxelSystemGPU's
        // own header comment on why THAT system is CPU-authoritative),
        // so a readback is unavoidable somewhere in this pipeline. Real
        // staged transfer + queue wait, same infrequent-blocking-call
        // shape Emit/EmitBox/EmitPoints already use for uploads - never
        // meant to run per-frame. Used by ClusterSystem.
        void ReadBackAll(std::vector<glm::vec4>& outPositions, std::vector<glm::vec4>& outVelocities,
                          std::vector<float>& outHeat, std::vector<uint32_t>& outChemicalIds) const;

        // Colors aren't part of ReadBackAll above (nothing needs them
        // alongside position/velocity/heat/chemicalId today) - separate,
        // dedicated readback for ClusterSystem, which needs each
        // reform-eligible cluster's actual (possibly diffused-away-from-
        // its-chemical's-default) color to hand to ReformSystem - see
        // ReformCluster::averageColor's comment.
        void ReadBackColors(std::vector<glm::vec4>& outColors) const;

        // Removes the given particle indices (each must be <
        // ParticleCount()) by reading everything back, compacting on
        // the CPU, and re-uploading the survivors - simplest correct
        // implementation, not a GPU compaction pass. Acceptable cost for
        // an event-driven reform action (a handful of times per
        // session), not something meant to run every frame. Used by
        // ReformSystem once a cluster's particles have been consumed
        // into a newly-seeded voxel volume.
        void KillParticles(const std::vector<uint32_t>& indices);

    private:
        struct PendingStep
        {
            float fixedDeltaTime = 0.0f;
            std::vector<ColliderGPU> colliders;
        };

        void CreateStorageBuffers();
        void CreateComputeDescriptors(VkBuffer voxelDensityBuffer);
        void CreateComputePipelines();

        // Staged upload into an existing DEVICE_LOCAL buffer at a byte
        // offset - the "existing buffer, partial range" case
        // VulkanBuffer::InitDeviceLocalWithData doesn't cover (that one
        // always creates+fills a brand new buffer).
        //
        // Used to block on vkQueueWaitIdle with a fresh staging buffer +
        // command buffer created and destroyed every call, on the
        // stated assumption that "Emit/EmitBox aren't per-frame calls."
        // That stopped being true the moment ProcessPourInput (main.cpp)
        // started calling EmitBox on a HELD key at ~12/sec - the exact
        // same "object churn in a path that turned out to become hot"
        // mistake VoxelSystemGPU's carve path had three separate
        // instances of (fence, staging buffer, command buffer - see that
        // class's own comments). Now uses the same persistent-resource
        // pattern: one staging buffer (grown, never recreated per call),
        // one reusable command buffer, one reusable fence.
        void UploadRange(VkBuffer dst, VkDeviceSize dstOffset, const void* data, VkDeviceSize size);

        // Read-side counterpart to UploadRange - stages through a
        // host-visible buffer + one-time command buffer + blocking
        // queue wait, same shape, just copying GPU->staging->CPU
        // instead of CPU->staging->GPU. See ReadBackAll's public
        // comment for why this exists at all. Still only called from
        // ClusterSystem's reform pass (H key, edge-triggered, not
        // held) - unlike UploadRange, nothing has turned this into a
        // hot path yet, so it's left as-is rather than fixed
        // speculatively. Worth revisiting with the same fix if that
        // ever changes.
        void DownloadRange(VkBuffer src, VkDeviceSize srcOffset, void* data, VkDeviceSize size) const;

        // (Re)creates m_UploadStagingBuffer only if requiredBytes
        // exceeds its current capacity - same shape as VoxelSystemGPU's
        // EnsureDensityStagingCapacity, see that function's own comment.
        void EnsureUploadStagingCapacity(VkDeviceSize requiredBytes);

        // Returns m_UploadCommandBuffer, allocating it once and
        // resetting (never reallocating) it on every later call - same
        // shape as VoxelSystemGPU::GetReusableCommandBuffer.
        VkCommandBuffer GetReusableUploadCommandBuffer();

        // Submits a fully-recorded m_UploadCommandBuffer and blocks only
        // until it finishes, via m_UploadFence (reset and reused every
        // call, never recreated) - same shape as VoxelSystemGPU::
        // SubmitAndWaitForCommandBuffer, see that function's own comment
        // for why a fence scoped to just this submission (not
        // vkQueueWaitIdle's whole-queue drain, and not a fresh VkFence
        // per call either) is the actual fix.
        void SubmitAndWaitForUploadCommandBuffer();

        SimParamsGPU BuildSimParams(float subDt, uint32_t numColliders) const;

        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;
        VkCommandPool m_CommandPool = VK_NULL_HANDLE;
        VkQueue m_GraphicsQueue = VK_NULL_HANDLE;

        // Persistent resources for UploadRange/EmitBox's batched upload -
        // see EnsureUploadStagingCapacity/GetReusableUploadCommandBuffer/
        // SubmitAndWaitForUploadCommandBuffer's own comments.
        VkBuffer m_UploadStagingBuffer = VK_NULL_HANDLE;
        VmaAllocation m_UploadStagingAllocation = VK_NULL_HANDLE;
        VkDeviceSize m_UploadStagingCapacityBytes = 0;
        void* m_UploadStagingMappedPtr = nullptr;
        VkCommandBuffer m_UploadCommandBuffer = VK_NULL_HANDLE;
        VkFence m_UploadFence = VK_NULL_HANDLE;

        VulkanBuffer m_Positions;
        VulkanBuffer m_Velocities;
        VulkanBuffer m_Densities;
        VulkanBuffer m_Pressures;
        VulkanBuffer m_Forces;
        VulkanBuffer m_Heats;
        // Chemical ID system (base build) - see ChemicalTable's own
        // comment for the table itself. m_ChemicalIds is per-particle
        // (like positions/heat - needs an initial upload at emit time and
        // compaction in KillParticles, since it's identity data, not
        // scratch). m_Colors is also per-particle and also needs initial
        // upload/compaction for the same reason (its value at any given
        // moment depends on this particle's history of diffusion, not
        // something recomputable from scratch each substep). m_ColorsAvg
        // is pure scratch, mirroring m_Forces' category exactly - written
        // in full by particle_force.comp every substep before
        // particle_integrate.comp reads it, never needs initial upload or
        // KillParticles compaction. m_ChemicalProperties is NOT indexed
        // by particle slot at all - it's the small fixed-size table
        // (ChemicalTable's GPU copy), entirely separate from per-particle
        // buffer management.
        VulkanBuffer m_ChemicalIds;
        VulkanBuffer m_Colors;
        VulkanBuffer m_ColorsAvg;
        VulkanBuffer m_ChemicalProperties;
        VulkanBuffer m_CellCounts;
        VulkanBuffer m_CellBuckets;

        // One collider buffer PER pending-step slot, not one shared
        // buffer - see the class comment on why Step() defers recording.
        // If collider uploads shared a single buffer, memcpying step
        // B's colliders in (at Step()-call CPU time) would corrupt step
        // A's already-recorded-but-not-yet-executed dispatches, since
        // recording happens long before submission. N independent
        // buffers (host-visible, persistently mapped) sidesteps that
        // entirely - each pending step's memcpy lands in its own slot.
        std::array<VulkanBuffer, GPU_MAX_PENDING_STEPS> m_ColliderBuffers;
        std::array<void*, GPU_MAX_PENDING_STEPS> m_ColliderBuffersMapped{};

        VkDescriptorSetLayout m_ComputeSetLayout = VK_NULL_HANDLE;
        VulkanDescriptorPool m_ComputeDescriptorPool;
        std::array<VkDescriptorSet, GPU_MAX_PENDING_STEPS> m_ComputeDescriptorSets{};

        VulkanPipelineLayout m_ComputePipelineLayout;
        VulkanComputePipeline m_BuildGridPipeline;
        VulkanComputePipeline m_DensityPipeline;
        VulkanComputePipeline m_ForcePipeline;
        VulkanComputePipeline m_IntegratePipeline;

        uint32_t m_ParticleCount = 0;
        std::vector<PendingStep> m_PendingSteps;
    };
}
