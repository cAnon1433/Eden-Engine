#pragma once

#include <glm/glm.hpp>
#include <cstdint>

namespace Eden
{
    // Fixed ceiling on how many raymarch objects can be live in one
    // frame - MUST match raymarch.frag's MAX_OBJECTS #define exactly
    // (that shader can't #include this header - see raymarch.frag's own
    // comment on why GLSL includes aren't used anywhere in Eden - so
    // this is a "kept in sync by hand" constant, same category as
    // ParticleGPUTypes.h's push-constant structs). Sizes BOTH the
    // ObjectBuffer array (small, one entry of metadata per object) AND
    // the descriptor-count math against MoltenVK's
    // maxPerStageDescriptorStorageBuffers=31 ceiling - unlike an
    // intermediate version of this file, this is back to being a real
    // hardware-bound ceiling, not a soft allocation choice.
    //
    // History, briefly (see git history / prior session notes for the
    // full account): this used to be 24, one descriptor slot per
    // object's OWN density buffer (N objects = N+1 descriptors,
    // directly hitting the 31-descriptor ceiling at a low object
    // count). An attempt to fix that via true bindless/descriptor-
    // indexing (unbounded densityBuffers[] array, update-after-bind)
    // was built and tested on this project's actual dev hardware, and
    // MoltenVK rejected the required layout outright
    // (vkGetDescriptorSetLayoutSupport returned supported=VK_FALSE,
    // independent of the requested count) - not a MoltenVK limits
    // question, a genuine "this flag combination isn't supported here"
    // result. The actual fix (this version): every volume's density
    // data now lives in ONE SHARED buffer
    // (VoxelSystemGPU::m_SharedDensityBuffer), with each object
    // carrying its own OFFSET into that buffer
    // (RaymarchObjectGPU::densityOffset) rather than its own descriptor
    // slot. That means raymarch.frag's DensityBuffer binding is back to
    // being exactly ONE descriptor, regardless of object count - this
    // constant now ONLY bounds ObjectBuffer's small metadata array and
    // m_RaymarchObjectBuffer's CPU-side upload size, not a descriptor-
    // count budget, so it can be raised freely without touching the
    // 31-descriptor ceiling at all. Left at a modest value (not raised
    // back to 256) simply because nothing currently needs more; raise
    // freely.
    constexpr uint32_t kRaymarchMaxObjects = 64;

    // Everything the raymarch fragment shader needs to sphere-trace one
    // object, laid out to match RaymarchObjectGPU in raymarch.frag
    // field-for-field (std430 layout - see alignas comments below,
    // matching this project's existing storage-buffer convention, e.g.
    // particle_density.comp's PositionsBuffer). This is the CPU-side mirror that gets packed
    // into the set=1/binding=0 storage buffer each frame; it is NOT
    // itself uploaded to a persistent per-object buffer, since the
    // whole array is small and rebuilt fresh every frame the same way
    // RenderSystem::BuildDrawList rebuilds the mesh drawList - see
    // RaymarchSystem::BuildObjectList.
    struct RaymarchObjectGPU
    {
        alignas(16) glm::mat4 invModel{ 1.0f }; // world -> object-local

        // Object-local AABB, centered on zero (TransformComponent::
        // position is this object's world-space CENTER, matching every
        // other TransformComponent convention in the codebase - see
        // RaymarchSystem.h's namespace comment for the full reasoning,
        // including why this replaced an earlier min-corner convention
        // from this system's first working version).
        alignas(16) glm::vec3 localMin{ 0.0f };
        float voxelSize = 0.1f;

        alignas(16) glm::vec3 localMax{ 0.0f };
        // Previously an unused pad float (pad0) - now this object's
        // starting index (in float ELEMENTS, not bytes) within
        // raymarch.frag's single shared DensityBuffer array, populated
        // directly from VoxelSystemGPU::GetDensityBufferOffset (see
        // RaymarchSystem::BuildObjectList). The shader adds this to
        // every local sample index before indexing density[] - see
        // raymarch.frag's SampleDensityTrilinear. NOT alignas'd
        // separately; this occupies the same std430 slot pad0 used to,
        // so the struct's total size/layout is unchanged from before
        // this field existed - only its meaning changed from "unused"
        // to "load-bearing."
        uint32_t densityOffset = 0;

        alignas(16) glm::ivec3 sampleDims{ 0 }; // SampleDims() - see VoxelVolumeDesc::SampleDims
        float pad1 = 0.0f;

        alignas(16) glm::vec3 tintColor{ 1.0f };
        float pad2 = 0.0f;

        // Added to a local-space position AFTER the object-space ray
        // march (which works in the -halfExtent..+halfExtent AABB
        // above) and BEFORE sampling the density field, which is still
        // stored starting at field-local (0,0,0) per
        // VoxelVolumeDesc/densityCPU's own convention (unchanged by
        // this object-position convention switch - only how the
        // OBJECT's AABB relates to TransformComponent::position moved,
        // not how VoxelSystemGPU itself lays out a volume's density
        // array). Always equals -localMin (i.e. +halfExtent) by
        // construction - see RaymarchSystem::BuildObjectList.
        alignas(16) glm::vec3 fieldOffset{ 0.0f };
        float pad3 = 0.0f;
    };

    // Matches raymarch.frag's PushConstants block exactly.
    struct RaymarchPushConstants
    {
        int32_t objectCount = 0;
        float nearPlane = 0.1f;
        float farPlane = 100.0f;
    };
}
