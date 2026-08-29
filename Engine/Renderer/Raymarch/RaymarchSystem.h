#pragma once

#include "RaymarchTypes.h"
#include "../../ECS/Registry.h"
#include "../../ECS/Components/TransformComponent.h"
#include "../../Voxel/VoxelField.h"
#include "../../Voxel/VoxelSystemGPU.h"

#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <iostream>

namespace Eden
{
    // Attach alongside TransformComponent + VoxelVolumeComponent to mark
    // a registered VoxelSystemGPU volume as raymarched (sphere-traced
    // per-pixel against its live density field) rather than
    // marching-cubes-rendered. A volume can carry VoxelVolumeComponent
    // WITHOUT this and still get real Sphere-vs-Voxel collision (see
    // ColliderComponent's Voxel shape) - this component controls DRAW
    // path only, same separation MeshComponent vs VoxelVolumeComponent
    // already had.
    //
    // tintColor replaces voxel_march.comp's hardcoded vec3(1.0) white -
    // see raymarch.frag's ObjectBuffer struct for where this ends up.
    // No colorOverride/off-switch shape like ColorComponent's alpha
    // channel - a RaymarchVolumeComponent entity always has a color,
    // there's no "use the mesh's own vertex color" fallback to opt out
    // of the way voxel_march.comp's per-vertex white currently is.
    struct RaymarchVolumeComponent
    {
        glm::vec3 tintColor{ 1.0f };
    };

    // TransformComponent::position on a raymarch-volume entity is the
    // volume's WORLD-SPACE CENTER - matching RegisterVoxelPhysicsEntity's
    // convention (and every other TransformComponent in this codebase:
    // mesh cubes, physics colliders, etc. all treat position as center,
    // not a corner). This REPLACES an earlier min-corner convention from
    // this system's first working version - min-corner was the odd one
    // out once objects started needing both a raymarch entity AND a
    // physics-collider entity built the normal way, and having two
    // different position conventions for what's conceptually "the same
    // object, two components" was a real footgun waiting to happen. See
    // localMin/localMax below for how this shifts the object-local AABB
    // to be centered on zero instead of starting at zero, to match.
    namespace RaymarchSystem
    {
        // Builds this frame's RaymarchObjectGPU array from every
        // TransformComponent + VoxelVolumeComponent + RaymarchVolumeComponent
        // entity - same "resolve storages once, loop entities once" shape
        // as RenderSystem::BuildDrawList. Capped at kRaymarchMaxObjects;
        // overflow is dropped with a one-time warning (same pattern
        // RenderSystem::BuildDrawList uses for MAX_INSTANCES_PER_FRAME),
        // not a crash.
        //
        // No frustum/occlusion culling here - raymarch.frag's own
        // ray-AABB rejection is the actual cost-control mechanism for
        // this path (see that shader's RayAabbLocal), not a CPU-side
        // visibility pass. An object fully outside the view frustum
        // still gets uploaded into the SSBO and tested per-pixel
        // against every screen ray's AABB, which is wasted work at
        // scale - worth revisiting alongside the voxel indirect-draw
        // path's own known-missing culling (see main.cpp's "NoCull
        // Stopgap" state) once object counts actually grow past what
        // this scene tests today.
        //
        // Returns both the object-metadata array AND its index-matched
        // density-buffer list together (rather than two separate
        // functions/callers each re-walking registry.View()) - the two
        // MUST stay paired by index (see Renderer::DrawFrame's param
        // comment), and building them from one shared iteration is the
        // only way to guarantee that without relying on
        // ComponentStorage::Entities() returning the same order across
        // two separate calls.
        //
        // densityBuffer is now a SINGLE shared VkBuffer (every
        // registered volume's density data lives in one buffer - see
        // VoxelSystemGPU::m_SharedDensityBuffer's comment for why this
        // replaced the earlier one-buffer-per-object design), not a
        // per-object vector - each object instead carries its own
        // offset INTO that one buffer via RaymarchObjectGPU::
        // densityOffset, populated below from
        // VoxelSystemGPU::GetDensityBufferOffset.
        struct RaymarchFrameData
        {
            std::vector<RaymarchObjectGPU> objects;
            VkBuffer densityBuffer = VK_NULL_HANDLE;
        };

        inline RaymarchFrameData BuildObjectList(Registry& registry, const VoxelSystemGPU& voxelSystem)
        {
            RaymarchFrameData frameData;

            auto entities = registry.View<TransformComponent, VoxelVolumeComponent, RaymarchVolumeComponent>();

            auto& transforms = registry.GetStorage<TransformComponent>();
            auto& volumes = registry.GetStorage<VoxelVolumeComponent>();
            auto& raymarch = registry.GetStorage<RaymarchVolumeComponent>();

            for (Entity entity : entities)
            {
                if (frameData.objects.size() >= kRaymarchMaxObjects)
                {
                    static bool s_WarnedAboveCap = false;
                    if (!s_WarnedAboveCap)
                    {
                        std::cerr << "Eden: frame requested more than " << kRaymarchMaxObjects
                                  << " raymarch objects, dropping the rest (see kRaymarchMaxObjects in RaymarchTypes.h)\n";
                        s_WarnedAboveCap = true;
                    }
                    break;
                }

                const TransformComponent& transform = transforms.Get(entity);
                VoxelVolumeHandle handle = volumes.Get(entity).handle;
                const RaymarchVolumeComponent& tint = raymarch.Get(entity);

                if (handle == InvalidVoxelVolumeHandle)
                {
                    continue;
                }

                RaymarchObjectGPU obj{};

                // World model = translate(worldCenter) * rotate * scale,
                // same TransformComponent::GetModelMatrix() every other
                // draw path uses - see this file's namespace comment on
                // why position means "center" here, matching every
                // other TransformComponent convention in the codebase.
                glm::mat4 model = transform.GetModelMatrix();
                obj.invModel = glm::inverse(model);

                // Object-local AABB centered on zero (-halfExtent to
                // +halfExtent), NOT VoxelVolumeDesc's own (0,0,0)-origin
                // convention - the density field's sample grid still
                // starts at local (0,0,0) internally (see
                // SampleSignedDistanceLocal in raymarch.frag, which
                // takes a field-local position), but the OBJECT's AABB
                // as seen from world space is shifted by -halfExtent so
                // that TransformComponent::position (the object's
                // center) lands in the middle of it. See localOffset
                // below for the actual shift applied before sampling.
                glm::ivec3 sampleDims = voxelSystem.GetSampleDims(handle);
                float voxelSize = voxelSystem.GetVoxelSize(handle);
                glm::ivec3 voxelDims = sampleDims - glm::ivec3(1); // SampleDims() = VoxelDims() + 1, see VoxelVolumeDesc
                glm::vec3 fieldExtent = glm::vec3(voxelDims) * voxelSize;
                glm::vec3 halfExtent = fieldExtent * 0.5f;

                obj.localMin = -halfExtent;
                obj.localMax = halfExtent;
                obj.fieldOffset = halfExtent; // added back before sampling the density field - see raymarch.frag's use of fieldOffset
                obj.voxelSize = voxelSize;
                obj.sampleDims = sampleDims;
                obj.tintColor = tint.tintColor;
                // This object's region within the ONE shared density
                // buffer - see RaymarchFrameData's own comment above and
                // raymarch.frag's DensityBuffer/SampleDensityTrilinear
                // for how the shader adds this before indexing.
                obj.densityOffset = static_cast<uint32_t>(voxelSystem.GetDensityBufferOffset(handle));

                frameData.objects.push_back(obj);

                // Every live object shares the SAME underlying buffer
                // now (see RaymarchFrameData's comment) - grabbing it
                // once per object is redundant but harmless (identical
                // VkBuffer handle every time; GetDensityBuffer's handle
                // parameter is unused now - see VoxelSystemGPU.cpp).
                // Kept inside the loop rather than hoisted out because
                // an EMPTY object list (frameData.objects stays empty)
                // should leave densityBuffer at its default
                // VK_NULL_HANDLE, not query a buffer nothing will read
                // from this frame.
                frameData.densityBuffer = voxelSystem.GetDensityBuffer(handle);
            }

            return frameData;
        }
    }
}
