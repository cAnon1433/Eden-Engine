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

                // Object-local AABB is NOT assumed symmetric around
                // TransformComponent::position - it's whatever
                // GetVolumeBounds actually reports, shifted into
                // object-local space. This used to be computed as
                // ±(fieldExtent*0.5), i.e. assuming the registered field
                // is exactly centered on the object's TransformComponent
                // position - which is only true when chunkDims/voxelSize
                // happen to add up to exactly 2x the shape's own
                // halfExtents+margin, with zero slack. They usually
                // don't (see e.g. ProcessWorkpieceSpawnInput's own fit-
                // check comment, which deliberately sizes the field with
                // slack to avoid a DIFFERENT bug - clipping), so this was
                // silently sampling the wrong region of the density field
                // for any box whose registered field wasn't perfectly
                // symmetric. This is the second half of the same bug
                // SpawnRaymarchBox's box-seeding math had (see that
                // function's own comment on the fix there) - fixing only
                // the seeding side left carving and rendering
                // disagreeing about where the box's center actually was,
                // instead of both being wrong together. GetVolumeBounds
                // (bounds.worldMin == VoxelVolumeDesc::origin exactly -
                // see that function's own implementation) is the one
                // already-correct source of truth every other consumer
                // (Carve, RaycastSurfaceSegments, collision) uses; this
                // now uses the same one instead of re-deriving its own,
                // symmetry-assuming answer.
                Eden::VoxelSystemGPU::VolumeBounds bounds = voxelSystem.GetVolumeBounds(handle);
                obj.localMin = bounds.worldMin - transform.position;
                obj.localMax = bounds.worldMax - transform.position;
                obj.fieldOffset = transform.position - bounds.worldMin; // see raymarch.frag's SampleSignedDistanceLocal: fieldPos = localPos + fieldOffset must equal worldPos - desc.origin
                glm::ivec3 sampleDims = voxelSystem.GetSampleDims(handle);
                float voxelSize = voxelSystem.GetVoxelSize(handle);
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
