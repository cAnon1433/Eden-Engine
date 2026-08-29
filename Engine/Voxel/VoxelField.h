#pragma once

#include "VoxelTypes.h"

#include <glm/glm.hpp>
#include <cstdint>

namespace Eden
{
    // Opaque handle into VoxelSystemGPU's volume registry - same
    // reasoning as MeshHandle/TextureHandle (RendererTypes.h): an index,
    // not a pointer, stable across container growth, trivially copyable
    // into an ECS component.
    //
    // Generational, packed exactly like Entity (Entity.h) - low 32 bits
    // are a slot index into VoxelSystemGPU's internal storage, high 32
    // bits are a generation counter for that slot. This is NOT
    // optional/defensive-only: UnregisterVolume actually reuses freed
    // slots now (real free-list, see VoxelSystemGPU.cpp), so without a
    // generation, a stale handle held after its volume was destroyed
    // and a NEW volume registered into the same slot would silently
    // resolve to that unrelated new volume - the exact bug Entity.h's
    // own comment describes for ECS entities. Kept as a plain uint64_t
    // rather than a wrapper struct for the same reason Entity is: usable
    // everywhere the old raw-size_t handle was (unordered_map keys via
    // default std::hash<uint64_t>, == comparisons, vector<VoxelVolumeHandle>)
    // with zero call-site changes outside VoxelField.h/VoxelSystemGPU.cpp
    // - confirmed nothing else in the codebase treats this value as a
    // raw array index.
    using VoxelVolumeHandle = uint64_t;
    using VoxelVolumeIndex = uint32_t;
    using VoxelVolumeGeneration = uint32_t;

    // index = 0xFFFFFFFF so it can never collide with a real slot index
    // (VoxelSystemGPU's slot count is bounded by kVoxelMaxVolumes, far
    // below that) regardless of what generation a real slot is on.
    constexpr VoxelVolumeHandle InvalidVoxelVolumeHandle = static_cast<VoxelVolumeHandle>(-1);

    constexpr VoxelVolumeHandle MakeVoxelVolumeHandle(VoxelVolumeIndex index, VoxelVolumeGeneration generation)
    {
        return (static_cast<VoxelVolumeHandle>(generation) << 32) | static_cast<VoxelVolumeHandle>(index);
    }

    constexpr VoxelVolumeIndex GetVoxelVolumeIndex(VoxelVolumeHandle handle)
    {
        return static_cast<VoxelVolumeIndex>(handle & 0xFFFFFFFFu);
    }

    constexpr VoxelVolumeGeneration GetVoxelVolumeGeneration(VoxelVolumeHandle handle)
    {
        return static_cast<VoxelVolumeGeneration>(handle >> 32);
    }

    // What the caller specifies when registering a new deformable
    // volume - VoxelSystemGPU derives everything else (sample counts,
    // buffer sizes, chunk count) from this. chunkDims is chunks per
    // axis, NOT voxels per axis - total voxel resolution along X is
    // chunkDims.x * kVoxelChunkSize.
    //
    // Kept deliberately modest as a default (see main.cpp's spawn call)
    // - a 2x2x2-chunk volume is 16^3 voxels, small enough to smoke-test
    // the whole seed->march->indirect-draw pipeline without the ~20MB/
    // volume worst-case vertex buffer a larger grid implies (see
    // kVoxelMaxVerticesPerChunk's comment in VoxelTypes.h). Raise
    // chunkDims once this is proven correct and you want more resolved
    // shapes.
    struct VoxelVolumeDesc
    {
        glm::vec3 origin{ 0.0f };        // world-space min corner
        float voxelSize = 0.1f;          // world units per voxel edge
        glm::ivec3 chunkDims{ 2, 2, 2 }; // chunks per axis

        glm::ivec3 VoxelDims() const
        {
            return chunkDims * static_cast<int32_t>(kVoxelChunkSize);
        }

        // +1: chunks share their boundary sample row/column/layer with
        // their neighbor rather than each owning an independent,
        // disjoint set of samples - this is what makes adjacent chunks'
        // marched surfaces line up with no seam, at the cost of storing
        // one shared field per object instead of one independent field
        // per chunk. See VoxelSystemGPU.h's class comment.
        glm::ivec3 SampleDims() const
        {
            return VoxelDims() + glm::ivec3(1);
        }

        glm::ivec3 ChunkDims() const { return chunkDims; }

        uint32_t NumChunks() const
        {
            return static_cast<uint32_t>(chunkDims.x * chunkDims.y * chunkDims.z);
        }

        uint32_t NumSamples() const
        {
            glm::ivec3 s = SampleDims();
            return static_cast<uint32_t>(s.x * s.y * s.z);
        }
    };

    // Attach to an entity alongside TransformComponent to make it a
    // marching-cubes-rendered deformable volume, drawn via
    // Renderer's indirect-draw voxel path (see Renderer::DrawFrame's
    // voxelSources parameter) instead of the ordinary instanced-mesh
    // path - a VoxelVolumeComponent entity does NOT also need/use
    // MeshComponent, these are two different draw paths.
    //
    // TransformComponent's position is currently used only to place the
    // volume's origin at registration time (VoxelSystemGPU::RegisterVolume
    // bakes origin into the density field's world-space sampling, it
    // isn't re-read per frame) - a physics-driven deformable volume that
    // needs to actually MOVE after spawning is a real gap, not silently
    // supported. Fine for now (corrosion/carving is the near-term goal,
    // not moving deformables), worth revisiting alongside the physics
    // chunk-solid-flag work.
    struct VoxelVolumeComponent
    {
        VoxelVolumeHandle handle = InvalidVoxelVolumeHandle;
    };
}
