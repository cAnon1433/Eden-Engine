#pragma once

#include "../../Renderer/Vulkan/RendererTypes.h"

namespace Eden
{
    // References a mesh resource owned by Renderer's mesh registry (see
    // Renderer::CreateMesh / CreateCubeMesh). Multiple entities can - and
    // usually should - point at the same MeshHandle: geometry is shared,
    // only the TransformComponent differs per entity. This is exactly what
    // fixes the "1000 cubes = 1000 GPU uploads" problem from before ECS
    // existed.
    struct MeshComponent
    {
        MeshHandle handle = InvalidMeshHandle;
    };
}
