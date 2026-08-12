#pragma once

#include "VulkanBuffer.h"
#include "../RendererTypes.h"

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>

namespace Eden
{
    // A drawable chunk of geometry: a vertex buffer + optional index
    // buffer. Purely geometry now - no position. Placement moved to
    // TransformComponent (ECS) once that existed; a Mesh is a shared
    // resource multiple entities can point at via MeshHandle, so it can't
    // own "its" position the way it did pre-ECS.
    class Mesh
    {
    public:
        Mesh() = default;
        ~Mesh();

        Mesh(const Mesh&) = delete;
        Mesh& operator=(const Mesh&) = delete;
        Mesh(Mesh&&) noexcept = default;
        Mesh& operator=(Mesh&&) noexcept = default;

        // Non-indexed - vertices drawn directly, in order. textureDescriptorSet
        // is the mesh's texture (set 1, binding 0 - see VulkanTextureSetLayout),
        // bound automatically every time DrawInstanced is called. Pass
        // Renderer's default/fallback texture handle's descriptor set if
        // this mesh has no real texture - see Renderer::CreateMesh.
        void Create(VmaAllocator allocator, VkDevice device, VkCommandPool commandPool,
                    VkQueue graphicsQueue, const std::vector<Vertex>& vertices,
                    VkDescriptorSet textureDescriptorSet);

        // Indexed - vertices deduplicated via an index buffer. Use this for
        // anything with shared corners (cubes, any closed mesh).
        void CreateIndexed(VmaAllocator allocator, VkDevice device, VkCommandPool commandPool,
                            VkQueue graphicsQueue, const std::vector<Vertex>& vertices,
                            const std::vector<uint32_t>& indices, VkDescriptorSet textureDescriptorSet);

        void Destroy();

        // Binds this mesh's own vertex/index buffer at binding 0, binds a
        // slice of the caller-owned instance buffer at binding 1 (starting
        // at instanceBufferOffset bytes, which Renderer::DrawFrame computes
        // per mesh group), binds this mesh's texture descriptor set at
        // Vulkan set 1, and issues ONE draw call covering instanceCount
        // instances. pipelineLayout is needed again (it wasn't when this
        // only bound vertex buffers) because binding a descriptor set
        // requires the pipeline layout it was built against. Call once per
        // mesh (not once per entity!) after the pipeline and the camera
        // descriptor set (set 0) are bound.
        void DrawInstanced(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout,
                            VkBuffer instanceBuffer, VkDeviceSize instanceBufferOffset,
                            uint32_t instanceCount) const;

        // Radius of a bounding sphere centered on this mesh's own local
        // origin (NOT world space - a mesh is a shared resource with no
        // position of its own, see the class comment above; callers
        // combine this with an entity's TransformComponent to get a
        // world-space bounding sphere for culling). Computed once at
        // Create()/CreateIndexed() time as the farthest vertex distance
        // from the origin - exact for any mesh whose local origin is
        // already its natural center (true for MakeCube() and every mesh
        // authored/exported that way, which is the common case), a
        // conservative overestimate otherwise (still a valid, safe
        // bounding sphere, just not the tightest possible one - fine for
        // frustum culling, which only needs "definitely won't wrongly
        // cull something visible", not a tight fit).
        float GetBoundingRadius() const { return m_BoundingRadius; }

    private:
        VulkanBuffer m_VertexBuffer;
        VulkanBuffer m_IndexBuffer;
        uint32_t m_VertexCount = 0;
        uint32_t m_IndexCount = 0;
        bool m_UseIndices = false;
        float m_BoundingRadius = 0.0f;
        VkDescriptorSet m_TextureDescriptorSet = VK_NULL_HANDLE; // not owned - see VulkanTexture, owned by Renderer's texture registry
    };
}
