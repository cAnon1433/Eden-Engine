#include "Mesh.h"

#include <glm/glm.hpp>

namespace Eden
{
    namespace
    {
        // Farthest vertex distance from the local origin - see
        // Mesh::GetBoundingRadius's comment for why origin-centered is
        // the assumption, not a computed centroid.
        float ComputeBoundingRadius(const std::vector<Vertex>& vertices)
        {
            float maxDistSq = 0.0f;
            for (const Vertex& v : vertices)
            {
                float distSq = glm::dot(v.position, v.position);
                if (distSq > maxDistSq)
                {
                    maxDistSq = distSq;
                }
            }
            return glm::sqrt(maxDistSq);
        }
    }

    void Mesh::Create(VmaAllocator allocator, VkDevice device, VkCommandPool commandPool,
                       VkQueue graphicsQueue, const std::vector<Vertex>& vertices,
                       VkDescriptorSet textureDescriptorSet)
    {
        m_VertexCount = static_cast<uint32_t>(vertices.size());
        m_UseIndices = false;
        m_TextureDescriptorSet = textureDescriptorSet;
        m_BoundingRadius = ComputeBoundingRadius(vertices);

        VkDeviceSize bufferSize = sizeof(Vertex) * vertices.size();
        m_VertexBuffer.InitDeviceLocalWithData(
            allocator, device, commandPool, graphicsQueue,
            vertices.data(), bufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    }

    void Mesh::CreateIndexed(VmaAllocator allocator, VkDevice device, VkCommandPool commandPool,
                              VkQueue graphicsQueue, const std::vector<Vertex>& vertices,
                              const std::vector<uint32_t>& indices, VkDescriptorSet textureDescriptorSet)
    {
        m_VertexCount = static_cast<uint32_t>(vertices.size());
        m_IndexCount = static_cast<uint32_t>(indices.size());
        m_UseIndices = true;
        m_TextureDescriptorSet = textureDescriptorSet;
        m_BoundingRadius = ComputeBoundingRadius(vertices);

        VkDeviceSize vertexBufferSize = sizeof(Vertex) * vertices.size();
        m_VertexBuffer.InitDeviceLocalWithData(
            allocator, device, commandPool, graphicsQueue,
            vertices.data(), vertexBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

        VkDeviceSize indexBufferSize = sizeof(uint32_t) * indices.size();
        m_IndexBuffer.InitDeviceLocalWithData(
            allocator, device, commandPool, graphicsQueue,
            indices.data(), indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    }

    void Mesh::DrawInstanced(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout,
                              VkBuffer instanceBuffer, VkDeviceSize instanceBufferOffset,
                              uint32_t instanceCount) const
    {
        // Set 1 = this mesh's texture. Set 0 (camera UBO) was already
        // bound once for the whole frame before any mesh draws happen -
        // see Renderer::DrawFrame - so only set 1 needs rebinding here,
        // per mesh.
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                                 1, 1, &m_TextureDescriptorSet, 0, nullptr);

        // Binding 0 = this mesh's own geometry (advances per-vertex),
        // binding 1 = the caller's instance data slice (advances
        // per-instance). One vkCmdBindVertexBuffers call, two bindings -
        // see InstanceData::GetBindingDescription in RendererTypes.h for
        // why they're numbered this way.
        VkBuffer vertexBuffers[] = { m_VertexBuffer.Get(), instanceBuffer };
        VkDeviceSize offsets[] = { 0, instanceBufferOffset };
        vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, offsets);

        if (m_UseIndices)
        {
            vkCmdBindIndexBuffer(commandBuffer, m_IndexBuffer.Get(), 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(commandBuffer, m_IndexCount, instanceCount, 0, 0, 0);
        }
        else
        {
            vkCmdDraw(commandBuffer, m_VertexCount, instanceCount, 0, 0);
        }
    }

    void Mesh::Destroy()
    {
        m_VertexBuffer.Shutdown();
        m_IndexBuffer.Shutdown();
        m_VertexCount = 0;
        m_IndexCount = 0;
        m_UseIndices = false;
    }

    Mesh::~Mesh()
    {
        Destroy();
    }
}
