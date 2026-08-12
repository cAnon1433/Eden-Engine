#pragma once

#include <vulkan/vulkan.h>
#include <string>
#include <vector>

namespace Eden
{
    class VulkanGraphicsPipeline
    {
    public:
        VulkanGraphicsPipeline() = default;
        ~VulkanGraphicsPipeline();

        VulkanGraphicsPipeline(const VulkanGraphicsPipeline&) = delete;
        VulkanGraphicsPipeline& operator=(const VulkanGraphicsPipeline&) = delete;

        // Uses dynamic viewport/scissor state, so this doesn't need to be
        // rebuilt on window resize - only the swapchain/framebuffers do.
        // bindingDescriptions/attributeDescriptions describe every vertex
        // input stream being fed in via vkCmdBindVertexBuffers - binding 0
        // is per-vertex geometry (Vertex::GetBindingDescription), binding 1
        // is per-instance data (InstanceData::GetBindingDescription). See
        // RendererTypes.h.
        //
        // `topology` defaults to TRIANGLE_LIST (every existing call site's
        // actual behavior, unchanged) - POINT_LIST is used for the
        // particle-points pipeline (see Renderer::m_ParticlePointsPipeline)
        // and needs no other changes here: point rasterization still goes
        // through the exact same vertex/instance input, viewport/scissor,
        // and descriptor-set binding path, it just assembles differently.
        void Init(VkDevice device, VkRenderPass renderPass, VkPipelineLayout layout,
                  const std::string& vertSpirvPath, const std::string& fragSpirvPath,
                  const std::vector<VkVertexInputBindingDescription>& bindingDescriptions,
                  const std::vector<VkVertexInputAttributeDescription>& attributeDescriptions,
                  VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        void Shutdown();

        VkPipeline Get() const { return m_Pipeline; }

    private:
        VkDevice m_DeviceHandle = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;
    };
}
