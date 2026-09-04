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
        //
        // `cullMode` defaults to VK_CULL_MODE_BACK_BIT, preserving every
        // existing call site's behavior unchanged. Pass VK_CULL_MODE_NONE
        // for meshes where a small missing/misordered patch of surface
        // (e.g. a marching-cubes ambiguous-case crack - see Voxel/
        // VoxelSystemGPU.cpp's notes on that open issue) should read as
        // "see the far interior wall through the gap" rather than "see
        // straight through to the skybox" - see Renderer::m_VoxelPipeline
        // for the one call site that does this today. This is a stopgap:
        // it papers over rare topology gaps by making backfaces visible
        // through them, it doesn't fix the gaps themselves.
        // depthTestEnable/depthWriteEnable default to true/true, matching
        // every call site's behavior before these parameters existed
        // (mesh, voxel, particle-point, raymarch pipelines all want depth
        // test+write ON - see VulkanGraphicsPipeline.cpp's depthStencil
        // block, previously hardcoded). Added for the fluid-surface blur
        // passes (see Renderer::m_FluidBlurPipeline), which render into a
        // render pass with NO depth attachment at all - passing false/false
        // there rather than relying on "an unused pDepthStencilState is
        // spec-ignored when the subpass has no depth attachment," which is
        // technically true per the Vulkan spec but not worth trusting
        // untested across MoltenVK/Windows drivers when an explicit,
        // zero-risk flag does the same thing.
        void Init(VkDevice device, VkRenderPass renderPass, VkPipelineLayout layout,
                  const std::string& vertSpirvPath, const std::string& fragSpirvPath,
                  const std::vector<VkVertexInputBindingDescription>& bindingDescriptions,
                  const std::vector<VkVertexInputAttributeDescription>& attributeDescriptions,
                  VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                  VkCullModeFlags cullMode = VK_CULL_MODE_BACK_BIT,
                  bool depthTestEnable = true,
                  bool depthWriteEnable = true);
        void Shutdown();

        VkPipeline Get() const { return m_Pipeline; }

    private:
        VkDevice m_DeviceHandle = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;
    };
}
