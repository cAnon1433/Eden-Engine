#pragma once

#include <vulkan/vulkan.h>
#include <vector>

namespace Eden
{
    class VulkanPipelineLayout
    {
    public:
        VulkanPipelineLayout() = default;
        ~VulkanPipelineLayout();

        VulkanPipelineLayout(const VulkanPipelineLayout&) = delete;
        VulkanPipelineLayout& operator=(const VulkanPipelineLayout&) = delete;

        // descriptorSetLayouts[i] becomes Vulkan set number i - so index 0
        // must be the camera UBO layout (VulkanDescriptorSetLayout, bound
        // once per frame) and index 1 must be the texture layout
        // (VulkanTextureSetLayout, bound once per mesh in
        // Mesh::DrawInstanced). No push constant range - per-object data
        // (model matrix, color override) lives in the per-instance vertex
        // buffer for instanced rendering (see RendererTypes.h InstanceData,
        // Frame/FrameContext.h). A push constant range is the wrong shape
        // for "N objects, one draw call" since it only ever carries one
        // object's worth of data.
        void Init(VkDevice device, const std::vector<VkDescriptorSetLayout>& descriptorSetLayouts);
        void Shutdown();

        VkPipelineLayout Get() const { return m_Layout; }

    private:
        VkDevice m_DeviceHandle = VK_NULL_HANDLE;
        VkPipelineLayout m_Layout = VK_NULL_HANDLE;
    };
}
