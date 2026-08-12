#pragma once

#include <vulkan/vulkan.h>

namespace Eden
{
    // Color attachment (swapchain image) + depth attachment, one subpass.
    class VulkanRenderPass
    {
    public:
        VulkanRenderPass() = default;
        ~VulkanRenderPass();

        VulkanRenderPass(const VulkanRenderPass&) = delete;
        VulkanRenderPass& operator=(const VulkanRenderPass&) = delete;

        void Init(VkDevice device, VkFormat colorFormat, VkFormat depthFormat);
        void Shutdown();

        VkRenderPass Get() const { return m_RenderPass; }

    private:
        VkDevice m_DeviceHandle = VK_NULL_HANDLE;
        VkRenderPass m_RenderPass = VK_NULL_HANDLE;
    };
}
