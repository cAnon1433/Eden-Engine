#pragma once

#include <vulkan/vulkan.h>

namespace Eden
{
    class VulkanCommandPool
    {
    public:
        VulkanCommandPool() = default;
        ~VulkanCommandPool();

        VulkanCommandPool(const VulkanCommandPool&) = delete;
        VulkanCommandPool& operator=(const VulkanCommandPool&) = delete;

        void Init(VkDevice device, uint32_t queueFamilyIndex);
        void Shutdown();

        VkCommandPool Get() const { return m_Pool; }

    private:
        VkDevice m_DeviceHandle = VK_NULL_HANDLE;
        VkCommandPool m_Pool = VK_NULL_HANDLE;
    };
}
