#pragma once

#include <vulkan/vulkan.h>

namespace Eden
{
    class VulkanSemaphore
    {
    public:
        VulkanSemaphore() = default;
        ~VulkanSemaphore();

        VulkanSemaphore(const VulkanSemaphore&) = delete;
        VulkanSemaphore& operator=(const VulkanSemaphore&) = delete;

        void Init(VkDevice device);
        void Shutdown();

        VkSemaphore Get() const { return m_Semaphore; }

    private:
        VkDevice m_DeviceHandle = VK_NULL_HANDLE;
        VkSemaphore m_Semaphore = VK_NULL_HANDLE;
    };
}
