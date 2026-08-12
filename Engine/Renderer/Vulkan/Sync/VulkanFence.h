#pragma once

#include <vulkan/vulkan.h>

namespace Eden
{
    class VulkanFence
    {
    public:
        VulkanFence() = default;
        ~VulkanFence();

        VulkanFence(const VulkanFence&) = delete;
        VulkanFence& operator=(const VulkanFence&) = delete;

        // signaled = true creates it already-signaled, which the first
        // frame needs so vkWaitForFences doesn't block forever on frame 0.
        void Init(VkDevice device, bool signaled = true);
        void Shutdown();

        void Wait() const;
        void Reset() const;

        VkFence Get() const { return m_Fence; }

    private:
        VkDevice m_DeviceHandle = VK_NULL_HANDLE;
        VkFence m_Fence = VK_NULL_HANDLE;
    };
}
