#pragma once

#include <vulkan/vulkan.h>

namespace Eden
{
    class VulkanPhysicalDevice;

    class VulkanDevice
    {
    public:
        VulkanDevice() = default;
        ~VulkanDevice();

        VulkanDevice(const VulkanDevice&) = delete;
        VulkanDevice& operator=(const VulkanDevice&) = delete;

        void Init(const VulkanPhysicalDevice& physicalDevice, bool enableValidation);
        void Shutdown();

        VkDevice Get() const { return m_Device; }
        VkQueue GetGraphicsQueue() const { return m_GraphicsQueue; }
        VkQueue GetPresentQueue() const { return m_PresentQueue; }

        void WaitIdle() const;

    private:
        VkDevice m_Device = VK_NULL_HANDLE;
        VkQueue m_GraphicsQueue = VK_NULL_HANDLE;
        VkQueue m_PresentQueue = VK_NULL_HANDLE;
    };
}
