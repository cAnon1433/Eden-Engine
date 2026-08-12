#pragma once

#include <vulkan/vulkan.h>

namespace Eden
{
    class VulkanInstance;

    // Routes Vulkan validation layer messages to stderr.
    // No-op if validation layers are disabled on the owning instance.
    class VulkanDebugMessenger
    {
    public:
        VulkanDebugMessenger() = default;
        ~VulkanDebugMessenger();

        VulkanDebugMessenger(const VulkanDebugMessenger&) = delete;
        VulkanDebugMessenger& operator=(const VulkanDebugMessenger&) = delete;

        void Init(const VulkanInstance& instance);
        void Shutdown();

        static void PopulateCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);

    private:
        VkInstance m_InstanceHandle = VK_NULL_HANDLE;
        VkDebugUtilsMessengerEXT m_Messenger = VK_NULL_HANDLE;
    };
}
