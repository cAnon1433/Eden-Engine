#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <optional>

namespace Eden
{
    struct QueueFamilyIndices
    {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;

        bool IsComplete() const
        {
            return graphicsFamily.has_value() && presentFamily.has_value();
        }
    };

    struct SwapchainSupportDetails
    {
        VkSurfaceCapabilitiesKHR capabilities{};
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };

    class VulkanPhysicalDevice
    {
    public:
        void Init(VkInstance instance, VkSurfaceKHR surface);

        VkPhysicalDevice Get() const { return m_PhysicalDevice; }
        const QueueFamilyIndices& GetQueueFamilies() const { return m_QueueFamilies; }

        SwapchainSupportDetails QuerySwapchainSupport(VkSurfaceKHR surface) const;

        static const std::vector<const char*>& RequiredDeviceExtensions();

    private:
        bool IsDeviceSuitable(VkPhysicalDevice device, VkSurfaceKHR surface, QueueFamilyIndices& outIndices) const;
        bool CheckDeviceExtensionSupport(VkPhysicalDevice device) const;
        QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface) const;

    private:
        VkPhysicalDevice m_PhysicalDevice = VK_NULL_HANDLE;
        QueueFamilyIndices m_QueueFamilies;
    };
}
