#pragma once

#include "Core/VulkanInstance.h"
#include "Core/VulkanDebugMessenger.h"
#include "Core/VulkanSurface.h"
#include "Core/VulkanPhysicalDevice.h"
#include "Core/VulkanDevice.h"
#include "Core/VulkanSwapchain.h"

struct GLFWwindow;

namespace Eden
{
    // Owns the Vulkan objects that exist for the lifetime of the app and
    // don't belong to any particular render pass/pipeline: instance, device,
    // surface, swapchain. Renderer builds render-pass-specific things on top.
    class VulkanContext
    {
    public:
        void Init(GLFWwindow* window, const std::string& appName, bool enableValidation);
        void Shutdown();

        void RecreateSwapchain(GLFWwindow* window);

        VulkanInstance& Instance() { return m_Instance; }
        VulkanDevice& Device() { return m_Device; }
        VulkanPhysicalDevice& PhysicalDevice() { return m_PhysicalDevice; }
        VulkanSwapchain& Swapchain() { return m_Swapchain; }
        VkSurfaceKHR Surface() const { return m_Surface.Get(); }

    private:
        VulkanInstance m_Instance;
        VulkanDebugMessenger m_DebugMessenger;
        VulkanSurface m_Surface;
        VulkanPhysicalDevice m_PhysicalDevice;
        VulkanDevice m_Device;
        VulkanSwapchain m_Swapchain;
    };
}
