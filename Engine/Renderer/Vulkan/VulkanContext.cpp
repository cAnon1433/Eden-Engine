#include "VulkanContext.h"

namespace Eden
{
    void VulkanContext::Init(GLFWwindow* window, const std::string& appName, bool enableValidation)
    {
        m_Instance.Init(appName, enableValidation);
        m_DebugMessenger.Init(m_Instance);
        m_Surface.Init(m_Instance, window);
        m_PhysicalDevice.Init(m_Instance.Get(), m_Surface.Get());
        m_Device.Init(m_PhysicalDevice, enableValidation);
        m_Swapchain.Init(m_PhysicalDevice, m_Device, m_Surface.Get(), window);
    }

    void VulkanContext::RecreateSwapchain(GLFWwindow* window)
    {
        m_Swapchain.Recreate(m_PhysicalDevice, m_Surface.Get(), window);
    }

    void VulkanContext::Shutdown()
    {
        // Order matters: tear down in reverse of creation.
        m_Swapchain.Shutdown();
        m_Device.Shutdown();
        m_Surface.Shutdown();
        m_DebugMessenger.Shutdown();
        m_Instance.Shutdown();
    }
}
