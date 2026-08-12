#pragma once

#include <vulkan/vulkan.h>

struct GLFWwindow;

namespace Eden
{
    class VulkanInstance;

    class VulkanSurface
    {
    public:
        VulkanSurface() = default;
        ~VulkanSurface();

        VulkanSurface(const VulkanSurface&) = delete;
        VulkanSurface& operator=(const VulkanSurface&) = delete;

        void Init(const VulkanInstance& instance, GLFWwindow* window);
        void Shutdown();

        VkSurfaceKHR Get() const { return m_Surface; }

    private:
        VkInstance m_InstanceHandle = VK_NULL_HANDLE;
        VkSurfaceKHR m_Surface = VK_NULL_HANDLE;
    };
}
