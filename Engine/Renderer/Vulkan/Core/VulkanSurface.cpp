#include "VulkanSurface.h"
#include "VulkanInstance.h"

#include <GLFW/glfw3.h>
#include <stdexcept>

namespace Eden
{
    void VulkanSurface::Init(const VulkanInstance& instance, GLFWwindow* window)
    {
        m_InstanceHandle = instance.Get();

        if (glfwCreateWindowSurface(m_InstanceHandle, window, nullptr, &m_Surface) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to create window surface");
        }
    }

    void VulkanSurface::Shutdown()
    {
        if (m_Surface != VK_NULL_HANDLE)
        {
            vkDestroySurfaceKHR(m_InstanceHandle, m_Surface, nullptr);
            m_Surface = VK_NULL_HANDLE;
        }
    }

    VulkanSurface::~VulkanSurface()
    {
        Shutdown();
    }
}
