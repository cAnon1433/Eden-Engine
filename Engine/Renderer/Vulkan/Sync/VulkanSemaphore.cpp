#include "VulkanSemaphore.h"

#include <stdexcept>

namespace Eden
{
    void VulkanSemaphore::Init(VkDevice device)
    {
        m_DeviceHandle = device;

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        if (vkCreateSemaphore(m_DeviceHandle, &semaphoreInfo, nullptr, &m_Semaphore) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to create semaphore");
        }
    }

    void VulkanSemaphore::Shutdown()
    {
        if (m_Semaphore != VK_NULL_HANDLE)
        {
            vkDestroySemaphore(m_DeviceHandle, m_Semaphore, nullptr);
            m_Semaphore = VK_NULL_HANDLE;
        }
    }

    VulkanSemaphore::~VulkanSemaphore()
    {
        Shutdown();
    }
}
