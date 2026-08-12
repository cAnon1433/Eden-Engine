#include "VulkanCommandPool.h"

#include <stdexcept>

namespace Eden
{
    void VulkanCommandPool::Init(VkDevice device, uint32_t queueFamilyIndex)
    {
        m_DeviceHandle = device;

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndex;

        if (vkCreateCommandPool(m_DeviceHandle, &poolInfo, nullptr, &m_Pool) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to create command pool");
        }
    }

    void VulkanCommandPool::Shutdown()
    {
        if (m_Pool != VK_NULL_HANDLE)
        {
            vkDestroyCommandPool(m_DeviceHandle, m_Pool, nullptr);
            m_Pool = VK_NULL_HANDLE;
        }
    }

    VulkanCommandPool::~VulkanCommandPool()
    {
        Shutdown();
    }
}
