#include "VulkanFence.h"

#include <stdexcept>
#include <limits>

namespace Eden
{
    void VulkanFence::Init(VkDevice device, bool signaled)
    {
        m_DeviceHandle = device;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = signaled ? VK_FENCE_CREATE_SIGNALED_BIT : 0;

        if (vkCreateFence(m_DeviceHandle, &fenceInfo, nullptr, &m_Fence) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to create fence");
        }
    }

    void VulkanFence::Wait() const
    {
        vkWaitForFences(m_DeviceHandle, 1, &m_Fence, VK_TRUE, std::numeric_limits<uint64_t>::max());
    }

    void VulkanFence::Reset() const
    {
        vkResetFences(m_DeviceHandle, 1, &m_Fence);
    }

    void VulkanFence::Shutdown()
    {
        if (m_Fence != VK_NULL_HANDLE)
        {
            vkDestroyFence(m_DeviceHandle, m_Fence, nullptr);
            m_Fence = VK_NULL_HANDLE;
        }
    }

    VulkanFence::~VulkanFence()
    {
        Shutdown();
    }
}
