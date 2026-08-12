#include "VulkanCommandBuffer.h"

#include <stdexcept>

namespace Eden
{
    void VulkanCommandBuffer::Allocate(VkDevice device, VkCommandPool pool)
    {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = pool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(device, &allocInfo, &m_CommandBuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to allocate command buffer");
        }
    }

    void VulkanCommandBuffer::Begin() const
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(m_CommandBuffer, &beginInfo) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to begin recording command buffer");
        }
    }

    void VulkanCommandBuffer::End() const
    {
        if (vkEndCommandBuffer(m_CommandBuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to end recording command buffer");
        }
    }

    void VulkanCommandBuffer::Reset() const
    {
        vkResetCommandBuffer(m_CommandBuffer, 0);
    }
}
