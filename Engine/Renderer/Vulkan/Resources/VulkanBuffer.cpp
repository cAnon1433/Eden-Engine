#include "VulkanBuffer.h"

#include <stdexcept>
#include <cstring>

namespace Eden
{
    void VulkanBuffer::Init(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage,
                             VmaAllocationCreateFlags extraAllocFlags, void** outMappedPtr)
    {
        m_AllocatorHandle = allocator;
        m_Size = size;

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = memoryUsage;
        allocInfo.flags = extraAllocFlags;

        VmaAllocationInfo allocResult{};
        if (vmaCreateBuffer(m_AllocatorHandle, &bufferInfo, &allocInfo, &m_Buffer, &m_Allocation,
                             outMappedPtr ? &allocResult : nullptr) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to create buffer");
        }

        if (outMappedPtr)
        {
            *outMappedPtr = allocResult.pMappedData;
        }
    }

    void VulkanBuffer::InitDeviceLocalWithData(
        VmaAllocator allocator,
        VkDevice device,
        VkCommandPool commandPool,
        VkQueue graphicsQueue,
        const void* data,
        VkDeviceSize size,
        VkBufferUsageFlags extraUsage)
    {
        m_AllocatorHandle = allocator;
        m_Size = size;

        // --- Staging buffer: host-visible, CPU writes straight into it ---
        VkBuffer stagingBuffer;
        VmaAllocation stagingAllocation;

        VkBufferCreateInfo stagingInfo{};
        stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        stagingInfo.size = size;
        stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo stagingAllocInfo{};
        stagingAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                  VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo stagingInfoOut{};
        if (vmaCreateBuffer(allocator, &stagingInfo, &stagingAllocInfo, &stagingBuffer, &stagingAllocation, &stagingInfoOut) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to create staging buffer");
        }

        std::memcpy(stagingInfoOut.pMappedData, data, static_cast<size_t>(size));

        // --- Destination buffer: device-local, fast for the GPU to read from every frame ---
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | extraUsage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

        if (vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &m_Buffer, &m_Allocation, nullptr) != VK_SUCCESS)
        {
            vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);
            throw std::runtime_error("Eden: failed to create device-local buffer");
        }

        // --- One-time command buffer to copy staging -> device-local ---
        VkCommandBufferAllocateInfo cmdAllocInfo{};
        cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAllocInfo.commandPool = commandPool;
        cmdAllocInfo.commandBufferCount = 1;

        VkCommandBuffer cmd;
        vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmd);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(cmd, stagingBuffer, m_Buffer, 1, &copyRegion);

        vkEndCommandBuffer(cmd);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;

        // Simple and correct, not fast: block until the copy finishes rather
        // than juggling a fence. Fine for one-off uploads at load time; if
        // you're doing this every frame later, switch to a fence + reused
        // command buffer instead of a full queue wait.
        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &cmd);
        vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);
    }

    void VulkanBuffer::Shutdown()
    {
        if (m_Buffer != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(m_AllocatorHandle, m_Buffer, m_Allocation);
            m_Buffer = VK_NULL_HANDLE;
            m_Allocation = VK_NULL_HANDLE;
        }
    }

    VulkanBuffer::VulkanBuffer(VulkanBuffer&& other) noexcept
        : m_AllocatorHandle(other.m_AllocatorHandle)
        , m_Buffer(other.m_Buffer)
        , m_Allocation(other.m_Allocation)
        , m_Size(other.m_Size)
    {
        other.m_Buffer = VK_NULL_HANDLE;
        other.m_Allocation = VK_NULL_HANDLE;
    }

    VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&& other) noexcept
    {
        if (this != &other)
        {
            Shutdown();
            m_AllocatorHandle = other.m_AllocatorHandle;
            m_Buffer = other.m_Buffer;
            m_Allocation = other.m_Allocation;
            m_Size = other.m_Size;
            other.m_Buffer = VK_NULL_HANDLE;
            other.m_Allocation = VK_NULL_HANDLE;
        }
        return *this;
    }

    VulkanBuffer::~VulkanBuffer()
    {
        Shutdown();
    }
}
