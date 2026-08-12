#pragma once

#include <vulkan/vulkan.h>
#include "VulkanMemoryAllocator.h"

namespace Eden
{
    // Generic VMA-backed buffer. Covers vertex, index, uniform, and storage
    // buffers - the difference between them is just the usage flags and
    // memory type you pass in, not the class.
    class VulkanBuffer
    {
    public:
        VulkanBuffer() = default;
        ~VulkanBuffer();

        VulkanBuffer(const VulkanBuffer&) = delete;
        VulkanBuffer& operator=(const VulkanBuffer&) = delete;
        VulkanBuffer(VulkanBuffer&& other) noexcept;
        VulkanBuffer& operator=(VulkanBuffer&& other) noexcept;

        // Raw allocation - no data upload. Use this directly for buffers you
        // intend to map yourself (e.g. host-visible uniform buffers).
        // Pass extraAllocFlags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
        // VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT and a
        // non-null outMappedPtr to get a persistent CPU-writable pointer
        // back (used by the per-frame camera uniform buffer).
        void Init(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage,
                  VmaAllocationCreateFlags extraAllocFlags = 0, void** outMappedPtr = nullptr);
        void Shutdown();

        // Creates a DEVICE_LOCAL buffer with `extraUsage` (e.g.
        // VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) and uploads `data` to it via a
        // temporary HOST_VISIBLE staging buffer + one-time command buffer
        // copy. This is the pattern you want for vertex/index data that's
        // set once and drawn many times - GPU-local memory is the fast path
        // for repeated reads, staging is just how the data gets there.
        void InitDeviceLocalWithData(
            VmaAllocator allocator,
            VkDevice device,
            VkCommandPool commandPool,
            VkQueue graphicsQueue,
            const void* data,
            VkDeviceSize size,
            VkBufferUsageFlags extraUsage);

        VkBuffer Get() const { return m_Buffer; }
        VkDeviceSize GetSize() const { return m_Size; }

    private:
        VmaAllocator m_AllocatorHandle = VK_NULL_HANDLE;
        VkBuffer m_Buffer = VK_NULL_HANDLE;
        VmaAllocation m_Allocation = VK_NULL_HANDLE;
        VkDeviceSize m_Size = 0;
    };
}
