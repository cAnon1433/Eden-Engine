#pragma once

#include <vulkan/vulkan.h>
#include "../../../../ThirdParty/vma/vk_mem_alloc.h"

namespace Eden
{
    // Thin wrapper around a VmaAllocator. Everything under Resources/
    // (VulkanBuffer, eventually VulkanImage) allocates through this instead
    // of calling vkAllocateMemory directly.
    class VulkanMemoryAllocator
    {
    public:
        VulkanMemoryAllocator() = default;
        ~VulkanMemoryAllocator();

        VulkanMemoryAllocator(const VulkanMemoryAllocator&) = delete;
        VulkanMemoryAllocator& operator=(const VulkanMemoryAllocator&) = delete;

        void Init(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device);
        void Shutdown();

        VmaAllocator Get() const { return m_Allocator; }

    private:
        VmaAllocator m_Allocator = VK_NULL_HANDLE;
    };
}
