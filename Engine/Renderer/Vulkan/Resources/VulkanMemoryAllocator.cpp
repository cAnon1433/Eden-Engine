// VMA's implementation is generated exactly once, in this translation unit.
#define VMA_IMPLEMENTATION
#include "VulkanMemoryAllocator.h"

#include <stdexcept>

namespace Eden
{
    void VulkanMemoryAllocator::Init(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device)
    {
        VmaAllocatorCreateInfo allocatorInfo{};
        allocatorInfo.instance = instance;
        allocatorInfo.physicalDevice = physicalDevice;
        allocatorInfo.device = device;
        allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_2;

        if (vmaCreateAllocator(&allocatorInfo, &m_Allocator) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to create VMA allocator");
        }
    }

    void VulkanMemoryAllocator::Shutdown()
    {
        if (m_Allocator != VK_NULL_HANDLE)
        {
            vmaDestroyAllocator(m_Allocator);
            m_Allocator = VK_NULL_HANDLE;
        }
    }

    VulkanMemoryAllocator::~VulkanMemoryAllocator()
    {
        Shutdown();
    }
}
