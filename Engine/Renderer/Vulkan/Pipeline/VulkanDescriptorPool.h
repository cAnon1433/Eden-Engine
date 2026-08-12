#pragma once

#include <vulkan/vulkan.h>
#include <vector>

namespace Eden
{
    // Generic descriptor pool: caller supplies exactly which descriptor
    // types and how many of each are needed (uniform buffers for the
    // per-frame camera set, combined image samplers for per-texture sets),
    // rather than this class hardcoding one type the way it used to when
    // only the camera UBO existed.
    class VulkanDescriptorPool
    {
    public:
        VulkanDescriptorPool() = default;
        ~VulkanDescriptorPool();

        VulkanDescriptorPool(const VulkanDescriptorPool&) = delete;
        VulkanDescriptorPool& operator=(const VulkanDescriptorPool&) = delete;

        void Init(VkDevice device, const std::vector<VkDescriptorPoolSize>& poolSizes, uint32_t maxSets);
        void Shutdown();

        VkDescriptorSet AllocateSet(VkDescriptorSetLayout layout);

        VkDescriptorPool Get() const { return m_Pool; }

    private:
        VkDevice m_DeviceHandle = VK_NULL_HANDLE;
        VkDescriptorPool m_Pool = VK_NULL_HANDLE;
    };
}
