#include "VulkanDescriptorPool.h"

#include <stdexcept>
#include <string>

namespace Eden
{
    void VulkanDescriptorPool::Init(VkDevice device, const std::vector<VkDescriptorPoolSize>& poolSizes, uint32_t maxSets)
    {
        m_DeviceHandle = device;

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = maxSets;

        VkResult result = vkCreateDescriptorPool(m_DeviceHandle, &poolInfo, nullptr, &m_Pool);
        if (result != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to create descriptor pool (VkResult " + std::to_string(result) + ")");
        }
    }

    VkDescriptorSet VulkanDescriptorPool::AllocateSet(VkDescriptorSetLayout layout)
    {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = m_Pool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &layout;

        VkDescriptorSet set;
        VkResult result = vkAllocateDescriptorSets(m_DeviceHandle, &allocInfo, &set);
        if (result != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to allocate descriptor set (VkResult " + std::to_string(result) + ")");
        }
        return set;
    }

    void VulkanDescriptorPool::Shutdown()
    {
        if (m_Pool != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(m_DeviceHandle, m_Pool, nullptr);
            m_Pool = VK_NULL_HANDLE;
        }
    }

    VulkanDescriptorPool::~VulkanDescriptorPool()
    {
        Shutdown();
    }
}
