#include "VulkanDescriptorSetLayout.h"

#include <stdexcept>

namespace Eden
{
    void VulkanDescriptorSetLayout::Init(VkDevice device)
    {
        m_DeviceHandle = device;

        VkDescriptorSetLayoutBinding uboBinding{};
        uboBinding.binding = 0;
        uboBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboBinding.descriptorCount = 1;
        uboBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &uboBinding;

        if (vkCreateDescriptorSetLayout(m_DeviceHandle, &layoutInfo, nullptr, &m_Layout) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to create descriptor set layout");
        }
    }

    void VulkanDescriptorSetLayout::Shutdown()
    {
        if (m_Layout != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(m_DeviceHandle, m_Layout, nullptr);
            m_Layout = VK_NULL_HANDLE;
        }
    }

    VulkanDescriptorSetLayout::~VulkanDescriptorSetLayout()
    {
        Shutdown();
    }
}
