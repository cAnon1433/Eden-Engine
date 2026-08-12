#include "VulkanPipelineLayout.h"

#include <stdexcept>

namespace Eden
{
    void VulkanPipelineLayout::Init(VkDevice device, const std::vector<VkDescriptorSetLayout>& descriptorSetLayouts)
    {
        m_DeviceHandle = device;

        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
        layoutInfo.pSetLayouts = descriptorSetLayouts.empty() ? nullptr : descriptorSetLayouts.data();

        layoutInfo.pushConstantRangeCount = 0;
        layoutInfo.pPushConstantRanges = nullptr;

        if (vkCreatePipelineLayout(m_DeviceHandle, &layoutInfo, nullptr, &m_Layout) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to create pipeline layout");
        }
    }

    void VulkanPipelineLayout::Shutdown()
    {
        if (m_Layout != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(m_DeviceHandle, m_Layout, nullptr);
            m_Layout = VK_NULL_HANDLE;
        }
    }

    VulkanPipelineLayout::~VulkanPipelineLayout()
    {
        Shutdown();
    }
}
