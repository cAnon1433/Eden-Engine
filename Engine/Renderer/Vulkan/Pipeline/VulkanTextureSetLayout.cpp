#include "VulkanTextureSetLayout.h"

#include <stdexcept>
#include <string>

namespace Eden
{
    void VulkanTextureSetLayout::Init(VkDevice device)
    {
        m_DeviceHandle = device;

        VkDescriptorSetLayoutBinding samplerBinding{};
        samplerBinding.binding = 0;
        samplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerBinding.descriptorCount = 1;
        samplerBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &samplerBinding;

        VkResult result = vkCreateDescriptorSetLayout(m_DeviceHandle, &layoutInfo, nullptr, &m_Layout);
        if (result != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to create texture descriptor set layout (VkResult " + std::to_string(result) + ")");
        }
    }

    void VulkanTextureSetLayout::Shutdown()
    {
        if (m_Layout != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(m_DeviceHandle, m_Layout, nullptr);
            m_Layout = VK_NULL_HANDLE;
        }
    }

    VulkanTextureSetLayout::~VulkanTextureSetLayout()
    {
        Shutdown();
    }
}
