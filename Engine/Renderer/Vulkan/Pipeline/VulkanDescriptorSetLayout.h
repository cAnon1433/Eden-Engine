#pragma once

#include <vulkan/vulkan.h>

namespace Eden
{
    // Single binding: one uniform buffer (the MVP matrices), visible to
    // the vertex shader only. Extend with more bindings (textures,
    // samplers) once VulkanTexture exists.
    class VulkanDescriptorSetLayout
    {
    public:
        VulkanDescriptorSetLayout() = default;
        ~VulkanDescriptorSetLayout();

        VulkanDescriptorSetLayout(const VulkanDescriptorSetLayout&) = delete;
        VulkanDescriptorSetLayout& operator=(const VulkanDescriptorSetLayout&) = delete;

        void Init(VkDevice device);
        void Shutdown();

        VkDescriptorSetLayout Get() const { return m_Layout; }

    private:
        VkDevice m_DeviceHandle = VK_NULL_HANDLE;
        VkDescriptorSetLayout m_Layout = VK_NULL_HANDLE;
    };
}
