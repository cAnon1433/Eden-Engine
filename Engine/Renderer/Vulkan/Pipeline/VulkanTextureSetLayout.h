#pragma once

#include <vulkan/vulkan.h>

namespace Eden
{
    // Single binding: one combined image sampler, visible to the fragment
    // shader only. Deliberately a SEPARATE VkDescriptorSetLayout from
    // VulkanDescriptorSetLayout (the camera UBO, set 0) rather than adding
    // a binding to that one - the camera UBO set is allocated once PER
    // FRAME-IN-FLIGHT and bound once per frame, but a texture is a
    // property of a MESH and needs to change between draw calls within
    // the same frame as different meshes get drawn. Two separate sets
    // (0 = camera, bound once per frame; 1 = texture, bound once per
    // mesh) is how Vulkan expects that kind of "changes at different
    // frequencies" data to be split - see VulkanPipelineLayout, which
    // takes both, and Mesh::DrawInstanced, which binds set 1 itself.
    class VulkanTextureSetLayout
    {
    public:
        VulkanTextureSetLayout() = default;
        ~VulkanTextureSetLayout();

        VulkanTextureSetLayout(const VulkanTextureSetLayout&) = delete;
        VulkanTextureSetLayout& operator=(const VulkanTextureSetLayout&) = delete;

        void Init(VkDevice device);
        void Shutdown();

        VkDescriptorSetLayout Get() const { return m_Layout; }

    private:
        VkDevice m_DeviceHandle = VK_NULL_HANDLE;
        VkDescriptorSetLayout m_Layout = VK_NULL_HANDLE;
    };
}
