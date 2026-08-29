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
        // Vertex AND fragment - originally vertex-only, which was
        // already technically inaccurate (triangle.frag has always read
        // camera.lightDirection/lightColor/ambientColor/cameraPosition
        // in the fragment stage - see that shader), just not one
        // validation happened to catch until the raymarch pipeline
        // (raymarch.vert declares no CameraUBO at all - it doesn't need
        // one, see that shader's own comment - so raymarch.frag is the
        // ONLY stage in that particular pipeline reading this UBO,
        // which made the fragment-stage gap in this binding
        // unambiguous and validation finally flagged it). Every
        // pipeline sharing this same VkDescriptorSetLayout object
        // (m_PipelineLayout, m_RaymarchPipelineLayout, and any future
        // one) needs both bits present regardless of whether that
        // specific pipeline's fragment shader happens to touch camera,
        // since the layout's stageFlags are a static property of the
        // shared object, not scoped per pipeline.
        uboBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

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
