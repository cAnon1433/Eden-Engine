#include "VulkanSampler.h"

#include <stdexcept>

namespace Eden
{
    void VulkanSampler::Init(VkDevice device, VkFilter filter, VkSamplerAddressMode addressMode)
    {
        m_DeviceHandle = device;

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = filter;
        samplerInfo.minFilter = filter;
        samplerInfo.addressModeU = addressMode;
        samplerInfo.addressModeV = addressMode;
        samplerInfo.addressModeW = addressMode;

        // Anisotropic filtering deliberately left off - it needs a
        // physical-device feature query (and enabling it in
        // VulkanDevice's logical device creation) to use safely, which
        // isn't wired up yet. Textures still look correct without it,
        // just slightly softer at oblique viewing angles. Worth revisiting
        // once texture quality actually matters more than "textures work
        // at all".
        samplerInfo.anisotropyEnable = VK_FALSE;
        samplerInfo.maxAnisotropy = 1.0f;

        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;

        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;

        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f; // single mip level for now - see VulkanTexture

        if (vkCreateSampler(device, &samplerInfo, nullptr, &m_Sampler) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to create sampler");
        }
    }

    void VulkanSampler::Shutdown()
    {
        if (m_Sampler != VK_NULL_HANDLE)
        {
            vkDestroySampler(m_DeviceHandle, m_Sampler, nullptr);
            m_Sampler = VK_NULL_HANDLE;
        }
    }

    VulkanSampler::~VulkanSampler()
    {
        Shutdown();
    }
}
