#pragma once

#include <vulkan/vulkan.h>

namespace Eden
{
    // Sampler state: how a shader reads a texture's pixels between texel
    // centers (filtering) and past its edges (address mode). Separate from
    // VulkanImage/VulkanTexture on purpose - a sampler has no pixel data of
    // its own, it's just a small config block, and in a bigger project
    // multiple textures could legitimately share one sampler instance.
    // Each VulkanTexture owns its own for now (see VulkanTexture.h) rather
    // than pulling in a shared-sampler cache, which is a real optimization
    // but not one "basics" needs yet.
    class VulkanSampler
    {
    public:
        VulkanSampler() = default;
        ~VulkanSampler();

        VulkanSampler(const VulkanSampler&) = delete;
        VulkanSampler& operator=(const VulkanSampler&) = delete;

        // filter: LINEAR for smooth/photographic textures, NEAREST for
        // pixel-art (keeps hard pixel edges instead of blurring them).
        // addressMode: REPEAT tiles past 0..1 UV range, CLAMP_TO_EDGE holds
        // the edge pixel instead - REPEAT is the sane default for most
        // textures, CLAMP matters more for things like UI atlases where
        // bleeding past the edge would show a neighboring sprite.
        void Init(VkDevice device,
                  VkFilter filter = VK_FILTER_LINEAR,
                  VkSamplerAddressMode addressMode = VK_SAMPLER_ADDRESS_MODE_REPEAT);
        void Shutdown();

        VkSampler Get() const { return m_Sampler; }

    private:
        VkDevice m_DeviceHandle = VK_NULL_HANDLE;
        VkSampler m_Sampler = VK_NULL_HANDLE;
    };
}
