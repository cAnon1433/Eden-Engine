#pragma once

#include "VulkanImage.h"
#include "VulkanSampler.h"
#include "VulkanMemoryAllocator.h"

#include <vulkan/vulkan.h>
#include <string>
#include <cstdint>

namespace Eden
{
    // A loaded, GPU-resident texture: image + sampler + a ready-to-bind
    // descriptor set (VulkanTextureSetLayout's set 1, binding 0). Neither
    // copyable nor movable (VulkanImage isn't movable, and there's no
    // reason for VulkanTexture to be either) - Renderer keeps these behind
    // unique_ptr in its texture registry instead, same pattern already
    // used for Renderer::m_RenderFinishedSemaphores.
    class VulkanTexture
    {
    public:
        VulkanTexture() = default;
        ~VulkanTexture();

        VulkanTexture(const VulkanTexture&) = delete;
        VulkanTexture& operator=(const VulkanTexture&) = delete;

        // Loads an image file from disk (PNG/JPG/BMP/TGA/etc - anything
        // stb_image supports, see ThirdParty/stb), decodes it to raw RGBA
        // pixels, and uploads it to a device-local GPU image via a
        // staging buffer + one-time command buffer, same pattern as
        // VulkanBuffer::InitDeviceLocalWithData and Mesh::Create. One-time
        // cost - call during load time (e.g. Renderer::CreateTexture), not
        // per-frame. Throws std::runtime_error if the file can't be found
        // or decoded.
        void LoadFromFile(
            VmaAllocator allocator, VkDevice device, VkCommandPool commandPool, VkQueue graphicsQueue,
            VkDescriptorPool descriptorPool, VkDescriptorSetLayout textureSetLayout,
            const std::string& path);

        // 1x1 solid-color texture. Used as Eden's default/fallback texture
        // (see Renderer::Init) so every mesh goes through the exact same
        // "sample texSampler, multiply by vertex/instance color" shader
        // path whether or not it actually has a real texture assigned - a
        // white pixel makes that multiply a no-op, preserving the vertex-
        // color/ColorComponent-override behavior every existing untextured
        // mesh already relies on.
        void CreateSolidColor(
            VmaAllocator allocator, VkDevice device, VkCommandPool commandPool, VkQueue graphicsQueue,
            VkDescriptorPool descriptorPool, VkDescriptorSetLayout textureSetLayout,
            uint8_t r, uint8_t g, uint8_t b, uint8_t a);

        void Shutdown();

        // Ready to bind at set 1 via vkCmdBindDescriptorSets - see
        // Mesh::DrawInstanced.
        VkDescriptorSet GetDescriptorSet() const { return m_DescriptorSet; }

    private:
        // Shared upload path for both LoadFromFile and CreateSolidColor:
        // creates the device-local image, uploads `pixels` (must be tightly
        // packed RGBA8, width*height*4 bytes) via staging buffer, transitions
        // it to shader-read-only, creates the sampler, and allocates +
        // writes the descriptor set.
        void UploadAndCreate(
            VmaAllocator allocator, VkDevice device, VkCommandPool commandPool, VkQueue graphicsQueue,
            VkDescriptorPool descriptorPool, VkDescriptorSetLayout textureSetLayout,
            const unsigned char* pixels, uint32_t width, uint32_t height);

        VulkanImage m_Image;
        VulkanSampler m_Sampler;
        VkDescriptorSet m_DescriptorSet = VK_NULL_HANDLE;
    };
}
