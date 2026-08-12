#pragma once

#include <vulkan/vulkan.h>
#include "VulkanMemoryAllocator.h"

namespace Eden
{
    // VkImage + VkImageView wrapper, VMA-backed. Currently only used for
    // the depth attachment - texture sampling (mips, transfer-dst uploads)
    // is a separate concern for VulkanTexture later.
    class VulkanImage
    {
    public:
        VulkanImage() = default;
        ~VulkanImage();

        VulkanImage(const VulkanImage&) = delete;
        VulkanImage& operator=(const VulkanImage&) = delete;

        void Init(VmaAllocator allocator, VkDevice device, VkExtent2D extent,
                  VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspectFlags);
        void Shutdown();

        VkImage Get() const { return m_Image; }
        VkImageView GetView() const { return m_ImageView; }
        VkFormat GetFormat() const { return m_Format; }

    private:
        VmaAllocator m_AllocatorHandle = VK_NULL_HANDLE;
        VkDevice m_DeviceHandle = VK_NULL_HANDLE;
        VkImage m_Image = VK_NULL_HANDLE;
        VmaAllocation m_Allocation = VK_NULL_HANDLE;
        VkImageView m_ImageView = VK_NULL_HANDLE;
        VkFormat m_Format = VK_FORMAT_UNDEFINED;
    };
}
