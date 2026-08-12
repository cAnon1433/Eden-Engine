#include "VulkanImage.h"

#include <stdexcept>

namespace Eden
{
    void VulkanImage::Init(VmaAllocator allocator, VkDevice device, VkExtent2D extent,
                            VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspectFlags)
    {
        m_AllocatorHandle = allocator;
        m_DeviceHandle = device;
        m_Format = format;

        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = extent.width;
        imageInfo.extent.height = extent.height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

        if (vmaCreateImage(allocator, &imageInfo, &allocInfo, &m_Image, &m_Allocation, nullptr) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to create image");
        }

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = m_Image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspectFlags;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device, &viewInfo, nullptr, &m_ImageView) != VK_SUCCESS)
        {
            vmaDestroyImage(allocator, m_Image, m_Allocation);
            throw std::runtime_error("Eden: failed to create image view");
        }
    }

    void VulkanImage::Shutdown()
    {
        if (m_ImageView != VK_NULL_HANDLE)
        {
            vkDestroyImageView(m_DeviceHandle, m_ImageView, nullptr);
            m_ImageView = VK_NULL_HANDLE;
        }
        if (m_Image != VK_NULL_HANDLE)
        {
            vmaDestroyImage(m_AllocatorHandle, m_Image, m_Allocation);
            m_Image = VK_NULL_HANDLE;
            m_Allocation = VK_NULL_HANDLE;
        }
    }

    VulkanImage::~VulkanImage()
    {
        Shutdown();
    }
}
