// stb_image's implementation is generated exactly once, in this
// translation unit - it's a single-header library (see ThirdParty/stb),
// and VulkanTexture is its only consumer in Eden.
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "VulkanTexture.h"

#include <stdexcept>
#include <cstring>
#include <array>

namespace Eden
{
    namespace
    {
        // One-time command buffer, same "allocate, record, submit, wait,
        // free" shape as VulkanBuffer::InitDeviceLocalWithData - simple
        // and correct for load-time work, not meant for per-frame use.
        VkCommandBuffer BeginOneTimeCommands(VkDevice device, VkCommandPool commandPool)
        {
            VkCommandBufferAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandPool = commandPool;
            allocInfo.commandBufferCount = 1;

            VkCommandBuffer cmd;
            vkAllocateCommandBuffers(device, &allocInfo, &cmd);

            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &beginInfo);

            return cmd;
        }

        void EndOneTimeCommands(VkDevice device, VkCommandPool commandPool, VkQueue graphicsQueue, VkCommandBuffer cmd)
        {
            vkEndCommandBuffer(cmd);

            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;

            vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(graphicsQueue);

            vkFreeCommandBuffers(device, commandPool, 1, &cmd);
        }

        // A newly-created VkImage starts in UNDEFINED layout, which the GPU
        // can't sample from or copy into directly - it has to be
        // transitioned through TRANSFER_DST_OPTIMAL (to receive the copy
        // from the staging buffer) and finally to SHADER_READ_ONLY_OPTIMAL
        // (what the fragment shader actually needs to sample it). This is
        // boilerplate every Vulkan texture-loading path needs; kept as a
        // free function here since only VulkanTexture calls it, but the
        // logic itself isn't texture-specific.
        void TransitionImageLayout(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout)
        {
            VkImageMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = oldLayout;
            barrier.newLayout = newLayout;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image;
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;

            VkPipelineStageFlags sourceStage;
            VkPipelineStageFlags destinationStage;

            if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
            {
                barrier.srcAccessMask = 0;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            }
            else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            {
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            }
            else
            {
                throw std::runtime_error("Eden: unsupported image layout transition");
            }

            vkCmdPipelineBarrier(cmd, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        }
    }

    void VulkanTexture::LoadFromFile(
        VmaAllocator allocator, VkDevice device, VkCommandPool commandPool, VkQueue graphicsQueue,
        VkDescriptorPool descriptorPool, VkDescriptorSetLayout textureSetLayout,
        const std::string& path)
    {
        int width, height, channelsInFile;
        // Force 4 channels (RGBA) regardless of the source file's actual
        // channel count - keeps the upload path uniform (always
        // VK_FORMAT_R8G8B8A8_SRGB) instead of branching on whether the
        // source was RGB/grayscale/etc.
        stbi_uc* pixels = stbi_load(path.c_str(), &width, &height, &channelsInFile, STBI_rgb_alpha);

        if (!pixels)
        {
            throw std::runtime_error("Eden: failed to load texture '" + path + "': " + stbi_failure_reason());
        }

        UploadAndCreate(allocator, device, commandPool, graphicsQueue, descriptorPool, textureSetLayout,
                         pixels, static_cast<uint32_t>(width), static_cast<uint32_t>(height));

        stbi_image_free(pixels);
    }

    void VulkanTexture::CreateSolidColor(
        VmaAllocator allocator, VkDevice device, VkCommandPool commandPool, VkQueue graphicsQueue,
        VkDescriptorPool descriptorPool, VkDescriptorSetLayout textureSetLayout,
        uint8_t r, uint8_t g, uint8_t b, uint8_t a)
    {
        std::array<uint8_t, 4> pixel = { r, g, b, a };
        UploadAndCreate(allocator, device, commandPool, graphicsQueue, descriptorPool, textureSetLayout,
                         pixel.data(), 1, 1);
    }

    void VulkanTexture::UploadAndCreate(
        VmaAllocator allocator, VkDevice device, VkCommandPool commandPool, VkQueue graphicsQueue,
        VkDescriptorPool descriptorPool, VkDescriptorSetLayout textureSetLayout,
        const unsigned char* pixels, uint32_t width, uint32_t height)
    {
        VkDeviceSize imageSize = static_cast<VkDeviceSize>(width) * height * 4;

        // --- Staging buffer: host-visible, CPU writes the decoded pixels straight into it ---
        VkBuffer stagingBuffer;
        VmaAllocation stagingAllocation;

        VkBufferCreateInfo stagingInfo{};
        stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        stagingInfo.size = imageSize;
        stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo stagingAllocInfo{};
        stagingAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                  VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo stagingInfoOut{};
        if (vmaCreateBuffer(allocator, &stagingInfo, &stagingAllocInfo, &stagingBuffer, &stagingAllocation, &stagingInfoOut) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to create texture staging buffer");
        }

        std::memcpy(stagingInfoOut.pMappedData, pixels, static_cast<size_t>(imageSize));

        // --- Device-local image ---
        m_Image.Init(allocator, device, VkExtent2D{ width, height }, VK_FORMAT_R8G8B8A8_SRGB,
                     VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                     VK_IMAGE_ASPECT_COLOR_BIT);

        // --- Upload: one command buffer, three steps (transition in, copy, transition to shader-readable) ---
        VkCommandBuffer cmd = BeginOneTimeCommands(device, commandPool);

        TransitionImageLayout(cmd, m_Image.Get(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;   // 0 = tightly packed, matches stb_image's output
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = { 0, 0, 0 };
        region.imageExtent = { width, height, 1 };

        vkCmdCopyBufferToImage(cmd, stagingBuffer, m_Image.Get(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        TransitionImageLayout(cmd, m_Image.Get(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        EndOneTimeCommands(device, commandPool, graphicsQueue, cmd);

        vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);

        // --- Sampler ---
        m_Sampler.Init(device);

        // --- Descriptor set: allocated once here (not per-frame - a
        // texture's descriptor set never changes after creation, so
        // there's no reason to reallocate/rewrite it every frame the way
        // the per-frame camera UBO set has to) ---
        VkDescriptorSetAllocateInfo setAllocInfo{};
        setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        setAllocInfo.descriptorPool = descriptorPool;
        setAllocInfo.descriptorSetCount = 1;
        setAllocInfo.pSetLayouts = &textureSetLayout;

        VkResult result = vkAllocateDescriptorSets(device, &setAllocInfo, &m_DescriptorSet);
        if (result != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to allocate texture descriptor set (VkResult "
                                      + std::to_string(result) + ")");
        }

        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = m_Image.GetView();
        imageInfo.sampler = m_Sampler.Get();

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = m_DescriptorSet;
        write.dstBinding = 0;
        write.dstArrayElement = 0;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.descriptorCount = 1;
        write.pImageInfo = &imageInfo;

        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }

    void VulkanTexture::Shutdown()
    {
        m_Sampler.Shutdown();
        m_Image.Shutdown();
        // m_DescriptorSet is NOT explicitly freed here - same reasoning as
        // FrameContext's descriptor set: it's freed automatically when its
        // pool (VkDescriptorPool) is destroyed, since that pool wasn't
        // created with VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT.
    }

    VulkanTexture::~VulkanTexture()
    {
        Shutdown();
    }
}
