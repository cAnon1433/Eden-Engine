#pragma once

#include "../Command/VulkanCommandBuffer.h"
#include "../Sync/VulkanFence.h"
#include "../Sync/VulkanSemaphore.h"
#include "../Resources/VulkanBuffer.h"
#include "../Resources/VulkanMemoryAllocator.h"
#include "../RendererTypes.h"

#include <cstring>
#include <stdexcept>

namespace Eden
{
    // Everything needed to record and submit exactly one frame while another
    // frame is still in flight on the GPU. Renderer owns MAX_FRAMES_IN_FLIGHT
    // of these and cycles through them. Also owns the per-frame camera
    // uniform buffer and instance buffer, since those are just as much
    // "per frame in flight" state as the command buffer and sync objects.
    //
    // NOTE: renderFinishedSemaphore is deliberately NOT here. A semaphore
    // signaled by a submit and waited on by vkQueuePresentKHR has to be
    // indexed by which SWAPCHAIN IMAGE was acquired, not by which
    // frame-in-flight slot is being used - those two things aren't
    // guaranteed to advance in lockstep, and indexing by frame-in-flight
    // instead can let a submit try to re-signal a semaphore the present
    // engine hasn't finished with yet. Validation catches this as
    // "semaphore may still be in use". See Renderer::m_RenderFinishedSemaphores
    // and Renderer::DrawFrame.
    struct FrameContext
    {
        VulkanCommandBuffer commandBuffer;
        VulkanSemaphore imageAvailableSemaphore;
        VulkanFence inFlightFence;

        VulkanBuffer uniformBuffer;
        void* uniformBufferMapped = nullptr;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

        // Per-instance data (model matrix + color override) for every
        // entity drawn this frame, across every mesh - Renderer::DrawFrame
        // groups DrawCommands by mesh and writes them here contiguously
        // per group, then binds a slice of this buffer (via byte offset)
        // for each mesh's instanced draw call. Same persistently-mapped
        // host-visible pattern as uniformBuffer above, for the same
        // reason: written fresh every frame, so staging through
        // device-local memory would be pure overhead.
        VulkanBuffer instanceBuffer;
        void* instanceBufferMapped = nullptr;

        void Init(VkDevice device, VkCommandPool commandPool, VmaAllocator allocator,
                  VkDescriptorPool descriptorPool, VkDescriptorSetLayout descriptorSetLayout)
        {
            commandBuffer.Allocate(device, commandPool);
            imageAvailableSemaphore.Init(device);
            inFlightFence.Init(device, /*signaled=*/true);

            // Persistently mapped, host-visible - written fresh every frame,
            // so there's no benefit to staging it through device-local
            // memory the way vertex data is.
            uniformBuffer.Init(
                allocator,
                sizeof(UniformBufferObject),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VMA_MEMORY_USAGE_AUTO,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                &uniformBufferMapped);

            instanceBuffer.Init(
                allocator,
                sizeof(InstanceData) * MAX_INSTANCES_PER_FRAME,
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                VMA_MEMORY_USAGE_AUTO,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                &instanceBufferMapped);

            VkDescriptorSetAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocInfo.descriptorPool = descriptorPool;
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts = &descriptorSetLayout;

            if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS)
            {
                throw std::runtime_error("Eden: failed to allocate descriptor set");
            }

            VkDescriptorBufferInfo bufferDescInfo{};
            bufferDescInfo.buffer = uniformBuffer.Get();
            bufferDescInfo.offset = 0;
            bufferDescInfo.range = sizeof(UniformBufferObject);

            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = descriptorSet;
            write.dstBinding = 0;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &bufferDescInfo;

            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
        }

        void UpdateUniformBuffer(const UniformBufferObject& ubo) const
        {
            std::memcpy(uniformBufferMapped, &ubo, sizeof(ubo));
        }

        void Shutdown()
        {
            imageAvailableSemaphore.Shutdown();
            inFlightFence.Shutdown();
            uniformBuffer.Shutdown();
            instanceBuffer.Shutdown();
            // commandBuffer and descriptorSet are freed automatically when
            // their pools (VkCommandPool / VkDescriptorPool) are destroyed.
        }
    };
}
