#pragma once

#include <vulkan/vulkan.h>

namespace Eden
{
    // Thin wrapper around a single primary command buffer. Recording of
    // actual draw commands happens in Renderer, which has render-pass /
    // pipeline context this class deliberately doesn't need to know about.
    class VulkanCommandBuffer
    {
    public:
        void Allocate(VkDevice device, VkCommandPool pool);

        void Begin() const;
        void End() const;
        void Reset() const;

        VkCommandBuffer Get() const { return m_CommandBuffer; }

    private:
        VkCommandBuffer m_CommandBuffer = VK_NULL_HANDLE;
    };
}
