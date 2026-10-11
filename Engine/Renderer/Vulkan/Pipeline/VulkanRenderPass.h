#pragma once

#include <vulkan/vulkan.h>

namespace Eden
{
    // Color attachment (swapchain image) + depth attachment, one subpass.
    //
    // Three variants of the SAME attachment layout (so pipelines created
    // against one are valid in the others, and one set of framebuffers
    // serves all three - Vulkan render pass compatibility ignores load/store
    // ops, layouts and, for single-subpass passes, dependencies):
    //   Single       - the original: clear everything, end ready to present.
    //   SceneSplit   - first half of a frame whose fluid surface needs the
    //                  scene's depth: clear, STORE both, leave color as a
    //                  color attachment and depth readable as a texture.
    //   Overlay      - second half: LOAD both, draw the fluid composite/UI,
    //                  end ready to present.
    enum class RenderPassMode
    {
        Single,
        SceneSplit,
        Overlay
    };

    class VulkanRenderPass
    {
    public:
        VulkanRenderPass() = default;
        ~VulkanRenderPass();

        VulkanRenderPass(const VulkanRenderPass&) = delete;
        VulkanRenderPass& operator=(const VulkanRenderPass&) = delete;

        void Init(VkDevice device, VkFormat colorFormat, VkFormat depthFormat, RenderPassMode mode = RenderPassMode::Single);
        void Shutdown();

        VkRenderPass Get() const { return m_RenderPass; }

    private:
        VkDevice m_DeviceHandle = VK_NULL_HANDLE;
        VkRenderPass m_RenderPass = VK_NULL_HANDLE;
    };
}
