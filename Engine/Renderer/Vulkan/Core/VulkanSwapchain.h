#pragma once

#include <vulkan/vulkan.h>
#include <vector>

struct GLFWwindow;

namespace Eden
{
    class VulkanPhysicalDevice;
    class VulkanDevice;

    class VulkanSwapchain
    {
    public:
        VulkanSwapchain() = default;
        ~VulkanSwapchain();

        VulkanSwapchain(const VulkanSwapchain&) = delete;
        VulkanSwapchain& operator=(const VulkanSwapchain&) = delete;

        void Init(const VulkanPhysicalDevice& physicalDevice, const VulkanDevice& device,
                  VkSurfaceKHR surface, GLFWwindow* window);
        void Shutdown();

        // Destroys and recreates the swapchain + image views, e.g. on window resize.
        void Recreate(const VulkanPhysicalDevice& physicalDevice, VkSurfaceKHR surface, GLFWwindow* window);

        // Framebuffers depend on a render pass, so they're created separately
        // once VulkanRenderPass exists. Called by Renderer during setup/recreate.
        // depthImageView must match the depth attachment the render pass expects.
        void CreateFramebuffers(VkRenderPass renderPass, VkImageView depthImageView);

        VkSwapchainKHR Get() const { return m_Swapchain; }
        VkFormat GetImageFormat() const { return m_ImageFormat; }
        VkExtent2D GetExtent() const { return m_Extent; }
        uint32_t GetImageCount() const { return static_cast<uint32_t>(m_Images.size()); }
        VkFramebuffer GetFramebuffer(uint32_t index) const { return m_Framebuffers[index]; }

    private:
        void CreateSwapchain(const VulkanPhysicalDevice& physicalDevice, VkSurfaceKHR surface, GLFWwindow* window);
        void CreateImageViews();
        void DestroyFramebuffers();

        VkSurfaceFormatKHR ChooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& available) const;
        VkPresentModeKHR ChoosePresentMode(const std::vector<VkPresentModeKHR>& available) const;
        VkExtent2D ChooseExtent(const VkSurfaceCapabilitiesKHR& capabilities, GLFWwindow* window) const;

    private:
        VkDevice m_DeviceHandle = VK_NULL_HANDLE;
        VkSwapchainKHR m_Swapchain = VK_NULL_HANDLE;

        std::vector<VkImage> m_Images;
        std::vector<VkImageView> m_ImageViews;
        std::vector<VkFramebuffer> m_Framebuffers;

        VkFormat m_ImageFormat = VK_FORMAT_UNDEFINED;
        VkExtent2D m_Extent{};
    };
}
