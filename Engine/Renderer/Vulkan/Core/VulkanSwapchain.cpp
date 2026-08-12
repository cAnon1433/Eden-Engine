#include "VulkanSwapchain.h"
#include "VulkanPhysicalDevice.h"
#include "VulkanDevice.h"

#include <GLFW/glfw3.h>
#include <stdexcept>
#include <algorithm>
#include <array>

namespace Eden
{
    void VulkanSwapchain::Init(const VulkanPhysicalDevice& physicalDevice, const VulkanDevice& device,
                                VkSurfaceKHR surface, GLFWwindow* window)
    {
        m_DeviceHandle = device.Get();
        CreateSwapchain(physicalDevice, surface, window);
        CreateImageViews();
    }

    void VulkanSwapchain::CreateSwapchain(const VulkanPhysicalDevice& physicalDevice, VkSurfaceKHR surface, GLFWwindow* window)
    {
        SwapchainSupportDetails support = physicalDevice.QuerySwapchainSupport(surface);

        VkSurfaceFormatKHR surfaceFormat = ChooseSurfaceFormat(support.formats);
        VkPresentModeKHR presentMode = ChoosePresentMode(support.presentModes);
        VkExtent2D extent = ChooseExtent(support.capabilities, window);

        uint32_t imageCount = support.capabilities.minImageCount + 1;
        if (support.capabilities.maxImageCount > 0 && imageCount > support.capabilities.maxImageCount)
        {
            imageCount = support.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        const QueueFamilyIndices& indices = physicalDevice.GetQueueFamilies();
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        if (indices.graphicsFamily != indices.presentFamily)
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        createInfo.preTransform = support.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(m_DeviceHandle, &createInfo, nullptr, &m_Swapchain) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to create swapchain");
        }

        vkGetSwapchainImagesKHR(m_DeviceHandle, m_Swapchain, &imageCount, nullptr);
        m_Images.resize(imageCount);
        vkGetSwapchainImagesKHR(m_DeviceHandle, m_Swapchain, &imageCount, m_Images.data());

        m_ImageFormat = surfaceFormat.format;
        m_Extent = extent;
    }

    void VulkanSwapchain::CreateImageViews()
    {
        m_ImageViews.resize(m_Images.size());

        for (size_t i = 0; i < m_Images.size(); i++)
        {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = m_Images[i];
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = m_ImageFormat;
            createInfo.components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                       VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;

            if (vkCreateImageView(m_DeviceHandle, &createInfo, nullptr, &m_ImageViews[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("Eden: failed to create swapchain image view");
            }
        }
    }

    void VulkanSwapchain::CreateFramebuffers(VkRenderPass renderPass, VkImageView depthImageView)
    {
        DestroyFramebuffers();
        m_Framebuffers.resize(m_ImageViews.size());

        for (size_t i = 0; i < m_ImageViews.size(); i++)
        {
            std::array<VkImageView, 2> attachments = { m_ImageViews[i], depthImageView };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = m_Extent.width;
            framebufferInfo.height = m_Extent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(m_DeviceHandle, &framebufferInfo, nullptr, &m_Framebuffers[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("Eden: failed to create framebuffer");
            }
        }
    }

    void VulkanSwapchain::DestroyFramebuffers()
    {
        for (auto framebuffer : m_Framebuffers)
        {
            vkDestroyFramebuffer(m_DeviceHandle, framebuffer, nullptr);
        }
        m_Framebuffers.clear();
    }

    void VulkanSwapchain::Recreate(const VulkanPhysicalDevice& physicalDevice, VkSurfaceKHR surface, GLFWwindow* window)
    {
        // Handle minimization: wait until the window has a non-zero size again.
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0)
        {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        Shutdown();
        CreateSwapchain(physicalDevice, surface, window);
        CreateImageViews();
        // Note: CreateFramebuffers must be called again by the owner (Renderer)
        // after Recreate(), since it needs the render pass handle.
    }

    void VulkanSwapchain::Shutdown()
    {
        DestroyFramebuffers();

        for (auto imageView : m_ImageViews)
        {
            vkDestroyImageView(m_DeviceHandle, imageView, nullptr);
        }
        m_ImageViews.clear();

        if (m_Swapchain != VK_NULL_HANDLE)
        {
            vkDestroySwapchainKHR(m_DeviceHandle, m_Swapchain, nullptr);
            m_Swapchain = VK_NULL_HANDLE;
        }
    }

    VulkanSwapchain::~VulkanSwapchain()
    {
        Shutdown();
    }

    VkSurfaceFormatKHR VulkanSwapchain::ChooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& available) const
    {
        for (const auto& format : available)
        {
            if (format.format == VK_FORMAT_B8G8R8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                return format;
            }
        }
        return available[0];
    }

    VkPresentModeKHR VulkanSwapchain::ChoosePresentMode(const std::vector<VkPresentModeKHR>& available) const
    {
        for (const auto& mode : available)
        {
            if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
            {
                return mode;
            }
        }
        return VK_PRESENT_MODE_FIFO_KHR; // guaranteed to be available
    }

    VkExtent2D VulkanSwapchain::ChooseExtent(const VkSurfaceCapabilitiesKHR& capabilities, GLFWwindow* window) const
    {
        if (capabilities.currentExtent.width != UINT32_MAX)
        {
            return capabilities.currentExtent;
        }

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        VkExtent2D actualExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };

        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

        return actualExtent;
    }
}
