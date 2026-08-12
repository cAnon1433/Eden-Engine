#include "VulkanDevice.h"
#include "VulkanPhysicalDevice.h"

#include <stdexcept>
#include <set>
#include <vector>
#include <string>

namespace Eden
{
    void VulkanDevice::Init(const VulkanPhysicalDevice& physicalDevice, bool enableValidation)
    {
        const QueueFamilyIndices& indices = physicalDevice.GetQueueFamilies();

        std::set<uint32_t> uniqueQueueFamilies = {
            indices.graphicsFamily.value(),
            indices.presentFamily.value()
        };

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        float queuePriority = 1.0f;
        for (uint32_t family : uniqueQueueFamilies)
        {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = family;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        // Query what THIS physical device actually supports before
        // requesting anything - requesting a feature the device doesn't
        // support fails device creation outright (not a graceful
        // partial-failure), so this has to check first.
        VkPhysicalDeviceFeatures supportedFeatures{};
        vkGetPhysicalDeviceFeatures(physicalDevice.Get(), &supportedFeatures);

        VkPhysicalDeviceFeatures deviceFeatures{};
        // largePoints: lets the vertex shader write gl_PointSize values
        // other than exactly 1.0 (writing anything else without this
        // enabled is undefined behavior per the Vulkan spec) - needed for
        // the particle-points render mode (see Renderer::m_ParticlePointsPipeline).
        // Only requested if supported; if not, the points pipeline still
        // works, it just can't be resized - gl_PointSize writes get
        // clamped to 1.0px regardless of what the shader asks for. Every
        // GPU class Eden targets (desktop, and MoltenVK on Apple Silicon
        // - Metal has always supported variable point size) reports this
        // as supported in practice, so the fallback path is a safety net,
        // not an expected outcome.
        deviceFeatures.largePoints = supportedFeatures.largePoints;

        // Start from the extensions every platform needs (VK_KHR_swapchain),
        // then add MoltenVK's portability subset if this physical device exposes it.
        std::vector<const char*> deviceExtensions = VulkanPhysicalDevice::RequiredDeviceExtensions();

        uint32_t availableExtCount = 0;
        vkEnumerateDeviceExtensionProperties(physicalDevice.Get(), nullptr, &availableExtCount, nullptr);
        std::vector<VkExtensionProperties> availableExts(availableExtCount);
        vkEnumerateDeviceExtensionProperties(physicalDevice.Get(), nullptr, &availableExtCount, availableExts.data());

        for (const auto& ext : availableExts)
        {
            if (std::string(ext.extensionName) == "VK_KHR_portability_subset")
            {
                deviceExtensions.push_back("VK_KHR_portability_subset");
                break;
            }
        }

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.pEnabledFeatures = &deviceFeatures;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        // Device-level validation layers were deprecated back in Vulkan 1.0
        // and current SDKs (1.4+) reject enabledLayerCount != 0 outright.
        // Validation is instance-wide via VulkanInstance/VulkanDebugMessenger.
        createInfo.enabledLayerCount = 0;
        createInfo.ppEnabledLayerNames = nullptr;

        if (vkCreateDevice(physicalDevice.Get(), &createInfo, nullptr, &m_Device) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to create logical device");
        }

        vkGetDeviceQueue(m_Device, indices.graphicsFamily.value(), 0, &m_GraphicsQueue);
        vkGetDeviceQueue(m_Device, indices.presentFamily.value(), 0, &m_PresentQueue);
    }

    void VulkanDevice::WaitIdle() const
    {
        if (m_Device != VK_NULL_HANDLE)
        {
            vkDeviceWaitIdle(m_Device);
        }
    }

    void VulkanDevice::Shutdown()
    {
        if (m_Device != VK_NULL_HANDLE)
        {
            vkDestroyDevice(m_Device, nullptr);
            m_Device = VK_NULL_HANDLE;
        }
    }

    VulkanDevice::~VulkanDevice()
    {
        Shutdown();
    }
}
