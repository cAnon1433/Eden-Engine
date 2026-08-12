#include "VulkanDebugMessenger.h"
#include "VulkanInstance.h"

#include <iostream>

namespace Eden
{
    static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT /*type*/,
        const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
        void* /*userData*/)
    {
        if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        {
            std::cerr << "[Vulkan] " << callbackData->pMessage << "\n";
        }
        return VK_FALSE;
    }

    static VkResult CreateDebugUtilsMessengerEXT(
        VkInstance instance,
        const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
        const VkAllocationCallbacks* pAllocator,
        VkDebugUtilsMessengerEXT* pMessenger)
    {
        auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
        if (func != nullptr)
        {
            return func(instance, pCreateInfo, pAllocator, pMessenger);
        }
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }

    static void DestroyDebugUtilsMessengerEXT(
        VkInstance instance,
        VkDebugUtilsMessengerEXT messenger,
        const VkAllocationCallbacks* pAllocator)
    {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if (func != nullptr)
        {
            func(instance, messenger, pAllocator);
        }
    }

    void VulkanDebugMessenger::PopulateCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
    {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                      VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = DebugCallback;
    }

    void VulkanDebugMessenger::Init(const VulkanInstance& instance)
    {
        if (!instance.IsValidationEnabled())
        {
            return;
        }

        m_InstanceHandle = instance.Get();

        VkDebugUtilsMessengerCreateInfoEXT createInfo{};
        PopulateCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(m_InstanceHandle, &createInfo, nullptr, &m_Messenger) != VK_SUCCESS)
        {
            std::cerr << "[Eden] Failed to set up debug messenger\n";
        }
    }

    void VulkanDebugMessenger::Shutdown()
    {
        if (m_Messenger != VK_NULL_HANDLE)
        {
            DestroyDebugUtilsMessengerEXT(m_InstanceHandle, m_Messenger, nullptr);
            m_Messenger = VK_NULL_HANDLE;
        }
    }

    VulkanDebugMessenger::~VulkanDebugMessenger()
    {
        Shutdown();
    }
}
