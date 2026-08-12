#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>

namespace Eden
{
    class VulkanInstance
    {
    public:
        VulkanInstance() = default;
        ~VulkanInstance();

        VulkanInstance(const VulkanInstance&) = delete;
        VulkanInstance& operator=(const VulkanInstance&) = delete;

        void Init(const std::string& appName, bool enableValidation);
        void Shutdown();

        VkInstance Get() const { return m_Instance; }
        bool IsValidationEnabled() const { return m_ValidationEnabled; }

    private:
        std::vector<const char*> GetRequiredExtensions() const;
        bool CheckValidationLayerSupport() const;

    private:
        VkInstance m_Instance = VK_NULL_HANDLE;
        bool m_ValidationEnabled = false;

        const std::vector<const char*> m_ValidationLayers = {
            "VK_LAYER_KHRONOS_validation"
        };
    };
}
