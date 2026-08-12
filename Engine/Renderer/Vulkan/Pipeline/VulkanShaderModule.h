#pragma once

#include <vulkan/vulkan.h>
#include <string>

namespace Eden
{
    // Loads a compiled SPIR-V (.spv) file from disk and wraps it in a VkShaderModule.
    class VulkanShaderModule
    {
    public:
        VulkanShaderModule() = default;
        ~VulkanShaderModule();

        VulkanShaderModule(const VulkanShaderModule&) = delete;
        VulkanShaderModule& operator=(const VulkanShaderModule&) = delete;

        void LoadFromFile(VkDevice device, const std::string& spirvPath);
        void Shutdown();

        VkShaderModule Get() const { return m_Module; }

    private:
        VkDevice m_DeviceHandle = VK_NULL_HANDLE;
        VkShaderModule m_Module = VK_NULL_HANDLE;
    };
}
