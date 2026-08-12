#include "VulkanShaderModule.h"

#include <fstream>
#include <stdexcept>
#include <vector>

namespace Eden
{
    static std::vector<char> ReadFile(const std::string& path)
    {
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error("Eden: failed to open shader file: " + path);
        }

        size_t fileSize = static_cast<size_t>(file.tellg());
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();

        return buffer;
    }

    void VulkanShaderModule::LoadFromFile(VkDevice device, const std::string& spirvPath)
    {
        m_DeviceHandle = device;

        auto code = ReadFile(spirvPath);

        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        if (vkCreateShaderModule(m_DeviceHandle, &createInfo, nullptr, &m_Module) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to create shader module from: " + spirvPath);
        }
    }

    void VulkanShaderModule::Shutdown()
    {
        if (m_Module != VK_NULL_HANDLE)
        {
            vkDestroyShaderModule(m_DeviceHandle, m_Module, nullptr);
            m_Module = VK_NULL_HANDLE;
        }
    }

    VulkanShaderModule::~VulkanShaderModule()
    {
        Shutdown();
    }
}
