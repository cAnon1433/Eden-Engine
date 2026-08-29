#include "VulkanShaderModule.h"
#include "../../../Core/PathUtils.h"

#include <fstream>
#include <stdexcept>
#include <vector>

namespace Eden
{
    // Every shader load in the engine (graphics AND compute pipelines -
    // see VulkanGraphicsPipeline.cpp/VulkanComputePipeline.cpp, both of
    // which construct a VulkanShaderModule and call LoadFromFile) funnels
    // through this one ReadFile, which is why resolving the exe-relative
    // path here - instead of at each of the ~7 call sites that pass a
    // literal "Shaders/Compiled/..." string - fixes resource loading for
    // the whole engine in one place. See PathUtils.h for why this can't
    // just be the process's working directory.
    static std::vector<char> ReadFile(const std::string& path)
    {
        std::string resolvedPath = PathUtils::ResolveResourcePath(path);
        std::ifstream file(resolvedPath, std::ios::ate | std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error("Eden: failed to open shader file: " + resolvedPath);
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
