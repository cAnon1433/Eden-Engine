#include "VulkanComputePipeline.h"
#include "VulkanShaderModule.h"

#include <stdexcept>

namespace Eden
{
    void VulkanComputePipeline::Init(VkDevice device, VkPipelineLayout layout, const std::string& compSpirvPath)
    {
        m_DeviceHandle = device;

        VulkanShaderModule shaderModule;
        shaderModule.LoadFromFile(device, compSpirvPath);

        VkPipelineShaderStageCreateInfo stageInfo{};
        stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo.module = shaderModule.Get();
        stageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = stageInfo;
        pipelineInfo.layout = layout;

        // shaderModule goes out of scope (and is destroyed) right after
        // this call - same lifetime pattern VulkanGraphicsPipeline uses
        // for its vert/frag modules, since vkCreateComputePipelines
        // consumes the module immediately and doesn't need it to
        // outlive the call.
        if (vkCreateComputePipelines(m_DeviceHandle, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_Pipeline) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to create compute pipeline (" + compSpirvPath + ")");
        }
    }

    void VulkanComputePipeline::Shutdown()
    {
        if (m_Pipeline != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(m_DeviceHandle, m_Pipeline, nullptr);
            m_Pipeline = VK_NULL_HANDLE;
        }
    }

    VulkanComputePipeline::~VulkanComputePipeline()
    {
        Shutdown();
    }
}
