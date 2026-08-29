#pragma once

#include <vulkan/vulkan.h>
#include <string>

namespace Eden
{
    // Compute pipeline builder - for AI/sim compute passes (agent state,
    // spatial hashing, etc). First real user: Engine/Particles/GPU/
    // ParticleSystemGPU, which owns one instance of this per compute
    // shader stage (grid build, density/pressure, forces, integrate).
    //
    // Deliberately as thin as VulkanGraphicsPipeline: takes an externally
    // built VkPipelineLayout (see VulkanPipelineLayout) rather than
    // constructing its own descriptor/push-constant layout internally -
    // ParticleSystemGPU's four compute stages all share ONE layout (same
    // descriptor set, same SimParamsGPU push constant range), so building
    // it once and passing it into four VulkanComputePipeline instances
    // avoids four redundant identical VkPipelineLayout objects.
    class VulkanComputePipeline
    {
    public:
        VulkanComputePipeline() = default;
        ~VulkanComputePipeline();

        VulkanComputePipeline(const VulkanComputePipeline&) = delete;
        VulkanComputePipeline& operator=(const VulkanComputePipeline&) = delete;

        void Init(VkDevice device, VkPipelineLayout layout, const std::string& compSpirvPath);
        void Shutdown();

        VkPipeline Get() const { return m_Pipeline; }

    private:
        VkDevice m_DeviceHandle = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;
    };
}
