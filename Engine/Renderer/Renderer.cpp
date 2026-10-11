#include "Renderer.h"
#include "Engine/Core/Profiler.h"
#include "Vulkan/RendererTypes.h"
#include "Primitives.h"
#include "ModelLoader.h"
#include "Engine/Particles/GPU/ParticleGPUTypes.h" // GPU_MAX_PARTICLES - see m_ParticlePreviousPositions' sizing in RegisterParticleGPUSource

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <GLFW/glfw3.h>
#include <stdexcept>
#include <limits>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cctype>
#include <cmath>
#include <iostream>
#include <array>

namespace Eden
{
    void Renderer::Init(GLFWwindow* window)
    {
        m_Window = window;

#ifdef NDEBUG
        constexpr bool enableValidation = false;
#else
        constexpr bool enableValidation = true;
#endif

        m_Context.Init(window, "Eden", enableValidation);

        m_Allocator.Init(m_Context.Instance().Get(), m_Context.PhysicalDevice().Get(), m_Context.Device().Get());

        CreateDepthResources();

        m_RenderPass.Init(m_Context.Device().Get(), m_Context.Swapchain().GetImageFormat(), m_DepthImage.GetFormat());
        m_SceneSplitRenderPass.Init(m_Context.Device().Get(), m_Context.Swapchain().GetImageFormat(), m_DepthImage.GetFormat(), RenderPassMode::SceneSplit);
        m_OverlayRenderPass.Init(m_Context.Device().Get(), m_Context.Swapchain().GetImageFormat(), m_DepthImage.GetFormat(), RenderPassMode::Overlay);
        m_Context.Swapchain().CreateFramebuffers(m_RenderPass.Get(), m_DepthImage.GetView());

        m_DescriptorSetLayout.Init(m_Context.Device().Get());
        m_TextureSetLayout.Init(m_Context.Device().Get());

        // Pool sizes: MAX_FRAMES_IN_FLIGHT uniform-buffer sets (the
        // per-frame camera UBO, set 0) plus MAX_TEXTURES combined-image-
        // sampler sets (one per loaded texture, set 1 - see CreateTexture).
        // maxSets has to cover both kinds of set being allocated from the
        // same pool.
        std::vector<VkDescriptorPoolSize> poolSizes = {
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MAX_FRAMES_IN_FLIGHT },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_TEXTURES },
        };
        m_DescriptorPool.Init(m_Context.Device().Get(), poolSizes, MAX_FRAMES_IN_FLIGHT + MAX_TEXTURES);

        CreateRenderFinishedSemaphores();

        VkVertexInputBindingDescription vertexBinding = Vertex::GetBindingDescription();
        VkVertexInputBindingDescription instanceBinding = InstanceData::GetBindingDescription();
        std::vector<VkVertexInputBindingDescription> bindingDescriptions = { vertexBinding, instanceBinding };

        auto vertexAttributesArray = Vertex::GetAttributeDescriptions();
        auto instanceAttributesArray = InstanceData::GetAttributeDescriptions();
        std::vector<VkVertexInputAttributeDescription> attributeDescriptions(
            vertexAttributesArray.begin(), vertexAttributesArray.end());
        attributeDescriptions.insert(
            attributeDescriptions.end(), instanceAttributesArray.begin(), instanceAttributesArray.end());

        // Set 0 = camera UBO, set 1 = texture - index in this vector IS
        // the Vulkan set number, see VulkanPipelineLayout::Init.
        m_PipelineLayout.Init(m_Context.Device().Get(), { m_DescriptorSetLayout.Get(), m_TextureSetLayout.Get() });
        m_GraphicsPipeline.Init(
            m_Context.Device().Get(),
            m_RenderPass.Get(),
            m_PipelineLayout.Get(),
            "Shaders/Compiled/triangle.vert.spv",
            "Shaders/Compiled/triangle.frag.spv",
            bindingDescriptions,
            attributeDescriptions);

        // Same shaders/layout/vertex-input as m_GraphicsPipeline - only
        // cullMode differs (NONE instead of the default BACK_BIT). See
        // m_VoxelPipeline's own comment in Renderer.h for why this
        // exists as a separate pipeline object rather than a global
        // culling change.
        m_VoxelPipeline.Init(
            m_Context.Device().Get(),
            m_RenderPass.Get(),
            m_PipelineLayout.Get(),
            "Shaders/Compiled/triangle.vert.spv",
            "Shaders/Compiled/triangle.frag.spv",
            bindingDescriptions,
            attributeDescriptions,
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            VK_CULL_MODE_NONE);

        // Same pipeline layout, render pass, and vertex/instance input
        // description as m_GraphicsPipeline above - only the shader
        // modules and topology differ. This is what actually lets
        // particles render as round point sprites (see EdenUI's particle
        // panel and ParticleSystem::BuildDrawList) instead of tiny cubes:
        // gl_PointSize in particle_point.vert controls screen-space size
        // directly, which no amount of world-space mesh scaling could do
        // (a cube's apparent size on screen still depends on distance
        // from camera; gl_PointSize does not - it's a fixed pixel
        // footprint regardless of depth, exactly what a debug/simulation
        // point-cloud visualization wants).
        m_ParticlePointsPipeline.Init(
            m_Context.Device().Get(),
            m_RenderPass.Get(),
            m_PipelineLayout.Get(),
            "Shaders/Compiled/particle_point.vert.spv",
            "Shaders/Compiled/particle_point.frag.spv",
            bindingDescriptions,
            attributeDescriptions,
            VK_PRIMITIVE_TOPOLOGY_POINT_LIST);

        m_CommandPool.Init(m_Context.Device().Get(), m_Context.PhysicalDevice().GetQueueFamilies().graphicsFamily.value());

        // Needs m_CommandPool (for the dummy density buffer's - none
        // actually needed, host-visible init only) and m_Allocator,
        // both already initialized above by this point.
        InitRaymarchPass();

        // Default/fallback texture - a single white pixel, bound to any
        // mesh created without an explicit TextureHandle so the shader's
        // "sample and multiply" path (see triangle.frag) works uniformly
        // whether or not a mesh actually has a real texture. Created here
        // (not lazily) so it always exists at index 0 of m_TextureRegistry
        // before any mesh gets created.
        {
            auto defaultTexture = std::make_unique<VulkanTexture>();
            defaultTexture->CreateSolidColor(
                m_Allocator.Get(), m_Context.Device().Get(), m_CommandPool.Get(), m_Context.Device().GetGraphicsQueue(),
                m_DescriptorPool.Get(), m_TextureSetLayout.Get(),
                255, 255, 255, 255);
            m_TextureRegistry.push_back(std::move(defaultTexture));
            m_DefaultTextureHandle = m_TextureRegistry.size() - 1;
        }

        // Single-vertex "point" mesh, used ONLY with m_ParticlePointsPipeline
        // (see GetParticlePointMesh) - the vertex position doesn't matter
        // beyond being defined (particle_point.vert hardcodes the
        // assumption that it's the origin; see that shader's comment), so
        // this exists purely to give DrawInstanced/vkCmdDraw a valid
        // vertex buffer with vertexCount=1: one point primitive drawn per
        // INSTANCE, at that instance's own model-matrix translation. Goes
        // through the same CreateMesh() every other mesh uses (untextured
        // - texture is irrelevant to a shader that doesn't sample one) so
        // it participates in the same instance-grouping/draw-call path in
        // DrawFrame without any special-casing there beyond the pipeline
        // swap.
        {
            Vertex originVertex{};
            originVertex.position = glm::vec3(0.0f);
            originVertex.color = glm::vec3(1.0f);
            originVertex.normal = glm::vec3(0.0f, 1.0f, 0.0f);
            m_ParticlePointMeshHandle = CreateMesh({ originVertex });
        }

        // Deliberately low resolution - see SoftwareOcclusionBuffer.h's
        // class comment on why this is a coarse conservative test, not a
        // second real depth buffer. 16:9-ish aspect independent of the
        // actual window/swapchain size (this buffer's grid mapping is
        // NDC-based, see ProjectAabb, so it works for any actual
        // viewport aspect - this resolution choice is just "how many
        // samples," not tied to matching pixels 1:1 with anything).
        m_OcclusionBuffer.Resize(160, 90);

        for (auto& frame : m_Frames)
        {
            frame.Init(m_Context.Device().Get(), m_CommandPool.Get(), m_Allocator.Get(),
                       m_DescriptorPool.Get(), m_DescriptorSetLayout.Get());
        }

        // Reasonable default vantage point: back a few units on +Z, looking
        // toward the origin where AddMesh'd geometry will sit by default.
        m_Camera.Position = glm::vec3(0.0f, 0.5f, 3.0f);

        InitImGui();
    }

    void Renderer::RegisterParticleGPUSource(VkBuffer positionBuffer, VkBuffer colorBuffer)
    {
        VkDevice device = m_Context.Device().Get();
        VmaAllocator allocator = m_Allocator.Get();

        m_ParticleLivePositionBuffer = positionBuffer;

        // See m_ParticlePreviousPositions' own comment (Renderer.h) for
        // why this exists instead of a velocity*deltaTime estimate.
        // TRANSFER_DST because Renderer::DrawFrame copies into this every
        // frame; STORAGE_BUFFER because fluid_depth.vert reads it exactly
        // like the live position buffer.
        m_ParticlePreviousPositions.Init(allocator, GPU_MAX_PARTICLES * sizeof(glm::vec4),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                          VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

        // Seed it from the live buffer's CURRENT contents (one-time copy,
        // same pattern as ClearFluidHistoryImages) so the very first
        // frame reprojects with zero motion rather than reading
        // uninitialized memory - harmless either way since
        // ClearFluidHistoryImages also guarantees no history exists yet
        // to reproject INTO on frame one, but cheap to do properly rather
        // than rely on that alone.
        {
            VkCommandBufferAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandPool = m_CommandPool.Get();
            allocInfo.commandBufferCount = 1;
            VkCommandBuffer cmd = VK_NULL_HANDLE;
            vkAllocateCommandBuffers(device, &allocInfo, &cmd);

            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &beginInfo);

            VkBufferCopy copyRegion{ 0, 0, GPU_MAX_PARTICLES * sizeof(glm::vec4) };
            vkCmdCopyBuffer(cmd, positionBuffer, m_ParticlePreviousPositions.Get(), 1, &copyRegion);

            vkEndCommandBuffer(cmd);
            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;
            vkQueueSubmit(m_Context.Device().GetGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(m_Context.Device().GetGraphicsQueue()); // one-time setup, not a per-frame path - see ClearFluidHistoryImages' identical reasoning
            vkFreeCommandBuffers(device, m_CommandPool.Get(), 1, &cmd);
        }

        // Three bindings: binding 0 the compute-owned positions storage
        // buffer (read-only from the vertex stage - see
        // particle_point_gpu.vert), binding 1 the per-particle color
        // buffer added for the Chemical ID base build (see
        // ParticleSystemGPU::ChemicalTable's comment) - read by
        // fluid_depth.vert/.frag so the fluid surface pass can show each
        // particle's actual (diffusing) color instead of one fixed tint -
        // and binding 2, m_ParticlePreviousPositions (see its own comment),
        // added for the fluid temporal reprojection pass (see
        // fluid_depth.vert's outPositionDelta comment and
        // fluid_temporal.frag). particle_point_gpu.vert still only
        // declares/uses binding 0 - adding more bindings to the layout
        // doesn't require every consumer of this set to read them.
        std::array<VkDescriptorSetLayoutBinding, 3> storageBindings{};
        storageBindings[0].binding = 0;
        storageBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        storageBindings[0].descriptorCount = 1;
        storageBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        storageBindings[1].binding = 1;
        storageBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        storageBindings[1].descriptorCount = 1;
        storageBindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        storageBindings[2].binding = 2;
        storageBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        storageBindings[2].descriptorCount = 1;
        storageBindings[2].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(storageBindings.size());
        layoutInfo.pBindings = storageBindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_ParticleGPUSetLayout) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to create particle-GPU descriptor set layout");
        }

        // Dedicated tiny pool - one set, three bindings - rather than
        // resizing m_DescriptorPool (which is sized exactly for the
        // camera-UBO/texture sets already allocated by the time this
        // runs). See VulkanDescriptorPool's own comment on why pool
        // sizing is caller-specified rather than hardcoded.
        std::vector<VkDescriptorPoolSize> poolSizes = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 } };
        m_ParticleGPUDescriptorPool.Init(device, poolSizes, 1);
        m_ParticleGPUStorageSet = m_ParticleGPUDescriptorPool.AllocateSet(m_ParticleGPUSetLayout);

        VkDescriptorBufferInfo positionInfo{ positionBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo colorInfo{ colorBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo previousPositionInfo{ m_ParticlePreviousPositions.Get(), 0, VK_WHOLE_SIZE };
        std::array<VkWriteDescriptorSet, 3> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = m_ParticleGPUStorageSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &positionInfo;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = m_ParticleGPUStorageSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &colorInfo;
        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = m_ParticleGPUStorageSet;
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo = &previousPositionInfo;
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

        // Set 0 = camera UBO (REUSES m_DescriptorSetLayout - the exact
        // same VkDescriptorSetLayout object m_PipelineLayout's set 0
        // already uses, which is what makes frame.descriptorSet, bound
        // every frame for the ordinary mesh draws, ALSO valid to bind
        // against this different pipeline layout at set 0 - see
        // DrawFrame). Set 1 = the storage buffer above.
        VkPushConstantRange colorPushConstant{};
        colorPushConstant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        colorPushConstant.offset = 0;
        colorPushConstant.size = sizeof(glm::vec4);

        m_ParticleGPUPipelineLayout.Init(device, { m_DescriptorSetLayout.Get(), m_ParticleGPUSetLayout }, { colorPushConstant });

        // No vertex/instance input at all - position comes from the
        // storage buffer via gl_InstanceIndex, not a bound vertex buffer
        // (see particle_point_gpu.vert). Reuses particle_point.frag
        // unchanged (round-point discard + flat color), same as
        // m_ParticlePointsPipeline does for its own fragment stage.
        m_ParticlePointsGPUPipeline.Init(
            device,
            m_RenderPass.Get(),
            m_ParticleGPUPipelineLayout.Get(),
            "Shaders/Compiled/particle_point_gpu.vert.spv",
            "Shaders/Compiled/particle_point.frag.spv",
            {},
            {},
            VK_PRIMITIVE_TOPOLOGY_POINT_LIST);

        // Fluid-surface pass setup lives HERE, not in Init() alongside
        // everything else in this file, because m_FluidDepthPipelineLayout's
        // set 1 IS m_ParticleGPUSetLayout (just created a few lines up) -
        // it reads the exact same position buffer this function just
        // registered, through a different pipeline/vertex shader, rather
        // than standing up a second copy of that binding (see
        // m_FluidDepthPipelineLayout's own comment in Renderer.h). Every
        // OTHER dependency (m_DescriptorSetLayout, m_TextureSetLayout,
        // m_RenderPass, m_CommandPool) was already created earlier in
        // Init(), before main.cpp ever calls RegisterParticleGPUSource -
        // so this is the earliest point everything this pass needs
        // actually exists together.
        InitFluidSurfacePass();
        CreateFluidSurfaceResources();
        m_FluidSurfaceInitialized = true;

        InitFluidBounds(); // needs m_ParticleLivePositionBuffer (set above) - see fluid_bounds.comp
    }


    void Renderer::InitRaymarchPass()
    {
        VkDevice device = m_Context.Device().Get();
        VmaAllocator allocator = m_Allocator.Get();

        // Set 1: two bindings, both ORDINARY fixed-count storage
        // buffers (descriptorCount = 1 each) - no descriptor indexing,
        // no update-after-bind, no bindless anything. This is
        // deliberate: an earlier version of this pass tried true
        // bindless (one descriptor slot per raymarch object's own
        // density buffer, later an unbounded densityBuffers[] array via
        // descriptor indexing) to get past MoltenVK's
        // maxPerStageDescriptorStorageBuffers=31 ceiling, and hit real,
        // confirmed dead ends on this project's actual dev hardware at
        // every step (see RaymarchTypes.h's kRaymarchMaxObjects comment
        // for the full history, including vkGetDescriptorSetLayoutSupport
        // returning supported=VK_FALSE for the bindless layout outright).
        //
        // The actual fix doesn't touch descriptor count at all: EVERY
        // volume's density data now lives in one shared buffer
        // (VoxelSystemGPU::m_SharedDensityBuffer), so binding 1 here is
        // just ONE ordinary storage-buffer descriptor regardless of how
        // many raymarch objects exist this frame - each object finds
        // its own region within that one buffer via
        // RaymarchObjectGPU::densityOffset (added in raymarch.frag's
        // SampleDensityTrilinear). This is portable by construction -
        // it only uses Vulkan 1.0-guaranteed-minimum storage-buffer
        // limits (every conformant implementation supports at least a
        // handful of per-stage storage buffers; this pass only ever
        // needs 2), not any optional/tier-dependent feature.
        //
        // Binding 0: the per-frame RaymarchObjectGPU array
        // (kRaymarchMaxObjects entries, rewritten wholesale every frame
        // - see UpdateRaymarchDescriptors).
        // Binding 1: the ONE shared density buffer - see above.
        VkDescriptorSetLayoutBinding objectBinding{};
        objectBinding.binding = 0;
        objectBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        objectBinding.descriptorCount = 1;
        objectBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding densityBinding{};
        densityBinding.binding = 1;
        densityBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        densityBinding.descriptorCount = 1;
        densityBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        std::array<VkDescriptorSetLayoutBinding, 2> bindings = { objectBinding, densityBinding };

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_RaymarchSetLayout) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to create raymarch descriptor set layout");
        }

        // Pool sized for MAX_FRAMES_IN_FLIGHT sets, each with 2 storage-
        // buffer descriptors - same "dedicated tiny pool" choice
        // RegisterParticleGPUSource makes, for the same reason (this
        // set's budget has nothing to do with m_DescriptorPool's mesh-
        // texture sizing). Ordinary vkCreateDescriptorPool, no
        // UPDATE_AFTER_BIND flag - not needed, per-frame sets solve the
        // same problem without it (see m_RaymarchSets' own comment).
        std::vector<VkDescriptorPoolSize> poolSizes = {
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * MAX_FRAMES_IN_FLIGHT }
        };

        VkDescriptorPoolCreateInfo raymarchPoolInfo{};
        raymarchPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        raymarchPoolInfo.maxSets = MAX_FRAMES_IN_FLIGHT;
        raymarchPoolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        raymarchPoolInfo.pPoolSizes = poolSizes.data();

        if (vkCreateDescriptorPool(device, &raymarchPoolInfo, nullptr, &m_RaymarchDescriptorPool) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to create raymarch descriptor pool");
        }

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            VkDescriptorSetAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocInfo.descriptorPool = m_RaymarchDescriptorPool;
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts = &m_RaymarchSetLayout;

            VkResult allocResult = vkAllocateDescriptorSets(device, &allocInfo, &m_RaymarchSets[i]);
            if (allocResult != VK_SUCCESS)
            {
                throw std::runtime_error(
                    "Eden: failed to allocate raymarch descriptor set (VkResult "
                    + std::to_string(static_cast<int>(allocResult)) + ")");
            }

            // Host-visible + persistently mapped - rewritten wholesale
            // every frame from RaymarchSystem::BuildObjectList's output,
            // same "small and constantly rewritten, skip the staging
            // round trip" reasoning as VoxelSystemGPU::Volume::instance.
            m_RaymarchObjectBuffers[i].Init(allocator, sizeof(RaymarchObjectGPU) * kRaymarchMaxObjects,
                                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                                             VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                                             &m_RaymarchObjectBuffersMapped[i]);
        }

        // Set 0 = camera UBO, reusing m_DescriptorSetLayout exactly like
        // m_ParticleGPUPipelineLayout does (see that function's own
        // comment on why the same VkDescriptorSetLayout object is valid
        // to bind at set 0 across different pipeline layouts). Set 1 =
        // m_RaymarchSetLayout above.
        VkPushConstantRange pushConstant{};
        pushConstant.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pushConstant.offset = 0;
        pushConstant.size = sizeof(RaymarchPushConstants);

        m_RaymarchPipelineLayout.Init(device, { m_DescriptorSetLayout.Get(), m_RaymarchSetLayout }, { pushConstant });

        // No vertex/instance input (see raymarch.vert), TRIANGLE_LIST
        // topology with a 3-vertex bufferless draw, cullMode=NONE - a
        // fullscreen triangle has no back face to cull.
        m_RaymarchPipeline.Init(
            device,
            m_RenderPass.Get(),
            m_RaymarchPipelineLayout.Get(),
            "Shaders/Compiled/raymarch.vert.spv",
            "Shaders/Compiled/raymarch.frag.spv",
            {},
            {},
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            VK_CULL_MODE_NONE);
    }

    void Renderer::UpdateRaymarchDescriptors(const std::vector<RaymarchObjectGPU>& objects,
                                              VkBuffer densityBuffer)
    {
        VkDevice device = m_Context.Device().Get();

        // objects.size() may legitimately exceed kRaymarchMaxObjects for
        // one frame (RaymarchSystem::BuildObjectList already clamps and
        // warns - see that function - but this is defensive in case a
        // caller ever builds the list some other way).
        uint32_t liveCount = std::min(static_cast<uint32_t>(objects.size()), kRaymarchMaxObjects);

        // Indexed by m_CurrentFrame throughout this function - see
        // m_RaymarchSets' own comment on why this can't be a single
        // shared set/buffer.
        if (liveCount > 0)
        {
            std::memcpy(m_RaymarchObjectBuffersMapped[m_CurrentFrame], objects.data(), sizeof(RaymarchObjectGPU) * liveCount);
        }

        VkDescriptorBufferInfo objectBufferInfo{ m_RaymarchObjectBuffers[m_CurrentFrame].Get(), 0, VK_WHOLE_SIZE };

        VkWriteDescriptorSet objectWrite{};
        objectWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        objectWrite.dstSet = m_RaymarchSets[m_CurrentFrame];
        objectWrite.dstBinding = 0;
        objectWrite.descriptorCount = 1;
        objectWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        objectWrite.pBufferInfo = &objectBufferInfo;

        std::array<VkWriteDescriptorSet, 2> writes{};
        writes[0] = objectWrite;

        // Only ONE density-buffer descriptor to write now, always -
        // see InitRaymarchPass's opening comment. When densityBuffer is
        // VK_NULL_HANDLE (no raymarch objects exist yet this frame -
        // DrawFrame only calls this when raymarchObjects is non-empty,
        // but defensively handled here too), skip this write entirely
        // rather than binding a null buffer, which would be invalid.
        uint32_t writeCount = 1;
        VkDescriptorBufferInfo densityInfo{};
        if (densityBuffer != VK_NULL_HANDLE)
        {
            densityInfo = { densityBuffer, 0, VK_WHOLE_SIZE };

            VkWriteDescriptorSet densityWrite{};
            densityWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            densityWrite.dstSet = m_RaymarchSets[m_CurrentFrame];
            densityWrite.dstBinding = 1;
            densityWrite.descriptorCount = 1;
            densityWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            densityWrite.pBufferInfo = &densityInfo;

            writes[1] = densityWrite;
            writeCount = 2;
        }

        vkUpdateDescriptorSets(device, writeCount, writes.data(), 0, nullptr);
    }

    // ---------------------------------------------------------------------
    // Fluid screen-space bounds
    // ---------------------------------------------------------------------
    void Renderer::InitFluidBounds()
    {
        VkDevice device = m_Context.Device().Get();
        VmaAllocator allocator = m_Allocator.Get();

        std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
        bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        bindings[2] = { 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_FluidBoundsSetLayout) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to create fluid bounds descriptor set layout");
        }

        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
        poolSizes[1] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 };
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_FluidBoundsPool) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to create fluid bounds descriptor pool");
        }
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = m_FluidBoundsPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_FluidBoundsSetLayout;
        if (vkAllocateDescriptorSets(device, &allocInfo, &m_FluidBoundsSet) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to allocate fluid bounds descriptor set");
        }

        m_FluidBoundsBuffer.Init(allocator, sizeof(FluidScreenBounds),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                  VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            m_FluidBoundsReadback[i].Init(allocator, sizeof(FluidScreenBounds), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                           VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                           &m_FluidBoundsReadbackMapped[i]);
            m_FluidBoundsReadbackPending[i] = false;
        }

        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(float) * 28; // mat4 + 3 x vec4 (see fluid_bounds.comp)
        m_FluidBoundsPipelineLayout.Init(device, { m_FluidBoundsSetLayout }, { pushRange });
        m_FluidBoundsPipeline.Init(device, m_FluidBoundsPipelineLayout.Get(), "Shaders/Compiled/fluid_bounds.comp.spv");

        WriteFluidBoundsDescriptors();
    }

    void Renderer::DestroyFluidBounds()
    {
        VkDevice device = m_Context.Device().Get();
        m_FluidBoundsPipeline.Shutdown();
        m_FluidBoundsPipelineLayout.Shutdown();
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            m_FluidBoundsReadback[i].Shutdown();
            m_FluidBoundsReadbackMapped[i] = nullptr;
        }
        m_FluidBoundsBuffer.Shutdown();
        if (m_FluidBoundsPool != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(device, m_FluidBoundsPool, nullptr);
            m_FluidBoundsPool = VK_NULL_HANDLE;
            m_FluidBoundsSet = VK_NULL_HANDLE;
        }
        if (m_FluidBoundsSetLayout != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(device, m_FluidBoundsSetLayout, nullptr);
            m_FluidBoundsSetLayout = VK_NULL_HANDLE;
        }
    }

    // Safe to call whenever something it depends on changes (particle source
    // registered, depth image recreated): does nothing until everything it
    // points at exists.
    void Renderer::WriteFluidBoundsDescriptors()
    {
        if (m_FluidBoundsSet == VK_NULL_HANDLE || m_ParticleLivePositionBuffer == VK_NULL_HANDLE ||
            m_FluidBoundsBuffer.Get() == VK_NULL_HANDLE || m_DepthImage.GetView() == VK_NULL_HANDLE ||
            m_FluidSampler.Get() == VK_NULL_HANDLE)
        {
            return;
        }

        VkDescriptorBufferInfo positionInfo{ m_ParticleLivePositionBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo boundsInfo{ m_FluidBoundsBuffer.Get(), 0, VK_WHOLE_SIZE };
        VkDescriptorImageInfo depthInfo{};
        depthInfo.sampler = m_FluidSampler.Get();
        depthInfo.imageView = m_DepthImage.GetView();
        depthInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        std::array<VkWriteDescriptorSet, 3> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = m_FluidBoundsSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &positionInfo;
        writes[1] = writes[0];
        writes[1].dstBinding = 1;
        writes[1].pBufferInfo = &boundsInfo;
        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = m_FluidBoundsSet;
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[2].pImageInfo = &depthInfo;
        vkUpdateDescriptorSets(m_Context.Device().Get(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    // Called right after this slot's fence wait (like ReadbackGpuTimers): the
    // copy recorded the last time this slot ran has finished.
    void Renderer::ReadbackFluidBounds(uint32_t slot)
    {
        if (!m_FluidBoundsReadbackPending[slot] || m_FluidBoundsReadbackMapped[slot] == nullptr)
        {
            return;
        }
        m_FluidBoundsReadbackPending[slot] = false;

        vmaInvalidateAllocation(m_Allocator.Get(), m_FluidBoundsReadback[slot].GetAllocation(), 0, sizeof(FluidScreenBounds));
        std::memcpy(&m_LastFluidBounds, m_FluidBoundsReadbackMapped[slot], sizeof(FluidScreenBounds));
        m_LastFluidBoundsValid = true;
        m_FluidEmptyReadings = (m_LastFluidBounds.visibleCount == 0) ? m_FluidEmptyReadings + 1 : 0;
    }

    void Renderer::RecordFluidBounds(VkCommandBuffer cmd, uint32_t particleCount, const glm::mat4& viewProj, VkExtent2D extent)
    {
        if (m_FluidBoundsSet == VK_NULL_HANDLE || particleCount == 0 || m_FluidBoundsPipeline.Get() == VK_NULL_HANDLE ||
            m_FluidBoundsReadbackMapped[m_CurrentFrame] == nullptr)
        {
            return;
        }

        // Reset the accumulator: min = +huge, max = -huge, counters = 0.
        FluidScreenBounds initial{ 0x3FFFFFFF, 0x3FFFFFFF, -0x3FFFFFFF, -0x3FFFFFFF, 0, 0, 0, 0 };
        vkCmdUpdateBuffer(cmd, m_FluidBoundsBuffer.Get(), 0, sizeof(initial), &initial);

        VkMemoryBarrier toCompute{};
        toCompute.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        toCompute.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toCompute.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &toCompute, 0, nullptr, 0, nullptr);

        struct
        {
            glm::mat4 viewProj;
            glm::vec4 cameraPosAndRadius;
            glm::vec4 screenAndScale;
            glm::uvec4 meta;
        } push{};
        push.viewProj = viewProj;
        const glm::mat4 proj = m_Camera.GetProjectionMatrix(static_cast<float>(extent.width) / static_cast<float>(extent.height));
        push.cameraPosAndRadius = glm::vec4(m_Camera.Position, FluidParticleRadius);
        push.screenAndScale = glm::vec4(static_cast<float>(extent.width), static_cast<float>(extent.height),
                                         proj[0][0] * 0.5f * static_cast<float>(extent.width),
                                         proj[1][1] * 0.5f * static_cast<float>(extent.height));
        push.meta = glm::uvec4(particleCount, 0, 0, 0);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_FluidBoundsPipeline.Get());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_FluidBoundsPipelineLayout.Get(), 0, 1, &m_FluidBoundsSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_FluidBoundsPipelineLayout.Get(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cmd, (particleCount + 255u) / 256u, 1, 1);

        // Copy the result to this frame slot's host-visible buffer for the CPU
        // to read after the slot's fence next time round.
        VkMemoryBarrier toCopy{};
        toCopy.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        toCopy.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        toCopy.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &toCopy, 0, nullptr, 0, nullptr);

        VkBufferCopy region{ 0, 0, sizeof(FluidScreenBounds) };
        vkCmdCopyBuffer(cmd, m_FluidBoundsBuffer.Get(), m_FluidBoundsReadback[m_CurrentFrame].Get(), 1, &region);

        VkMemoryBarrier toHost{};
        toHost.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        toHost.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toHost.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &toHost, 0, nullptr, 0, nullptr);

        m_FluidBoundsReadbackPending[m_CurrentFrame] = true;
    }

    // ---------------------------------------------------------------------
    // GPU pass timers
    // ---------------------------------------------------------------------
    void Renderer::InitGpuTimers()
    {
        VkPhysicalDevice physicalDevice = m_Context.PhysicalDevice().Get();
        VkDevice device = m_Context.Device().Get();

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physicalDevice, &props);

        // The graphics queue must support timestamps (validBits > 0) and
        // timestamps must be comparable across graphics+compute work, since
        // the SPH dispatches are timed alongside render passes.
        uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, families.data());
        const uint32_t graphicsFamily = m_Context.PhysicalDevice().GetQueueFamilies().graphicsFamily.value();

        const bool queueOk = graphicsFamily < familyCount && families[graphicsFamily].timestampValidBits > 0;
        if (!queueOk || props.limits.timestampPeriod <= 0.0f || !props.limits.timestampComputeAndGraphics)
        {
            m_GpuTimingSupported = false;
            return;
        }
        m_GpuTimestampPeriodNs = props.limits.timestampPeriod;

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            VkQueryPoolCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            info.queryType = VK_QUERY_TYPE_TIMESTAMP;
            info.queryCount = kMaxGpuStamps;
            if (vkCreateQueryPool(device, &info, nullptr, &m_GpuTimerPools[i]) != VK_SUCCESS)
            {
                DestroyGpuTimers();
                m_GpuTimingSupported = false;
                return;
            }
        }
        m_GpuTimingSupported = true;
    }

    void Renderer::DestroyGpuTimers()
    {
        VkDevice device = m_Context.Device().Get();
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            if (m_GpuTimerPools[i] != VK_NULL_HANDLE)
            {
                vkDestroyQueryPool(device, m_GpuTimerPools[i], nullptr);
                m_GpuTimerPools[i] = VK_NULL_HANDLE;
            }
            m_GpuTimerLabels[i].clear();
        }
    }

    // Called right after this slot's fence wait, so the previous use of the
    // slot (MAX_FRAMES_IN_FLIGHT frames ago) has fully finished on the GPU.
    void Renderer::ReadbackGpuTimers(uint32_t slot)
    {
        std::vector<const char*>& labels = m_GpuTimerLabels[slot];
        if (labels.size() < 2 || m_GpuTimerPools[slot] == VK_NULL_HANDLE)
        {
            labels.clear();
            return;
        }

        std::array<uint64_t, kMaxGpuStamps> stamps{};
        VkResult result = vkGetQueryPoolResults(
            m_Context.Device().Get(), m_GpuTimerPools[slot], 0, static_cast<uint32_t>(labels.size()),
            labels.size() * sizeof(uint64_t), stamps.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);

        if (result == VK_SUCCESS)
        {
            for (size_t i = 1; i < labels.size(); ++i)
            {
                if (stamps[i] >= stamps[i - 1])
                {
                    const double ns = static_cast<double>(stamps[i] - stamps[i - 1]) * static_cast<double>(m_GpuTimestampPeriodNs);
                    Profiler::Get().SetGpu(labels[i], static_cast<float>(ns * 1.0e-6));
                }
            }
        }
        labels.clear();
    }

    void Renderer::GpuStamp(VkCommandBuffer cmd, const char* label)
    {
        if (!GpuTimingEnabled || !m_GpuTimingSupported)
        {
            return;
        }
        std::vector<const char*>& labels = m_GpuTimerLabels[m_CurrentFrame];
        if (labels.size() >= kMaxGpuStamps)
        {
            return;
        }
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_GpuTimerPools[m_CurrentFrame],
                            static_cast<uint32_t>(labels.size()));
        labels.push_back(label);
    }

    void Renderer::InitFluidSurfacePass()
    {
        VkDevice device = m_Context.Device().Get();

        InitGpuTimers();

        m_FluidSampler.Init(device, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
        m_FluidLinearSampler.Init(device, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

        // --- Depth prepass render pass: color(R32_SFLOAT) + depth ------
        // Color attachment ends in SHADER_READ_ONLY_OPTIMAL (sampled by
        // the blur pass right after), not PRESENT_SRC_KHR like the main
        // render pass's swapchain attachment.
        {
            VkAttachmentDescription colorAttachment{};
            colorAttachment.format = VK_FORMAT_R32G32B32A32_SFLOAT; // R = world distance, GBA = particle color - see fluid_depth.frag
            colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
            colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkAttachmentDescription depthAttachment{};
            depthAttachment.format = FindDepthFormat();
            depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
            depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; // local to this pass only, see m_FluidDepthPassDSs' comment
            depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            // Second color output - per-particle screen-space motion
            // vector (see fluid_depth.vert's motion-vector comment and
            // fluid_depth.frag's second output), read directly by
            // fluid_temporal.frag for reprojection - unlike the distance+
            // color attachment, this is NOT blurred/polished, so it skips
            // those passes entirely. RG is enough (a 2D screen-space
            // delta); 16-bit float is more than enough precision/range
            // for a quantity that's normally a small fraction of the
            // screen per frame.
            VkAttachmentDescription motionAttachment{};
            motionAttachment.format = VK_FORMAT_R16G16_SFLOAT;
            motionAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
            motionAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // cleared to 0 motion - see this pass's clear values in DrawFrame
            motionAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            motionAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            motionAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            motionAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            motionAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            // Third color output: analytic world-space sphere normal (see
            // fluid_depth.frag's outNormal). RGBA16F is plenty for a unit vector.
            VkAttachmentDescription normalAttachment{};
            normalAttachment.format = VK_FORMAT_R16G16B16A16_SFLOAT;
            normalAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
            normalAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // cleared to 0 - composite treats a ~zero normal as "none here"
            normalAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            normalAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            normalAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            normalAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            normalAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkAttachmentReference colorRef{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
            VkAttachmentReference motionRef{ 1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
            VkAttachmentReference normalRef{ 2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
            VkAttachmentReference depthRef{ 3, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
            std::array<VkAttachmentReference, 3> colorRefs = { colorRef, motionRef, normalRef };

            VkSubpassDescription subpass{};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount = static_cast<uint32_t>(colorRefs.size());
            subpass.pColorAttachments = colorRefs.data();
            subpass.pDepthStencilAttachment = &depthRef;

            // Entry dependency - same shape as VulkanRenderPass.cpp's own
            // (external write hazard into a fresh color+depth attachment).
            VkSubpassDependency enterDependency{};
            enterDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
            enterDependency.dstSubpass = 0;
            enterDependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            enterDependency.srcAccessMask = 0;
            enterDependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            enterDependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

            // Exit dependency - this pass's color output is about to be
            // SAMPLED by the blur pass right after, not presented, so
            // (unlike the main render pass) this one also needs to
            // publish a color-attachment-write -> fragment-shader-read
            // barrier before that next pass's draw call is safe.
            VkSubpassDependency exitDependency{};
            exitDependency.srcSubpass = 0;
            exitDependency.dstSubpass = VK_SUBPASS_EXTERNAL;
            exitDependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            exitDependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            exitDependency.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            exitDependency.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            std::array<VkAttachmentDescription, 4> attachments = { colorAttachment, motionAttachment, normalAttachment, depthAttachment };
            std::array<VkSubpassDependency, 2> dependencies = { enterDependency, exitDependency };

            VkRenderPassCreateInfo renderPassInfo{};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            renderPassInfo.pAttachments = attachments.data();
            renderPassInfo.subpassCount = 1;
            renderPassInfo.pSubpasses = &subpass;
            renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
            renderPassInfo.pDependencies = dependencies.data();

            if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &m_FluidDepthPrepassRenderPass) != VK_SUCCESS)
            {
                throw std::runtime_error("Eden: failed to create fluid depth prepass render pass");
            }
        }

        // --- Blur render pass: single color attachment, no depth -------
        // Reused for BOTH the horizontal and vertical blur draw (see
        // DrawFrame) - same format/config either way, only which
        // framebuffer/descriptor set gets bound differs.
        {
            VkAttachmentDescription colorAttachment{};
            colorAttachment.format = VK_FORMAT_R32G32B32A32_SFLOAT; // R = world distance, GBA = particle color - see fluid_depth.frag
            colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
            colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; // fullscreen triangle overwrites every pixel, no need to clear first
            colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkAttachmentReference colorRef{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

            VkSubpassDescription subpass{};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = &colorRef;

            // Enter dependency: this pass writing to its OWN fresh color
            // attachment (same "undefined -> color attachment optimal"
            // shape as the depth prepass's enter dependency above and
            // VulkanRenderPass.cpp's existing one) - NOT the hazard of
            // reading the OTHER pass's output texture as a sampler input.
            // That second hazard (must-happen-before-this-pass-samples-it)
            // is already covered by the SOURCE render pass's own EXIT
            // dependency (see the depth prepass's exitDependency above,
            // and this same pass's exitDependency below when it's the
            // producer for the next blur direction) - a producer's exit
            // dependency into VK_SUBPASS_EXTERNAL correctly orders
            // against a later, separate render pass's matching access,
            // it doesn't need to be repeated as this pass's entry
            // dependency too.
            VkSubpassDependency enterDependency{};
            enterDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
            enterDependency.dstSubpass = 0;
            // FRAGMENT_SHADER + SHADER_READ added to the source scope: this
            // render pass is reused for the temporal pass, which WRITES a
            // history image that the previous frame's temporal pass and
            // composite SAMPLED (ping-pong). Without the fragment-shader
            // read in the first scope that write-after-read had no
            // execution dependency, so on GPUs that overlap consecutive
            // frames (tile-based / MoltenVK) a new frame could overwrite a
            // history image the old frame was still reading.
            enterDependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            enterDependency.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            enterDependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            enterDependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

            VkSubpassDependency exitDependency{};
            exitDependency.srcSubpass = 0;
            exitDependency.dstSubpass = VK_SUBPASS_EXTERNAL;
            exitDependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            exitDependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            exitDependency.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            exitDependency.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            std::array<VkSubpassDependency, 2> dependencies = { enterDependency, exitDependency };

            VkRenderPassCreateInfo renderPassInfo{};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            renderPassInfo.attachmentCount = 1;
            renderPassInfo.pAttachments = &colorAttachment;
            renderPassInfo.subpassCount = 1;
            renderPassInfo.pSubpasses = &subpass;
            renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
            renderPassInfo.pDependencies = dependencies.data();

            if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &m_FluidBlurRenderPass) != VK_SUCCESS)
            {
                throw std::runtime_error("Eden: failed to create fluid blur render pass");
            }
        }

        // --- Thickness render pass: single R16_SFLOAT color, CLEAR to 0 ---
        // Additively accumulated by fluid_thickness.frag, sampled by the
        // composite pass (inside the MAIN render pass) for opacity. CLEAR
        // (not DONT_CARE like the blur pass) because additive blending
        // needs a known 0 to accumulate onto.
        {
            VkAttachmentDescription colorAttachment{};
            colorAttachment.format = VK_FORMAT_R16_SFLOAT;
            colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
            colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkAttachmentReference colorRef{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

            VkSubpassDescription subpass{};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = &colorRef;

            // Enter: this slot's image was last SAMPLED by a previous
            // frame's composite (fragment shader read) - wait for that
            // before overwriting it, not just for earlier color writes.
            VkSubpassDependency enterDependency{};
            enterDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
            enterDependency.dstSubpass = 0;
            enterDependency.srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            enterDependency.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            enterDependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            enterDependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT; // READ: blending reads the destination

            VkSubpassDependency exitDependency{};
            exitDependency.srcSubpass = 0;
            exitDependency.dstSubpass = VK_SUBPASS_EXTERNAL;
            exitDependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            exitDependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            exitDependency.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            exitDependency.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            std::array<VkSubpassDependency, 2> dependencies = { enterDependency, exitDependency };

            VkRenderPassCreateInfo renderPassInfo{};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            renderPassInfo.attachmentCount = 1;
            renderPassInfo.pAttachments = &colorAttachment;
            renderPassInfo.subpassCount = 1;
            renderPassInfo.pSubpasses = &subpass;
            renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
            renderPassInfo.pDependencies = dependencies.data();

            if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &m_FluidThicknessRenderPass) != VK_SUCCESS)
            {
                throw std::runtime_error("Eden: failed to create fluid thickness render pass");
            }
        }

        // --- Depth prepass pipeline -------------------------------------
        // Set 0 = camera UBO (m_DescriptorSetLayout, reused exactly like
        // every other pipeline layout in this file). Set 1 =
        // m_ParticleGPUSetLayout - the SAME layout (and, at draw time,
        // the same already-allocated m_ParticleGPUStorageSet)
        // RegisterParticleGPUSource created for the raw-point path; this
        // pipeline reads the identical position buffer through a
        // different vertex shader, not a second copy of the binding.
        VkPushConstantRange fluidDepthPushConstant{};
        // Back to vertex-only, radius-only - deltaTime no longer flows
        // through this push constant at all now that motion vectors are
        // built from m_ParticlePreviousPositions (an exact buffer
        // snapshot) instead of velocity*deltaTime - see that buffer's own
        // comment (Renderer.h) for why.
        fluidDepthPushConstant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        fluidDepthPushConstant.offset = 0;
        fluidDepthPushConstant.size = sizeof(float) * 4; // radius (vertex), 1/targetW, 1/targetH, pad (fragment) - see fluid_depth.vert/.frag

        m_FluidDepthPipelineLayout.Init(device, { m_DescriptorSetLayout.Get(), m_ParticleGPUSetLayout, m_TextureSetLayout.Get() /* scene depth */ },
                                         { fluidDepthPushConstant });

        m_FluidDepthPipeline.Init(
            device,
            m_FluidDepthPrepassRenderPass,
            m_FluidDepthPipelineLayout.Get(),
            "Shaders/Compiled/fluid_depth.vert.spv",
            "Shaders/Compiled/fluid_depth.frag.spv",
            {}, {},
            // TRIANGLE_LIST billboard quads, not POINT_LIST - see
            // fluid_depth.vert's file comment for why the point-sprite
            // version got replaced (gl_PointSize hardware clamp bug).
            // cullMode explicitly NONE: a billboard's winding relative to
            // the view direction isn't guaranteed by construction the
            // way ordinary front-facing mesh geometry's is, and the
            // default VK_CULL_MODE_BACK_BIT would silently discard every
            // quad if it came out backwards - points were never subject
            // to face culling at all, so this risk didn't exist before
            // this pass switched to real triangles.
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            VK_CULL_MODE_NONE,
            // depthTestEnable/depthWriteEnable left at their true/true
            // defaults - overlapping sphere impostors need real depth
            // resolution within this pass, see m_FluidDepthPassDSs.
            /*depthTestEnable=*/true,
            /*depthWriteEnable=*/true,
            // 2 color attachments now - distance+color AND the motion
            // vector target, see m_FluidDepthPrepassRenderPass's own
            // comment on why this pass got a second output.
            /*colorAttachmentCount=*/3);

        // --- Blur pipeline -----------------------------------------------
        // Set 0 = m_TextureSetLayout (one combined image sampler) - no
        // camera UBO, pure image-space work. depthTestEnable/
        // depthWriteEnable explicitly FALSE - this render pass has no
        // depth attachment at all (see m_FluidBlurRenderPass above).
        VkPushConstantRange blurPushConstant{};
        blurPushConstant.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        blurPushConstant.offset = 0;
        blurPushConstant.size = sizeof(float) * 6; // texelSize.xy, direction.xy, pixelsPerWorldUnitAtUnitDistance, particleRadius - see fluid_blur.frag

        m_FluidBlurPipelineLayout.Init(device, { m_TextureSetLayout.Get() }, { blurPushConstant });

        m_FluidBlurPipeline.Init(
            device,
            m_FluidBlurRenderPass,
            m_FluidBlurPipelineLayout.Get(),
            "Shaders/Compiled/raymarch.vert.spv", // generic fullscreen triangle - see fluid_blur.frag's file comment
            "Shaders/Compiled/fluid_blur.frag.spv",
            {}, {},
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            VK_CULL_MODE_NONE,
            /*depthTestEnable=*/false,
            /*depthWriteEnable=*/false);

        // --- Polish pipeline (fluid_blur_polish.frag) ---------------------
        // Same set 0 layout as the blur pipeline; smaller push constant
        // (no direction - this pass isn't separable, see that shader's
        // own comment).
        VkPushConstantRange polishPushConstant{};
        polishPushConstant.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        polishPushConstant.offset = 0;
        polishPushConstant.size = sizeof(float) * 4; // texelSize.xy, pixelsPerWorldUnitAtUnitDistance, particleRadius - see fluid_blur_polish.frag

        m_FluidPolishPipelineLayout.Init(device, { m_TextureSetLayout.Get() }, { polishPushConstant });

        m_FluidPolishPipeline.Init(
            device,
            m_FluidBlurRenderPass, // reused - identical single-color-attachment shape, see m_FluidPolishFramebuffers' comment
            m_FluidPolishPipelineLayout.Get(),
            "Shaders/Compiled/raymarch.vert.spv",
            "Shaders/Compiled/fluid_blur_polish.frag.spv",
            {}, {},
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            VK_CULL_MODE_NONE,
            /*depthTestEnable=*/false,
            /*depthWriteEnable=*/false);

        // --- Temporal pipeline (fluid_temporal.frag) -----------------------
        // Three set slots, ALL using m_TextureSetLayout (one combined
        // image sampler each) - set 0 "new frame", set 1 "motion vector",
        // set 2 "history", matching that shader's three sampler bindings.
        // No push constant - see that shader's own file comment.
        m_FluidTemporalPipelineLayout.Init(
            device, { m_TextureSetLayout.Get(), m_TextureSetLayout.Get(), m_TextureSetLayout.Get() }, {});

        m_FluidTemporalPipeline.Init(
            device,
            m_FluidBlurRenderPass, // reused - identical single-color-attachment shape, see m_FluidHistoryFramebuffers' comment
            m_FluidTemporalPipelineLayout.Get(),
            "Shaders/Compiled/raymarch.vert.spv",
            "Shaders/Compiled/fluid_temporal.frag.spv",
            {}, {},
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            VK_CULL_MODE_NONE,
            /*depthTestEnable=*/false,
            /*depthWriteEnable=*/false);

        // --- Composite pipeline -------------------------------------------
        // Renders INTO THE MAIN render pass (m_RenderPass), not one of
        // the offscreen ones above - see fluid_composite.frag's file
        // comment for why (needs to depth-test against, and be occluded
        // by, whatever opaque/raymarched geometry the main pass already
        // drew this frame). Set 0 = camera UBO, set 1 = the final
        // blurred depth+color texture. No push constant - the Chemical ID
        // base build removed the old uniform tintColor in favor of
        // reading actual per-particle diffused color out of the texture
        // itself (see fluid_composite.frag's surfaceColor).
        // Set 2 = the thickness texture (opacity), plus 4 floats of
        // fragment push constants (absorption, minAlpha, maxAlpha, pad) -
        // see fluid_composite.frag's CompositePushConstants. Blending is
        // PREMULTIPLIED alpha: the shader outputs rgb already scaled by
        // alpha plus un-attenuated specular, so thin liquid shows the
        // scene behind it but keeps its highlights.
        VkPushConstantRange compositePushConstant{};
        compositePushConstant.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        compositePushConstant.offset = 0;
        compositePushConstant.size = sizeof(float) * 4;

        m_FluidCompositePipelineLayout.Init(
            device, { m_DescriptorSetLayout.Get(), m_TextureSetLayout.Get(), m_TextureSetLayout.Get(), m_TextureSetLayout.Get() },
            { compositePushConstant });

        m_FluidCompositePipeline.Init(
            device,
            m_RenderPass.Get(),
            m_FluidCompositePipelineLayout.Get(),
            "Shaders/Compiled/raymarch.vert.spv",
            "Shaders/Compiled/fluid_composite.frag.spv",
            {}, {},
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            VK_CULL_MODE_NONE,
            /*depthTestEnable=*/true,
            /*depthWriteEnable=*/true,
            /*colorAttachmentCount=*/1,
            /*enableAlphaBlend=*/true,
            /*premultipliedBlend=*/true);

        // --- Thickness pipeline (fluid_thickness.vert/.frag) --------------
        // Same particle storage set as the depth pass; additive blend, no
        // depth attachment (order-independent accumulation).
        VkPushConstantRange thicknessPushConstant{};
        thicknessPushConstant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        thicknessPushConstant.offset = 0;
        thicknessPushConstant.size = sizeof(float) * 4; // radius (vertex), 1/targetW, 1/targetH, window (fragment)

        m_FluidThicknessPipelineLayout.Init(device, { m_DescriptorSetLayout.Get(), m_ParticleGPUSetLayout, m_TextureSetLayout.Get() /* scene depth */, m_TextureSetLayout.Get() /* nearest liquid surface */ },
                                             { thicknessPushConstant });

        m_FluidThicknessPipeline.Init(
            device,
            m_FluidThicknessRenderPass,
            m_FluidThicknessPipelineLayout.Get(),
            "Shaders/Compiled/fluid_thickness.vert.spv",
            "Shaders/Compiled/fluid_thickness.frag.spv",
            {}, {},
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            VK_CULL_MODE_NONE,
            /*depthTestEnable=*/false,
            /*depthWriteEnable=*/false,
            /*colorAttachmentCount=*/1,
            /*enableAlphaBlend=*/true,
            /*premultipliedBlend=*/false,
            /*additiveBlend=*/true);
            // depthTestEnable/depthWriteEnable left at true/true - same
            // "write real depth, let the fixed-function test handle
            // occlusion" approach m_RaymarchPipeline already uses.

        // --- Soft-billboard pipeline (see FluidUseSoftBlobs/m_FluidBlobPipeline's
        // own comment, Renderer.h) ------------------------------------------
        // Also renders into the main render pass, same as composite just
        // above - the two are mutually exclusive alternatives for the
        // same draw slot in DrawFrame, not a sequence.
        VkPushConstantRange blobPushConstant{};
        blobPushConstant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        blobPushConstant.offset = 0;
        blobPushConstant.size = sizeof(float); // radius - see fluid_blob.vert

        m_FluidBlobPipelineLayout.Init(device, { m_DescriptorSetLayout.Get(), m_ParticleGPUSetLayout },
                                        { blobPushConstant });

        m_FluidBlobPipeline.Init(
            device,
            m_RenderPass.Get(),
            m_FluidBlobPipelineLayout.Get(),
            "Shaders/Compiled/fluid_blob.vert.spv",
            "Shaders/Compiled/fluid_blob.frag.spv",
            {}, {},
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            VK_CULL_MODE_NONE,
            /*depthTestEnable=*/true,
            /*depthWriteEnable=*/false, // see this pipeline's own comment (Renderer.h) for why
            /*colorAttachmentCount=*/1,
            /*enableAlphaBlend=*/true);
    }

    void Renderer::CreateFluidSurfaceResources()
    {
        VkDevice device = m_Context.Device().Get();
        VmaAllocator allocator = m_Allocator.Get();
        const VkExtent2D swapchainExtent = m_Context.Swapchain().GetExtent();
        const float fluidScale = std::min(1.0f, std::max(0.25f, FluidResolutionScale));
        m_AppliedFluidScale = FluidResolutionScale;
        VkExtent2D extent{
            std::max(1u, static_cast<uint32_t>(static_cast<float>(swapchainExtent.width) * fluidScale + 0.5f)),
            std::max(1u, static_cast<uint32_t>(static_cast<float>(swapchainExtent.height) * fluidScale + 0.5f))
        };
        m_FluidExtent = extent;
        // Bounds read back so far were for the old size/targets - start over
        // (full screen until fresh readings arrive).
        m_LastFluidBoundsValid = false;
        m_FluidEmptyReadings = 0;
        for (uint32_t s = 0; s < MAX_FRAMES_IN_FLIGHT; ++s) { m_FluidBoundsReadbackPending[s] = false; }
        // The thickness pass is already half-res at full fluid scale; once the
        // whole chain is scaled down, don't halve it again (it would get
        // too coarse for the rim opacity falloff).
        m_FluidThicknessDivisor = (fluidScale >= 0.99f) ? kFluidThicknessDivisor : 1u;

        // Dedicated tiny pool. 5 sets PER FRAME SLOT (blur X, blur Y,
        // polish, temporal-new, temporal-motion - see the per-slot arrays
        // below) plus 4 STATIC sets that exist only twice each regardless
        // of MAX_FRAMES_IN_FLIGHT (temporal-history, composite - see
        // m_FluidHistoryImages' own comment in Renderer.h for why those
        // are sized 2, not MAX_FRAMES_IN_FLIGHT). Was 3 * MAX_FRAMES_IN_FLIGHT
        // before the polish pass, 4 * MAX_FRAMES_IN_FLIGHT before the
        // temporal pass, 4 * MAX_FRAMES_IN_FLIGHT + 4 before reprojection
        // added the per-slot motion-vector input.
        const uint32_t kFluidDescriptorSetCount = 7 * MAX_FRAMES_IN_FLIGHT + 5; // +1: scene depth (single, not per frame slot) // thickness + analytic-normal textures are per-slot too
        std::vector<VkDescriptorPoolSize> poolSizes = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kFluidDescriptorSetCount }
        };
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = kFluidDescriptorSetCount;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_FluidDescriptorPool) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to create fluid descriptor pool");
        }

        auto allocateAndWrite = [&](VkImageView view, VkSampler samplerOverride = VK_NULL_HANDLE,
                                     VkImageLayout viewLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) -> VkDescriptorSet
        {
            VkDescriptorSetLayout layout = m_TextureSetLayout.Get();
            VkDescriptorSetAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocInfo.descriptorPool = m_FluidDescriptorPool;
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts = &layout;

            VkDescriptorSet set = VK_NULL_HANDLE;
            if (vkAllocateDescriptorSets(device, &allocInfo, &set) != VK_SUCCESS)
            {
                throw std::runtime_error("Eden: failed to allocate fluid descriptor set");
            }

            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = viewLayout;
            imageInfo.imageView = view;
            imageInfo.sampler = samplerOverride != VK_NULL_HANDLE ? samplerOverride : m_FluidSampler.Get();

            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set;
            write.dstBinding = 0;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.pImageInfo = &imageInfo;
            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

            return set;
        };

        // One full set of images/framebuffers/descriptor sets PER FRAME
        // SLOT - see m_FluidDepthFramebuffers' own comment (Renderer.h)
        // for why a single shared set raced with still-in-flight GPU
        // work. Each slot's descriptor sets are written ONCE here, not
        // every frame - the image views a slot points at never change
        // after creation (only recreated wholesale on resize, via
        // DestroyFluidSurfaceResources + this function running again),
        // unlike m_RaymarchSets which rewrites its bound CONTENTS every
        // frame because the underlying object data itself changes.
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            m_FluidDepthImages[i].Init(allocator, device, extent, VK_FORMAT_R32G32B32A32_SFLOAT,
                                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                        VK_IMAGE_ASPECT_COLOR_BIT);
            m_FluidMotionImages[i].Init(allocator, device, extent, VK_FORMAT_R16G16_SFLOAT,
                                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                         VK_IMAGE_ASPECT_COLOR_BIT);
            m_FluidDepthPassDSs[i].Init(allocator, device, extent, FindDepthFormat(),
                                         VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
            m_FluidBlurImagesA[i].Init(allocator, device, extent, VK_FORMAT_R32G32B32A32_SFLOAT,
                                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                        VK_IMAGE_ASPECT_COLOR_BIT);
            m_FluidBlurImagesB[i].Init(allocator, device, extent, VK_FORMAT_R32G32B32A32_SFLOAT,
                                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                        VK_IMAGE_ASPECT_COLOR_BIT);
            m_FluidPolishImages[i].Init(allocator, device, extent, VK_FORMAT_R32G32B32A32_SFLOAT,
                                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                         VK_IMAGE_ASPECT_COLOR_BIT);

            const VkExtent2D thicknessExtent{ std::max(1u, (extent.width + m_FluidThicknessDivisor - 1) / m_FluidThicknessDivisor),
                                               std::max(1u, (extent.height + m_FluidThicknessDivisor - 1) / m_FluidThicknessDivisor) };
            m_FluidNormalImages[i].Init(allocator, device, extent, VK_FORMAT_R16G16B16A16_SFLOAT,
                                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                         VK_IMAGE_ASPECT_COLOR_BIT);
            m_FluidThicknessImages[i].Init(allocator, device, thicknessExtent, VK_FORMAT_R16_SFLOAT,
                                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                            VK_IMAGE_ASPECT_COLOR_BIT);
            {
                VkImageView attachment = m_FluidThicknessImages[i].GetView();
                VkFramebufferCreateInfo framebufferInfo{};
                framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                framebufferInfo.renderPass = m_FluidThicknessRenderPass;
                framebufferInfo.attachmentCount = 1;
                framebufferInfo.pAttachments = &attachment;
                framebufferInfo.width = thicknessExtent.width;
                framebufferInfo.height = thicknessExtent.height;
                framebufferInfo.layers = 1;
                if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &m_FluidThicknessFramebuffers[i]) != VK_SUCCESS)
                {
                    throw std::runtime_error("Eden: failed to create fluid thickness framebuffer");
                }
            }
            {
                std::array<VkImageView, 4> attachments = { m_FluidDepthImages[i].GetView(), m_FluidMotionImages[i].GetView(),
                                                             m_FluidNormalImages[i].GetView(), m_FluidDepthPassDSs[i].GetView() };
                VkFramebufferCreateInfo framebufferInfo{};
                framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                framebufferInfo.renderPass = m_FluidDepthPrepassRenderPass;
                framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
                framebufferInfo.pAttachments = attachments.data();
                framebufferInfo.width = extent.width;
                framebufferInfo.height = extent.height;
                framebufferInfo.layers = 1;
                if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &m_FluidDepthFramebuffers[i]) != VK_SUCCESS)
                {
                    throw std::runtime_error("Eden: failed to create fluid depth framebuffer");
                }
            }
            {
                VkImageView attachment = m_FluidBlurImagesA[i].GetView();
                VkFramebufferCreateInfo framebufferInfo{};
                framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                framebufferInfo.renderPass = m_FluidBlurRenderPass;
                framebufferInfo.attachmentCount = 1;
                framebufferInfo.pAttachments = &attachment;
                framebufferInfo.width = extent.width;
                framebufferInfo.height = extent.height;
                framebufferInfo.layers = 1;
                if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &m_FluidBlurFramebuffersA[i]) != VK_SUCCESS)
                {
                    throw std::runtime_error("Eden: failed to create fluid blur framebuffer A");
                }
            }
            {
                VkImageView attachment = m_FluidBlurImagesB[i].GetView();
                VkFramebufferCreateInfo framebufferInfo{};
                framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                framebufferInfo.renderPass = m_FluidBlurRenderPass;
                framebufferInfo.attachmentCount = 1;
                framebufferInfo.pAttachments = &attachment;
                framebufferInfo.width = extent.width;
                framebufferInfo.height = extent.height;
                framebufferInfo.layers = 1;
                if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &m_FluidBlurFramebuffersB[i]) != VK_SUCCESS)
                {
                    throw std::runtime_error("Eden: failed to create fluid blur framebuffer B");
                }
            }
            {
                VkImageView attachment = m_FluidPolishImages[i].GetView();
                VkFramebufferCreateInfo framebufferInfo{};
                framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                framebufferInfo.renderPass = m_FluidBlurRenderPass; // reused, see m_FluidPolishFramebuffers' comment
                framebufferInfo.attachmentCount = 1;
                framebufferInfo.pAttachments = &attachment;
                framebufferInfo.width = extent.width;
                framebufferInfo.height = extent.height;
                framebufferInfo.layers = 1;
                if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &m_FluidPolishFramebuffers[i]) != VK_SUCCESS)
                {
                    throw std::runtime_error("Eden: failed to create fluid polish framebuffer");
                }
            }

            m_FluidBlurXSets[i] = allocateAndWrite(m_FluidDepthImages[i].GetView());
            m_FluidBlurYSets[i] = allocateAndWrite(m_FluidBlurImagesA[i].GetView());
            m_FluidPolishSets[i] = allocateAndWrite(m_FluidBlurImagesB[i].GetView());
            m_FluidTemporalNewSets[i] = allocateAndWrite(m_FluidPolishImages[i].GetView());
            m_FluidMotionSets[i] = allocateAndWrite(m_FluidMotionImages[i].GetView());
            m_FluidThicknessSets[i] = allocateAndWrite(m_FluidThicknessImages[i].GetView(), m_FluidLinearSampler.Get());
            m_FluidNormalSets[i] = allocateAndWrite(m_FluidNormalImages[i].GetView());
            if (i == 0)
            {
                // One set for the (single, shared) scene depth image. Depth is
                // sampled in DEPTH_STENCIL_READ_ONLY_OPTIMAL - the layout
                // m_SceneSplitRenderPass leaves it in. Uses the NEAREST sampler.
                m_SceneDepthSet = allocateAndWrite(m_DepthImage.GetView(), VK_NULL_HANDLE,
                                                    VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
                WriteFluidBoundsDescriptors(); // the depth image may have just been recreated (resize)
            }
        }

        // --- Temporal history buffers - see m_FluidHistoryImages' own
        // comment (Renderer.h) for why these are created OUTSIDE the
        // per-frame-slot loop above: exactly 2, ping-ponged once per
        // rendered frame, not per frame-in-flight slot.
        for (uint32_t i = 0; i < 2; ++i)
        {
            // TRANSFER_DST, in addition to the usual COLOR_ATTACHMENT |
            // SAMPLED every other fluid image gets, is required here and
            // only here: ClearFluidHistoryImages uses vkCmdClearColorImage
            // to seed these with the sentinel value once at creation, and
            // that command (and the TRANSFER_DST_OPTIMAL layout it needs)
            // requires the image to have been created with this bit set -
            // missing it is a validation error (and, on hardware that
            // doesn't silently tolerate the violation, undefined image
            // contents - likely exactly what produced the garbage-history
            // artifact seen before this was caught).
            m_FluidHistoryImages[i].Init(allocator, device, extent, VK_FORMAT_R32G32B32A32_SFLOAT,
                                          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                                              | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                          VK_IMAGE_ASPECT_COLOR_BIT);

            VkImageView attachment = m_FluidHistoryImages[i].GetView();
            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = m_FluidBlurRenderPass; // reused, see m_FluidHistoryFramebuffers' comment
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = &attachment;
            framebufferInfo.width = extent.width;
            framebufferInfo.height = extent.height;
            framebufferInfo.layers = 1;
            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &m_FluidHistoryFramebuffers[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("Eden: failed to create fluid history framebuffer");
            }

            m_FluidTemporalHistorySets[i] = allocateAndWrite(m_FluidHistoryImages[i].GetView());
            m_FluidCompositeSets[i] = allocateAndWrite(m_FluidHistoryImages[i].GetView());
        }

        // Fresh/resized history holds undefined memory, not the sentinel
        // - see ClearFluidHistoryImages' own comment for why that has to
        // be fixed explicitly rather than left to the next frame's
        // reject-gate.
        ClearFluidHistoryImages();
        m_FluidHistoryIndex = 0;
    }

    void Renderer::ClearFluidHistoryImages()
    {
        VkDevice device = m_Context.Device().Get();

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = m_CommandPool.Get();
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer cmd = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(device, &allocInfo, &cmd) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to allocate one-time command buffer for fluid history clear");
        }

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        // Both history images were just created with initialLayout =
        // VK_IMAGE_LAYOUT_UNDEFINED (see VulkanImage::Init) - transition
        // to TRANSFER_DST, clear, then to SHADER_READ_ONLY_OPTIMAL so
        // they match the layout fluid_temporal.frag's sampled reads (and
        // the render pass that writes them next frame) already expect
        // every other fluid pass's output to be in between uses.
        for (VulkanImage& image : m_FluidHistoryImages)
        {
            VkImageMemoryBarrier toTransferDst{};
            toTransferDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            toTransferDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            toTransferDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toTransferDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toTransferDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toTransferDst.image = image.Get();
            toTransferDst.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            toTransferDst.srcAccessMask = 0;
            toTransferDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  0, 0, nullptr, 0, nullptr, 1, &toTransferDst);

            // R = SENTINEL_VALUE (1e9, see fluid_temporal.frag) so the
            // very first frame's reject-gate sees "no history anywhere"
            // and falls back to using only that frame's new data
            // everywhere - never blends toward undefined driver memory.
            VkClearColorValue clearValue{};
            clearValue.float32[0] = 1.0e9f;
            clearValue.float32[1] = 0.0f;
            clearValue.float32[2] = 0.0f;
            clearValue.float32[3] = 0.0f;
            VkImageSubresourceRange range{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            vkCmdClearColorImage(cmd, image.Get(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearValue, 1, &range);

            VkImageMemoryBarrier toShaderRead{};
            toShaderRead.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            toShaderRead.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toShaderRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            toShaderRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toShaderRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toShaderRead.image = image.Get();
            toShaderRead.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            toShaderRead.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            toShaderRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                  0, 0, nullptr, 0, nullptr, 1, &toShaderRead);
        }

        vkEndCommandBuffer(cmd);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        vkQueueSubmit(m_Context.Device().GetGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
        // One-time setup work, not a per-frame path - a full stall here
        // is the right tradeoff for simplicity, same reasoning
        // UnregisterVolume's vkDeviceWaitIdle already uses for a
        // similarly rare, deliberate operation (see that function).
        vkQueueWaitIdle(m_Context.Device().GetGraphicsQueue());

        vkFreeCommandBuffers(device, m_CommandPool.Get(), 1, &cmd);
    }

    void Renderer::DestroyFluidSurfaceResources()
    {
        VkDevice device = m_Context.Device().Get();

        // Destroying the pool implicitly frees every set allocated from
        // it (all MAX_FRAMES_IN_FLIGHT slots' worth) - same reasoning as
        // m_RaymarchDescriptorPool's teardown.
        if (m_FluidDescriptorPool != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(device, m_FluidDescriptorPool, nullptr);
            m_FluidDescriptorPool = VK_NULL_HANDLE;
        }

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            m_FluidBlurXSets[i] = VK_NULL_HANDLE;
            m_FluidBlurYSets[i] = VK_NULL_HANDLE;
            m_FluidPolishSets[i] = VK_NULL_HANDLE;
            m_FluidTemporalNewSets[i] = VK_NULL_HANDLE;
            m_FluidMotionSets[i] = VK_NULL_HANDLE;
            m_FluidThicknessSets[i] = VK_NULL_HANDLE;
            m_FluidNormalSets[i] = VK_NULL_HANDLE;
            m_SceneDepthSet = VK_NULL_HANDLE;
            m_FluidNormalImages[i].Shutdown();

            if (m_FluidThicknessFramebuffers[i] != VK_NULL_HANDLE)
            {
                vkDestroyFramebuffer(device, m_FluidThicknessFramebuffers[i], nullptr);
                m_FluidThicknessFramebuffers[i] = VK_NULL_HANDLE;
            }
            m_FluidThicknessImages[i].Shutdown();

            if (m_FluidDepthFramebuffers[i] != VK_NULL_HANDLE)
            {
                vkDestroyFramebuffer(device, m_FluidDepthFramebuffers[i], nullptr);
                m_FluidDepthFramebuffers[i] = VK_NULL_HANDLE;
            }
            if (m_FluidBlurFramebuffersA[i] != VK_NULL_HANDLE)
            {
                vkDestroyFramebuffer(device, m_FluidBlurFramebuffersA[i], nullptr);
                m_FluidBlurFramebuffersA[i] = VK_NULL_HANDLE;
            }
            if (m_FluidBlurFramebuffersB[i] != VK_NULL_HANDLE)
            {
                vkDestroyFramebuffer(device, m_FluidBlurFramebuffersB[i], nullptr);
                m_FluidBlurFramebuffersB[i] = VK_NULL_HANDLE;
            }
            if (m_FluidPolishFramebuffers[i] != VK_NULL_HANDLE)
            {
                vkDestroyFramebuffer(device, m_FluidPolishFramebuffers[i], nullptr);
                m_FluidPolishFramebuffers[i] = VK_NULL_HANDLE;
            }

            m_FluidDepthImages[i].Shutdown();
            m_FluidMotionImages[i].Shutdown();
            m_FluidDepthPassDSs[i].Shutdown();
            m_FluidBlurImagesA[i].Shutdown();
            m_FluidBlurImagesB[i].Shutdown();
            m_FluidPolishImages[i].Shutdown();
        }

        // Static (not per-frame-slot) history resources - see
        // m_FluidHistoryImages' own comment (Renderer.h) for why these
        // are a separate loop of 2, not MAX_FRAMES_IN_FLIGHT.
        for (uint32_t i = 0; i < 2; ++i)
        {
            m_FluidTemporalHistorySets[i] = VK_NULL_HANDLE;
            m_FluidCompositeSets[i] = VK_NULL_HANDLE;

            if (m_FluidHistoryFramebuffers[i] != VK_NULL_HANDLE)
            {
                vkDestroyFramebuffer(device, m_FluidHistoryFramebuffers[i], nullptr);
                m_FluidHistoryFramebuffers[i] = VK_NULL_HANDLE;
            }
            m_FluidHistoryImages[i].Shutdown();
        }
    }

    void Renderer::InitImGui()
    {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();

        // install_callbacks=true - chains to whatever GLFW callbacks
        // main.cpp already installed (framebuffer resize, cursor pos for
        // the free-fly camera) rather than replacing them, as long as
        // this runs AFTER main.cpp's glfwSetXxxCallback calls. Renderer::
        // Init() does, since main.cpp sets its callbacks before calling
        // renderer.Init(window). If camera mouse-look or window-resize
        // ever stops working after this, check that ordering first.
        ImGui_ImplGlfw_InitForVulkan(m_Window, true);

        ImGui_ImplVulkan_InitInfo initInfo{};
        initInfo.ApiVersion = VK_API_VERSION_1_2; // must match VulkanInstance's appInfo.apiVersion
        initInfo.Instance = m_Context.Instance().Get();
        initInfo.PhysicalDevice = m_Context.PhysicalDevice().Get();
        initInfo.Device = m_Context.Device().Get();
        initInfo.QueueFamily = m_Context.PhysicalDevice().GetQueueFamilies().graphicsFamily.value();
        initInfo.Queue = m_Context.Device().GetGraphicsQueue();
        // DescriptorPoolSize (not DescriptorPool) - lets the backend create
        // and own a small internal descriptor pool itself, rather than
        // sharing Eden's own pool (which is sized for mesh textures, a
        // different budget with different growth characteristics).
        initInfo.DescriptorPoolSize = 16;
        initInfo.MinImageCount = MAX_FRAMES_IN_FLIGHT;
        initInfo.ImageCount = m_Context.Swapchain().GetImageCount();
        initInfo.PipelineInfoMain.RenderPass = m_RenderPass.Get();
        initInfo.PipelineInfoMain.Subpass = 0;
        initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

        ImGui_ImplVulkan_Init(&initInfo);
    }

    void Renderer::ShutdownImGui()
    {
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }

    void Renderer::CreateRenderFinishedSemaphores()
    {
        uint32_t imageCount = m_Context.Swapchain().GetImageCount();

        m_RenderFinishedSemaphores.clear();
        m_RenderFinishedSemaphores.reserve(imageCount);

        for (uint32_t i = 0; i < imageCount; ++i)
        {
            auto semaphore = std::make_unique<VulkanSemaphore>();
            semaphore->Init(m_Context.Device().Get());
            m_RenderFinishedSemaphores.push_back(std::move(semaphore));
        }
    }

    VkFormat Renderer::FindDepthFormat() const
    {
        // D32_SFLOAT is required by the Vulkan spec to support depth
        // attachment usage on every conformant implementation - no need to
        // query support, unlike combined depth/stencil formats.
        return VK_FORMAT_D32_SFLOAT;
    }

    void Renderer::CreateDepthResources()
    {
        VkFormat depthFormat = FindDepthFormat();
        m_DepthImage.Init(
            m_Allocator.Get(),
            m_Context.Device().Get(),
            m_Context.Swapchain().GetExtent(),
            depthFormat,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, // SAMPLED: the fluid passes read scene depth
            VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    MeshHandle Renderer::CreateMesh(const std::vector<Vertex>& vertices, TextureHandle texture)
    {
        Mesh mesh;
        mesh.Create(m_Allocator.Get(), m_Context.Device().Get(), m_CommandPool.Get(),
                    m_Context.Device().GetGraphicsQueue(), vertices, ResolveTextureDescriptorSet(texture));
        m_MeshRegistry.push_back(std::move(mesh));
        return m_MeshRegistry.size() - 1;
    }

    MeshHandle Renderer::CreateCubeMesh(float size, TextureHandle texture)
    {
        Primitives::IndexedMesh cubeData = Primitives::MakeCube(size);

        Mesh mesh;
        mesh.CreateIndexed(m_Allocator.Get(), m_Context.Device().Get(), m_CommandPool.Get(),
                            m_Context.Device().GetGraphicsQueue(), cubeData.vertices, cubeData.indices,
                            ResolveTextureDescriptorSet(texture));
        m_MeshRegistry.push_back(std::move(mesh));
        return m_MeshRegistry.size() - 1;
    }

    MeshHandle Renderer::CreateIndexedMesh(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, TextureHandle texture)
    {
        Mesh mesh;
        mesh.CreateIndexed(m_Allocator.Get(), m_Context.Device().Get(), m_CommandPool.Get(),
                            m_Context.Device().GetGraphicsQueue(), vertices, indices,
                            ResolveTextureDescriptorSet(texture));
        m_MeshRegistry.push_back(std::move(mesh));
        return m_MeshRegistry.size() - 1;
    }

    MeshHandle Renderer::CreateMeshFromFile(const std::string& path, TextureHandle texture)
    {
        // Extension-based dispatch, case-insensitive - lets callers (and
        // EdenUI's "Load Model" field) use one entry point regardless of
        // format instead of needing to know/call a different function per
        // extension. Falls through to OBJ for anything unrecognized rather
        // than throwing immediately, since that produces tinyobjloader's
        // own (often more specific) parse error instead of a generic
        // "unknown extension" message.
        std::string extension;
        size_t dotPos = path.find_last_of('.');
        if (dotPos != std::string::npos)
        {
            extension = path.substr(dotPos);
            for (char& c : extension)
            {
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
        }

        Primitives::IndexedMesh loaded = (extension == ".gltf" || extension == ".glb")
            ? ModelLoader::LoadGltf(path)
            : ModelLoader::LoadObj(path);

        return CreateIndexedMesh(loaded.vertices, loaded.indices, texture);
    }

    TextureHandle Renderer::CreateTexture(const std::string& path)
    {
        auto texture = std::make_unique<VulkanTexture>();
        texture->LoadFromFile(
            m_Allocator.Get(), m_Context.Device().Get(), m_CommandPool.Get(), m_Context.Device().GetGraphicsQueue(),
            m_DescriptorPool.Get(), m_TextureSetLayout.Get(), path);
        m_TextureRegistry.push_back(std::move(texture));
        return m_TextureRegistry.size() - 1;
    }

    glm::mat4 Renderer::GetViewProjectionMatrix()
    {
        // Deliberately identical to DrawFrame's own ubo.view/ubo.proj
        // calculation (see the "Camera matrices" block there) - kept as
        // a literal duplicate rather than having DrawFrame call this
        // (DrawFrame's version isn't const, and refactoring the frame's
        // hot path to share this small a calculation isn't worth the
        // risk to code that's already working); if one changes, the
        // other must too, which is exactly the discipline this comment
        // exists to flag.
        VkExtent2D extent = m_Context.Swapchain().GetExtent();
        float aspect = extent.height > 0
            ? static_cast<float>(extent.width) / static_cast<float>(extent.height)
            : 1.0f;

        return m_Camera.GetProjectionMatrix(aspect) * m_Camera.GetViewMatrix();
    }

    Frustum Renderer::GetViewFrustum()
    {
        return Frustum::FromViewProjection(GetViewProjectionMatrix(), m_Camera.Position, m_Camera.Front,
                                            m_Camera.NearPlane, m_Camera.FarPlane);
    }

    float Renderer::GetMeshBoundingRadius(MeshHandle handle) const
    {
        if (handle == InvalidMeshHandle || handle >= m_MeshRegistry.size())
        {
            return 0.0f;
        }
        return m_MeshRegistry[handle].GetBoundingRadius();
    }

    VkDescriptorSet Renderer::ResolveTextureDescriptorSet(TextureHandle texture) const
    {
        // InvalidTextureHandle (the default for CreateMesh/CreateCubeMesh's
        // `texture` parameter) means "no real texture" - fall back to the
        // 1x1 white texture so the shader's sample-and-multiply path still
        // works, just as a no-op. See VulkanTexture::CreateSolidColor.
        TextureHandle resolved = (texture == InvalidTextureHandle) ? m_DefaultTextureHandle : texture;
        return m_TextureRegistry[resolved]->GetDescriptorSet();
    }

    void Renderer::RecreateSwapchainResources()
    {
        // Everything torn down and rebuilt below (framebuffers, depth
        // image, and now the per-swapchain-image semaphores) might still
        // be referenced by GPU work that was in flight when this got
        // called - a submit from a frame or two ago, or the present that
        // was just issued when this triggered from a SUBOPTIMAL/OUT_OF_DATE
        // result. Waiting here first is the difference between "recreate
        // resources nothing is using anymore" and "destroy a semaphore the
        // driver is still about to touch".
        WaitIdle();

        m_Context.RecreateSwapchain(m_Window);

        // Depth buffer must match the new swapchain extent.
        m_DepthImage.Shutdown();
        CreateDepthResources();

        m_Context.Swapchain().CreateFramebuffers(m_RenderPass.Get(), m_DepthImage.GetView());

        // Fluid-surface offscreen targets are sized to the swapchain
        // extent too - tear down and rebuild alongside the depth image
        // above. Guarded on m_FluidSurfaceInitialized (see that field's
        // comment) since a resize landing before RegisterParticleGPUSource
        // has ever run would otherwise touch still-null render passes.
        if (m_FluidSurfaceInitialized)
        {
            DestroyFluidSurfaceResources();
            CreateFluidSurfaceResources();
        }

        // Image count CAN change on recreate (rare, but the spec doesn't
        // guarantee it stays the same) - rebuild to match rather than
        // assume the old count still applies.
        CreateRenderFinishedSemaphores();
    }

    void Renderer::DrawFrame(const std::vector<DrawCommand>& drawList, const std::function<void()>& buildUI,
                              const std::function<void(VkCommandBuffer)>& recordComputeWork, uint32_t particleGPUCount,
                              const std::vector<VoxelDrawSource>& voxelSources,
                              const std::vector<RaymarchObjectGPU>& raymarchObjects,
                              VkBuffer raymarchDensityBuffer, float deltaTime)
    {
        FrameContext& frame = m_Frames[m_CurrentFrame];

        // Auto resolution: judge last frame from the profiler's numbers (the
        // CPU-side "waiting for GPU" share is a reliable GPU-bound signal on
        // every platform, unlike GPU timestamps on Apple GPUs).
        if (FluidAutoResolution && m_FluidSurfaceInitialized)
        {
            if (m_AutoResolutionCooldown > 0)
            {
                --m_AutoResolutionCooldown;
            }
            else
            {
                const Profiler& profiler = Profiler::Get();
                const float frameMs = profiler.FrameMs();
                const float frameAvgMs = std::max(profiler.FrameAvgMs(), 0.001f);
                static const bool forceGpuBound = (std::getenv("EDEN_AUTO_RES_FORCE") != nullptr); // test hook: pretend we're GPU-bound (software renderers can't show the real signal)
                const float waitShare = forceGpuBound ? 1.0f : profiler.AvgOf("Frame: waiting for GPU (fence)") / frameAvgMs;
                const bool liquidDrawnLastFrame = m_FluidChainRanLastFrame;

                // Slow (under ~45 fps), mostly waiting on the GPU, and the
                // fluid chain actually ran - i.e. the liquid is plausibly why.
                const bool looksGpuBoundOnLiquid = liquidDrawnLastFrame && frameMs > 22.0f && waitShare > 0.4f;
                m_AutoResolutionSlowFrames = looksGpuBoundOnLiquid ? m_AutoResolutionSlowFrames + 1
                                                                   : std::max(m_AutoResolutionSlowFrames - 2, 0);

                if (m_AutoResolutionSlowFrames >= 60) // ~1.3 s of it at 45 fps
                {
                    m_AutoResolutionSlowFrames = 0;
                    const float levels[] = { 1.0f, 0.75f, 0.5f, 0.35f };
                    for (int i = 0; i < 3; ++i)
                    {
                        if (FluidResolutionScale > levels[i + 1] + 0.01f && FluidResolutionScale <= levels[i] + 0.01f)
                        {
                            FluidResolutionScale = levels[i + 1];
                            std::printf("[Eden] Auto fluid resolution: GPU-bound with liquid on screen -> %.0f%%\n", FluidResolutionScale * 100.0f);
                            m_AutoResolutionCooldown = 180; // let the rebuild hitch pass and the averages settle
                            break;
                        }
                    }
                }
            }
        }

        // Fluid resolution scale changed (UI): rebuild the fluid targets at
        // the new size. WaitIdle first - same reason as
        // RecreateSwapchainResources, nothing may still be reading them.
        if (m_FluidSurfaceInitialized && std::fabs(FluidResolutionScale - m_AppliedFluidScale) > 0.001f)
        {
            WaitIdle();
            DestroyFluidSurfaceResources();
            CreateFluidSurfaceResources();
        }

        {
            EDEN_PROFILE_SCOPE("Frame: waiting for GPU (fence)");
            frame.inFlightFence.Wait();
        }
        ReadbackGpuTimers(m_CurrentFrame);
        ReadbackFluidBounds(m_CurrentFrame);

        uint32_t imageIndex;
        const auto acquireStart = Profiler::Now();
        VkResult acquireResult = vkAcquireNextImageKHR(
            m_Context.Device().Get(),
            m_Context.Swapchain().Get(),
            std::numeric_limits<uint64_t>::max(),
            frame.imageAvailableSemaphore.Get(),
            VK_NULL_HANDLE,
            &imageIndex);

        Profiler::Get().Add("Frame: acquire image / vsync", Profiler::MsSince(acquireStart));

        if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR)
        {
            RecreateSwapchainResources();
            return;
        }
        else if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR)
        {
            throw std::runtime_error("Eden: failed to acquire swapchain image");
        }

        frame.inFlightFence.Reset();

        // Builds ImGui's CPU-side draw data for this frame - doesn't touch
        // the command buffer at all yet, that happens later via
        // ImGui_ImplVulkan_RenderDrawData once the render pass is active.
        // Safe to call even when buildUI is empty (ImGui::Render() on a
        // frame with no widgets just produces an empty draw list).
        {
            EDEN_PROFILE_SCOPE("Frame: UI build (ImGui)");
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            if (buildUI)
            {
                buildUI();
            }
            ImGui::Render();
        }

        // Camera matrices, uploaded fresh every frame via the persistently
        // mapped uniform buffer - no vkMap/vkUnmap needed per frame.
        VkExtent2D extent = m_Context.Swapchain().GetExtent();
        const VkExtent2D fluidExtent = m_FluidExtent; // fluid offscreen passes run at this (possibly scaled-down) size
        float aspect = extent.height > 0
            ? static_cast<float>(extent.width) / static_cast<float>(extent.height)
            : 1.0f;

        UniformBufferObject ubo{};
        ubo.view = m_Camera.GetViewMatrix();
        ubo.proj = m_Camera.GetProjectionMatrix(aspect);
        ubo.cameraPosition = m_Camera.Position;
        ubo.particlePointSize = ParticlePointSize;
        // Eden's one hardcoded global light for now - no LightComponent
        // yet, so this isn't placed/aimed per-scene, just a reasonable
        // default "sun coming from up and to the side" setup. Revisit as
        // a real ECS-driven light (position/color/intensity as component
        // data, Renderer reading it from the registry each frame) once
        // more than one light - or any control over this one - is needed.
        //
        // Expressed as a zenith angle (degrees off straight-down) instead
        // of a raw direction vector, kept as a named constant rather than
        // baked into the vec3 - a raw (0.4,-1.0,0.3) direction doesn't
        // communicate "how steep is this sun" the way a zenith angle
        // does, and this is exactly the kind of value someone will want
        // to retune by eye. 33 degrees (bumped up from the original,
        // much steeper ~26-degree-elevation default) - shallow enough
        // that even gentle terrain slopes produce real dot(N,L) contrast
        // instead of reading as flat. Azimuth (the original 0.4/0.3 XZ
        // ratio - light coming from the +X/+Z-ish direction) is preserved
        // exactly, only the steepness changes.
        constexpr float kSunZenithDegrees = 33.0f;
        float sunZenithRad = glm::radians(kSunZenithDegrees);
        glm::vec2 sunAzimuthDir = glm::normalize(glm::vec2(0.4f, 0.3f));
        ubo.lightDirection = glm::normalize(glm::vec3(
            sunAzimuthDir.x * glm::sin(sunZenithRad),
            -glm::cos(sunZenithRad),
            sunAzimuthDir.y * glm::sin(sunZenithRad)));
        ubo.lightColor = glm::vec3(1.0f, 1.0f, 0.95f);
        ubo.ambientColor = glm::vec3(0.15f, 0.15f, 0.18f);
        // See UniformBufferObject::cameraRight's comment - only consumed
        // by the fluid-surface pass's shaders.
        ubo.cameraRight = m_Camera.Right;
        ubo.cameraUp = m_Camera.Up;
        ubo.cameraForward = m_Camera.Front;
        // See UniformBufferObject::prevViewProj's comment - this is
        // last frame's value, captured at the END of the previous
        // DrawFrame call (search this function for the assignment to
        // m_PreviousViewProj), not recomputed here from the current
        // camera state.
        ubo.prevViewProj = m_PreviousViewProj;
        ubo.invViewProj = glm::inverse(ubo.proj * ubo.view);
        frame.UpdateUniformBuffer(ubo);

        frame.commandBuffer.Reset();
        frame.commandBuffer.Begin();

        VkCommandBuffer cmd = frame.commandBuffer.Get();
        const auto recordStart = Profiler::Now(); // closed just before frame.commandBuffer.End() below

        if (GpuTimingEnabled && m_GpuTimingSupported)
        {
            // Queries must be reset outside a render pass, before first use
            // this frame.
            m_GpuTimerLabels[m_CurrentFrame].clear();
            vkCmdResetQueryPool(cmd, m_GpuTimerPools[m_CurrentFrame], 0, kMaxGpuStamps);
            GpuStamp(cmd, "start");
        }

        // Compute dispatches (if any) go here - BEFORE the render pass
        // begins, since compute work can't be recorded inside one. See
        // DrawFrame's own header comment on recordComputeWork/
        // ParticleSystemGPU::RecordPendingSteps for why this is the one
        // spot in the frame where that's true. The barrier after it
        // covers compute-shader-write -> vertex-shader-read, which is
        // exactly what particle_point_gpu.vert needs before the render
        // pass's draw calls run (see particleGPUCount below).
        if (recordComputeWork)
        {
            recordComputeWork(cmd);
            GpuStamp(cmd, "SPH particle simulation (compute)");

            VkMemoryBarrier computeToVertexBarrier{};
            computeToVertexBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            computeToVertexBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            // TRANSFER_READ added alongside SHADER_READ: this frame's
            // later copy of the live position buffer into
            // m_ParticlePreviousPositions (see that copy's own comment,
            // further down this function) reads the SAME buffer this
            // compute dispatch just wrote, at the TRANSFER stage - a
            // barrier naming only VERTEX_SHADER as the destination stage
            // does NOT make the write visible to a later TRANSFER-stage
            // read, even later in the same command buffer. Missing this
            // was a real, confirmed bug: the previous-position snapshot
            // had no guaranteed-correct data to copy FROM, and on
            // hardware that overlaps compute and transfer work, could
            // read a write still in flight. That corruption then fed the
            // temporal history buffer every frame after, compounding
            // instead of washing out - consistent with the artifact
            // visibly worsening over time rather than looking like a
            // one-off glitch.
            computeToVertexBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  0, 1, &computeToVertexBarrier, 0, nullptr, 0, nullptr);
        }

        // --- Fluid surface offscreen passes (see FluidSurfaceEnabled's
        // comment in Renderer.h) - three separate, serial render pass
        // instances, all recorded BEFORE the main render pass begins,
        // same "can't nest render passes" constraint recordComputeWork's
        // comment above already flags for compute work. fluidSurfaceThisFrame
        // also gates the composite draw further down (inside the main
        // render pass) and the raw-point fallback draw right after this
        // block - the three are mutually exclusive per frame, not
        // independent toggles.
        bool fluidSurfaceThisFrame = FluidSurfaceEnabled && m_FluidSurfaceInitialized &&
                                      particleGPUCount > 0 && m_ParticleGPUStorageSet != VK_NULL_HANDLE;
        // The entire offscreen pipeline below (depth splat, previous-
        // position copy, blur, polish, temporal) only exists to feed
        // m_FluidCompositePipeline - the soft-blob path (FluidUseSoftBlobs)
        // needs none of it, it draws directly into the main render pass
        // further down (see that draw's own comment). Skipping this
        // block entirely in blob mode isn't just a performance win - it
        // also means blob mode can never be affected by whatever is
        // still unresolved in the temporal reprojection system, since it
        // simply never runs.
        const bool splitMainPass = fluidSurfaceThisFrame && !FluidUseSoftBlobs;

        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = { { 0.02f, 0.02f, 0.05f, 1.0f } };
        clearValues[1].depthStencil = { 1.0f, 0 };

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        // Frames that draw the fluid surface split the main pass: this scene pass
        // stores its depth (readable) so the fluid passes can occlude against the
        // opaque scene, then an overlay pass loads color+depth and draws the
        // composite and UI. Every other frame keeps the original single pass.
        renderPassInfo.renderPass = splitMainPass ? m_SceneSplitRenderPass.Get() : m_RenderPass.Get();
        renderPassInfo.framebuffer = m_Context.Swapchain().GetFramebuffer(imageIndex);
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_GraphicsPipeline.Get());

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(extent.width);
        viewport.height = static_cast<float>(extent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = extent;
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_PipelineLayout.Get(),
                                 0, 1, &frame.descriptorSet, 0, nullptr);

        // Group by mesh: this is the whole point of instancing. Before,
        // every entity was its own vkCmdDraw* call even when 500 of them
        // shared the exact same MeshHandle. Now every entity sharing a
        // mesh becomes one InstanceData entry in a contiguous run, and
        // that run becomes ONE draw call covering all of them.
        //
        // insertion order within a group doesn't matter (nothing depends
        // on draw order between same-mesh instances). m_GroupedByMeshScratch
        // is a member reused across frames (see Renderer.h) - clear each
        // vector's contents here, but don't erase the map entries, so
        // repeat mesh handles keep whatever capacity they already grew to
        // instead of reallocating from empty every single frame.
        for (auto& [mesh, instances] : m_GroupedByMeshScratch)
        {
            instances.clear();
        }

        for (const auto& drawCmd : drawList)
        {
            InstanceData& instance = m_GroupedByMeshScratch[drawCmd.mesh].emplace_back();
            instance.model = drawCmd.model;
            instance.colorOverride = drawCmd.colorOverride;
        }

        size_t totalInstances = drawList.size();
        if (totalInstances > MAX_INSTANCES_PER_FRAME)
        {
            // Known limitation, not a crash: the instance buffer is a
            // fixed-size allocation (see FrameContext::Init). Silently
            // dropping the overflow keeps the frame rendering instead of
            // writing past the buffer; raise MAX_INSTANCES_PER_FRAME in
            // RendererTypes.h if a scene legitimately needs more than 64k
            // instances in one frame.
            //
            // In the normal path this shouldn't actually fire anymore -
            // RenderSystem::BuildDrawList caps drawList at
            // MAX_INSTANCES_PER_FRAME before it ever gets here, and warns
            // about it there instead (where the CPU-side cost of building
            // the oversized list was actually being paid). This stays as
            // a defense-in-depth backstop for any other caller that might
            // hand DrawFrame an oversized list directly - warned once, not
            // every frame, for the same reason as the one in RenderSystem.h.
            static bool s_WarnedAboveCap = false;
            if (!s_WarnedAboveCap)
            {
                std::cerr << "Eden: frame requested " << totalInstances << " instances, capping at "
                          << MAX_INSTANCES_PER_FRAME << " (see MAX_INSTANCES_PER_FRAME in RendererTypes.h)\n";
                s_WarnedAboveCap = true;
            }
        }

        // Write every group's instances into one contiguous per-frame
        // buffer, back to back, tracking where each group starts so its
        // draw call can bind just that slice via a byte offset. groupRanges
        // stays local/per-frame (unlike m_GroupedByMeshScratch above) -
        // it's sized by distinct mesh count, not entity count, so rebuilding
        // it from scratch every frame is negligible regardless of scene size.
        auto* instanceDest = static_cast<InstanceData*>(frame.instanceBufferMapped);
        size_t writtenSoFar = 0;
        std::unordered_map<MeshHandle, std::pair<VkDeviceSize, uint32_t>> groupRanges; // mesh -> (byteOffset, count)

        for (auto& [mesh, instances] : m_GroupedByMeshScratch)
        {
            if (instances.empty())
            {
                continue;
            }

            size_t remainingCapacity = MAX_INSTANCES_PER_FRAME - writtenSoFar;
            size_t countToWrite = std::min(instances.size(), remainingCapacity);

            if (countToWrite == 0)
            {
                continue;
            }

            std::memcpy(instanceDest + writtenSoFar, instances.data(), countToWrite * sizeof(InstanceData));
            groupRanges[mesh] = { writtenSoFar * sizeof(InstanceData), static_cast<uint32_t>(countToWrite) };
            writtenSoFar += countToWrite;
        }

        // ONE draw call per mesh now, not per entity - a scene of 500
        // same-mesh cubes used to mean 500 vkCmdDrawIndexed calls, each
        // with its own push constant upload. Now it's exactly
        // groupRanges.size() draw calls (one per distinct MeshHandle
        // actually used this frame), each covering however many instances
        // share that mesh.
        //
        // Pipeline bind happens INSIDE this loop, per mesh, rather than
        // once before it - most frames only ever need m_GraphicsPipeline
        // (the common triangle-mesh case) or occasionally also
        // m_ParticlePointsPipeline for the one dedicated point mesh (see
        // GetParticlePointMesh) - at most one extra vkCmdBindPipeline call
        // per frame in practice, not per draw call, since groupRanges has
        // one entry per distinct MeshHandle, and there's only ever one
        // point-mesh handle.
        VkPipeline lastBoundPipeline = VK_NULL_HANDLE;
        for (const auto& [mesh, range] : groupRanges)
        {
            if (!DebugDrawMeshes)
            {
                break;
            }
            VkPipeline requiredPipeline = (mesh == m_ParticlePointMeshHandle)
                ? m_ParticlePointsPipeline.Get()
                : m_GraphicsPipeline.Get();

            if (requiredPipeline != lastBoundPipeline)
            {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, requiredPipeline);
                lastBoundPipeline = requiredPipeline;
            }

            const auto& [byteOffset, count] = range;
            m_MeshRegistry[mesh].DrawInstanced(cmd, m_PipelineLayout.Get(), frame.instanceBuffer.Get(), byteOffset, count);
        }

        // GPU-resident particles - entirely separate draw path from the
        // mesh loop above (see RegisterParticleGPUSource's class
        // comment): no InstanceData, no vertex buffer, position comes
        // straight from the storage buffer via gl_InstanceIndex. Only
        // fires once RegisterParticleGPUSource has actually been called
        // (m_ParticleGPUStorageSet stays VK_NULL_HANDLE until then).
        if (particleGPUCount > 0 && m_ParticleGPUStorageSet != VK_NULL_HANDLE && !fluidSurfaceThisFrame)
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_ParticlePointsGPUPipeline.Get());

            std::array<VkDescriptorSet, 2> particleSets = { frame.descriptorSet, m_ParticleGPUStorageSet };
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_ParticleGPUPipelineLayout.Get(),
                                     0, static_cast<uint32_t>(particleSets.size()), particleSets.data(), 0, nullptr);

            vkCmdPushConstants(cmd, m_ParticleGPUPipelineLayout.Get(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::vec4), &ParticleGPUColor);

            // vertexCount=1: gl_VertexIndex is always 0, unused by
            // particle_point_gpu.vert (position comes from
            // gl_InstanceIndex into the storage buffer instead) -
            // instanceCount=particleGPUCount is what actually produces
            // one point per particle.
            vkCmdDraw(cmd, 1, particleGPUCount, 0, 0);
        }

        // GPU-generated indirect-draw geometry (marching-cubes voxel
        // volumes today, see VoxelDrawSource's comment in RendererTypes.h
        // for why this isn't voxel-specific by name) - uses
        // m_VoxelPipeline (same Blinn-Phong/texture shader every
        // ordinary mesh uses, cullMode=NONE instead of m_GraphicsPipeline's
        // BACK_BIT - see m_VoxelPipeline's comment in Renderer.h), fed
        // from a compute-written buffer via vkCmdDrawIndirect instead of
        // a CPU-uploaded one via vkCmdDrawIndexed. Rebinding the
        // pipeline defensively here rather than trusting
        // lastBoundPipeline from the mesh loop above - that loop may
        // have left m_ParticlePointsPipeline (or now m_GraphicsPipeline)
        // bound depending on what was drawn last.
        if (!voxelSources.empty() && DebugDrawVoxelGeometry)
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_VoxelPipeline.Get());

            // Explicitly rebind set 0 (camera UBO) here, not just set 1 -
            // it does NOT reliably survive from the frame-start bind
            // (line ~806) through to here. The particle pass just above
            // rebinds set 0 itself using m_ParticleGPUPipelineLayout,
            // which has different push constant ranges than
            // m_PipelineLayout - per Vulkan's pipeline layout
            // compatibility rules, that invalidates set 0's binding for
            // any layout that isn't compatible with it, and rebinding
            // set 1 alone below doesn't restore it. (The raymarch pass
            // right after this block already rebinds set 0 for the same
            // reason - this block was the one place that didn't.)
            VkDescriptorSet cameraSet = frame.descriptorSet;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_PipelineLayout.Get(),
                                     0, 1, &cameraSet, 0, nullptr);

            // Untextured - bind Eden's default white fallback at set 1,
            // same as any untextured ordinary mesh (see CreateMesh's
            // resolved TextureHandle).
            VkDescriptorSet defaultTextureSet = m_TextureRegistry[m_DefaultTextureHandle]->GetDescriptorSet();
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_PipelineLayout.Get(),
                                     1, 1, &defaultTextureSet, 0, nullptr);

            for (const VoxelDrawSource& source : voxelSources)
            {
                if (source.drawCount == 0 || source.vertexBuffer == VK_NULL_HANDLE || source.indirectBuffer == VK_NULL_HANDLE)
                {
                    continue;
                }

                // Binding 0 = this source's compute-generated geometry
                // (advances per-vertex), binding 1 = its 1-entry
                // instance buffer (advances per-instance, but every
                // indirect draw below sets instanceCount=1 anyway - see
                // VoxelSystemGPU::RegisterVolume) - same two-binding
                // shape as Mesh::DrawInstanced, just sourced from
                // different buffers.
                VkBuffer vertexBuffers[] = { source.vertexBuffer, source.instanceBuffer };
                VkDeviceSize offsets[] = { 0, 0 };
                vkCmdBindVertexBuffers(cmd, 0, 2, vertexBuffers, offsets);

                // One vkCmdDrawIndirect per chunk, NOT one call with
                // drawCount=source.drawCount - that relied on the
                // multiDrawIndirect device feature (multiple indirect
                // commands consumed from one call), which is NEVER
                // enabled (see VulkanDevice.cpp - deviceFeatures is
                // zero-initialized aside from largePoints). Per the
                // Vulkan spec, drawCount must be 0 or 1 without that
                // feature - anything higher is undefined behavior, not
                // just a validation warning to silence. This had been
                // silently wrong since the very first multi-chunk
                // volume (chunkDims=(2,2,2)=8 chunks on the original
                // test sphere), apparently tolerated well enough in
                // practice at that scale to go unnoticed - terrain's
                // 16-chunk volumes made it actually visible as missing
                // squares of geometry along chunk boundaries, which is
                // what surfaced this.
                //
                // Looping here (rather than querying and conditionally
                // enabling multiDrawIndirect) is the deliberately more
                // portable fix - this project targets Mac (MoltenVK) and
                // Windows, and this way rendering is correct on ANY
                // device regardless of whether that feature happens to
                // be supported, with no runtime feature-detection branch
                // to maintain. CPU cost is negligible - source.drawCount
                // cheap indirect-draw calls per source per frame, not
                // per-chunk CPU geometry work.
                for (uint32_t chunkIndex = 0; chunkIndex < source.drawCount; ++chunkIndex)
                {
                    VkDeviceSize chunkOffset = static_cast<VkDeviceSize>(chunkIndex) * sizeof(VkDrawIndirectCommand);
                    vkCmdDrawIndirect(cmd, source.indirectBuffer, chunkOffset, 1, sizeof(VkDrawIndirectCommand));
                }
            }
        }

        GpuStamp(cmd, "Main: meshes + voxel geometry");

        // Raymarch pass - fullscreen sphere-tracing of every "raymarch
        // object" (see Raymarch/RaymarchSystem.h), recorded after
        // rasterized mesh/voxel geometry so this frame's depth buffer
        // already holds terrain's real depth values (raymarch.frag
        // relies on depth test being ON to composite correctly against
        // that, and writes its own gl_FragDepth per hit so it can in
        // turn be occluded by geometry drawn in a future frame's
        // earlier passes - see that shader's depth-write comment).
        // Raymarch visibility, on the CPU, BEFORE the (otherwise fullscreen)
        // sphere-tracing pass: cull objects whose bounds are entirely outside
        // the view frustum, and find the screen rectangle the survivors can
        // possibly cover. The pass is then scissored to that rectangle (or
        // skipped entirely if nothing is visible) - without this, every pixel
        // of the screen ran the object loop every frame, including when the
        // objects were small, far away, or behind the camera. Conservative:
        // only culls when ALL 8 corners of an object's box are outside the
        // SAME frustum plane, and falls back to the whole screen if any
        // corner is at/behind the camera (its projection is meaningless).
        m_VisibleRaymarchScratch.clear();
        VkRect2D raymarchScissor{ { 0, 0 }, extent };
        if (!raymarchObjects.empty() && DebugDrawRaymarch)
        {
            const glm::mat4 viewProj = ubo.proj * ubo.view;
            float rectMinX = 1.0e9f, rectMinY = 1.0e9f, rectMaxX = -1.0e9f, rectMaxY = -1.0e9f;
            bool needFullScreen = false;

            for (const RaymarchObjectGPU& object : raymarchObjects)
            {
                const glm::mat4 model = glm::inverse(object.invModel);
                const glm::mat4 modelViewProj = viewProj * model;

                bool allOutside[6] = { true, true, true, true, true, true }; // x<-w, x>w, y<-w, y>w, z<0, z>w
                bool anyCornerBehindCamera = false;
                float objMinX = 1.0e9f, objMinY = 1.0e9f, objMaxX = -1.0e9f, objMaxY = -1.0e9f;

                for (int corner = 0; corner < 8; ++corner)
                {
                    const glm::vec3 localCorner((corner & 1) ? object.localMax.x : object.localMin.x,
                                                 (corner & 2) ? object.localMax.y : object.localMin.y,
                                                 (corner & 4) ? object.localMax.z : object.localMin.z);
                    const glm::vec4 clip = modelViewProj * glm::vec4(localCorner, 1.0f);

                    allOutside[0] = allOutside[0] && (clip.x < -clip.w);
                    allOutside[1] = allOutside[1] && (clip.x > clip.w);
                    allOutside[2] = allOutside[2] && (clip.y < -clip.w);
                    allOutside[3] = allOutside[3] && (clip.y > clip.w);
                    allOutside[4] = allOutside[4] && (clip.z < 0.0f);
                    allOutside[5] = allOutside[5] && (clip.z > clip.w);

                    if (clip.w <= 1.0e-4f)
                    {
                        anyCornerBehindCamera = true;
                    }
                    else
                    {
                        objMinX = std::min(objMinX, clip.x / clip.w);
                        objMaxX = std::max(objMaxX, clip.x / clip.w);
                        objMinY = std::min(objMinY, clip.y / clip.w);
                        objMaxY = std::max(objMaxY, clip.y / clip.w);
                    }
                }

                if (allOutside[0] || allOutside[1] || allOutside[2] || allOutside[3] || allOutside[4] || allOutside[5])
                {
                    continue; // entirely outside the frustum - nothing of it can appear
                }

                m_VisibleRaymarchScratch.push_back(object);
                if (anyCornerBehindCamera)
                {
                    needFullScreen = true;
                }
                else
                {
                    rectMinX = std::min(rectMinX, objMinX);
                    rectMaxX = std::max(rectMaxX, objMaxX);
                    rectMinY = std::min(rectMinY, objMinY);
                    rectMaxY = std::max(rectMaxY, objMaxY);
                }
            }

            if (!m_VisibleRaymarchScratch.empty() && !needFullScreen)
            {
                // NDC -> framebuffer pixels (viewport: x=y=0, positive height;
                // the projection already carries the Vulkan Y flip, so this
                // mapping is direct). 2 px of margin for rasterization rounding.
                const float width = static_cast<float>(extent.width);
                const float height = static_cast<float>(extent.height);
                const int32_t x0 = static_cast<int32_t>(std::floor(std::max((rectMinX * 0.5f + 0.5f) * width - 2.0f, 0.0f)));
                const int32_t y0 = static_cast<int32_t>(std::floor(std::max((rectMinY * 0.5f + 0.5f) * height - 2.0f, 0.0f)));
                const int32_t x1 = static_cast<int32_t>(std::ceil(std::min((rectMaxX * 0.5f + 0.5f) * width + 2.0f, width)));
                const int32_t y1 = static_cast<int32_t>(std::ceil(std::min((rectMaxY * 0.5f + 0.5f) * height + 2.0f, height)));
                raymarchScissor.offset = { x0, y0 };
                raymarchScissor.extent = { static_cast<uint32_t>(std::max(x1 - x0, 0)), static_cast<uint32_t>(std::max(y1 - y0, 0)) };
            }
        }

        if (!m_VisibleRaymarchScratch.empty() && raymarchScissor.extent.width > 0 && raymarchScissor.extent.height > 0)
        {
            UpdateRaymarchDescriptors(m_VisibleRaymarchScratch, raymarchDensityBuffer);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_RaymarchPipeline.Get());

            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_RaymarchPipelineLayout.Get(),
                                     0, 1, &frame.descriptorSet, 0, nullptr);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_RaymarchPipelineLayout.Get(),
                                     1, 1, &m_RaymarchSets[m_CurrentFrame], 0, nullptr);

            RaymarchPushConstants pushConstants{};
            pushConstants.objectCount = static_cast<int32_t>(std::min(m_VisibleRaymarchScratch.size(),
                                                                        static_cast<size_t>(kRaymarchMaxObjects)));
            pushConstants.nearPlane = m_Camera.NearPlane;
            pushConstants.farPlane = m_Camera.FarPlane;
            vkCmdPushConstants(cmd, m_RaymarchPipelineLayout.Get(), VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(RaymarchPushConstants), &pushConstants);

            // No vertex/instance buffers bound - see raymarch.vert,
            // vertices are generated from gl_VertexIndex.
            // Only the pixels the visible objects' bounds can cover.
            vkCmdSetScissor(cmd, 0, 1, &raymarchScissor);
            vkCmdDraw(cmd, 3, 1, 0, 0);
            vkCmdSetScissor(cmd, 0, 1, &scissor); // restore for everything drawn after this
        }

        GpuStamp(cmd, "Main: raymarched objects");

        if (splitMainPass)
        {
            vkCmdEndRenderPass(cmd); // opaque scene done; its depth is now readable by the fluid passes below

            if (FluidBoundsCullingEnabled)
            {
                // Where is the visible liquid? Read back a couple of frames
                // later to size/skip the fluid passes - see fluid_bounds.comp.
                RecordFluidBounds(cmd, particleGPUCount, ubo.proj * ubo.view, extent);
                GpuStamp(cmd, "Fluid: screen bounds (compute)");
            }
        }

        // --- Where on screen the fluid passes need to run -----------------
        // From fluid_bounds.comp's result (read back 2 frames late, see
        // ReadbackFluidBounds). Everything outside this rectangle is skipped
        // by every fluid pass and by the composite. The rectangle is padded
        // generously: (a) by more than the widest blur/polish/normal-stencil
        // reach, so pixels near its edge never read stale texels from outside
        // it, and (b) by a screen-fraction margin to cover the camera/liquid
        // moving during the 2-frame readback delay (turn very fast toward
        // liquid and its first frames can be clipped at the edge).
        const bool fluidAllEmpty = FluidBoundsCullingEnabled && m_LastFluidBoundsValid && m_FluidEmptyReadings >= 2;
        const bool fluidChainActive = fluidSurfaceThisFrame && !FluidUseSoftBlobs && !fluidAllEmpty;
        m_FluidChainRanLastFrame = fluidChainActive;

        VkRect2D fluidRect{ { 0, 0 }, fluidExtent };
        VkRect2D compositeRect{ { 0, 0 }, extent };
        if (FluidBoundsCullingEnabled && m_LastFluidBoundsValid && m_LastFluidBounds.needsFullScreen == 0 &&
            m_LastFluidBounds.visibleCount > 0 && m_LastFluidBounds.maxX >= m_LastFluidBounds.minX &&
            m_LastFluidBounds.maxY >= m_LastFluidBounds.minY)
        {
            const float screenW = static_cast<float>(extent.width);
            const float screenH = static_cast<float>(extent.height);
            const float toFluid = static_cast<float>(fluidExtent.width) / screenW;
            const float marginScreen = 0.08f * std::max(screenW, screenH) + 56.0f / std::max(toFluid, 0.01f);

            const float x0 = std::max(static_cast<float>(m_LastFluidBounds.minX) - marginScreen, 0.0f);
            const float y0 = std::max(static_cast<float>(m_LastFluidBounds.minY) - marginScreen, 0.0f);
            const float x1 = std::min(static_cast<float>(m_LastFluidBounds.maxX) + marginScreen, screenW);
            const float y1 = std::min(static_cast<float>(m_LastFluidBounds.maxY) + marginScreen, screenH);

            if (x1 > x0 && y1 > y0)
            {
                compositeRect.offset = { static_cast<int32_t>(std::floor(x0)), static_cast<int32_t>(std::floor(y0)) };
                compositeRect.extent = { static_cast<uint32_t>(std::ceil(x1 - x0)) + 1u, static_cast<uint32_t>(std::ceil(y1 - y0)) + 1u };

                const int32_t fx0 = static_cast<int32_t>(std::floor(x0 * toFluid));
                const int32_t fy0 = static_cast<int32_t>(std::floor(y0 * toFluid));
                const int32_t fx1 = std::min(static_cast<int32_t>(std::ceil(x1 * toFluid)) + 1, static_cast<int32_t>(fluidExtent.width));
                const int32_t fy1 = std::min(static_cast<int32_t>(std::ceil(y1 * toFluid)) + 1, static_cast<int32_t>(fluidExtent.height));
                fluidRect.offset = { fx0, fy0 };
                fluidRect.extent = { static_cast<uint32_t>(std::max(fx1 - fx0, 1)), static_cast<uint32_t>(std::max(fy1 - fy0, 1)) };
            }
        }

        // Which texture the composite pass reads as "the fluid surface":
        // the temporal pass's output when temporal accumulation is on,
        // otherwise the polish pass's output directly. Filled in below.
        VkDescriptorSet fluidCompositeSurfaceSet = VK_NULL_HANDLE;
        if (fluidChainActive)
        {
            VkViewport fluidViewport{};
            fluidViewport.x = 0.0f;
            fluidViewport.y = 0.0f;
            fluidViewport.width = static_cast<float>(fluidExtent.width);
            fluidViewport.height = static_cast<float>(fluidExtent.height);
            fluidViewport.minDepth = 0.0f;
            fluidViewport.maxDepth = 1.0f;

            VkRect2D fluidScissor = fluidRect;

            // Pass 1: sphere-impostor depth splat - see fluid_depth.vert/.frag.
            // 3 attachments now (see m_FluidDepthPrepassRenderPass):
            // color(0), motion(1), depth(2).
            std::array<VkClearValue, 4> depthPassClears{};
            // Sentinel "no fluid here" value - see fluid_blur.frag/
            // fluid_composite.frag's SENTINEL_THRESHOLD, which both treat
            // anything above 1e6 as empty.
            depthPassClears[0].color = { { 1.0e9f, 0.0f, 0.0f, 0.0f } };
            // Zero motion - harmless even where a particle later writes a
            // real value (fully overwritten), and exactly correct for any
            // pixel that stays empty all frame (no particle = nothing to
            // reproject, fluid_temporal.frag never reads motion for an
            // empty pixel anyway since it gates on the depth/color
            // texture's own sentinel first).
            depthPassClears[1].color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
            depthPassClears[2].color = { { 0.0f, 0.0f, 0.0f, 0.0f } }; // no normal
            depthPassClears[3].depthStencil = { 1.0f, 0 };

            VkRenderPassBeginInfo depthPassInfo{};
            depthPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            depthPassInfo.renderPass = m_FluidDepthPrepassRenderPass;
            depthPassInfo.framebuffer = m_FluidDepthFramebuffers[m_CurrentFrame];
            depthPassInfo.renderArea = fluidRect;
            depthPassInfo.clearValueCount = static_cast<uint32_t>(depthPassClears.size());
            depthPassInfo.pClearValues = depthPassClears.data();

            vkCmdBeginRenderPass(cmd, &depthPassInfo, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_FluidDepthPipeline.Get());
            vkCmdSetViewport(cmd, 0, 1, &fluidViewport);
            vkCmdSetScissor(cmd, 0, 1, &fluidScissor);

            std::array<VkDescriptorSet, 3> depthSets = { frame.descriptorSet, m_ParticleGPUStorageSet, m_SceneDepthSet };
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_FluidDepthPipelineLayout.Get(),
                                     0, static_cast<uint32_t>(depthSets.size()), depthSets.data(), 0, nullptr);

            struct { float radius, invTargetW, invTargetH, pad; } depthPush{
                FluidParticleRadius, 1.0f / static_cast<float>(fluidExtent.width), 1.0f / static_cast<float>(fluidExtent.height), 0.0f
            };
            vkCmdPushConstants(cmd, m_FluidDepthPipelineLayout.Get(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                0, sizeof(depthPush), &depthPush);

            // 6 vertices/instance - a billboard quad (see
            // fluid_depth.vert), not 1 point/instance.
            vkCmdDraw(cmd, 6, particleGPUCount, 0, 0);
            vkCmdEndRenderPass(cmd);
            GpuStamp(cmd, "Fluid: depth pass");

            // Thickness pass - every particle additively writes the chord
            // length a view ray travels through it (see fluid_thickness.frag).
            // Independent of the depth/blur chain above, so it just needs to
            // finish before the main pass's composite samples it (the render
            // pass's exit dependency covers that).
            {
                VkClearValue thicknessClear{};
                thicknessClear.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };

                VkRenderPassBeginInfo thicknessPassInfo{};
                thicknessPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
                thicknessPassInfo.renderPass = m_FluidThicknessRenderPass;
                thicknessPassInfo.framebuffer = m_FluidThicknessFramebuffers[m_CurrentFrame];
                const VkExtent2D thicknessExtentDraw{ std::max(1u, (fluidExtent.width + m_FluidThicknessDivisor - 1) / m_FluidThicknessDivisor),
                                                       std::max(1u, (fluidExtent.height + m_FluidThicknessDivisor - 1) / m_FluidThicknessDivisor) };
                // Same rectangle, in thickness-target pixels.
                const float thicknessScaleX = static_cast<float>(thicknessExtentDraw.width) / static_cast<float>(fluidExtent.width);
                const float thicknessScaleY = static_cast<float>(thicknessExtentDraw.height) / static_cast<float>(fluidExtent.height);
                VkRect2D thicknessRect;
                thicknessRect.offset = { static_cast<int32_t>(std::floor(static_cast<float>(fluidRect.offset.x) * thicknessScaleX)),
                                          static_cast<int32_t>(std::floor(static_cast<float>(fluidRect.offset.y) * thicknessScaleY)) };
                thicknessRect.extent = { std::min(static_cast<uint32_t>(std::ceil(static_cast<float>(fluidRect.extent.width) * thicknessScaleX)) + 1u,
                                                   thicknessExtentDraw.width - static_cast<uint32_t>(thicknessRect.offset.x)),
                                          std::min(static_cast<uint32_t>(std::ceil(static_cast<float>(fluidRect.extent.height) * thicknessScaleY)) + 1u,
                                                   thicknessExtentDraw.height - static_cast<uint32_t>(thicknessRect.offset.y)) };
                thicknessPassInfo.renderArea = thicknessRect;
                thicknessPassInfo.clearValueCount = 1;
                thicknessPassInfo.pClearValues = &thicknessClear;

                VkViewport thicknessViewport = fluidViewport;
                thicknessViewport.width = static_cast<float>(thicknessExtentDraw.width);
                thicknessViewport.height = static_cast<float>(thicknessExtentDraw.height);
                VkRect2D thicknessScissor = thicknessRect;

                vkCmdBeginRenderPass(cmd, &thicknessPassInfo, VK_SUBPASS_CONTENTS_INLINE);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_FluidThicknessPipeline.Get());
                vkCmdSetViewport(cmd, 0, 1, &thicknessViewport);
                vkCmdSetScissor(cmd, 0, 1, &thicknessScissor);

                std::array<VkDescriptorSet, 4> thicknessSets = {
                    frame.descriptorSet, m_ParticleGPUStorageSet,
                    m_SceneDepthSet,                      // set 2 - opaque scene depth (occlusion)
                    m_FluidBlurXSets[m_CurrentFrame]      // set 3 - nearest visible liquid surface (this frame's UNBLURRED depth-pass output)
                };
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_FluidThicknessPipelineLayout.Get(),
                                         0, static_cast<uint32_t>(thicknessSets.size()), thicknessSets.data(), 0, nullptr);

                // window: how far behind the visible liquid surface liquid still
                // counts toward thickness - see fluid_thickness.frag. ~5 particle
                // radii: deep pools still saturate, separate bodies behind don't.
                struct { float radius, invTargetW, invTargetH, window; } thicknessPush{
                    FluidParticleRadius,
                    1.0f / static_cast<float>(thicknessExtentDraw.width),
                    1.0f / static_cast<float>(thicknessExtentDraw.height),
                    FluidParticleRadius * 5.0f
                };
                vkCmdPushConstants(cmd, m_FluidThicknessPipelineLayout.Get(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                    0, sizeof(thicknessPush), &thicknessPush);

                vkCmdDraw(cmd, 6, particleGPUCount, 0, 0);
                vkCmdEndRenderPass(cmd);
            }
            GpuStamp(cmd, "Fluid: thickness pass");

            // Refresh m_ParticlePreviousPositions for NEXT frame's
            // reprojection, now that THIS frame's depth pass has finished
            // reading it (that read is what the barrier below waits on).
            // See that buffer's own comment (Renderer.h) for why it
            // exists; this is the one place it's written. Only runs when
            // the depth pass itself ran (same gate as everything else in
            // this block) - if fluid surface rendering gets toggled back
            // on after being off for a while, the first frame back may
            // reproject against a stale snapshot from whenever it was
            // last on, which can show a brief, one-frame artifact; not
            // worth the extra per-frame cost of updating this buffer on
            // frames that never read it.
            // Only needed by the (optional) temporal pass - skipped entirely
            // when FluidTemporalEnabled is off.
            if (FluidTemporalEnabled)
            {
                VkBufferMemoryBarrier toTransferWrite{};
                toTransferWrite.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                toTransferWrite.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
                toTransferWrite.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                toTransferWrite.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toTransferWrite.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toTransferWrite.buffer = m_ParticlePreviousPositions.Get();
                toTransferWrite.offset = 0;
                toTransferWrite.size = VK_WHOLE_SIZE;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      0, 0, nullptr, 1, &toTransferWrite, 0, nullptr);

                VkBufferCopy copyRegion{ 0, 0, GPU_MAX_PARTICLES * sizeof(glm::vec4) };
                vkCmdCopyBuffer(cmd, m_ParticleLivePositionBuffer, m_ParticlePreviousPositions.Get(), 1, &copyRegion);

                // Post-copy barrier. The earlier comment here claimed
                // submission order was enough - it is not: Vulkan only
                // orders execution/visibility across commands through
                // explicit dependencies, and the next frame's depth-pass
                // vertex shader (which reads this buffer) can otherwise
                // start before this copy finished or became visible. That
                // gives garbage motion vectors on some frames, which is
                // exactly what feeds the temporal reprojection. A barrier's
                // second scope covers commands LATER IN SUBMISSION ORDER,
                // including the next frame's command buffer.
                //
                // Same barrier also covers the other direction: the copy
                // READS m_ParticleLivePositionBuffer, and next frame's SPH
                // compute WRITES it - write-after-read needs an execution
                // dependency too.
                VkMemoryBarrier afterCopy{};
                afterCopy.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                afterCopy.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
                afterCopy.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                      0, 1, &afterCopy, 0, nullptr, 0, nullptr);
            }

            // Pass 2/3: separable bilateral blur, horizontal then
            // vertical (see fluid_blur.frag) - X reads
            // m_FluidDepthImages[slot] and writes m_FluidBlurImagesA[slot];
            // Y reads m_FluidBlurImagesA[slot] and writes
            // m_FluidBlurImagesB[slot], the final texture
            // m_FluidCompositeSets[slot] (and so fluid_composite.frag)
            // samples.
            // Standard "focal length in pixels" quantity - how many
            // screen pixels a world-space unit projects to at exactly 1
            // unit of distance from the camera, along the vertical axis.
            // Together with a particle's own linear depth (already
            // stored per-pixel by the depth pass - see fluid_depth.frag's
            // comment on why it's linear depth, not Euclidean distance),
            // this is what lets fluid_blur.frag work out how many actual
            // screen pixels FluidParticleRadius covers for THIS
            // particle, instead of assuming a fixed pixel count - see
            // that shader's own comment for why a fixed kernel radius
            // only blended particles into a smooth surface far from the
            // camera and fell apart up close.
            float pixelsPerWorldUnitAtUnitDistance =
                static_cast<float>(fluidExtent.height) / (2.0f * std::tan(glm::radians(m_Camera.FovDegrees) * 0.5f));

            auto recordBlurPass = [&](VkFramebuffer targetFramebuffer, VkDescriptorSet sourceSet, float dirX, float dirY)
            {
                VkClearValue blurClear{};
                blurClear.color = { { 0.0f, 0.0f, 0.0f, 0.0f } }; // LOAD_OP_DONT_CARE for this attachment - never actually read, see m_FluidBlurRenderPass's comment

                VkRenderPassBeginInfo blurPassInfo{};
                blurPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
                blurPassInfo.renderPass = m_FluidBlurRenderPass;
                blurPassInfo.framebuffer = targetFramebuffer;
                blurPassInfo.renderArea = fluidRect;
                blurPassInfo.clearValueCount = 1;
                blurPassInfo.pClearValues = &blurClear;

                vkCmdBeginRenderPass(cmd, &blurPassInfo, VK_SUBPASS_CONTENTS_INLINE);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_FluidBlurPipeline.Get());
                vkCmdSetViewport(cmd, 0, 1, &fluidViewport);
                vkCmdSetScissor(cmd, 0, 1, &fluidScissor);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_FluidBlurPipelineLayout.Get(),
                                         0, 1, &sourceSet, 0, nullptr);

                struct { float texelSizeX, texelSizeY, dirX, dirY, pixelsPerWorldUnitAtUnitDistance, particleRadius; } blurPush{
                    1.0f / static_cast<float>(fluidExtent.width), 1.0f / static_cast<float>(fluidExtent.height), dirX, dirY,
                    pixelsPerWorldUnitAtUnitDistance, FluidParticleRadius
                };
                vkCmdPushConstants(cmd, m_FluidBlurPipelineLayout.Get(), VK_SHADER_STAGE_FRAGMENT_BIT,
                                    0, sizeof(blurPush), &blurPush);

                vkCmdDraw(cmd, 3, 1, 0, 0);
                vkCmdEndRenderPass(cmd);
            };

            recordBlurPass(m_FluidBlurFramebuffersA[m_CurrentFrame], m_FluidBlurXSets[m_CurrentFrame], 1.0f, 0.0f);
            recordBlurPass(m_FluidBlurFramebuffersB[m_CurrentFrame], m_FluidBlurYSets[m_CurrentFrame], 0.0f, 1.0f);
            GpuStamp(cmd, "Fluid: blur (2 passes)");

            // Pass 3: fixed-radius 2D polish (see fluid_blur_polish.frag
            // for why this exists and why its cost is bounded regardless
            // of scene/camera distance, unlike the full-radius 2D pass it
            // replaced). Reads m_FluidBlurImagesB (the separable blur's
            // output), writes m_FluidPolishImages - what the temporal
            // pass's "new frame" input reads next.
            {
                VkClearValue polishClear{};
                polishClear.color = { { 0.0f, 0.0f, 0.0f, 0.0f } }; // never read, same reasoning as the blur pass's own clear value

                VkRenderPassBeginInfo polishPassInfo{};
                polishPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
                polishPassInfo.renderPass = m_FluidBlurRenderPass; // reused - see m_FluidPolishFramebuffers' comment
                polishPassInfo.framebuffer = m_FluidPolishFramebuffers[m_CurrentFrame];
                polishPassInfo.renderArea = fluidRect;
                polishPassInfo.clearValueCount = 1;
                polishPassInfo.pClearValues = &polishClear;

                vkCmdBeginRenderPass(cmd, &polishPassInfo, VK_SUBPASS_CONTENTS_INLINE);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_FluidPolishPipeline.Get());
                vkCmdSetViewport(cmd, 0, 1, &fluidViewport);
                vkCmdSetScissor(cmd, 0, 1, &fluidScissor);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_FluidPolishPipelineLayout.Get(),
                                         0, 1, &m_FluidPolishSets[m_CurrentFrame], 0, nullptr);

                struct { float texelSizeX, texelSizeY, pixelsPerWorldUnitAtUnitDistance, particleRadius; } polishPush{
                    1.0f / static_cast<float>(fluidExtent.width), 1.0f / static_cast<float>(fluidExtent.height),
                    pixelsPerWorldUnitAtUnitDistance, FluidParticleRadius
                };
                vkCmdPushConstants(cmd, m_FluidPolishPipelineLayout.Get(), VK_SHADER_STAGE_FRAGMENT_BIT,
                                    0, sizeof(polishPush), &polishPush);

                vkCmdDraw(cmd, 3, 1, 0, 0);
                vkCmdEndRenderPass(cmd);
            }

            GpuStamp(cmd, "Fluid: polish pass");

            // Pass 4: temporal accumulation via reprojection (see
            // fluid_temporal.frag for the full design notes and
            // fluid_depth.vert's outMotion comment for where the motion
            // vectors it reprojects with come from).
            //
            // Index bookkeeping: m_FluidHistoryIndex holds whichever
            // history image is valid GOING INTO this frame (last frame's
            // output). historyReadIdx captures that; historyWriteIdx is
            // simply the other slot - this frame writes there, and it
            // becomes next frame's valid history once m_FluidHistoryIndex
            // is updated below. This alternates once per CALL TO
            // DrawFrame, i.e. once per actually-rendered frame - NOT tied
            // to m_CurrentFrame (which cycles for frame-in-flight
            // pipelining and has no relationship to this index - see
            // m_FluidHistoryImages' own comment in Renderer.h).
            int historyReadIdx = m_FluidHistoryIndex;
            int historyWriteIdx = 1 - historyReadIdx;
            if (FluidTemporalEnabled)
            {
                VkClearValue temporalClear{};
                temporalClear.color = { { 0.0f, 0.0f, 0.0f, 0.0f } }; // never read - the fullscreen triangle overwrites every pixel

                VkRenderPassBeginInfo temporalPassInfo{};
                temporalPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
                temporalPassInfo.renderPass = m_FluidBlurRenderPass; // reused - see m_FluidHistoryFramebuffers' comment
                temporalPassInfo.framebuffer = m_FluidHistoryFramebuffers[historyWriteIdx];
                temporalPassInfo.renderArea = fluidRect;
                temporalPassInfo.clearValueCount = 1;
                temporalPassInfo.pClearValues = &temporalClear;

                vkCmdBeginRenderPass(cmd, &temporalPassInfo, VK_SUBPASS_CONTENTS_INLINE);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_FluidTemporalPipeline.Get());
                vkCmdSetViewport(cmd, 0, 1, &fluidViewport);
                vkCmdSetScissor(cmd, 0, 1, &fluidScissor);

                std::array<VkDescriptorSet, 3> temporalSets = {
                    m_FluidTemporalNewSets[m_CurrentFrame],      // set 0 - this frame's polish output
                    m_FluidMotionSets[m_CurrentFrame],           // set 1 - this frame's unblurred motion vectors
                    m_FluidTemporalHistorySets[historyReadIdx]   // set 2 - last frame's accumulated result
                };
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_FluidTemporalPipelineLayout.Get(),
                                         0, static_cast<uint32_t>(temporalSets.size()), temporalSets.data(), 0, nullptr);

                vkCmdDraw(cmd, 3, 1, 0, 0);
                vkCmdEndRenderPass(cmd);
            }

            // This frame's blended result (historyWriteIdx) is what
            // composite reads below, AND becomes next frame's history -
            // see m_FluidCompositeSets' own comment (Renderer.h) for why
            // it's indexed by this instead of m_CurrentFrame now.
            if (FluidTemporalEnabled)
            {
                m_FluidHistoryIndex = historyWriteIdx;
                fluidCompositeSurfaceSet = m_FluidCompositeSets[m_FluidHistoryIndex];
            }
            else
            {
                // No temporal pass: composite reads the polish output
                // directly (this slot's m_FluidPolishImages).
                fluidCompositeSurfaceSet = m_FluidTemporalNewSets[m_CurrentFrame];
            }
        }

        if (splitMainPass)
        {
            GpuStamp(cmd, "Fluid: temporal pass + setup");

            VkRenderPassBeginInfo overlayPassInfo = renderPassInfo;
            overlayPassInfo.renderPass = m_OverlayRenderPass.Get();
            overlayPassInfo.clearValueCount = 0; // both attachments are LOADed
            overlayPassInfo.pClearValues = nullptr;
            vkCmdBeginRenderPass(cmd, &overlayPassInfo, VK_SUBPASS_CONTENTS_INLINE);

            // Dynamic state doesn't carry through the fluid passes' own
            // viewports - restore the full-screen one for the overlay.
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            vkCmdSetScissor(cmd, 0, 1, &scissor);
        }


        // Fluid surface composite - drawn after raymarch, same render
        // pass/depth buffer, so it depth-tests against (and can be
        // occluded by) both rasterized geometry and raymarched objects.
        // See fluid_composite.frag's file comment. Mutually exclusive
        // with the soft-blob draw just below - see FluidUseSoftBlobs.
        if (fluidChainActive)
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_FluidCompositePipeline.Get());

            std::array<VkDescriptorSet, 4> compositeSets = {
                frame.descriptorSet,
                fluidCompositeSurfaceSet,                  // set 1 - blurred depth + color
                m_FluidThicknessSets[m_CurrentFrame],      // set 2 - accumulated thickness (opacity), half-res + linear
                m_FluidNormalSets[m_CurrentFrame]          // set 3 - unblurred analytic normals (isolated droplets)
            };
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_FluidCompositePipelineLayout.Get(),
                                     0, static_cast<uint32_t>(compositeSets.size()), compositeSets.data(), 0, nullptr);

            struct { float absorption, minAlpha, maxAlpha, particleRadius; } compositePush{
                FluidAbsorption, FluidMinAlpha, FluidMaxAlpha, FluidParticleRadius
            };
            vkCmdPushConstants(cmd, m_FluidCompositePipelineLayout.Get(), VK_SHADER_STAGE_FRAGMENT_BIT,
                                0, sizeof(compositePush), &compositePush);

            // No push constants anymore - color comes from the texture
            // itself now (see fluid_composite.frag's surfaceColor).
            // No vertex/instance buffers - fullscreen triangle, same as
            // the raymarch draw just above.
            // Only the rectangle the fluid passes ran in (anywhere else the
            // fluid textures hold stale data and there is no liquid to draw).
            vkCmdSetScissor(cmd, 0, 1, &compositeRect);
            vkCmdDraw(cmd, 3, 1, 0, 0);
            vkCmdSetScissor(cmd, 0, 1, &scissor);
        }
        GpuStamp(cmd, "Main: fluid composite");

        // Soft-billboard liquid (see FluidUseSoftBlobs, m_FluidBlobPipeline's
        // own comment in Renderer.h) - one forward alpha-blended draw,
        // same render pass/depth buffer as everything else here, no
        // offscreen passes of its own at all. particleGPUCount instances
        // of a 6-vertex (2-triangle) billboard quad, same draw shape as
        // the screen-space technique's own depth prepass.
        if (fluidSurfaceThisFrame && FluidUseSoftBlobs)
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_FluidBlobPipeline.Get());

            std::array<VkDescriptorSet, 2> blobSets = { frame.descriptorSet, m_ParticleGPUStorageSet };
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_FluidBlobPipelineLayout.Get(),
                                     0, static_cast<uint32_t>(blobSets.size()), blobSets.data(), 0, nullptr);

            float blobRadius = FluidParticleRadius;
            vkCmdPushConstants(cmd, m_FluidBlobPipelineLayout.Get(), VK_SHADER_STAGE_VERTEX_BIT,
                                0, sizeof(blobRadius), &blobRadius);

            vkCmdDraw(cmd, 6, particleGPUCount, 0, 0);
        }

        // Recorded last, within the same render pass, so UI draws on top
        // of the scene rather than being overwritten by it.
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

        vkCmdEndRenderPass(cmd);

        GpuStamp(cmd, "Main: ImGui + particle points");
        Profiler::Get().Add("Frame: record command buffer (CPU)", Profiler::MsSince(recordStart));
        frame.commandBuffer.End();

        VkSemaphore waitSemaphores[] = { frame.imageAvailableSemaphore.Get() };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        // Indexed by imageIndex (which swapchain image got acquired), NOT
        // by m_CurrentFrame (which frame-in-flight slot is being used).
        // Those two only happen to match by coincidence with 2 frames in
        // flight and simple/light workloads - under real GPU load
        // (variable frame timing, images acquired out of the naive
        // round-robin order) they diverge, and reusing a semaphore by
        // frame-in-flight index means resubmitting it before the present
        // engine's prior use of it is guaranteed done. See the NOTE on
        // FrameContext in Frame/FrameContext.h.
        VkSemaphore signalSemaphores[] = { m_RenderFinishedSemaphores[imageIndex]->Get() };

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(m_Context.Device().GetGraphicsQueue(), 1, &submitInfo, frame.inFlightFence.Get()) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to submit draw command buffer");
        }

        VkSwapchainKHR swapchains[] = { m_Context.Swapchain().Get() };

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapchains;
        presentInfo.pImageIndices = &imageIndex;

        VkResult presentResult = vkQueuePresentKHR(m_Context.Device().GetPresentQueue(), &presentInfo);

        if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR || m_FramebufferResized)
        {
            m_FramebufferResized = false;
            RecreateSwapchainResources();
        }
        else if (presentResult != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to present swapchain image");
        }

        // Captured for NEXT frame's UniformBufferObject::prevViewProj -
        // see that field's own comment. Deliberately uses the exact
        // view/proj THIS frame's UBO was built from (not a fresh
        // m_Camera read), so it genuinely reflects "how the camera
        // looked when this frame was drawn," even if something mutates
        // m_Camera between now and the next DrawFrame call.
        m_PreviousViewProj = ubo.proj * ubo.view;

        m_CurrentFrame = (m_CurrentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void Renderer::WaitIdle()
    {
        m_Context.Device().WaitIdle();
    }

    void Renderer::Shutdown()
    {
        WaitIdle();

        ShutdownImGui();

        for (auto& frame : m_Frames)
        {
            frame.Shutdown();
        }

        for (auto& semaphore : m_RenderFinishedSemaphores)
        {
            semaphore->Shutdown();
        }
        m_RenderFinishedSemaphores.clear();

        m_DescriptorPool.Shutdown();
        m_DescriptorSetLayout.Shutdown();
        m_TextureSetLayout.Shutdown();

        m_CommandPool.Shutdown();
        m_GraphicsPipeline.Shutdown();
        m_VoxelPipeline.Shutdown();
        m_ParticlePointsPipeline.Shutdown();
        m_PipelineLayout.Shutdown();

        m_ParticlePointsGPUPipeline.Shutdown();
        m_ParticleGPUPipelineLayout.Shutdown();
        m_ParticleGPUDescriptorPool.Shutdown();
        m_ParticlePreviousPositions.Shutdown();
        if (m_ParticleGPUSetLayout != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(m_Context.Device().Get(), m_ParticleGPUSetLayout, nullptr);
            m_ParticleGPUSetLayout = VK_NULL_HANDLE;
        }

        m_SceneSplitRenderPass.Shutdown();
        m_OverlayRenderPass.Shutdown();
        m_RenderPass.Shutdown();

        // Raymarch pass resources (see InitRaymarchPass) - these were
        // previously never torn down at all (a pre-existing gap, not
        // introduced by the bindless rewrite - m_RaymarchSetLayout has
        // always been a plain VkDescriptorSetLayout with no destructor
        // to rely on). Fixed here while this area was already being
        // touched, following the same explicit-destroy pattern
        // m_ParticleGPUSetLayout uses above.
        m_RaymarchPipeline.Shutdown();
        m_RaymarchPipelineLayout.Shutdown();
        for (auto& buffer : m_RaymarchObjectBuffers)
        {
            buffer.Shutdown(); // was never torn down at all before - a separate pre-existing gap, fixed alongside the per-frame rework
        }
        for (VkDescriptorSet& set : m_RaymarchSets)
        {
            // No explicit vkFreeDescriptorSets call needed - destroying
            // the pool below implicitly frees every set allocated from
            // it (VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT was
            // never requested, so per-set freeing isn't even available -
            // this is just documenting that the pool destroy below is
            // sufficient, not a missing step).
            set = VK_NULL_HANDLE;
        }
        if (m_RaymarchDescriptorPool != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(m_Context.Device().Get(), m_RaymarchDescriptorPool, nullptr);
            m_RaymarchDescriptorPool = VK_NULL_HANDLE;
        }
        if (m_RaymarchSetLayout != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(m_Context.Device().Get(), m_RaymarchSetLayout, nullptr);
            m_RaymarchSetLayout = VK_NULL_HANDLE;
        }

        m_DepthImage.Shutdown();

        // Fluid-surface pass (see InitFluidSurfacePass/RegisterParticleGPUSource) -
        // safe to call unconditionally even if RegisterParticleGPUSource
        // was never invoked: DestroyFluidSurfaceResources/VulkanImage::
        // Shutdown/VulkanGraphicsPipeline::Shutdown/VulkanPipelineLayout::
        // Shutdown all already guard on their handles being non-null
        // before destroying anything, same as every other Shutdown() in
        // this file.
        DestroyFluidSurfaceResources();
        m_FluidDepthPipeline.Shutdown();
        m_FluidDepthPipelineLayout.Shutdown();
        m_FluidBlurPipeline.Shutdown();
        m_FluidBlurPipelineLayout.Shutdown();
        m_FluidPolishPipeline.Shutdown();
        m_FluidPolishPipelineLayout.Shutdown();
        m_FluidTemporalPipeline.Shutdown();
        m_FluidTemporalPipelineLayout.Shutdown();
        m_FluidCompositePipeline.Shutdown();
        m_FluidCompositePipelineLayout.Shutdown();
        m_FluidThicknessPipeline.Shutdown();
        m_FluidThicknessPipelineLayout.Shutdown();
        if (m_FluidThicknessRenderPass != VK_NULL_HANDLE)
        {
            vkDestroyRenderPass(m_Context.Device().Get(), m_FluidThicknessRenderPass, nullptr);
            m_FluidThicknessRenderPass = VK_NULL_HANDLE;
        }
        m_FluidBlobPipeline.Shutdown();
        m_FluidBlobPipelineLayout.Shutdown();
        m_FluidSampler.Shutdown();
        m_FluidLinearSampler.Shutdown();
        DestroyGpuTimers();
        DestroyFluidBounds();
        if (m_FluidDepthPrepassRenderPass != VK_NULL_HANDLE)
        {
            vkDestroyRenderPass(m_Context.Device().Get(), m_FluidDepthPrepassRenderPass, nullptr);
            m_FluidDepthPrepassRenderPass = VK_NULL_HANDLE;
        }
        if (m_FluidBlurRenderPass != VK_NULL_HANDLE)
        {
            vkDestroyRenderPass(m_Context.Device().Get(), m_FluidBlurRenderPass, nullptr);
            m_FluidBlurRenderPass = VK_NULL_HANDLE;
        }

        m_MeshRegistry.clear(); // each Mesh's destructor frees its VulkanBuffer
        m_TextureRegistry.clear(); // each VulkanTexture's destructor frees its VulkanImage/VulkanSampler
        m_Allocator.Shutdown();

        m_Context.Shutdown();
    }
}
