#include "Renderer.h"
#include "Vulkan/RendererTypes.h"
#include "Primitives.h"
#include "ModelLoader.h"

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
#include <cctype>
#include <iostream>

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
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
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

        // Image count CAN change on recreate (rare, but the spec doesn't
        // guarantee it stays the same) - rebuild to match rather than
        // assume the old count still applies.
        CreateRenderFinishedSemaphores();
    }

    void Renderer::DrawFrame(const std::vector<DrawCommand>& drawList, const std::function<void()>& buildUI)
    {
        FrameContext& frame = m_Frames[m_CurrentFrame];

        frame.inFlightFence.Wait();

        uint32_t imageIndex;
        VkResult acquireResult = vkAcquireNextImageKHR(
            m_Context.Device().Get(),
            m_Context.Swapchain().Get(),
            std::numeric_limits<uint64_t>::max(),
            frame.imageAvailableSemaphore.Get(),
            VK_NULL_HANDLE,
            &imageIndex);

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
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        if (buildUI)
        {
            buildUI();
        }
        ImGui::Render();

        // Camera matrices, uploaded fresh every frame via the persistently
        // mapped uniform buffer - no vkMap/vkUnmap needed per frame.
        VkExtent2D extent = m_Context.Swapchain().GetExtent();
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
        ubo.lightDirection = glm::normalize(glm::vec3(0.4f, -1.0f, 0.3f));
        ubo.lightColor = glm::vec3(1.0f, 1.0f, 0.95f);
        ubo.ambientColor = glm::vec3(0.15f, 0.15f, 0.18f);
        frame.UpdateUniformBuffer(ubo);

        frame.commandBuffer.Reset();
        frame.commandBuffer.Begin();

        VkCommandBuffer cmd = frame.commandBuffer.Get();

        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = { { 0.02f, 0.02f, 0.05f, 1.0f } };
        clearValues[1].depthStencil = { 1.0f, 0 };

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = m_RenderPass.Get();
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

        // Recorded last, within the same render pass, so UI draws on top
        // of the scene rather than being overwritten by it.
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

        vkCmdEndRenderPass(cmd);

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
        m_ParticlePointsPipeline.Shutdown();
        m_PipelineLayout.Shutdown();
        m_RenderPass.Shutdown();

        m_DepthImage.Shutdown();

        m_MeshRegistry.clear(); // each Mesh's destructor frees its VulkanBuffer
        m_TextureRegistry.clear(); // each VulkanTexture's destructor frees its VulkanImage/VulkanSampler
        m_Allocator.Shutdown();

        m_Context.Shutdown();
    }
}
