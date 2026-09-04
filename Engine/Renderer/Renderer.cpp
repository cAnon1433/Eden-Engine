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

    void Renderer::RegisterParticleGPUSource(VkBuffer positionBuffer)
    {
        VkDevice device = m_Context.Device().Get();

        // Single binding: the compute-owned positions storage buffer,
        // read-only from the vertex stage - see particle_point_gpu.vert.
        VkDescriptorSetLayoutBinding storageBinding{};
        storageBinding.binding = 0;
        storageBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        storageBinding.descriptorCount = 1;
        storageBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &storageBinding;

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_ParticleGPUSetLayout) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to create particle-GPU descriptor set layout");
        }

        // Dedicated tiny pool - one set, one binding - rather than
        // resizing m_DescriptorPool (which is sized exactly for the
        // camera-UBO/texture sets already allocated by the time this
        // runs). See VulkanDescriptorPool's own comment on why pool
        // sizing is caller-specified rather than hardcoded.
        std::vector<VkDescriptorPoolSize> poolSizes = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 } };
        m_ParticleGPUDescriptorPool.Init(device, poolSizes, 1);
        m_ParticleGPUStorageSet = m_ParticleGPUDescriptorPool.AllocateSet(m_ParticleGPUSetLayout);

        VkDescriptorBufferInfo bufferInfo{ positionBuffer, 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = m_ParticleGPUStorageSet;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &bufferInfo;
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

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

    void Renderer::DrawFrame(const std::vector<DrawCommand>& drawList, const std::function<void()>& buildUI,
                              const std::function<void(VkCommandBuffer)>& recordComputeWork, uint32_t particleGPUCount,
                              const std::vector<VoxelDrawSource>& voxelSources,
                              const std::vector<RaymarchObjectGPU>& raymarchObjects,
                              VkBuffer raymarchDensityBuffer)
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
        frame.UpdateUniformBuffer(ubo);

        frame.commandBuffer.Reset();
        frame.commandBuffer.Begin();

        VkCommandBuffer cmd = frame.commandBuffer.Get();

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

            VkMemoryBarrier computeToVertexBarrier{};
            computeToVertexBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            computeToVertexBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            computeToVertexBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                                  0, 1, &computeToVertexBarrier, 0, nullptr, 0, nullptr);
        }

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

        // GPU-resident particles - entirely separate draw path from the
        // mesh loop above (see RegisterParticleGPUSource's class
        // comment): no InstanceData, no vertex buffer, position comes
        // straight from the storage buffer via gl_InstanceIndex. Only
        // fires once RegisterParticleGPUSource has actually been called
        // (m_ParticleGPUStorageSet stays VK_NULL_HANDLE until then).
        if (particleGPUCount > 0 && m_ParticleGPUStorageSet != VK_NULL_HANDLE)
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
        if (!voxelSources.empty())
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

        // Raymarch pass - fullscreen sphere-tracing of every "raymarch
        // object" (see Raymarch/RaymarchSystem.h), recorded after
        // rasterized mesh/voxel geometry so this frame's depth buffer
        // already holds terrain's real depth values (raymarch.frag
        // relies on depth test being ON to composite correctly against
        // that, and writes its own gl_FragDepth per hit so it can in
        // turn be occluded by geometry drawn in a future frame's
        // earlier passes - see that shader's depth-write comment).
        if (!raymarchObjects.empty())
        {
            UpdateRaymarchDescriptors(raymarchObjects, raymarchDensityBuffer);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_RaymarchPipeline.Get());

            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_RaymarchPipelineLayout.Get(),
                                     0, 1, &frame.descriptorSet, 0, nullptr);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_RaymarchPipelineLayout.Get(),
                                     1, 1, &m_RaymarchSets[m_CurrentFrame], 0, nullptr);

            RaymarchPushConstants pushConstants{};
            pushConstants.objectCount = static_cast<int32_t>(std::min(raymarchObjects.size(),
                                                                        static_cast<size_t>(kRaymarchMaxObjects)));
            pushConstants.nearPlane = m_Camera.NearPlane;
            pushConstants.farPlane = m_Camera.FarPlane;
            vkCmdPushConstants(cmd, m_RaymarchPipelineLayout.Get(), VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(RaymarchPushConstants), &pushConstants);

            // No vertex/instance buffers bound - see raymarch.vert,
            // vertices are generated from gl_VertexIndex.
            vkCmdDraw(cmd, 3, 1, 0, 0);
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
        m_VoxelPipeline.Shutdown();
        m_ParticlePointsPipeline.Shutdown();
        m_PipelineLayout.Shutdown();

        m_ParticlePointsGPUPipeline.Shutdown();
        m_ParticleGPUPipelineLayout.Shutdown();
        m_ParticleGPUDescriptorPool.Shutdown();
        if (m_ParticleGPUSetLayout != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(m_Context.Device().Get(), m_ParticleGPUSetLayout, nullptr);
            m_ParticleGPUSetLayout = VK_NULL_HANDLE;
        }

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

        m_MeshRegistry.clear(); // each Mesh's destructor frees its VulkanBuffer
        m_TextureRegistry.clear(); // each VulkanTexture's destructor frees its VulkanImage/VulkanSampler
        m_Allocator.Shutdown();

        m_Context.Shutdown();
    }
}
