#pragma once

#include "Vulkan/VulkanContext.h"
#include "Vulkan/Pipeline/VulkanRenderPass.h"
#include "Vulkan/Pipeline/VulkanPipelineLayout.h"
#include "Vulkan/Pipeline/VulkanGraphicsPipeline.h"
#include "Vulkan/Pipeline/VulkanDescriptorSetLayout.h"
#include "Vulkan/Pipeline/VulkanDescriptorPool.h"
#include "Vulkan/Pipeline/VulkanTextureSetLayout.h"
#include "Vulkan/Command/VulkanCommandPool.h"
#include "Vulkan/Frame/FrameContext.h"
#include "Vulkan/Resources/VulkanMemoryAllocator.h"
#include "Vulkan/Resources/VulkanImage.h"
#include "Vulkan/Resources/VulkanTexture.h"
#include "Vulkan/Resources/Mesh.h"
#include "Camera.h"
#include "Frustum.h"

#include <array>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>

struct GLFWwindow;

namespace Eden
{
    // Top-level facade the rest of Eden calls into. Everything else in
    // Engine/Renderer/Vulkan is an implementation detail behind this class.
    class Renderer
    {
    public:
        static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

        void Init(GLFWwindow* window);
        void Shutdown();

        // Records and submits one frame from the given draw list, presents
        // it, and advances the frame-in-flight index. Call once per main
        // loop iteration - the draw list is normally built fresh each
        // frame by Systems/RenderSystem.h from ECS Transform+Mesh pairs.
        //
        // buildUI, if provided, is called between ImGui::NewFrame() and
        // ImGui::Render() - put arbitrary ImGui:: calls (ImGui::Begin,
        // ImGui::Button, etc.) in there to add in-engine UI for that
        // frame. Deliberately a callback rather than Renderer owning any
        // UI logic itself - Renderer has no business knowing about
        // Registry/ECS/what a "create cube" button should do, that's an
        // application-level concern (see Engine/UI/EdenUI.h for Eden's
        // actual mesh create/destroy panel, wired in from main.cpp).
        // Left empty, no UI renders - existing call sites need no changes.
        void DrawFrame(const std::vector<DrawCommand>& drawList, const std::function<void()>& buildUI = {});

        // Call from the GLFW framebuffer-resize callback.
        void NotifyFramebufferResized() { m_FramebufferResized = true; }

        void WaitIdle();

        // Uploads geometry to the GPU once and returns a stable handle for
        // it. Call during/after Init(), before the main loop starts - this
        // does a GPU upload (and a full queue wait), it's not meant for
        // per-frame use. Many entities can share one handle via
        // MeshComponent - that's the whole point of separating "geometry
        // resource" from "where it's drawn this frame".
        //
        // `texture` is a per-MESH property, not per-entity - see
        // RendererTypes.h's comment on TextureHandle for why. Leave it
        // default (InvalidTextureHandle) for an untextured mesh; it'll be
        // bound to Eden's built-in white fallback so vertex color /
        // ColorComponent overrides still work exactly as before.
        MeshHandle CreateMesh(const std::vector<Vertex>& vertices, TextureHandle texture = InvalidTextureHandle);
        MeshHandle CreateCubeMesh(float size = 1.0f, TextureHandle texture = InvalidTextureHandle);

        // General indexed-mesh entry point - CreateCubeMesh's underlying
        // path, exposed directly for anything that already has its own
        // vertex/index data (e.g. a future OBJ/glTF loader).
        MeshHandle CreateIndexedMesh(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices,
                                      TextureHandle texture = InvalidTextureHandle);

        // Loads a 3D model file (currently OBJ - see ModelLoader.h) from
        // disk and uploads it, same one-time-cost/load-time-only caveat as
        // CreateMesh/CreateTexture. Put model files under Assets/Models/
        // (see Assets/README.md for why paths are relative to build/, not
        // the project root).
        MeshHandle CreateMeshFromFile(const std::string& path, TextureHandle texture = InvalidTextureHandle);

        // Loads an image file from disk and uploads it to the GPU. Same
        // load-time-only caveat as CreateMesh - does a real GPU upload,
        // don't call this per-frame. Returns a handle to pass into
        // CreateMesh/CreateCubeMesh/CreateIndexedMesh.
        TextureHandle CreateTexture(const std::string& path);

        Camera& GetCamera() { return m_Camera; }

        // View * projection using the CURRENT swapchain aspect ratio -
        // same calculation DrawFrame itself uses for the camera UBO (see
        // its `ubo.view`/`ubo.proj` lines), exposed here so callers that
        // need to reason about what's actually visible this frame (frustum
        // culling in RenderSystem::BuildDrawList) don't duplicate the
        // aspect-ratio lookup and risk it drifting out of sync with what
        // DrawFrame actually renders.
        glm::mat4 GetViewProjectionMatrix();

        // Builds this frame's view Frustum from the current camera state
        // - see Frustum::FromViewProjection's own comment for why near/
        // far are constructed from Camera's scalar fields rather than
        // extracted from the matrix. Preferred over calling
        // GetViewProjectionMatrix() and building the Frustum yourself -
        // keeps that convention-safety detail in one place.
        Frustum GetViewFrustum();

        // Bounding-sphere radius of `handle`'s mesh, in the mesh's own
        // LOCAL (unscaled, untranslated) space - see Mesh::GetBoundingRadius.
        // Returns 0 for InvalidMeshHandle or an out-of-range handle rather
        // than asserting/throwing, so a caller mid-culling-loop over
        // possibly-stale MeshComponents degrades to "treat as a point"
        // instead of crashing.
        float GetMeshBoundingRadius(MeshHandle handle) const;

        // MeshHandle for the single-vertex "point" mesh (see
        // CreateParticlePointResources in Renderer.cpp) - pass this as a
        // DrawCommand's mesh field to render that instance as a round
        // point sprite via m_ParticlePointsPipeline instead of ordinary
        // triangle geometry. Created once in Init(), always valid after
        // that - InvalidMeshHandle is never a real possibility here the
        // way it is for CreateMesh's caller-supplied handles, so this
        // doesn't need the same defensive check GetMeshBoundingRadius
        // above does.
        MeshHandle GetParticlePointMesh() const { return m_ParticlePointMeshHandle; }

        // Screen-space diameter, in pixels, that particle points render
        // at (gl_PointSize in particle_point.vert) - adjustable at
        // runtime (see EdenUI's particle panel), unlike everything else
        // about a mesh's appearance which is fixed at creation time. This
        // is deliberately public and mutable rather than going through a
        // setter - matches ParticleSystem's own tunables (smoothingRadius,
        // stiffness, etc.), which are public fields for the same reason:
        // they're meant to be live-tweaked from a UI panel, not
        // encapsulated behind getter/setter ceremony.
        float ParticlePointSize = 6.0f;

    private:
        void RecreateSwapchainResources();
        void CreateDepthResources();
        VkFormat FindDepthFormat() const;
        void CreateRenderFinishedSemaphores();
        void InitImGui();
        void ShutdownImGui();

        // Maps a (possibly InvalidTextureHandle) TextureHandle to the
        // actual descriptor set to bind - resolving InvalidTextureHandle
        // to the default white texture happens here, once, so
        // CreateMesh/CreateCubeMesh/CreateIndexedMesh don't each need
        // their own copy of that fallback logic.
        VkDescriptorSet ResolveTextureDescriptorSet(TextureHandle texture) const;

    private:
        GLFWwindow* m_Window = nullptr;

        VulkanContext m_Context;
        VulkanRenderPass m_RenderPass;
        VulkanDescriptorSetLayout m_DescriptorSetLayout;
        VulkanTextureSetLayout m_TextureSetLayout;
        VulkanDescriptorPool m_DescriptorPool;
        VulkanPipelineLayout m_PipelineLayout;
        VulkanGraphicsPipeline m_GraphicsPipeline;
        // Second pipeline, same VkPipelineLayout/render pass/vertex input
        // description as m_GraphicsPipeline (see CreateParticlePointResources
        // in Renderer.cpp) but VK_PRIMITIVE_TOPOLOGY_POINT_LIST topology and
        // particle_point.vert/.frag shader modules instead of triangle's -
        // DrawFrame binds whichever of the two is appropriate right before
        // each mesh's draw call (see the per-mesh loop there).
        VulkanGraphicsPipeline m_ParticlePointsPipeline;
        MeshHandle m_ParticlePointMeshHandle = InvalidMeshHandle;
        VulkanCommandPool m_CommandPool;
        VulkanMemoryAllocator m_Allocator;
        VulkanImage m_DepthImage;
        std::vector<Mesh> m_MeshRegistry;

        // Texture registry, same "index = handle" shape as m_MeshRegistry.
        // unique_ptr, not a plain vector<VulkanTexture>, because
        // VulkanImage (which VulkanTexture owns) is neither copyable nor
        // movable - same reasoning as m_RenderFinishedSemaphores below.
        std::vector<std::unique_ptr<VulkanTexture>> m_TextureRegistry;
        // Always index 0 in practice (created first, in Init()) - kept as
        // its own named handle rather than hardcoding 0 everywhere so the
        // "index 0 is special" assumption lives in exactly one place.
        TextureHandle m_DefaultTextureHandle = InvalidTextureHandle;

        // Scratch space for grouping DrawCommands by mesh in DrawFrame,
        // kept as a member (not a local rebuilt from scratch every call)
        // so its vectors' heap capacity survives across frames instead of
        // being freed and reallocated every single frame regardless of
        // how static the scene is. DrawFrame clears each vector's
        // CONTENTS at the top of every call but doesn't erase map entries,
        // so a mesh used every frame keeps its already-grown capacity
        // indefinitely - only genuinely new MeshHandles cause a fresh
        // allocation.
        std::unordered_map<MeshHandle, std::vector<InstanceData>> m_GroupedByMeshScratch;

        Camera m_Camera;

        std::array<FrameContext, MAX_FRAMES_IN_FLIGHT> m_Frames;

        // One semaphore per SWAPCHAIN IMAGE, not per frame-in-flight - see
        // the NOTE on FrameContext for why. Sized to
        // m_Context.Swapchain().GetImageCount() in Init() (and again in
        // RecreateSwapchainResources(), since image count can technically
        // change on recreate). VulkanSemaphore is non-copyable and has no
        // implicit move (it declares a destructor), so this has to be a
        // vector of owning pointers rather than a vector of VulkanSemaphore
        // directly - the vector needs to be resizable/reallocatable and
        // VulkanSemaphore itself can't tolerate that.
        std::vector<std::unique_ptr<VulkanSemaphore>> m_RenderFinishedSemaphores;
        uint32_t m_CurrentFrame = 0;
        bool m_FramebufferResized = false;
    };
}
