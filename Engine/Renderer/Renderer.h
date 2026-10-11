#pragma once

#include "Vulkan/VulkanContext.h"
#include "Vulkan/Pipeline/VulkanRenderPass.h"
#include "Vulkan/Pipeline/VulkanPipelineLayout.h"
#include "Vulkan/Pipeline/VulkanGraphicsPipeline.h"
#include "Vulkan/Pipeline/VulkanComputePipeline.h"
#include "Vulkan/Pipeline/VulkanDescriptorSetLayout.h"
#include "Vulkan/Pipeline/VulkanDescriptorPool.h"
#include "Vulkan/Pipeline/VulkanTextureSetLayout.h"
#include "Vulkan/Command/VulkanCommandPool.h"
#include "Vulkan/Frame/FrameContext.h"
#include "Vulkan/Resources/VulkanMemoryAllocator.h"
#include "Vulkan/Resources/VulkanImage.h"
#include "Vulkan/Resources/VulkanTexture.h"
#include "Vulkan/Resources/VulkanSampler.h"
#include "Vulkan/Resources/Mesh.h"
#include "Camera.h"
#include "Frustum.h"
#include "SoftwareOcclusionBuffer.h"
#include "Raymarch/RaymarchTypes.h"

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
        //
        // recordComputeWork, if provided, is invoked with this frame's
        // command buffer right after it enters the recording state and
        // BEFORE vkCmdBeginRenderPass - compute dispatches can't happen
        // inside a render pass, and this is the one point in DrawFrame
        // where the command buffer is open but nothing else has been
        // recorded into it yet. Built for ParticleSystemGPU::
        // RecordPendingSteps (pass `[&](VkCommandBuffer cmd){
        // particleSystem.RecordPendingSteps(cmd); }`), but not coupled
        // to that class specifically - any compute work a caller wants
        // folded into this frame's single submission can go through
        // here. If provided, DrawFrame inserts one memory barrier
        // (compute-shader-write -> vertex-shader-read) right after the
        // callback returns, before the render pass begins - see
        // particleGPUCount below for why vertex-shader-read is the
        // relevant destination stage.
        //
        // particleGPUCount, if nonzero, draws that many points using
        // RegisterParticleGPUSource's storage buffer directly (see that
        // method) - a second, independent particle path alongside the
        // ECS/CPU-particle drawList above, not a replacement for it.
        // voxelSources, if non-empty, draws each entry via
        // vkCmdDrawIndirect against m_GraphicsPipeline (same pipeline as
        // the ordinary drawList mesh loop - see VoxelDrawSource's own
        // comment in RendererTypes.h for why no dedicated pipeline is
        // needed). Entirely separate draw path from drawList/
        // particleGPUCount, same "own section, own loop" treatment
        // particleGPUCount already gets below.
        // raymarchObjects, if non-empty, sphere-traces each entry as a
        // fullscreen pass AFTER the ordinary mesh/voxel draws above
        // (same render pass, same depth buffer - see
        // m_RaymarchPipeline's comment for why depth write stays on).
        // raymarchDensityBuffer is the ONE GPU buffer every object in
        // raymarchObjects samples from - each object finds its own
        // region within it via RaymarchObjectGPU::densityOffset (see
        // RaymarchSystem::BuildObjectList, which builds both, and
        // VoxelSystemGPU::m_SharedDensityBuffer's comment for why every
        // volume now shares one buffer instead of each having its own).
        // A single VkBuffer, not a per-object vector, since that's the
        // whole point of this design - see RaymarchTypes.h's
        // kRaymarchMaxObjects comment for why an earlier per-object-
        // buffer version hit MoltenVK's descriptor-count ceiling.
        // deltaTime: wall-clock seconds since the PREVIOUS call to
        // DrawFrame (main.cpp already computes this for camera/input
        // handling - see that file's main loop). NOT currently read by
        // this function - an earlier version of the fluid temporal pass
        // used it to approximate each particle's previous position
        // (currentPosition - velocity * deltaTime), but that was replaced
        // by an exact previous-position buffer snapshot instead (see
        // m_ParticlePreviousPositions' comment for why the approximation
        // was wrong: ParticleSystemGPU steps on a fixed-timestep
        // accumulator, so render-frame deltaTime doesn't reliably match
        // how far the simulation actually advanced). Left as a parameter
        // rather than removed since main.cpp already computes and passes
        // it, and a future per-frame rendering feature may still want it;
        // defaults to 0 so other callers aren't forced to supply a real
        // value for a parameter nothing currently reads.
        void DrawFrame(const std::vector<DrawCommand>& drawList, const std::function<void()>& buildUI = {},
                        const std::function<void(VkCommandBuffer)>& recordComputeWork = {},
                        uint32_t particleGPUCount = 0,
                        const std::vector<VoxelDrawSource>& voxelSources = {},
                        const std::vector<RaymarchObjectGPU>& raymarchObjects = {},
                        VkBuffer raymarchDensityBuffer = VK_NULL_HANDLE,
                        float deltaTime = 0.0f);

        // One-time registration (call once after the source's position
        // buffer is created and stable - see ParticleSystemGPU::Init/
        // GetPositionBuffer) of a GPU-resident particle source to draw
        // directly from, with zero CPU readback: builds the dedicated
        // pipeline/layout/descriptor set that let particle_point_gpu.vert
        // read `positionBuffer` straight from the vertex shader, indexed
        // by gl_InstanceIndex - see that shader's file comment. Safe to
        // call only once; the position buffer must not be recreated
        // afterward (it's baked into a descriptor set here, which isn't
        // automatically kept in sync with a buffer handle changing).
        void RegisterParticleGPUSource(VkBuffer positionBuffer, VkBuffer colorBuffer);

        // Raw handle accessors - needed by ParticleSystemGPU::Init (which
        // owns its own Vulkan resources, outside Renderer's normal mesh/
        // texture registries, so it needs to allocate against the same
        // device/allocator/command pool Renderer itself uses rather than
        // Renderer owning it directly). Not meant for general use -
        // reach for CreateMesh/CreateTexture/etc. above for anything
        // that fits Renderer's existing resource-registry pattern.
        VkDevice GetDevice() { return m_Context.Device().Get(); }
        VkPhysicalDevice GetPhysicalDevice() { return m_Context.PhysicalDevice().Get(); }
        VmaAllocator GetAllocator() const { return m_Allocator.Get(); }
        VkCommandPool GetCommandPool() const { return m_CommandPool.Get(); }
        VkQueue GetGraphicsQueue() { return m_Context.Device().GetGraphicsQueue(); }

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
        float ParticlePointSize = 16.0f;

        // Flat debug-viz color for GPU-resident particles (see
        // RegisterParticleGPUSource / DrawFrame's particleGPUCount) -
        // same default as ParticleSystem::BuildDrawList's colorOverride
        // for the CPU path. Not a per-call parameter the way the CPU
        // path's is, since DrawFrame's particleGPUCount is just a count,
        // not a list of per-particle draw data - public and mutable for
        // the same live-tweak-from-UI reasoning as ParticlePointSize.
        glm::vec4 ParticleGPUColor{ 0.2f, 0.5f, 1.0f, 1.0f };

        // --- Fluid surface rendering (see InitFluidSurfacePass) --------
        // When true, DrawFrame replaces the raw round-point particle draw
        // above (particleGPUCount/m_ParticlePointsGPUPipeline) with a
        // 3-pass screen-space reconstruction that reads from the exact
        // same GPU-resident position buffer: particles are splatted as
        // sphere impostors into an offscreen depth texture
        // (fluid_depth.vert/.frag), bilaterally blurred twice
        // (fluid_blur.frag - see that shader for why this fuses separate
        // particles into one surface instead of just softening the raw
        // dots), then composited into the main render pass with a normal
        // reconstructed from the blurred depth (fluid_composite.frag).
        // Purely a rendering technique - doesn't touch ParticleSystemGPU
        // or particle physics at all. Same live-tweak-from-UI reasoning
        // as ParticlePointSize for why these are public fields.
        bool FluidSurfaceEnabled = true;
        // When FluidSurfaceEnabled is true, this picks WHICH liquid-like
        // technique actually draws: true (default) uses soft alpha-
        // blended billboards (fluid_blob.vert/.frag) - each particle is
        // its own translucent sphere with a soft radial falloff instead
        // of a hard edge, and overlapping particles read as more solid
        // simply because their alpha stacks, which is also a cheap,
        // free approximation of "thicker liquid looks more opaque."
        // False uses the screen-space reconstruction described above
        // (depth splat -> blur -> polish -> temporal -> composite).
        // Soft blobs was briefly the default - it has no history buffer, no
        // reprojection, no blur, nothing that depends on a previous
        // frame or a neighboring pixel at all, so the hatching/vibration/
        // ghosting/streaking failure modes that technique went through
        // cannot occur here structurally, not just "were fixed." The
        // screen-space path is kept available (not deleted) since it can
        // still look more like one continuous liquid surface at its best
        // - this is a real visual tradeoff, not a strict downgrade.
        bool FluidUseSoftBlobs = false;
        // --- Continuous-surface transparency (see fluid_thickness.frag /
        // fluid_composite.frag) - only used by the screen-space path. The
        // liquid's opacity comes from how much liquid a view ray passes
        // through: alpha = 1 - exp(-FluidAbsorption * thickness). Thin
        // splashes/edges/droplets read see-through, deep pools read solid.
        // All live-tunable from the UI.
        float FluidAbsorption = 0.87f; // higher = reads solid sooner
        float FluidMinAlpha = 0.41f;   // opacity floor for thin liquid so lone droplets stay visible (lower = glassier droplets)
        float FluidMaxAlpha = 1.0f;    // opacity ceiling
        // Temporal accumulation (fluid_temporal.frag) for the screen-space
        // path. ON in this build (with the reworked shader and the barrier fixes);
        // set false to drop back to no temporal. It was the source of the "particle outlines
        // stuck on screen for a few frames" ghosting (see that shader's
        // comments). When off, the passes, the previous-position copy and
        // the history buffers' per-frame work are all skipped and the
        // composite reads the polish pass's output directly. The shader now
        // has neighborhood clamping/motion-adaptive weighting so it can be
        // re-tried, but it has not been verified visually - flip this on to
        // A/B it, don't assume it's fixed.
        bool FluidTemporalEnabled = true;

        // Resolution of the fluid surface passes (depth, blur, polish,
        // temporal, normals) relative to the window: 1.0 = full, 0.75, 0.5.
        // These passes are the main GPU cost when liquid fills the screen,
        // and their cost scales with pixel count, so 0.5 is ~4x cheaper.
        // The final composite still draws at full resolution; silhouettes
        // get slightly stair-stepped at lower scales. Changing it rebuilds
        // the fluid targets at the start of the next frame (brief hitch -
        // use the discrete UI choices, not a continuously dragged slider).
        float FluidResolutionScale = 1.0f;
        // Auto mode (default on): when the frame is GPU-bound with liquid on
        // screen for about a second, step FluidResolutionScale down one level
        // (1.0 -> 0.75 -> 0.5 -> 0.35) so slow GPUs get playable frame rates
        // without anyone having to find the setting. It only ever steps DOWN
        // (stepping up would need a trial-and-error loop with a rebuild hitch
        // each time); choose a fixed level in the UI to take control back.
        // Each step rebuilds the fluid targets once (a brief hitch).
        bool FluidAutoResolution = true;

        // --- Profiling (see Engine/Core/Profiler.h) -------------------------
        // Per-pass GPU timings via timestamp queries. OFF by default: when
        // off, nothing is recorded or read back, so this costs nothing.
        // Results land in Profiler::Get() and the Performance panel.
        // Approximate on Apple GPUs (MoltenVK) - see Profiler.h.
        bool GpuTimingEnabled = false;

        // Debug draw toggles for isolating where scene GPU time goes (Performance
        // panel). All default ON; turning one off just skips that draw block.
        bool DebugDrawMeshes = true;         // ordinary instanced meshes
        bool DebugDrawVoxelGeometry = true;  // marching-cubes terrain + voxel objects
        bool DebugDrawRaymarch = true;       // raymarched SDF objects (fullscreen pass)

        // Work out where on screen the VISIBLE liquid is (GPU, read back a couple
        // of frames late) and confine the fluid passes to that rectangle - or
        // skip them entirely when none is visible. Without it every fluid pass
        // touches every screen pixel whenever any particle exists. Costs one
        // tiny compute dispatch per frame. Turn off to compare / if liquid is
        // ever clipped at a hard edge while turning fast.
        bool FluidBoundsCullingEnabled = true;
        bool GpuTimingSupported() const { return m_GpuTimingSupported; }
        // World-space visual sphere radius per particle - independent of
        // ParticleSystemGPU::boundaryRadius (the hard collision radius)
        // and smoothingRadius (the SPH kernel radius); this only controls
        // how big particles are drawn, same relationship ParticlePointSize
        // already has to the raw-point path. Larger values overlap more
        // (reads as more cohesive/continuous) at the cost of the fluid
        // visually sitting a bit "puffier" than its actual simulated
        // volume - a rendering compromise, not a physics one.
        float FluidParticleRadius = 0.12f;
        // FluidTintColor removed - the Chemical ID base build replaced
        // the old single uniform tint with real per-particle diffusing
        // color (see ParticleSystemGPU::ChemicalTable and
        // fluid_composite.frag's surfaceColor). Tune color via
        // ChemicalTable[id].baseColor now, not here.

        // CPU software occlusion buffer, cleared and repopulated fresh
        // every frame by RenderSystem::BuildDrawList - see
        // SoftwareOcclusionBuffer.h for the technique. Owned by Renderer
        // for the same reason Camera/Frustum are: it's part of "what's
        // actually visible this frame" state, which is a rendering
        // concern independent of any single call site. Resized once in
        // Init() and reused every frame after that (no per-frame
        // reallocation - only Clear()'d).
        //
        // COARSE, AABB-BASED - a real tradeoff worth knowing, not a
        // hidden limitation: occluders are tested/rasterized as their
        // world-space bounding boxes, not their exact triangle
        // silhouettes. For an axis-aligned, box-shaped occluder this is
        // exact; for a ROTATED or non-box-shaped one, the AABB can be
        // noticeably larger than what's actually opaque on screen,
        // which means something genuinely visible through a gap near
        // such an occluder could, in principle, get wrongly culled.
        // Standard, accepted tradeoff for this class of technique - the
        // fix, if it's ever visibly wrong, is either restricting what
        // qualifies as an occluder (see MinOccluderFootprintCells) or a
        // real triangle-rasterization version, not a quick patch here.
        SoftwareOcclusionBuffer& GetOcclusionBuffer() { return m_OcclusionBuffer; }

        // Master on/off switch - RenderSystem::BuildDrawList checks this
        // and skips the whole occlusion pass (falling back to frustum-
        // culling-only) when false. Useful for A/B-testing whether
        // occlusion culling is actually helping in a given scene, or for
        // diagnosing a suspected false-cull (see GetOcclusionBuffer's
        // comment on the AABB tradeoff) by turning it off and confirming
        // the "missing" object reappears.
        bool EnableOcclusionCulling = true;

        // Minimum screen-space footprint (in occlusion-buffer grid
        // cells) an entity's bounding box needs to cover before it's
        // considered worth rasterizing as an occluder for OTHER entities
        // to be tested against. Small/distant objects are skipped as
        // occluder sources - not because they can't occlude anything in
        // principle, but because their contribution is negligible and
        // testing+rasterizing every tiny far-away object for little
        // benefit defeats the point of a performance optimization. Every
        // entity still gets TESTED against the buffer regardless of its
        // own size - this only controls who gets to contribute depth,
        // not who gets checked against it.
        int MinOccluderFootprintCells = 9; // roughly a 3x3 cell area

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
        // Frames that draw the fluid surface split the main pass in two so the
        // fluid passes can read the opaque scene's depth (see DrawFrame):
        // m_SceneSplitRenderPass (clear, store depth readable) then the fluid
        // passes then m_OverlayRenderPass (load, composite + UI). Same
        // attachments as m_RenderPass, so pipelines/framebuffers are shared.
        VulkanRenderPass m_SceneSplitRenderPass;
        VulkanRenderPass m_OverlayRenderPass;
        // Samples m_DepthImage (the opaque scene's depth) for the fluid
        // depth/thickness passes. Rebuilt with the fluid resources (the depth
        // image is recreated on resize).
        VkDescriptorSet m_SceneDepthSet = VK_NULL_HANDLE;
        VulkanDescriptorSetLayout m_DescriptorSetLayout;
        VulkanTextureSetLayout m_TextureSetLayout;
        VulkanDescriptorPool m_DescriptorPool;
        VulkanPipelineLayout m_PipelineLayout;
        VulkanGraphicsPipeline m_GraphicsPipeline;
        // Same shaders/layout/vertex-input as m_GraphicsPipeline, only
        // difference is cullMode=NONE instead of BACK_BIT - used
        // exclusively for voxel/marching-cubes meshes (see
        // VoxelSystemGPU's draw path in DrawFrame), so that a rare
        // marching-cubes topology gap (see VoxelSystemGPU.cpp's notes on
        // the open ambiguous-face crack issue) shows the mesh's own far
        // interior wall through the gap instead of the skybox straight
        // through to nothing - a cheap visual stopgap, not a fix for the
        // gap itself. Deliberately its OWN pipeline object rather than
        // flipping m_GraphicsPipeline's cullMode globally, so ordinary
        // (non-voxel) meshes keep normal backface culling and its
        // performance benefit untouched. Extra cost versus a culled
        // pipeline is occasional back-face overdraw on voxel/terrain
        // meshes specifically (mostly absorbed by early depth testing
        // on solid closed geometry) - NOT a second draw call or a
        // second pass over the scene.
        VulkanGraphicsPipeline m_VoxelPipeline;
        // Second pipeline, same VkPipelineLayout/render pass/vertex input
        // description as m_GraphicsPipeline (see CreateParticlePointResources
        // in Renderer.cpp) but VK_PRIMITIVE_TOPOLOGY_POINT_LIST topology and
        // particle_point.vert/.frag shader modules instead of triangle's -
        // DrawFrame binds whichever of the two is appropriate right before
        // each mesh's draw call (see the per-mesh loop there).
        VulkanGraphicsPipeline m_ParticlePointsPipeline;
        MeshHandle m_ParticlePointMeshHandle = InvalidMeshHandle;

        // --- GPU-resident particle rendering (see RegisterParticleGPUSource) ---
        // Separate pipeline/layout/descriptor-set trio from
        // m_ParticlePointsPipeline above: that one draws the CPU
        // ParticleSystem's positions via the normal instanced-draw path
        // (InstanceData per particle, built by ParticleSystem::
        // BuildDrawList). This one draws directly from a compute-owned
        // storage buffer with NO per-instance vertex data at all - see
        // particle_point_gpu.vert. Only valid (non-VK_NULL_HANDLE) after
        // RegisterParticleGPUSource has been called.
        VkDescriptorSetLayout m_ParticleGPUSetLayout = VK_NULL_HANDLE;
        VulkanPipelineLayout m_ParticleGPUPipelineLayout;
        VulkanGraphicsPipeline m_ParticlePointsGPUPipeline;
        VulkanDescriptorPool m_ParticleGPUDescriptorPool;
        VkDescriptorSet m_ParticleGPUStorageSet = VK_NULL_HANDLE;

        // Renderer-owned snapshot of the particle position buffer AS OF
        // THE END OF THE PREVIOUS RENDERED FRAME - what
        // fluid_depth.vert/.frag actually reproject against (binding 2
        // of m_ParticleGPUSetLayout). NOT a velocity*deltaTime estimate -
        // that was tried first and was the actual bug behind the
        // lingering-outline artifact: ParticleSystemGPU steps on a FIXED
        // 60Hz accumulator (see FIXED_TIMESTEP/MAX_PHYSICS_STEPS_PER_FRAME,
        // main.cpp), 0 to 5 times per RENDERED frame depending on
        // framerate - so a render frame's wall-clock deltaTime routinely
        // does not match how far the simulation actually advanced that
        // frame. On any frame where 0 physics steps ran, the true
        // previous position equals the current one exactly, but
        // velocity*deltaTime always produced a nonzero (fictional)
        // estimate anyway - precisely the kind of per-frame error that
        // would reproject an edge slightly wrong and read as a faint,
        // decaying double-exposed outline. This buffer sidesteps the
        // whole problem: it holds the ACTUAL simulated position from last
        // render, zero extrapolation involved. Updated once per DrawFrame
        // (see that function's own comment at the copy site) via a
        // straight GPU buffer copy FROM the live position buffer
        // (m_ParticleLivePositionBuffer) - not owned by ParticleSystemGPU
        // because it's a rendering-side history snapshot, not simulation
        // state.
        VulkanBuffer m_ParticlePreviousPositions;
        // Cached from RegisterParticleGPUSource's positionBuffer argument -
        // the copy source for m_ParticlePreviousPositions each frame. Not
        // owned here (ParticleSystemGPU owns the real buffer; this is
        // just the handle needed to copy FROM it).
        VkBuffer m_ParticleLivePositionBuffer = VK_NULL_HANDLE;

        // --- Raymarch pass (see Raymarch/RaymarchSystem.h) -------------
        // Fullscreen-triangle pipeline, no vertex/instance input (see
        // raymarch.vert - vertices are generated from gl_VertexIndex),
        // cullMode=NONE (a fullscreen triangle has no "back face" to
        // cull), depth test+write ON so raymarched hits correctly
        // occlude/are-occluded-by rasterized terrain drawn earlier in
        // the same render pass (see DrawFrame's raymarch section).
        // Set 0 reuses m_DescriptorSetLayout (camera UBO) exactly like
        // m_ParticleGPUPipelineLayout does. Set 1 is m_RaymarchSetLayout
        // below - one RaymarchObjectGPU array (binding 0) + ONE shared
        // density buffer (binding 1). Both bindings are ordinary,
        // fixed-count (descriptorCount=1) storage buffers - NOT
        // bindless/descriptor-indexed. An earlier version of this pass
        // tried true bindless (one descriptor per object, later an
        // unbounded array via descriptor indexing + update-after-bind)
        // specifically to get past MoltenVK's
        // maxPerStageDescriptorStorageBuffers=31 ceiling, and hit
        // confirmed dead ends on this project's actual dev hardware at
        // every step, ending with vkGetDescriptorSetLayoutSupport
        // reporting the required layout as unsupported outright,
        // independent of descriptor count (see RaymarchTypes.h's
        // kRaymarchMaxObjects comment for the full history). The actual
        // fix doesn't touch descriptor count at all: every raymarch
        // object's density data now lives in ONE shared buffer
        // (VoxelSystemGPU::m_SharedDensityBuffer), with each object
        // finding its own region within it via
        // RaymarchObjectGPU::densityOffset - so binding 1 only ever
        // needs to be ONE descriptor, however many objects exist. This
        // is portable by construction (Vulkan 1.0-guaranteed-minimum
        // storage-buffer limits only - every conformant implementation
        // supports far more than the 2 this pass needs), not just a
        // MoltenVK-specific workaround.
        VkDescriptorSetLayout m_RaymarchSetLayout = VK_NULL_HANDLE;
        VulkanPipelineLayout m_RaymarchPipelineLayout;
        VulkanGraphicsPipeline m_RaymarchPipeline;
        // Plain handle, not VulkanDescriptorPool - created/destroyed
        // directly in InitRaymarchPass/the renderer's teardown path.
        // (Was briefly a VulkanDescriptorPool, then briefly needed
        // VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT for the
        // bindless attempt mentioned above - neither is true anymore,
        // this is back to an ordinary pool, just not routed through
        // that shared wrapper since InitRaymarchPass also needs direct
        // vkAllocateDescriptorSets control.)
        VkDescriptorPool m_RaymarchDescriptorPool = VK_NULL_HANDLE;

        // One set/buffer per frame-in-flight, NOT a single shared one -
        // this used to be a single VkDescriptorSet/VulkanBuffer, which
        // was a real, confirmed bug: UpdateRaymarchDescriptors
        // vkUpdateDescriptorSets's this set's bindings every frame
        // (raymarch objects can change every frame), but with
        // MAX_FRAMES_IN_FLIGHT=2, that update could race a PREVIOUS
        // frame's command buffer that was still pending on the GPU and
        // still referenced the old contents - exactly the Vulkan
        // validation error class this caused ("VkDescriptorSet ... is
        // in use by VkCommandBuffer ... only possible with
        // VK_EXT_descriptor_indexing"), which then cascaded into
        // unrelated-looking validation failures on nearby draw calls
        // and eventually a lost device. Same fix shape as
        // FrameContext's own descriptorSet/uniformBuffer - indexed by
        // m_CurrentFrame, exactly like those.
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_RaymarchSets{};

        // Rewritten every frame from raymarchObjects (see DrawFrame) -
        // host-visible + persistently mapped, same reasoning as
        // VoxelSystemGPU::Volume::instance: small, written every frame,
        // not worth a staging round trip. Per-frame array for the same
        // reason m_RaymarchSets above is - a single shared buffer being
        // memcpy'd into while a previous frame's GPU work might still
        // be reading it is the same race, even though it doesn't trip
        // this specific validation layer check the descriptor set does.
        std::array<VulkanBuffer, MAX_FRAMES_IN_FLIGHT> m_RaymarchObjectBuffers;
        std::array<void*, MAX_FRAMES_IN_FLIGHT> m_RaymarchObjectBuffersMapped{};

        void InitRaymarchPass();
        // densityBuffer: the ONE shared buffer every live raymarch
        // object samples from this frame (see m_RaymarchSetLayout's
        // comment above) - VK_NULL_HANDLE is valid (no raymarch objects
        // this frame; DrawFrame only calls this when raymarchObjects is
        // non-empty, but this function handles the null case
        // defensively regardless).
        void UpdateRaymarchDescriptors(const std::vector<RaymarchObjectGPU>& objects,
                                        VkBuffer densityBuffer);

        // --- Fluid surface rendering (see FluidSurfaceEnabled above and
        // fluid_depth.vert/fluid_blur.frag/fluid_composite.frag) --------
        // Three offscreen-ish stages, all hand-managed raw VkRenderPass/
        // VkFramebuffer objects rather than going through VulkanRenderPass
        // (that class is hardcoded to "swapchain color + depth, one
        // subpass" - see its own class comment - which doesn't fit any of
        // these three: the depth prepass needs a COLOR attachment that's
        // R32_SFLOAT and gets SAMPLED afterward, not presented, and the
        // blur passes need no depth attachment at all).
        void InitFluidSurfacePass();

        // --- Fluid screen-space bounds (see fluid_bounds.comp) ---
        struct FluidScreenBounds
        {
            int32_t minX, minY, maxX, maxY; // screen pixels, swapchain resolution; includes the particles' projected radius
            uint32_t visibleCount;          // particles in view and not hidden by the opaque scene
            uint32_t needsFullScreen;       // a particle was too close to the camera for a trustworthy projection
            uint32_t pad0, pad1;
        };
        VkDescriptorSetLayout m_FluidBoundsSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_FluidBoundsPool = VK_NULL_HANDLE;
        VkDescriptorSet m_FluidBoundsSet = VK_NULL_HANDLE;
        VulkanPipelineLayout m_FluidBoundsPipelineLayout;
        VulkanComputePipeline m_FluidBoundsPipeline;
        VulkanBuffer m_FluidBoundsBuffer; // device-local; reset + written by the compute pass each frame
        std::array<VulkanBuffer, MAX_FRAMES_IN_FLIGHT> m_FluidBoundsReadback; // host-visible copy per frame slot
        std::array<void*, MAX_FRAMES_IN_FLIGHT> m_FluidBoundsReadbackMapped{};
        std::array<bool, MAX_FRAMES_IN_FLIGHT> m_FluidBoundsReadbackPending{}; // this slot recorded a copy that hasn't been read yet
        FluidScreenBounds m_LastFluidBounds{};
        bool m_LastFluidBoundsValid = false;   // at least one readback has arrived
        int m_FluidEmptyReadings = 0;          // consecutive readbacks with no visible liquid
        void InitFluidBounds();
        void DestroyFluidBounds();
        void WriteFluidBoundsDescriptors();
        void ReadbackFluidBounds(uint32_t slot);
        void RecordFluidBounds(VkCommandBuffer cmd, uint32_t particleCount, const glm::mat4& viewProj, VkExtent2D extent);

        // --- GPU pass timers (timestamp queries) ---
        static constexpr uint32_t kMaxGpuStamps = 16;
        std::array<VkQueryPool, MAX_FRAMES_IN_FLIGHT> m_GpuTimerPools{};
        // Label for each stamp written this slot's last use; stamp i's label
        // names the interval ENDING at stamp i. Cleared once read back.
        std::array<std::vector<const char*>, MAX_FRAMES_IN_FLIGHT> m_GpuTimerLabels;
        float m_GpuTimestampPeriodNs = 0.0f;
        bool m_GpuTimingSupported = false;
        void InitGpuTimers();
        void DestroyGpuTimers();
        void ReadbackGpuTimers(uint32_t slot);
        void GpuStamp(VkCommandBuffer cmd, const char* label);
        // Split from InitFluidSurfacePass (which only runs once, in
        // Init()) because the image/framebuffer half - unlike the
        // pipelines/descriptor-set-layouts/sampler half - is sized to the
        // swapchain extent and has to be torn down and rebuilt on every
        // resize, same as m_DepthImage/CreateDepthResources. Order
        // matters: CreateFluidSurfaceResources allocates the descriptor
        // sets that read these images, so DestroyFluidSurfaceResources
        // must run before a resize recreates them, and
        // CreateFluidSurfaceResources must run again after.
        void CreateFluidSurfaceResources();
        void DestroyFluidSurfaceResources();
        // One-time clear of both history images to the sentinel value -
        // called once right after they're (re)created (first init, and
        // again after any resize). Without this, a fresh/resized history
        // image holds undefined driver memory, and fluid_temporal.frag's
        // reject-gate has no way to tell that from a legitimate low
        // "depth" value - it could wrongly treat garbage as trustworthy
        // history and blend toward it for however many frames it takes
        // the reject gate to happen to trip. Uses a direct one-time
        // command buffer (see .cpp) rather than a render pass clear,
        // since this needs to run once at creation time, not every frame.
        void ClearFluidHistoryImages();

        VkRenderPass m_FluidDepthPrepassRenderPass = VK_NULL_HANDLE;
        VkRenderPass m_FluidBlurRenderPass = VK_NULL_HANDLE;

        // Per-frame-in-flight, same fix shape and same reason as
        // m_RaymarchSets/m_RaymarchObjectBuffers (see those fields' own
        // comments for the full mechanism and the validation-error/
        // lost-device history behind it) - these were singular for a
        // while after the raymarch fix was already established
        // elsewhere in this file, which meant frame N's depth/blur
        // passes could start clearing and writing these SAME images
        // while frame N-1's still-in-flight GPU work was still reading
        // or writing them (composite still sampling, or the depth pass
        // still Z-testing against m_FluidDepthPassDSs below). Visible
        // result: flickering/ghosting artifacts around moving particles
        // - old, not-yet-overwritten data surviving into a frame that
        // should have fully replaced it, worse the more frames'
        // rendering overlaps in time (i.e. the GPU falling behind, or a
        // wider blur kernel taking longer per frame) - not a shader-math
        // bug at all, despite looking like one.
        std::array<VkFramebuffer, MAX_FRAMES_IN_FLIGHT> m_FluidDepthFramebuffers{};
        std::array<VkFramebuffer, MAX_FRAMES_IN_FLIGHT> m_FluidBlurFramebuffersA{};
        std::array<VkFramebuffer, MAX_FRAMES_IN_FLIGHT> m_FluidBlurFramebuffersB{};
        // Target of the fixed-radius 2D "polish" pass (fluid_blur_polish.frag)
        // that now runs after the existing separable blur - see that
        // shader's own comment for why it exists and why its cost is
        // bounded, unlike the full-radius 2D attempt it replaced. Reuses
        // m_FluidBlurRenderPass (identical single-color-attachment shape),
        // just a third framebuffer target.
        std::array<VkFramebuffer, MAX_FRAMES_IN_FLIGHT> m_FluidPolishFramebuffers{};
        // Targets of the temporal accumulation pass (fluid_temporal.frag)
        // - see m_FluidHistoryImages' own comment for why there are only
        // 2 of these, not MAX_FRAMES_IN_FLIGHT. Also reuses
        // m_FluidBlurRenderPass.
        std::array<VkFramebuffer, 2> m_FluidHistoryFramebuffers{};

        // R32_SFLOAT, world-space distance from the camera to the
        // reconstructed sphere surface at each pixel - see
        // fluid_depth.frag's comment for why world-distance (not raw
        // view-space Z) is what gets stored, matching raymarch.frag's
        // existing ray-reconstruction technique that fluid_composite.frag
        // reuses. "No particle here" pixels hold whatever
        // CreateFluidSurfaceResources clears this to (see that function) -
        // every downstream shader treats anything past
        // SENTINEL_THRESHOLD as empty, never as a real depth. Per-frame
        // array - see m_FluidDepthFramebuffers' own comment just above.
        std::array<VulkanImage, MAX_FRAMES_IN_FLIGHT> m_FluidDepthImages;
        // Second output of the SAME depth prepass draw (MRT - see
        // m_FluidDepthPrepassRenderPass) - each pixel's particle's
        // screen-space motion since last frame (see fluid_depth.vert's
        // motion-vector comment). Deliberately NOT run through
        // fluid_blur.frag/fluid_blur_polish.frag - those passes blend
        // DEPTH, and different overlapping particles can be moving in
        // different directions, so blending their motion vectors together
        // would reproject toward nonsense. fluid_temporal.frag samples
        // this directly, unblurred. Per-frame array, same reason as
        // m_FluidDepthImages above.
        std::array<VulkanImage, MAX_FRAMES_IN_FLIGHT> m_FluidMotionImages;
        // LOCAL depth-test-only buffer for the depth prepass - resolves
        // which of several overlapping sphere impostors is actually
        // nearest at a given pixel (ordinary hardware depth test). Never
        // sampled by anything afterward, unlike m_FluidDepthImages above -
        // exists purely so vkCmdDraw'ing particles as unsorted point
        // sprites still occludes correctly, the same reason ANY depth
        // buffer exists. Per-frame too, despite never being read outside
        // its own pass - two overlapping frames' GPU work both WRITING
        // (Z-testing/clearing) the same shared depth attachment is still
        // a real race even when neither frame cares about the other's
        // leftover contents afterward; it can corrupt THIS frame's own
        // Z-test results, not just leak stale data into a later one.
        std::array<VulkanImage, MAX_FRAMES_IN_FLIGHT> m_FluidDepthPassDSs;
        // Ping-pong targets for the separable bilateral blur (see
        // fluid_blur.frag) - A holds the horizontal pass's output/the
        // vertical pass's input, B holds the vertical pass's (i.e. the
        // whole blur's) final output, which feeds into the fixed-radius
        // polish pass (fluid_blur_polish.frag) next. Per-frame, same
        // reason as above.
        std::array<VulkanImage, MAX_FRAMES_IN_FLIGHT> m_FluidBlurImagesA;
        std::array<VulkanImage, MAX_FRAMES_IN_FLIGHT> m_FluidBlurImagesB;
        // Output of fluid_blur_polish.frag - what the temporal pass's
        // "new frame" input reads (used to be what m_FluidCompositeSets
        // read directly; composite now reads the temporal pass's output
        // instead - see m_FluidHistoryImages below).
        std::array<VulkanImage, MAX_FRAMES_IN_FLIGHT> m_FluidPolishImages;
        // Per-pixel accumulated liquid thickness (fluid_thickness.vert/.frag),
        // R16_SFLOAT, additively blended, cleared to 0 each frame. Per frame
        // slot like every other fluid target. Sampled by the composite pass
        // for opacity. Has its own render pass (m_FluidThicknessRenderPass)
        // since it needs a CLEAR load op and a different format from
        // m_FluidBlurRenderPass.
        // Unblurred analytic sphere normals (depth pass's 3rd output, see
        // fluid_depth.frag) - sampled by the composite for isolated droplets.
        std::array<VulkanImage, MAX_FRAMES_IN_FLIGHT> m_FluidNormalImages;
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_FluidNormalSets{};
        // Thickness is rendered at 1/kFluidThicknessDivisor resolution per
        // axis and sampled with this LINEAR sampler. It is a smooth additive
        // sum so nothing is lost, and up close (where each particle quad
        // covers hundreds of pixels and overdraw explodes) it is 4x fewer
        // blended fragments.
        VulkanSampler m_FluidLinearSampler;
        static constexpr uint32_t kFluidThicknessDivisor = 2;
        // Actual fluid target size (swapchain extent * FluidResolutionScale
        // at the time CreateFluidSurfaceResources last ran) and the scale
        // that produced it, so DrawFrame can detect a change.
        VkExtent2D m_FluidExtent{};
        float m_AppliedFluidScale = 1.0f;
        int m_AutoResolutionSlowFrames = 0;     // consecutive-ish frames that looked GPU-bound with liquid on screen
        bool m_FluidChainRanLastFrame = false;  // set each frame in DrawFrame, read by the NEXT frame's auto-resolution check
        int m_AutoResolutionCooldown = 0;       // frames to wait after a step before judging again (the rebuild hitch would otherwise count against the new level)
        uint32_t m_FluidThicknessDivisor = kFluidThicknessDivisor; // 1 when the fluid targets are already scaled down
        std::array<VulkanImage, MAX_FRAMES_IN_FLIGHT> m_FluidThicknessImages;
        std::array<VkFramebuffer, MAX_FRAMES_IN_FLIGHT> m_FluidThicknessFramebuffers{};
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_FluidThicknessSets{}; // reads m_FluidThicknessImages[slot] - set 2 of the composite pass
        VkRenderPass m_FluidThicknessRenderPass = VK_NULL_HANDLE;
        VulkanPipelineLayout m_FluidThicknessPipelineLayout;
        VulkanGraphicsPipeline m_FluidThicknessPipeline;

        // The temporal accumulation pass's (fluid_temporal.frag) two
        // ping-ponged history buffers. Deliberately sized 2, NOT
        // MAX_FRAMES_IN_FLIGHT - these alternate once per RENDERED FRAME
        // (see m_FluidHistoryIndex), a completely different concern from
        // the frame-in-flight slots everything else in this feature uses
        // (which exist purely so the CPU can record frame N+1 while the
        // GPU still executes frame N, and hold no state that needs to
        // survive between frames on purpose). History here is the
        // opposite: it's SPECIFICALLY the thing that must survive from
        // one rendered frame to the next. It's a coincidence that
        // MAX_FRAMES_IN_FLIGHT is also 2 - the two numbers are not the
        // same thing and must not be assumed interchangeable if either
        // ever changes independently.
        std::array<VulkanImage, 2> m_FluidHistoryImages;
        // Which of m_FluidHistoryImages holds the most recently written
        // (i.e. currently valid/readable-as-history) result, going into
        // THIS frame. Toggled exactly once per Renderer::DrawFrame call -
        // see that function's own comment at the fluid temporal pass for
        // the read/write index math. Reset to 0 whenever the history
        // images are recreated (resize - see ClearFluidHistoryImages),
        // since their contents are invalid at that point regardless of
        // which index this was pointing at before.
        int m_FluidHistoryIndex = 0;

        // NEAREST, not LINEAR - see fluid_blur.frag's comment: this
        // texture holds either a real depth value or a huge sentinel with
        // nothing in between, and hardware bilinear filtering would blend
        // the two right at every silhouette edge before the shader's own
        // sentinel check ever sees the sample, corrupting exactly the
        // boundary the whole bilateral-weight scheme exists to preserve.
        VulkanSampler m_FluidSampler;

        // Set 1 = m_ParticleGPUSetLayout (the same one-storage-buffer
        // layout RegisterParticleGPUSource already created) - this
        // pipeline reads the identical position buffer that pipeline
        // does, just through a different pipeline/vertex shader, so it
        // reuses that layout AND that already-allocated m_ParticleGPUStorageSet
        // directly rather than standing up a second copy of the same
        // binding.
        VulkanPipelineLayout m_FluidDepthPipelineLayout;
        VulkanGraphicsPipeline m_FluidDepthPipeline;

        // Set 0 = m_TextureSetLayout (one combined image sampler) - no
        // camera UBO needed for a pure image-space blur. Same pipeline
        // object is bound for BOTH the horizontal and vertical pass
        // (see DrawFrame) - only the bound descriptor set and the
        // direction push constant differ between the two draw calls.
        VulkanPipelineLayout m_FluidBlurPipelineLayout;
        VulkanGraphicsPipeline m_FluidBlurPipeline;

        // Set 0 = m_TextureSetLayout, same shape as m_FluidBlurPipelineLayout
        // - the fixed-radius 2D polish pass (fluid_blur_polish.frag) needs
        // no direction push constant (it isn't separable), just texelSize.
        VulkanPipelineLayout m_FluidPolishPipelineLayout;
        VulkanGraphicsPipeline m_FluidPolishPipeline;

        // Set 0 = m_TextureSetLayout reused at TWO slots (new frame,
        // history - see fluid_temporal.frag's two sampler bindings). No
        // push constants - both inputs are sampled at the same
        // fragScreenUV, no offset math needed.
        VulkanPipelineLayout m_FluidTemporalPipelineLayout;
        VulkanGraphicsPipeline m_FluidTemporalPipeline;

        // Set 0 = m_DescriptorSetLayout (camera UBO, reused exactly like
        // m_RaymarchPipelineLayout does), set 1 = m_TextureSetLayout
        // (the final blurred depth texture).
        VulkanPipelineLayout m_FluidCompositePipelineLayout;
        VulkanGraphicsPipeline m_FluidCompositePipeline;

        // --- Soft-billboard liquid rendering (see FluidUseSoftBlobs) ------
        // Draws DIRECTLY into the main scene render pass (m_RenderPass),
        // same place m_FluidCompositePipeline draws, mutually exclusive
        // with it - this technique has no offscreen passes at all, so
        // there's nothing else to initialize. Set 0 = m_DescriptorSetLayout
        // (camera UBO), set 1 = m_ParticleGPUSetLayout (reused from the
        // screen-space technique's own particle storage set - fluid_blob.vert
        // only reads bindings 0/1 of it, the unused PreviousPositionBuffer
        // binding is harmless). depthTestEnable=true/depthWriteEnable=false
        // (test against opaque geometry so blobs sink into walls/floor
        // correctly; don't write, so overlapping translucent blobs blend
        // with each other instead of occluding), alpha blending ON (the
        // one pipeline in this renderer that needs it - see
        // VulkanGraphicsPipeline::Init's enableAlphaBlend parameter).
        VulkanPipelineLayout m_FluidBlobPipelineLayout;
        VulkanGraphicsPipeline m_FluidBlobPipeline;

        // Dedicated tiny pool, same "own pool, sized for exactly what
        // this feature needs" reasoning as m_ParticleGPUDescriptorPool/
        // m_RaymarchDescriptorPool - 3 sets PER FRAME IN FLIGHT (see
        // m_FluidBlurXSets etc. below), one combined-image-sampler
        // descriptor each. Reallocated in CreateFluidSurfaceResources
        // every time the underlying images are recreated (resize), since
        // a VkDescriptorSet reads a specific VkImageView, not an image
        // "slot" that keeps working after the view it points to is
        // destroyed and recreated.
        VkDescriptorPool m_FluidDescriptorPool = VK_NULL_HANDLE;
        // Per-frame, same reason as m_FluidDepthFramebuffers etc. above -
        // each slot's set points at THAT slot's own images (written once
        // per slot in CreateFluidSurfaceResources, not rewritten every
        // frame the way m_RaymarchSets is, since which image view a slot
        // points at never changes after creation - only the raymarch
        // case needs a per-frame REWRITE, because raymarch's object data
        // itself changes frame to frame).
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_FluidBlurXSets{};       // reads m_FluidDepthImages[slot]
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_FluidBlurYSets{};       // reads m_FluidBlurImagesA[slot]
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_FluidPolishSets{};      // reads m_FluidBlurImagesB[slot]
        // Temporal pass inputs - two SEPARATE descriptor sets bound at
        // once (set 0 and set 1, see m_FluidTemporalPipelineLayout).
        // "New" is per-frame-in-flight-slot (same reason as every other
        // *Sets array above - it points at m_FluidPolishImages[slot]).
        // "History" is sized 2, NOT MAX_FRAMES_IN_FLIGHT - same reasoning
        // as m_FluidHistoryImages itself; these are the two STATIC sets
        // pointing at m_FluidHistoryImages[0] and [1] respectively,
        // created once and picked from by index at draw time rather than
        // rewritten per frame.
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_FluidTemporalNewSets{}; // reads m_FluidPolishImages[slot]
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_FluidMotionSets{};      // reads m_FluidMotionImages[slot] - set 2 of the temporal pass, see m_FluidTemporalPipelineLayout
        std::array<VkDescriptorSet, 2> m_FluidTemporalHistorySets{};                // [i] reads m_FluidHistoryImages[i]
        // Composite now reads the TEMPORAL pass's output
        // (m_FluidHistoryImages[m_FluidHistoryIndex], this frame's
        // freshly-blended result) rather than the polish pass's output
        // directly. Sized 2 for the same reason as m_FluidTemporalHistorySets
        // - picked by m_FluidHistoryIndex at draw time, not per frame-in-
        // flight slot.
        std::array<VkDescriptorSet, 2> m_FluidCompositeSets{};                      // [i] reads m_FluidHistoryImages[i]

        // Guards RecreateSwapchainResources/DrawFrame's fluid-surface
        // branches - InitFluidSurfacePass/CreateFluidSurfaceResources run
        // lazily from RegisterParticleGPUSource (see that function's
        // comment for why: the depth prepass pipeline's set 1 IS
        // m_ParticleGPUSetLayout, which doesn't exist until that call
        // happens), not from Init() like everything else in this file -
        // so a resize or a frame that lands before RegisterParticleGPUSource
        // has ever been called must skip fluid-surface work entirely
        // rather than touching still-null render passes/pipelines.
        bool m_FluidSurfaceInitialized = false;

        SoftwareOcclusionBuffer m_OcclusionBuffer;
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
        std::vector<RaymarchObjectGPU> m_VisibleRaymarchScratch; // this frame's frustum-culled raymarch objects (rebuilt every frame; kept to avoid reallocating)

        Camera m_Camera;
        // See UniformBufferObject::prevViewProj's comment - "how the
        // camera looked at the end of the previous DrawFrame call,"
        // updated at the very end of DrawFrame, read at the start of the
        // next one. Identity on the first frame (before any DrawFrame
        // call has run) - harmless, since fluid_temporal.frag's
        // reject-gate sees no valid history on the first frame regardless
        // (see ClearFluidHistoryImages), so a meaningless motion vector
        // that frame has nothing to reproject against yet.
        glm::mat4 m_PreviousViewProj = glm::mat4(1.0f);

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
