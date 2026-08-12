#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <array>

namespace Eden
{
    // Shared enums/structs used across the renderer.

    struct Vertex
    {
        glm::vec3 position;
        glm::vec3 color;
        glm::vec2 texCoord;
        glm::vec3 normal;

        static VkVertexInputBindingDescription GetBindingDescription()
        {
            VkVertexInputBindingDescription binding{};
            binding.binding = 0;
            binding.stride = sizeof(Vertex);
            binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            return binding;
        }

        // locations 7 and 8, not 2/3: locations 2-6 belong to InstanceData
        // (binding 1) - see its GetAttributeDescriptions below. Vulkan
        // doesn't require attribute locations to be contiguous, just
        // unique among whatever's actually bound, so leaving gaps here to
        // keep this struct's own locations (0, 1, 7, 8) stable regardless
        // of what InstanceData does is fine.
        static std::array<VkVertexInputAttributeDescription, 4> GetAttributeDescriptions()
        {
            std::array<VkVertexInputAttributeDescription, 4> attributes{};

            attributes[0].binding = 0;
            attributes[0].location = 0;
            attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
            attributes[0].offset = offsetof(Vertex, position);

            attributes[1].binding = 0;
            attributes[1].location = 1;
            attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
            attributes[1].offset = offsetof(Vertex, color);

            attributes[2].binding = 0;
            attributes[2].location = 7;
            attributes[2].format = VK_FORMAT_R32G32_SFLOAT;
            attributes[2].offset = offsetof(Vertex, texCoord);

            attributes[3].binding = 0;
            attributes[3].location = 8;
            attributes[3].format = VK_FORMAT_R32G32B32_SFLOAT;
            attributes[3].offset = offsetof(Vertex, normal);

            return attributes;
        }
    };

    // One per frame-in-flight, uploaded to a uniform buffer and read by the
    // vertex and fragment shaders. Camera state (view/proj/position) plus
    // a single hardcoded directional light (see Renderer::DrawFrame for
    // where the light's actual direction/color/ambient values come from -
    // there's no LightComponent yet, so this is Eden's one global light,
    // not something placed per-entity). alignas(16) matches std140/std430
    // layout rules Vulkan expects - omitting it is a classic source of
    // subtly corrupted data on some drivers, especially mixing vec3/mat4
    // like this struct does.
    struct UniformBufferObject
    {
        alignas(16) glm::mat4 view;
        alignas(16) glm::mat4 proj;
        alignas(16) glm::vec3 cameraPosition; // needed for specular - Blinn-Phong's halfway vector depends on view direction
        alignas(16) glm::vec3 lightDirection; // direction the light travels, e.g. {0.3, -1.0, 0.2} normalized = mostly downward, slightly angled
        alignas(16) glm::vec3 lightColor;
        alignas(16) glm::vec3 ambientColor;
        // Lands in ambientColor's own trailing std140 padding (a vec3
        // member is padded out to 16 bytes, leaving exactly 4 unused
        // bytes right after it) - this doesn't grow the struct's size,
        // it just uses space that was already being reserved. The GLSL
        // side (particle_point.vert's CameraUBO block, and triangle.vert/
        // .frag's for consistency even though they don't read it) must
        // declare this field in the SAME position for the two sides'
        // implicit std140 layouts to agree - if you ever reorder these
        // fields, reorder them identically on the GLSL side too.
        float particlePointSize = 6.0f;
    };

    // Opaque handle into Renderer's mesh resource registry (see
    // Renderer::CreateMesh / CreateCubeMesh). Just an index - deliberately
    // not a pointer/reference, so it stays valid across container
    // reallocation and is trivially copyable into ECS components.
    using MeshHandle = size_t;
    constexpr MeshHandle InvalidMeshHandle = static_cast<MeshHandle>(-1);

    // Opaque handle into Renderer's texture resource registry (see
    // Renderer::CreateTexture). Same reasoning as MeshHandle - an index, not
    // a pointer, stable across container growth.
    //
    // Textures are a property of a MESH, not of an individual entity -
    // when you call CreateMesh/CreateCubeMesh/CreateIndexedMesh you can
    // pass a TextureHandle to bind to that mesh permanently. This is
    // deliberately coarser-grained than ColorComponent (which IS
    // per-entity): instancing draws every entity sharing a MeshHandle in
    // one draw call, and Vulkan can only bind one descriptor set per draw
    // call, so per-entity textures within a single instanced draw would
    // need a texture array/bindless indexing scheme - real, but a bigger
    // feature than "basics" calls for right now. Per-mesh texturing covers
    // the common case (this model uses this texture) with the same
    // grouping Renderer::DrawFrame already does for meshes.
    using TextureHandle = size_t;
    constexpr TextureHandle InvalidTextureHandle = static_cast<TextureHandle>(-1);

    // Fixed capacity for the texture descriptor pool (see
    // VulkanDescriptorPool sizing in Renderer::Init) - same "documented,
    // raisable ceiling" pattern as MAX_INSTANCES_PER_FRAME. 256 loaded
    // textures is well beyond what hand-placed scenes need; raise this if
    // a real asset pipeline ever needs more.
    constexpr uint32_t MAX_TEXTURES = 256;

    // One draw: which mesh resource, at what world transform, with an
    // optional per-entity color override. Built fresh each frame by
    // Systems/RenderSystem.h from ECS Transform+Mesh(+Color+Visibility)
    // pairs, consumed by Renderer::DrawFrame.
    struct DrawCommand
    {
        MeshHandle mesh = InvalidMeshHandle;
        glm::mat4 model{ 1.0f };

        // w = 1.0 means "use colorOverride.rgb instead of vertex color",
        // w = 0.0 means "use the mesh's own vertex color". Packed as a
        // single vec4 so the vert shader can pick with mix() instead of
        // branching - see triangle.vert.
        glm::vec4 colorOverride{ 0.0f };
    };

    // What actually gets uploaded to the per-frame instance buffer and fed
    // to the vertex shader as per-instance vertex attributes (binding 1,
    // VK_VERTEX_INPUT_RATE_INSTANCE) - see GetBindingDescription/
    // GetAttributeDescriptions below and Frame/FrameContext.h.
    //
    // This replaces what used to be a push constant. Push constants only
    // carry ONE object's data per draw call, which is fine for one draw
    // call per object but a contradiction in terms for instancing (the
    // entire point is many objects, one draw call). Per-instance vertex
    // data doesn't have that ceiling - it's one buffer, read once per
    // instance by the vertex shader automatically.
    struct InstanceData
    {
        alignas(16) glm::mat4 model{ 1.0f };
        alignas(16) glm::vec4 colorOverride{ 0.0f };

        // Binding 1 - separate from Vertex's binding 0 (per-vertex geometry)
        // so the same mesh geometry can be reused across every instance
        // while this stream advances once per instance instead of once
        // per vertex.
        static VkVertexInputBindingDescription GetBindingDescription()
        {
            VkVertexInputBindingDescription binding{};
            binding.binding = 1;
            binding.stride = sizeof(InstanceData);
            binding.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
            return binding;
        }

        // A mat4 has to be split into four vec4 attributes (locations 2-5)
        // - there's no single VkFormat for a whole 4x4 matrix. colorOverride
        // follows at location 6. Locations 0-1 are Vertex's (position,
        // color); this starts at 2 to not collide with those.
        static std::array<VkVertexInputAttributeDescription, 5> GetAttributeDescriptions()
        {
            std::array<VkVertexInputAttributeDescription, 5> attributes{};

            for (uint32_t column = 0; column < 4; ++column)
            {
                attributes[column].binding = 1;
                attributes[column].location = 2 + column;
                attributes[column].format = VK_FORMAT_R32G32B32A32_SFLOAT;
                attributes[column].offset = static_cast<uint32_t>(offsetof(InstanceData, model)) + column * sizeof(glm::vec4);
            }

            attributes[4].binding = 1;
            attributes[4].location = 6;
            attributes[4].format = VK_FORMAT_R32G32B32A32_SFLOAT;
            attributes[4].offset = static_cast<uint32_t>(offsetof(InstanceData, colorOverride));

            return attributes;
        }
    };

    // Hard cap on how many instances can be drawn in a single frame - the
    // per-frame instance buffer is allocated once at this fixed size
    // (see FrameContext::Init) rather than resized dynamically. 65536 is
    // comfortably above what this project needs today; if a scene ever
    // wants more than that, this constant (and the buffer size derived
    // from it) is the place to raise it. Renderer::DrawFrame silently
    // drops instances beyond this cap rather than crashing - see the
    // warning logged there.
    constexpr size_t MAX_INSTANCES_PER_FRAME = 65536;
}
