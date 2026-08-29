#version 450

// Fullscreen triangle, no vertex/instance buffers bound - see
// Renderer::m_RaymarchPipeline's comment for why this pairs with an
// empty bindingDescriptions/attributeDescriptions Init() call. Standard
// "3 vertices covering the whole screen" trick: at gl_VertexIndex 0/1/2,
// clipPos lands at (-1,-1), (3,-1), (-1,3) - a triangle twice the size
// of the screen on two sides, whose visible portion exactly covers the
// viewport with no seam down the diagonal a two-triangle quad would have.
layout(location = 0) out vec2 fragScreenUV;

void main()
{
    vec2 clipPos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2) * 2.0 - 1.0;
    gl_Position = vec4(clipPos, 0.0, 1.0);

    // 0..1 UV, Y-down (matches Vulkan's framebuffer convention) - the
    // fragment shader uses this to reconstruct a world-space camera ray
    // per pixel, not to sample a texture.
    fragScreenUV = clipPos * 0.5 + 0.5;
}
