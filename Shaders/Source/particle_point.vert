#version 450

layout(binding = 0) uniform CameraUBO
{
    mat4 view;
    mat4 proj;
    vec3 cameraPosition;
    vec3 lightDirection;
    vec3 lightColor;
    vec3 ambientColor;
    float particlePointSize;
} camera;

// inPosition is always (0,0,0) for the dedicated single-vertex point mesh
// (see Renderer's particle-point mesh setup) - kept as a real vertex
// attribute (rather than hardcoding vec3(0) in-shader) purely so this
// shader's vertex input layout matches triangle.vert's exactly, letting
// both pipelines share ONE VkPipelineVertexInputStateCreateInfo
// description instead of needing a second one just for this.
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

// Per-instance attributes - identical meaning and layout to triangle.vert,
// see its own comment on why a mat4 needs four consecutive locations.
layout(location = 2) in vec4 inModelCol0;
layout(location = 3) in vec4 inModelCol1;
layout(location = 4) in vec4 inModelCol2;
layout(location = 5) in vec4 inModelCol3;
layout(location = 6) in vec4 inColorOverride;

// Unused by this shader (no texturing or lighting on point particles -
// see this file's fragment shader) - declared anyway, same reason as
// inPosition above: keeps the vertex input description identical to
// triangle.vert's so both pipelines can share it.
layout(location = 7) in vec2 inTexCoord;
layout(location = 8) in vec3 inNormal;

layout(location = 0) out vec3 fragColor;

void main()
{
    mat4 model = mat4(inModelCol0, inModelCol1, inModelCol2, inModelCol3);

    // inPosition is always the origin (see comment above), so scale
    // baked into `model` has NO effect on gl_Position here - scaling the
    // origin is still the origin. Only translation/rotation move a
    // point. gl_PointSize below - a SCREEN-SPACE pixel size, not a
    // world-space one - is what actually controls how big each particle
    // appears; this is a genuinely different mechanism from the old
    // cube-mesh path's world-space visualScale, not just a renamed
    // version of it.
    vec4 worldPosition = model * vec4(inPosition, 1.0);
    gl_Position = camera.proj * camera.view * worldPosition;
    gl_PointSize = camera.particlePointSize;

    // Same hard-switch mix() as triangle.vert - inColorOverride.a is
    // always exactly 0.0 or 1.0, set from ColorComponent presence
    // upstream (or, for particles, always 1.0 - see
    // ParticleSystem::BuildDrawList).
    fragColor = mix(inColor, inColorOverride.rgb, inColorOverride.a);
}
