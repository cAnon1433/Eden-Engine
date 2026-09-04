#version 450

// Pass 1 of the screen-space fluid reconstruction (see fluid_composite.frag
// for the overview and Renderer::InitFluidSurfacePass's comment for the
// full pipeline). Reads positions straight from the same GPU-resident
// storage buffer particle_point_gpu.vert already reads (see
// RegisterParticleGPUSource - this pipeline reuses that exact
// m_ParticleGPUStorageSet/m_ParticleGPUSetLayout rather than a second
// registration path), indexed by gl_InstanceIndex, same as that shader.
//
// Draws each particle as a camera-facing BILLBOARD QUAD (6 vertices,
// gl_VertexIndex-generated - no vertex buffer, same "no vertex buffer
// needed" trick raymarch.vert's fullscreen triangle already uses, just a
// 4-corner quad instead of a 3-corner fullscreen shape) rather than a
// GL_POINTS point sprite sized via gl_PointSize. This was a real bug in
// the first version of this pass: gl_PointSize is clamped by
// VkPhysicalDeviceLimits::pointSizeRange, and the computed size was
// evidently saturating that clamp across the entire visual-radius slider
// range, making every particle render at the same (wrong, oversized)
// pixel footprint regardless of the radius control - and, downstream of
// that, keeping neighboring particles' depths too far apart for
// fluid_blur.frag's bilateral weight to ever fuse them. Billboard quads
// are ordinary rasterized triangles sized through the normal perspective
// projection, exactly like every other draw call in this renderer - no
// device-specific size ceiling applies to them at all.

layout(binding = 0) uniform CameraUBO
{
    mat4 view;
    mat4 proj;
    vec3 cameraPosition;
    vec3 lightDirection;
    vec3 lightColor;
    vec3 ambientColor;
    float particlePointSize; // unused here
    vec3 cameraRight;
    vec3 cameraUp;
    vec3 cameraForward;
} camera;

layout(std430, set = 1, binding = 0) readonly buffer PositionBuffer
{
    vec4 positions[];
};

layout(push_constant) uniform FluidDepthPushConstants
{
    float radius;
} pc;

layout(location = 0) out vec3 outWorldCenter;
layout(location = 1) out float outRadius;
layout(location = 2) out vec2 outLocalCoord;

// Two triangles, CCW, quad corners in [-1,1] local billboard space -
// local.x follows camera.cameraRight, local.y follows camera.cameraUp
// (both "positive" in the intuitive sense - unlike gl_PointCoord, which
// is Y-down, this needs no sign flip when fluid_depth.frag reconstructs
// the same point).
const vec2 CORNERS[6] = vec2[6](
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
    vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0)
);

void main()
{
    vec3 worldPos = positions[gl_InstanceIndex].xyz;

    // Same void-kill sentinel check as particle_point_gpu.vert - a
    // particle parked far below the world (SimParamsGPU::voidKillY) must
    // stay excluded from the fluid surface too.
    if (worldPos.y < -1.0e8)
    {
        gl_Position = vec4(1e9, 1e9, 1e9, 1.0);
        outWorldCenter = worldPos;
        outRadius = 0.0;
        outLocalCoord = vec2(0.0);
        return;
    }

    vec2 corner = CORNERS[gl_VertexIndex];
    vec3 cornerWorldPos = worldPos
        + camera.cameraRight * (corner.x * pc.radius)
        + camera.cameraUp * (corner.y * pc.radius);

    gl_Position = camera.proj * camera.view * vec4(cornerWorldPos, 1.0);
    outWorldCenter = worldPos;
    outRadius = pc.radius;
    outLocalCoord = corner;
}
