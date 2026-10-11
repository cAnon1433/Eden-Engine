#version 450

// Soft-billboard liquid rendering - the alternative to the screen-space
// depth/blur/polish/temporal pipeline (fluid_depth.vert and everything
// downstream of it). See Renderer::FluidUseSoftBlobs for the full
// reasoning on why this exists alongside that technique rather than
// replacing it outright.
//
// Each particle draws as its own translucent, alpha-blended billboard
// with a soft radial falloff (fluid_blob.frag) - no offscreen passes, no
// history, no reprojection, nothing that depends on a neighboring pixel
// or a previous frame. Overlapping particles read as more solid purely
// because alpha blending naturally stacks - dense/thick regions
// accumulate more opacity, sparse regions stay closer to transparent,
// which is a free, cheap approximation of "thicker liquid looks more
// opaque" with no separate thickness pass required.
//
// Billboard setup (camera-facing quad, Gram-Schmidt per-particle basis)
// is the EXACT same approach fluid_depth.vert uses and for the same
// reason - see that shader's own comment on why a per-particle basis
// (not the screen-wide camera.cameraRight/cameraUp) is what actually
// keeps a billboard's silhouette correct as it moves off dead-center,
// and why that specifically matters for avoiding the "swimming" artifact
// this project already fixed once for the other technique. No reason to
// re-derive or re-learn that here - it's copied, not reinvented.

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
    mat4 prevViewProj; // unused here - this technique has no previous-frame dependency at all
} camera;

layout(std430, set = 1, binding = 0) readonly buffer PositionBuffer
{
    vec4 positions[];
};

layout(std430, set = 1, binding = 1) readonly buffer ColorBuffer
{
    vec4 colors[];
};

layout(push_constant) uniform FluidBlobPushConstants
{
    float radius;
} pc;

layout(location = 0) out vec3 outWorldCenter;
layout(location = 1) out float outRadius;
layout(location = 2) out vec2 outLocalCoord;
layout(location = 3) out vec3 outColor;
layout(location = 4) out vec3 outRight; // this particle's own billboard basis - fluid_blob.frag needs it for the analytic sphere normal, cheaper to pass through than recompute from worldCenter twice
layout(location = 5) out vec3 outUp;
layout(location = 6) out vec3 outViewDir;

const vec2 CORNERS[6] = vec2[6](
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
    vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0)
);

void main()
{
    vec3 worldPos = positions[gl_InstanceIndex].xyz;

    // Same void-kill sentinel check as particle_point_gpu.vert/
    // fluid_depth.vert - a particle parked far below the world
    // (SimParamsGPU::voidKillY) must stay excluded here too.
    if (worldPos.y < -1.0e8)
    {
        gl_Position = vec4(1e9, 1e9, 1e9, 1.0);
        outWorldCenter = worldPos;
        outRadius = 0.0;
        outLocalCoord = vec2(0.0);
        outColor = vec3(1.0);
        outRight = vec3(1.0, 0.0, 0.0);
        outUp = vec3(0.0, 1.0, 0.0);
        outViewDir = vec3(0.0, 0.0, 1.0);
        return;
    }

    vec2 corner = CORNERS[gl_VertexIndex];

    // See this file's top comment - identical basis construction to
    // fluid_depth.vert, same reasoning.
    vec3 viewDir = normalize(worldPos - camera.cameraPosition);
    vec3 right = normalize(camera.cameraRight - viewDir * dot(camera.cameraRight, viewDir));
    vec3 up = cross(right, viewDir);

    vec3 cornerWorldPos = worldPos
        + right * (corner.x * pc.radius)
        + up * (corner.y * pc.radius);

    gl_Position = camera.proj * camera.view * vec4(cornerWorldPos, 1.0);
    outWorldCenter = worldPos;
    outRadius = pc.radius;
    outLocalCoord = corner;
    outColor = colors[gl_InstanceIndex].rgb;
    outRight = right;
    outUp = up;
    outViewDir = viewDir;
}
