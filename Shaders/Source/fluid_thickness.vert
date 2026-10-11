#version 450

// Thickness pass (vertex half) - gives the screen-space fluid surface its
// TRANSPARENCY. See fluid_thickness.frag for the overview.
//
// Same billboard quad construction as fluid_depth.vert (per-particle
// Gram-Schmidt basis, 6 vertices per instance, no vertex buffer), minus
// everything this pass doesn't need: no color, no motion.

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
    mat4 prevViewProj; // unused here
} camera;

layout(std430, set = 1, binding = 0) readonly buffer PositionBuffer
{
    vec4 positions[];
};

layout(push_constant) uniform FluidThicknessPushConstants
{
    float radius;
    float invTargetW; // fragment stage
    float invTargetH; // fragment stage
    float window;     // fragment stage
} pc;

layout(location = 0) out vec2 outLocalCoord;
layout(location = 1) out float outRadius;
// Per-particle, flat: distance of the particle CENTER from the camera
// measured along camera.cameraForward (same "linear depth" convention
// fluid_depth.frag stores), and cos(angle between this particle's view ray
// and cameraForward) - converts chord lengths between "along the ray" and
// "along the forward axis".
layout(location = 2) flat out vec2 outCenterDepthAndCos;

const vec2 CORNERS[6] = vec2[6](
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
    vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0)
);

void main()
{
    vec3 worldPos = positions[gl_InstanceIndex].xyz;

    // Void-kill sentinel - same check every other particle shader uses.
    if (worldPos.y < -1.0e8)
    {
        gl_Position = vec4(1e9, 1e9, 1e9, 1.0);
        outLocalCoord = vec2(0.0);
        outRadius = 0.0;
        outCenterDepthAndCos = vec2(0.0, 1.0);
        return;
    }

    vec2 corner = CORNERS[gl_VertexIndex];

    vec3 viewDir = normalize(worldPos - camera.cameraPosition);
    vec3 right = normalize(camera.cameraRight - viewDir * dot(camera.cameraRight, viewDir));
    vec3 up = cross(right, viewDir);

    vec3 cornerWorldPos = worldPos + right * (corner.x * pc.radius) + up * (corner.y * pc.radius);

    gl_Position = camera.proj * camera.view * vec4(cornerWorldPos, 1.0);
    outLocalCoord = corner;
    outRadius = pc.radius;
    outCenterDepthAndCos = vec2(dot(worldPos - camera.cameraPosition, camera.cameraForward),
                                max(dot(viewDir, camera.cameraForward), 0.05));
}
