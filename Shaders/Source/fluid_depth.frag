#version 450

// Pass 1 (fragment half) - see fluid_depth.vert's file comment for the
// overview. For each fragment inside this point sprite's circle,
// reconstructs the actual point on the particle's sphere SURFACE (not
// just its center) and outputs two things: the world-space distance from
// the camera to that surface point (this pass's whole purpose - the
// color attachment, sampled by fluid_blur.frag/fluid_composite.frag
// later), and a real per-fragment gl_FragDepth so overlapping particles
// correctly occlude each other via this pass's own LOCAL depth attachment
// (Renderer::m_FluidDepthPassDS - never sampled afterward, exists only to
// make nearest-particle-wins work within this one pass, same reason any
// depth buffer exists).

layout(binding = 0) uniform CameraUBO
{
    mat4 view;
    mat4 proj;
    vec3 cameraPosition;
    vec3 lightDirection;
    vec3 lightColor;
    vec3 ambientColor;
    float particlePointSize;
    vec3 cameraRight;
    vec3 cameraUp;
    vec3 cameraForward;
} camera;

layout(location = 0) in vec3 worldCenter;
layout(location = 1) in float radius;
layout(location = 2) in vec2 localCoord;

layout(location = 0) out float outWorldDistance;

void main()
{
    // localCoord arrives pre-mapped to -1..1 with local.y already
    // matching camera.cameraUp's sign (see fluid_depth.vert's CORNERS
    // comment) - no gl_PointCoord-style Y-flip needed here, unlike the
    // point-sprite version this replaced.
    vec2 local = localCoord;
    float r2 = dot(local, local);
    if (r2 > 1.0)
    {
        discard; // outside the circle inscribed in this quad - not part of this sphere's silhouette
    }

    // Height of the sphere surface above the local.xy plane, in sphere-
    // radius units (0 at the silhouette edge, 1 at dead center) - the
    // standard point-sprite-as-sphere reconstruction, unchanged by the
    // quad-vs-point-sprite switch.
    float heightUnits = sqrt(1.0 - r2);

    vec3 surfacePos = worldCenter
        + camera.cameraRight * (local.x * radius)
        + camera.cameraUp * (local.y * radius)
        - camera.cameraForward * (heightUnits * radius);

    vec4 clipHit = camera.proj * camera.view * vec4(surfacePos, 1.0);
    gl_FragDepth = clipHit.z / clipHit.w;

    outWorldDistance = length(surfacePos - camera.cameraPosition);
}
