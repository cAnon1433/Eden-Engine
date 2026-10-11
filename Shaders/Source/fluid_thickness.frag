#version 450

// Thickness pass (fragment half). Every particle adds the length of the
// chord a view ray travels THROUGH its sphere at this pixel, into an
// additively-blended single-channel float target (cleared to 0 each frame).
// fluid_composite.frag turns the sum into opacity with Beer-Lambert
// (alpha = 1 - exp(-absorption * thickness)): deep liquid reads solid, thin
// edges / splashes / lone droplets read see-through.
//
// WHICH liquid counts matters, and a plain "sum every particle along the
// ray" is wrong in two visible ways, both fixed here:
//
//  * Liquid hidden behind an opaque object (a solid, the terrain) must not
//    count. Without this, particles behind a solid darkened the liquid in
//    front of it - ghost "shadows" of hidden particles showing through the
//    object. Chords are cut off at the scene depth.
//
//  * Liquid FAR behind the visible surface (a separate puddle, other
//    particles on the far side of the room) isn't part of this surface's
//    body. Counting it made a lone droplet in front of distant liquid look
//    thick/opaque and "connected" (wrong normals, dark outline around it).
//    Only liquid within `window` behind the visible liquid surface counts -
//    deep pools still saturate (the window is several particle radii), lone
//    droplets stay lone.
//
// Additive blending is order-independent, so no sorting and no depth
// attachment.

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
    mat4 prevViewProj;
} camera;

layout(set = 2, binding = 0) uniform sampler2D sceneDepth;   // opaque scene, NDC depth
layout(set = 3, binding = 0) uniform sampler2D fluidFront;   // fluid_depth.frag's UNBLURRED output: R = linear depth of the nearest visible liquid surface, 1e9 where none

layout(push_constant) uniform FluidThicknessPushConstants
{
    float radius;
    float invTargetW;
    float invTargetH;
    float window; // how far behind the visible surface liquid still counts (world units)
} pc;

layout(location = 0) in vec2 localCoord;
layout(location = 1) in float radius;
layout(location = 2) flat in vec2 centerDepthAndCos;

layout(location = 0) out float outThickness;

const float SENTINEL_THRESHOLD = 1.0e8;

// NDC depth -> distance along the camera's forward axis. Derived from the
// projection matrix itself (clip.z = a*zv + b, clip.w = c*zv + e for a point
// at view-space depth zv on the optical axis), so it is correct for whatever
// depth convention/near/far the renderer uses.
float LinearizeDepth(float ndcDepth)
{
    float a = camera.proj[2][2];
    float b = camera.proj[3][2];
    float c = camera.proj[2][3];
    float e = camera.proj[3][3];
    float zv = (e * ndcDepth - b) / (a - c * ndcDepth);
    return abs(zv);
}

void main()
{
    float r2 = dot(localCoord, localCoord);
    if (r2 > 1.0)
    {
        discard;
    }

    vec2 uv = gl_FragCoord.xy * vec2(pc.invTargetW, pc.invTargetH);

    // Nearest visible liquid surface at this pixel (already scene-occluded by
    // fluid_depth.frag). None -> nothing visible here to be thick for.
    float frontLinear = texture(fluidFront, uv).r;
    if (frontLinear >= SENTINEL_THRESHOLD)
    {
        discard;
    }

    float sceneLinear = LinearizeDepth(texture(sceneDepth, uv).r);

    // This sphere's chord along the camera's forward axis at this pixel.
    float centerDepth = centerDepthAndCos.x;
    float cosToForward = centerDepthAndCos.y;
    float halfChordRay = radius * sqrt(1.0 - r2);
    float halfChordForward = halfChordRay * cosToForward;

    // Clip to [visible surface, surface + window] and to in front of the scene.
    float lo = max(centerDepth - halfChordForward, frontLinear);
    float hi = min(centerDepth + halfChordForward, min(frontLinear + pc.window, sceneLinear));
    if (hi <= lo)
    {
        discard;
    }

    // Back to length along the view ray (what the unclipped chord was).
    outThickness = (hi - lo) / cosToForward;
}
