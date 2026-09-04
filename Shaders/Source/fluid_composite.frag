#version 450

// Pass 3 (final) of the screen-space fluid reconstruction - runs as an
// ordinary fullscreen pass WITHIN the main render pass, same "depth test
// ON against whatever rasterized/raymarched geometry already wrote this
// frame's depth buffer" approach raymarch.frag already uses (see that
// shader's file comment - this one mirrors its ray-reconstruction and
// gl_FragDepth-reprojection technique deliberately, rather than inventing
// a second way to do the same thing).
//
// Vertex stage is raymarch.vert unchanged (generic fullscreen triangle).
//
// This is the pass that actually makes the fluid look like a continuous
// surface instead of a field of dots: everything before this point
// (fluid_depth.*, fluid_blur.frag) only ever decided WHERE the surface is
// per pixel; this pass reconstructs a NORMAL from the blurred depth
// buffer's own screen-space derivatives and lights it like any other
// surface. The normal has no idea it used to be a cloud of SPH particles
// - that's the whole point.

layout(set = 0, binding = 0) uniform CameraUBO
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

layout(set = 1, binding = 0) uniform sampler2D fluidDepth;

layout(push_constant) uniform FluidCompositePushConstants
{
    vec3 tintColor;
} pc;

layout(location = 0) in vec2 fragScreenUV;
layout(location = 0) out vec4 outColor;

const float SENTINEL_THRESHOLD = 1.0e6;
// Tightened from the first pass's (48, 0.6, 4.0) - broad, soft specular
// reads as rubber/slime; a tight, bright highlight is a big part of what
// actually reads as "water" without doing real refraction/transmission
// (still deferred - see the Design Document's Explicitly Deferred
// section). This alone won't fully solve it - there's no thickness-based
// darkening at all yet (a puddle and a thick blob currently shade
// identically) - but it's the cheap half of the fix; the tint color
// itself (FluidTintColor, live-tunable) is the other half.
const float SHININESS = 200.0;
const float SPECULAR_STRENGTH = 1.1;
const float FRESNEL_POWER = 5.0;

// Reconstructs a world-space point at world-space distance `t` along the
// camera ray through `uv`, using the SAME inverse-view-projection
// unprojection raymarch.frag's main() already does (see that shader) -
// deliberately the identical derivation, not a second, different one, so
// the two passes agree with each other at any boundary between a
// raymarched object and the fluid surface.
vec3 WorldPointAtDistance(vec2 uv, float t)
{
    vec2 ndc = uv * 2.0 - 1.0;
    mat4 invView = inverse(camera.view);
    mat4 invProj = inverse(camera.proj);
    vec4 nearPoint4 = invView * invProj * vec4(ndc, 0.0, 1.0);
    vec4 farPoint4 = invView * invProj * vec4(ndc, 1.0, 1.0);
    vec3 nearPoint = nearPoint4.xyz / nearPoint4.w;
    vec3 farPoint = farPoint4.xyz / farPoint4.w;
    vec3 rayDir = normalize(farPoint - nearPoint);
    return camera.cameraPosition + rayDir * t;
}

void main()
{
    float worldT = texture(fluidDepth, fragScreenUV).r;
    if (worldT > SENTINEL_THRESHOLD)
    {
        discard; // no fluid at this pixel - let whatever's rasterized underneath show through
    }

    vec3 worldPos = WorldPointAtDistance(fragScreenUV, worldT);

    // Normal from screen-space finite differences of the BLURRED depth
    // buffer - two neighboring reconstructed world points give two
    // tangent vectors, whose cross product is the surface normal. Falls
    // back to the center sample at a blob's silhouette edge (where a
    // neighbor texel has no fluid at all) rather than differencing
    // against the sentinel, which would otherwise produce one wrong,
    // huge-magnitude normal exactly at every blob's outline.
    vec2 texelSize = 1.0 / vec2(textureSize(fluidDepth, 0));

    float tX = texture(fluidDepth, fragScreenUV + vec2(texelSize.x, 0.0)).r;
    vec3 worldPosX = (tX < SENTINEL_THRESHOLD)
        ? WorldPointAtDistance(fragScreenUV + vec2(texelSize.x, 0.0), tX)
        : worldPos;

    float tY = texture(fluidDepth, fragScreenUV + vec2(0.0, texelSize.y)).r;
    vec3 worldPosY = (tY < SENTINEL_THRESHOLD)
        ? WorldPointAtDistance(fragScreenUV + vec2(0.0, texelSize.y), tY)
        : worldPos;

    vec3 N = normalize(cross(worldPosX - worldPos, worldPosY - worldPos));
    vec3 V = normalize(camera.cameraPosition - worldPos);
    if (dot(N, V) < 0.0)
    {
        N = -N; // finite-difference cross product can come out facing either way - always face the camera
    }

    vec3 L = normalize(-camera.lightDirection);
    vec3 H = normalize(L + V);

    float diffuseFactor = max(dot(N, L), 0.0);
    float specularFactor = diffuseFactor > 0.0 ? pow(max(dot(N, H), 0.0), SHININESS) : 0.0;
    float fresnel = pow(1.0 - max(dot(N, V), 0.0), FRESNEL_POWER);

    vec3 ambient = camera.ambientColor;
    vec3 diffuse = diffuseFactor * camera.lightColor;
    vec3 specular = specularFactor * SPECULAR_STRENGTH * camera.lightColor;
    vec3 base = pc.tintColor * (ambient + diffuse);
    // Cheap stand-in for real refraction/transmission (deferred - see
    // the "Explicitly Deferred" section of the design doc's rendering
    // notes): brighten toward the light color at grazing angles, same
    // shape as real Fresnel reflectance without an actual environment
    // sample to reflect.
    vec3 lit = mix(base, camera.lightColor, fresnel * 0.5) + specular;

    outColor = vec4(lit, 1.0);

    // Reproject through the SAME matrices the rasterizer's own
    // gl_Position pipeline uses, exactly like raymarch.frag's identical
    // line - so this pass's depth test against m_DepthImage (already
    // written by opaque mesh/voxel/raymarch draws earlier this render
    // pass) agrees with them at any boundary instead of one path always
    // winning regardless of actual depth.
    vec4 clipHit = camera.proj * camera.view * vec4(worldPos, 1.0);
    gl_FragDepth = clipHit.z / clipHit.w;
}
