#version 450

// See fluid_blob.vert's top comment for the overview. This is the whole
// technique in one pass: discard outside a circle, soft alpha falloff
// inside it, light an analytic sphere normal, done. No neighboring pixel
// is ever read - every value here comes from this fragment's own
// interpolated inputs, which is what makes this structurally immune to
// the failure modes (hatching, vibration, ghosting, streaking) the
// screen-space technique went through, rather than just "currently
// working."

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

layout(location = 0) in vec3 worldCenter;
layout(location = 1) in float radius;
layout(location = 2) in vec2 localCoord;
layout(location = 3) in vec3 particleColor;
layout(location = 4) in vec3 right;
layout(location = 5) in vec3 up;
layout(location = 6) in vec3 viewDir;

layout(location = 0) out vec4 outColor;

// --- Tuning constants - this is the "look" of the liquid; retune these
// live rather than touching the structure above them. ---

// How much of the billboard's radius is soft falloff vs. fully solid
// center. Higher = softer/foggier particles that blend into each other
// more readily; lower = crisper individual spheres that read more like
// distinct glass marbles. 1.5-2.5 is a reasonable starting range.
const float ALPHA_FALLOFF_POWER = 1.8;

// Opacity at the particle's own center (before any overlap-stacking).
// Lower = more see-through / thin-liquid-like, needs more overlapping
// particles to read as solid. Higher = each particle is more opaque on
// its own, overlap matters less. 0.35-0.6 is a reasonable starting range
// for "translucent, cohesive body" rather than either "fog" or "solid
// marbles."
const float BASE_OPACITY = 0.45;

const float SHININESS = 120.0;
const float SPEC_STRENGTH = 0.9;
const float FRESNEL_STRENGTH = 0.4;
const float AMBIENT = 0.35;
const float DIFFUSE_STRENGTH = 0.75;

void main()
{
    float r2 = dot(localCoord, localCoord);
    if (r2 > 1.0)
    {
        discard;
    }

    float height = sqrt(1.0 - r2); // 1 at center (nearest the camera), 0 at the rim - same reconstruction fluid_depth.frag uses

    // Analytic sphere normal, built from this fragment's own position on
    // the billboard - no neighboring pixel involved, so nothing here can
    // be noisy frame to frame the way a screen-space derivative can.
    // Minus viewDir*height so the center (height=1) faces the camera
    // (normal ~= -viewDir there) - same sign convention fluid_depth.frag
    // uses for surfacePos's bulge direction.
    vec3 normal = normalize(right * localCoord.x + up * localCoord.y - viewDir * height);

    vec3 V = -viewDir;
    vec3 L = normalize(-camera.lightDirection);
    vec3 H = normalize(L + V);
    float diff = clamp(dot(normal, L), 0.0, 1.0);
    float spec = diff > 0.0 ? pow(clamp(dot(normal, H), 0.0, 1.0), SHININESS) : 0.0;
    float fres = pow(1.0 - clamp(dot(normal, V), 0.0, 1.0), 5.0);

    vec3 base = particleColor * (AMBIENT + diff * DIFFUSE_STRENGTH);
    vec3 lit = mix(base, vec3(1.0), fres * FRESNEL_STRENGTH) + spec * SPEC_STRENGTH;

    float edge = 1.0 - r2; // 1 at center, 0 at the rim
    float alpha = pow(clamp(edge, 0.0, 1.0), ALPHA_FALLOFF_POWER) * BASE_OPACITY;

    // Per-fragment depth from the actual bulged surface point, not the
    // flat billboard quad - so this particle sinks into solid geometry
    // (floor, walls, terrain) the same convincing way fluid_depth.frag's
    // impostor does, even though depthWriteEnable is off for this
    // pipeline (see Renderer::InitFluidSurfacePass's blob pipeline
    // comment for why: depth TEST stays on so blobs are correctly
    // occluded by solid geometry, but WRITE stays off so overlapping
    // translucent blobs don't occlude each other in the depth buffer -
    // they need to blend, not z-fight).
    vec3 surfacePos = worldCenter + right * (localCoord.x * radius) + up * (localCoord.y * radius) - viewDir * (height * radius);
    vec4 clipPos = camera.proj * camera.view * vec4(surfacePos, 1.0);
    gl_FragDepth = clipPos.z / clipPos.w;

    outColor = vec4(lit, alpha);
}
