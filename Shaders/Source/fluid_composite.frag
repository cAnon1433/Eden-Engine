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
    mat4 prevViewProj; // unused here, declared only so invViewProj lands at the right offset
    mat4 invViewProj;  // inverse(proj * view), computed ONCE per frame on the CPU
} camera;

layout(set = 1, binding = 0) uniform sampler2D fluidDepth;

// Accumulated per-pixel liquid thickness (fluid_thickness.vert/.frag) -
// what makes this surface TRANSPARENT instead of a solid shell. Turned into
// opacity below with Beer-Lambert.
layout(set = 2, binding = 0) uniform sampler2D fluidThickness;

// Unblurred per-pixel analytic sphere normal (world space) from the depth
// pass's third output - used for isolated droplets, see main().
layout(set = 3, binding = 0) uniform sampler2D fluidAnalyticNormal;

layout(push_constant) uniform CompositePushConstants
{
    float absorption; // opacity gain per unit of thickness - higher = reads solid sooner
    float minAlpha;   // opacity floor for thin liquid (fades out toward the rim, see below) so lone droplets stay visible
    float maxAlpha;   // opacity ceiling - below 1.0 so even deep liquid keeps a hint of see-through
    float particleRadius; // Renderer::FluidParticleRadius - lets this shader tell "one lone droplet" thickness from "stacked liquid" thickness
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

// Reconstructs a world-space point given `t`, the LINEAR VIEW-SPACE
// DEPTH at `uv` - distance from the camera measured along
// camera.cameraForward, not raw Euclidean distance along this pixel's
// own ray (see fluid_depth.frag's own comment on why that distinction
// matters and what silently reconstructing the wrong position used to
// cause). Uses the SAME inverse-view-projection unprojection
// raymarch.frag's main() already does to get rayDir (see that shader) -
// deliberately the identical derivation for the ray itself, so this
// pass and raymarch.frag still agree with each other at any boundary
// between a raymarched object and the fluid surface. The division below
// is the actual linear-depth-to-world-position step: rayDir's own
// component along cameraForward is dot(rayDir, cameraForward), so
// scaling rayDir by t/dot(...) is exactly the length that makes the
// walked point's own forward-axis distance equal t, regardless of how
// far this particular ray diverges from the camera's central axis.
// invViewProj is inverse(view) * inverse(proj), computed ONCE per fragment
// in main() and passed in - this function runs up to five times per
// fragment now (center + four normal-stencil neighbors), and it used to
// call inverse() on two 4x4 matrices every single time.
vec3 WorldPointAtDistance(mat4 invViewProj, vec2 uv, float t)
{
    vec2 ndc = uv * 2.0 - 1.0;
    vec4 nearPoint4 = invViewProj * vec4(ndc, 0.0, 1.0);
    vec4 farPoint4 = invViewProj * vec4(ndc, 1.0, 1.0);
    vec3 nearPoint = nearPoint4.xyz / nearPoint4.w;
    vec3 farPoint = farPoint4.xyz / farPoint4.w;
    vec3 rayDir = normalize(farPoint - nearPoint);
    return camera.cameraPosition + rayDir * (t / dot(rayDir, camera.cameraForward));
}

// A neighbor whose linear depth differs from the center's by more than
// this is treated as a DIFFERENT surface (another blob behind/in front,
// or a silhouette step), not part of this pixel's own tangent plane.
// World units, same scale/ballpark as fluid_blur.frag's DEPTH_SIGMA.
const float NORMAL_MAX_DEPTH_JUMP = 0.3;

// One tangent vector of the reconstructed surface at `uv`, along the
// screen-space axis `stepUV`. Looks BOTH ways (+step and -step) and uses
// whichever side is a valid neighbor (real fluid, no big depth jump);
// if both are valid, uses the SMALLER difference, the standard trick from
// screen-space fluid rendering (van der Laan et al.) for keeping normals
// from being dragged by a depth discontinuity on one side.
//
// This replaces the old one-sided forward difference, which had a real
// bug at every silhouette: when the +x or +y neighbor was empty it
// substituted the pixel's own position, so the difference vector was
// EXACTLY ZERO, the cross product was zero, and normalize(vec3(0)) is
// undefined in GLSL (NaN/garbage on real hardware). A 1-pixel-staircase
// silhouette has lots of such pixels, and a small isolated particle is
// mostly silhouette - which is the "cross-hatching" across whole dots
// and along pool rims. Returns vec3(0) only if neither side is usable;
// main() handles that case explicitly instead of normalizing it.
//
// When BOTH sides are valid, this uses their CENTRAL (averaged)
// difference, not a one-sided pick. An earlier version of this function
// picked whichever single side had the smaller magnitude, on the theory
// that the closer neighbor is more trustworthy near a depth
// discontinuity - true, but it introduced a worse problem: which side is
// "smaller" is a hard, binary decision, and on an ordinary curved
// surface (not a discontinuity - just two legitimate, slightly different
// neighbors) that decision sits right on a knife edge for MANY pixels at
// once. The smallest camera movement can flip which side wins across a
// large patch of pixels simultaneously, and a flipped tangent means a
// visibly different normal - which read as particles flickering on
// camera movement. Confirmed with a numeric test: perturbing the input
// depth field by a sub-pixel amount (simulating a tiny camera move) and
// comparing the resulting normal field, the min-pick version changed
// roughly 2x more frame-to-frame than a plain central difference. A
// central difference has no side to "pick," so there's nothing to flip -
// and it's the textbook more-accurate option anyway (second-order
// accurate vs. first-order for a one-sided difference) whenever both
// neighbors are legitimately part of the same surface. The one-sided
// fallback below is still exactly what runs at a REAL silhouette (only
// one side valid) - that case has no stability problem to fix, since
// there's no second side to have picked instead.
vec3 TangentAlong(mat4 invViewProj, vec2 uv, float centerT, vec3 centerPos, vec2 stepUV)
{
    float tF = texture(fluidDepth, uv + stepUV).r;
    float tB = texture(fluidDepth, uv - stepUV).r;
    bool validF = (tF < SENTINEL_THRESHOLD) && (abs(tF - centerT) < NORMAL_MAX_DEPTH_JUMP);
    bool validB = (tB < SENTINEL_THRESHOLD) && (abs(tB - centerT) < NORMAL_MAX_DEPTH_JUMP);

    if (validF && validB)
    {
        vec3 pF = WorldPointAtDistance(invViewProj, uv + stepUV, tF);
        vec3 pB = WorldPointAtDistance(invViewProj, uv - stepUV, tB);
        return (pF - pB) * 0.5;
    }
    if (validF)
    {
        return WorldPointAtDistance(invViewProj, uv + stepUV, tF) - centerPos;
    }
    if (validB)
    {
        return centerPos - WorldPointAtDistance(invViewProj, uv - stepUV, tB);
    }
    return vec3(0.0); // neither side usable - main() handles this case explicitly instead of normalizing it
}

void main()
{
    vec4 sampled = texture(fluidDepth, fragScreenUV);
    float worldT = sampled.r;
    if (worldT > SENTINEL_THRESHOLD)
    {
        discard; // no fluid at this pixel - let whatever's rasterized underneath show through
    }

    // Per-pixel color comes from the Chemical ID base build's diffusing
    // per-particle color (see fluid_depth.vert/.frag and
    // particle_force.comp/particle_integrate.comp's color-diffusion
    // blocks) - already bilaterally blurred alongside depth in
    // fluid_blur.frag, using the same weights, so it stays aligned with
    // the reconstructed surface rather than being a flat, unrelated
    // overlay. There's no separate uniform tint anymore - this IS the
    // fluid's actual simulated color at this point on its surface.
    vec3 surfaceColor = sampled.gba;

    // Was inverse(view) * inverse(proj) HERE, i.e. two 4x4 matrix inversions
    // per pixel - on a close-up, full-screen fluid that was a large part of
    // the frame cost. Now one inverse per frame on the CPU.
    mat4 invViewProj = camera.invViewProj;
    vec3 worldPos = WorldPointAtDistance(invViewProj, fragScreenUV, worldT);

    // Normal from screen-space finite differences of the BLURRED depth
    // buffer - two tangent vectors (see TangentAlong for how each is
    // picked and why), whose cross product is the surface normal.
    vec2 texelSize = 1.0 / vec2(textureSize(fluidDepth, 0));

    // NORMAL_STENCIL widens TangentAlong's sample offset beyond the
    // immediate 1-texel neighbor. A derivative amplifies whatever noise
    // is left in its input - and even with TangentAlong's central
    // difference (no more discrete side-picking, see that function's own
    // comment) there's still some genuine per-pixel noise surviving the
    // blur/polish passes, small enough not to matter for the VALUES
    // themselves but still enough to visibly perturb a DERIVATIVE of
    // them frame to frame as the camera moves - measured as particles
    // "vibrating" rather than flickering once the discrete flip was
    // fixed. A wider baseline is a lower-frequency (gentler) derivative,
    // so it's less sensitive to that leftover noise - confirmed
    // numerically: widening from 1 to 2 texels cut frame-to-frame normal
    // change by roughly a third in testing, with a smoothly diminishing
    // (not cliff-edged) return going wider still. 2 is a conservative
    // middle ground - wide enough to meaningfully help, not so wide that
    // it starts averaging over real curvature on a small particle rather
    // than just noise.
    const float NORMAL_STENCIL = 2.0;
    vec2 stencilTexelSize = texelSize * NORMAL_STENCIL;
    vec3 tangentX = TangentAlong(invViewProj, fragScreenUV, worldT, worldPos, vec2(stencilTexelSize.x, 0.0));
    vec3 tangentY = TangentAlong(invViewProj, fragScreenUV, worldT, worldPos, vec2(0.0, stencilTexelSize.y));

    // Thickness drives both opacity (below) and how "isolated" this pixel is.
    float thickness = texture(fluidThickness, fragScreenUV).r;

    // 1 = a lone droplet/particle (thickness no more than about one sphere
    // across, 2r), 0 = stacked/connected liquid (about 2.5+ spheres deep).
    // This is what keeps pools looking exactly as they do while droplets get
    // their own treatment.
    float isolated = 1.0 - smoothstep(2.0 * pc.particleRadius, 5.0 * pc.particleRadius, thickness);

    vec3 V = normalize(camera.cameraPosition - worldPos);
    vec3 N = cross(tangentX, tangentY);
    float normalLength = length(N);
    if (normalLength > 1.0e-12)
    {
        N /= normalLength;
        if (dot(N, V) < 0.0)
        {
            N = -N; // finite-difference cross product can come out facing either way - always face the camera
        }
    }
    else
    {
        // No usable tangent on at least one axis (a feature 1 pixel wide) -
        // never normalize a zero vector; face the camera instead.
        N = V;
    }

    // Isolated droplets: swap the blurred-depth normal for the exact
    // analytic sphere normal. A droplet is nearly all silhouette, so there
    // are no valid neighbors to build a screen-space normal from, and the
    // depth profile is flattened at the rim on purpose (see fluid_depth.frag's
    // edgeFade) - which is what made them read as flat-topped domes with a
    // dark wall instead of round drops.
    if (isolated > 0.001)
    {
        vec3 analyticN = texture(fluidAnalyticNormal, fragScreenUV).xyz;
        if (dot(analyticN, analyticN) > 0.25)
        {
            N = normalize(mix(N, normalize(analyticN), isolated));
            if (dot(N, V) < 0.0) { N = -N; }
        }
    }

    vec3 L = normalize(-camera.lightDirection);
    vec3 H = normalize(L + V);

    float ndl = dot(N, L);
    float diffuseFactor = max(ndl, 0.0);
    float specularFactor = diffuseFactor > 0.0 ? pow(max(dot(N, H), 0.0), SHININESS) : 0.0;
    float fresnel = pow(1.0 - max(dot(N, V), 0.0), FRESNEL_POWER);

    vec3 ambient = camera.ambientColor;
    // Thin, translucent liquid is lit by light passing THROUGH it, so its
    // shadowed side shouldn't go black like an opaque solid's. Blend toward
    // half-Lambert (wrapped) lighting by how isolated this pixel is; thick
    // pools keep the original shading.
    float wrappedDiffuse = mix(diffuseFactor, 0.5 + 0.5 * ndl, isolated * 0.8);
    vec3 diffuse = wrappedDiffuse * camera.lightColor;
    vec3 specular = specularFactor * SPECULAR_STRENGTH * camera.lightColor;
    vec3 base = surfaceColor * (ambient + diffuse);
    // Cheap stand-in for real refraction/transmission (deferred - see
    // the "Explicitly Deferred" section of the design doc's rendering
    // notes): brighten toward the light color at grazing angles, same
    // shape as real Fresnel reflectance without an actual environment
    // sample to reflect.

    // --- Transparency -------------------------------------------------
    // Opacity from accumulated thickness (Beer-Lambert). minAlpha is
    // scaled in by smoothstep over the first sliver of thickness so the
    // very rim of a blob/droplet still feathers out to fully transparent
    // instead of ending in a hard minAlpha-opacity edge.
    float alpha = 1.0 - exp(-pc.absorption * thickness);
    alpha = max(alpha, pc.minAlpha * smoothstep(0.0, 0.05, thickness));
    // Thin water is see-through head-on but reflective at grazing angles -
    // raise opacity with Fresnel for isolated droplets so they read as glassy
    // drops with a bright rim instead of flat translucent discs.
    alpha = max(alpha, fresnel * isolated * 0.85);
    alpha = clamp(alpha, 0.0, pc.maxAlpha);

    // Premultiplied output (pipeline blend: src ONE, dst ONE_MINUS_SRC_ALPHA).
    // The body color is scaled by alpha like any translucent material, but
    // specular is added at FULL strength: a highlight is light reflected off
    // the surface, it doesn't get dimmer because the liquid underneath is
    // thin. That's also what keeps thin liquid readable as liquid.
    vec3 body = mix(base, camera.lightColor, fresnel * 0.5);
    outColor = vec4(body * alpha + specular, alpha);

    // Reproject through the SAME matrices the rasterizer's own
    // gl_Position pipeline uses, exactly like raymarch.frag's identical
    // line - so this pass's depth test against m_DepthImage (already
    // written by opaque mesh/voxel/raymarch draws earlier this render
    // pass) agrees with them at any boundary instead of one path always
    // winning regardless of actual depth.
    vec4 clipHit = camera.proj * camera.view * vec4(worldPos, 1.0);
    gl_FragDepth = clipHit.z / clipHit.w;
}
