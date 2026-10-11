#version 450

// Pass 3 of the screen-space fluid reconstruction - a small, FIXED-radius
// true 2D bilateral "polish" pass, run once after fluid_blur.frag's
// existing two-pass (horizontal, then vertical) separable blur.
//
// Why this exists: fluid_blur.frag's separable blur is two independent 1D
// passes, which is only an approximation of a real bilateral filter (the
// range weight makes it value-dependent/non-linear, and a non-linear
// filter doesn't decompose into two 1D passes the way an ordinary
// position-only Gaussian does). That approximation error is what was
// producing fine cross-hatching on individual particles and blob
// silhouettes - confirmed with a numeric replica of the exact filter math.
//
// A full-radius (up to MAX_KERNEL_RADIUS) true 2D pass was tried first
// and reverted - see this project's own history/commit notes - because
// its cost is O(radius^2), and radius is DATA-DEPENDENT (grows with how
// close/dense the particles on screen are). At radius 32 that's up to
// ~3300 texture reads in a single fragment invocation, and when enough
// pixels hit that simultaneously (camera close to a lot of liquid) the
// frame can take long enough that the OS/driver's GPU watchdog resets the
// device mid-frame - corrupted frame contents and a severe stall, not
// just "slower."
//
// This pass sidesteps that by keeping its radius a FIXED CONSTANT
// (POLISH_RADIUS), never data-dependent - so its cost is the same,
// bounded, known number on every pixel regardless of scene, camera
// distance, or particle density. It doesn't need to reach far: reach/
// cohesion (merging distant particles into one surface) is already
// fluid_blur.frag's job. This pass's only job is cleaning up the
// small-scale, few-pixel-wide noise the separable approximation leaves
// behind, and a few-pixel-wide problem only needs a few-pixel-wide true
// 2D kernel to fix - see POLISH_RADIUS's own comment for the actual
// number and how it was chosen.
//
// Vertex stage is raymarch.vert, same as every other fullscreen pass in
// this pipeline (see fluid_blur.frag's own comment).

layout(set = 0, binding = 0) uniform sampler2D sourceDepth;

layout(push_constant) uniform PolishPushConstants
{
    vec2 texelSize; // 1/width, 1/height, in texels
    // Same two values fluid_blur.frag gets - used here only to know how big
    // a particle is on screen at this pixel, see the stride logic in main().
    float pixelsPerWorldUnitAtUnitDistance;
    float particleRadius;
} pc;

layout(location = 0) in vec2 fragScreenUV;
layout(location = 0) out vec4 outPolished;

const float SENTINEL_THRESHOLD = 1.0e6;

// Fixed, NOT data-dependent - see this file's top comment for why that
// matters. 8 was the smallest radius in testing that got measured
// pixel-to-pixel depth jitter down near the same floor as an unbounded
// full-radius 2D pass; smaller (radius 6) left visibly more residual
// noise, larger (10-12) measured only marginally better for
// meaningfully more cost. Worst-case cost with this value is
// pi*8*8 =~ 201 texture reads/pixel, ALWAYS - not a function of scene
// content, unlike the full-radius attempt this replaced.
const int POLISH_RADIUS = 8;
const int POLISH_RADIUS_SQ = POLISH_RADIUS * POLISH_RADIUS;
const float POLISH_SPATIAL_SIGMA = float(POLISH_RADIUS) / 2.0;

// Same DEPTH_SIGMA base as fluid_blur.frag, widened by the same kind of
// factor and for the same reason - see that file's DEPTH_SIGMA_RADIUS_SCALE
// comment. This pass's radius is fixed rather than adaptive, so the scale
// here is just a flat constant instead of scaling with a per-pixel
// kernelRadius; re-tuned empirically for THIS pass's fixed radius rather
// than reused verbatim from fluid_blur.frag's value.
const float DEPTH_SIGMA = 0.2 * 4.0;

void main()
{
    vec4 center = texture(sourceDepth, fragScreenUV);
    float centerDepth = center.r;
    if (centerDepth > SENTINEL_THRESHOLD)
    {
        outPolished = center; // nothing to polish - see fluid_blur.frag's identical early-out for why this is passed through untouched, not blended
        return;
    }

    float totalWeight = 0.0;
    float totalDepth = 0.0;
    vec3 totalColor = vec3(0.0);

    // Polish only exists to clean a few-pixel-wide residue. When a particle
    // covers many pixels on screen (camera close), that residue is tiny
    // relative to the surface and a dense 201-tap disc is wasted work - and
    // those are exactly the frames where the fluid covers the whole screen.
    // So the disc is sampled on a coarser grid when particles are big:
    //   <= 16 px radius: every pixel (unchanged, ~201 taps)
    //   <= 64 px radius: every 2nd pixel (~50 taps)
    //    > 64 px radius: every 4th pixel (~13 taps)
    float pixelRadius = (pc.particleRadius / max(centerDepth, 0.001)) * pc.pixelsPerWorldUnitAtUnitDistance;
    int stride = pixelRadius > 64.0 ? 4 : (pixelRadius > 16.0 ? 2 : 1);
    int tapsEachSide = POLISH_RADIUS / stride;

    for (int ky = -tapsEachSide; ky <= tapsEachSide; ++ky)
    {
        for (int kx = -tapsEachSide; kx <= tapsEachSide; ++kx)
        {
            int dx = kx * stride;
            int dy = ky * stride;
            int distSq = dx * dx + dy * dy;
            if (distSq > POLISH_RADIUS_SQ)
            {
                continue; // circular kernel, not square
            }

            vec2 offset = vec2(float(dx), float(dy)) * pc.texelSize;
            vec4 sampleValue = texture(sourceDepth, fragScreenUV + offset);
            float sampleDepth = sampleValue.r;
            if (sampleDepth > SENTINEL_THRESHOLD)
            {
                continue;
            }

            float spatialWeight = exp(-float(distSq) / (2.0 * POLISH_SPATIAL_SIGMA * POLISH_SPATIAL_SIGMA));
            float depthDiff = sampleDepth - centerDepth;
            float rangeWeight = exp(-(depthDiff * depthDiff) / (2.0 * DEPTH_SIGMA * DEPTH_SIGMA));
            float weight = spatialWeight * rangeWeight;

            totalDepth += sampleDepth * weight;
            totalColor += sampleValue.gba * weight;
            totalWeight += weight;
        }
    }

    outPolished = totalWeight > 0.0
        ? vec4(totalDepth / totalWeight, totalColor / totalWeight)
        : center;
}
