#version 450

// Pass 2 of the screen-space fluid reconstruction - a separable
// (horizontal, then vertical - see Renderer::DrawFrame's two draw calls
// through this same pipeline) bilateral blur over the depth texture Pass
// 1 (fluid_depth.vert/.frag) wrote. This is what turns a field of
// separate sphere silhouettes into one smooth, continuous surface - see
// the project's fluid-sim literature notes doc's "Bilateral Filtering"
// section, this is that exact technique applied to real per-particle
// depth instead of a screen-space splat of the whole scene.
//
// Vertex stage is raymarch.vert unchanged (see Renderer::m_FluidBlurPipeline)
// - a generic fullscreen triangle producing fragScreenUV, no dependency on
// anything raymarch-specific.
//
// Bilateral = weighted by BOTH screen-space distance (an ordinary
// Gaussian) AND depth difference from the center sample - the second
// term is what keeps this from blurring straight across a silhouette
// edge into whatever's behind/beside a blob (which a plain Gaussian blur
// would do, softening the actual shape instead of just smoothing its
// surface detail).

layout(set = 0, binding = 0) uniform sampler2D sourceDepth;

layout(push_constant) uniform BlurPushConstants
{
    vec2 texelSize;  // 1/width, 1/height, in texels
    vec2 direction;  // (1,0) for the horizontal pass, (0,1) for the vertical pass
} pc;

layout(location = 0) in vec2 fragScreenUV;
layout(location = 0) out float outBlurred;

// See fluid_depth.frag - this pass's sentinel for "no particle touched
// this pixel" is whatever CreateFluidSurfaceResources clears the color
// attachment to (see that function's comment); anything above this
// threshold is treated as empty, never as a real depth value.
const float SENTINEL_THRESHOLD = 1.0e6;

const int KERNEL_RADIUS = 6;
const float SPATIAL_SIGMA = 3.0;
// World-space units (meters, in this project's convention) - a depth
// difference smaller than this blends normally; bigger than this is
// treated as "a different surface" and excluded, which is what preserves
// silhouette edges instead of smearing across them. Widened from the
// first pass's 0.08 - that was tuned before FluidParticleRadius (the
// live-tunable visual sphere size, currently tested around 0.12) existed
// as a separate, adjustable control, and ended up tighter than a single
// particle's own radius: two neighboring particles' surface depths can
// differ by close to a full radius even when they're touching, so 0.08
// was suppressing legitimate blending between adjacent, connected
// particles - not just correctly preserving real silhouette edges - and
// showing up as the individual-bump/orange-peel look on a blob's
// surface. Background pixels are excluded from blending entirely via the
// sentinel check above regardless of this value, so widening it only
// affects blending BETWEEN actual fluid-covered pixels, not the
// fluid-vs-background edge - there's no silhouette-bleed risk from this
// change. If FluidParticleRadius gets tuned much larger or smaller than
// ~0.12 going forward, this should move with it (roughly on the order of
// the visual radius itself, not a fixed fraction of it - hasn't been
// tested across a wide radius range yet).
const float DEPTH_SIGMA = 0.2;

void main()
{
    float centerDepth = texture(sourceDepth, fragScreenUV).r;
    if (centerDepth > SENTINEL_THRESHOLD)
    {
        // No fluid at this pixel - pass the sentinel through untouched
        // rather than letting neighboring real depth values bleed INTO
        // empty space, which would grow every blob's silhouette outward
        // a little on every blur pass.
        outBlurred = centerDepth;
        return;
    }

    float totalWeight = 0.0;
    float totalDepth = 0.0;

    for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; ++i)
    {
        vec2 offset = pc.direction * pc.texelSize * float(i);
        float sampleDepth = texture(sourceDepth, fragScreenUV + offset).r;
        if (sampleDepth > SENTINEL_THRESHOLD)
        {
            continue; // don't let empty background pull this surface's depth outward either
        }

        float fi = float(i);
        float spatialWeight = exp(-(fi * fi) / (2.0 * SPATIAL_SIGMA * SPATIAL_SIGMA));
        float depthDiff = sampleDepth - centerDepth;
        float rangeWeight = exp(-(depthDiff * depthDiff) / (2.0 * DEPTH_SIGMA * DEPTH_SIGMA));
        float weight = spatialWeight * rangeWeight;

        totalDepth += sampleDepth * weight;
        totalWeight += weight;
    }

    outBlurred = totalWeight > 0.0 ? (totalDepth / totalWeight) : centerDepth;
}
