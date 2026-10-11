#version 450

// Pass 2 of the screen-space fluid reconstruction - a separable
// (horizontal, then vertical - see Renderer::DrawFrame's two draw calls
// through this same pipeline) bilateral blur over the depth+color texture
// Pass 1 (fluid_depth.vert/.frag) wrote (R = world distance, GBA =
// particle color - see Renderer::CreateFluidSurfaceResources' format
// comment). This is what turns a field of separate sphere silhouettes
// into one smooth, continuous surface - see the project's fluid-sim
// literature notes doc's "Bilateral Filtering" section, this is that
// exact technique applied to real per-particle depth instead of a
// screen-space splat of the whole scene.
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
// surface detail). Color is blended using the EXACT SAME per-sample
// weight the depth channel computes - deliberate: two particles the
// depth weighting has already decided are "different surfaces" (e.g.
// across a real silhouette edge) shouldn't have their COLORS blend
// either, or you'd get a color halo bleeding past the depth boundary
// that produced it.
//
// Kernel radius is now computed PER-PIXEL from how many screen pixels
// FluidParticleRadius actually covers at this pixel's own depth - see
// ComputeKernelRadius's own comment for why. It used to be a fixed
// texel count regardless of distance, which meant the SAME kernel had
// to work for a particle covering 4 pixels (far away) and one covering
// 200 (right in front of the camera) - tuned for the far case, it did
// nothing close up (not wide enough to reach a neighboring particle's
// footprint at all, so blobs stayed looking like a cluster of separate
// spheres instead of one liquid surface); tuned for the close case, it
// would have blurred distant, unrelated particles together into mush.
// Scaling with actual on-screen size fixes both at once.

layout(set = 0, binding = 0) uniform sampler2D sourceDepth;

layout(push_constant) uniform BlurPushConstants
{
    vec2 texelSize;  // 1/width, 1/height, in texels
    vec2 direction;  // (1,0) for the horizontal pass, (0,1) for the vertical pass
    // Screen pixels one world-space unit projects to at exactly 1 unit
    // of distance from the camera (vertical axis) - computed once per
    // frame on the CPU from vertical FOV and framebuffer height (see
    // Renderer::DrawFrame's own comment on this exact value). Combined
    // with a pixel's own linear depth (centerDepth below) and
    // particleRadius, this is what lets ComputeKernelRadius work out how
    // many actual screen pixels a particle covers, without needing to
    // know the camera's full projection matrix in this shader at all.
    float pixelsPerWorldUnitAtUnitDistance;
    float particleRadius; // world-space units - Renderer::FluidParticleRadius, same value fluid_depth.vert used to size the billboard quads themselves
} pc;

layout(location = 0) in vec2 fragScreenUV;
layout(location = 0) out vec4 outBlurred;

// See fluid_depth.frag - this pass's sentinel for "no particle touched
// this pixel" is whatever CreateFluidSurfaceResources clears the color
// attachment to (see that function's comment); anything above this
// threshold is treated as empty, never as a real depth value. Only the R
// channel (distance) is ever compared against this - the GBA (color)
// channels of the clear value are irrelevant since a sentinel pixel is
// always fully discarded/passed-through, never displayed.
const float SENTINEL_THRESHOLD = 1.0e6;

const int KERNEL_RADIUS_FLOOR = 6; // the ORIGINAL fixed value - kept as a floor so a far/small particle never gets a narrower kernel than the proven-good original behavior, only ever the same or wider
const int MAX_KERNEL_RADIUS = 32; // caps the cost of a particle extremely close to the camera - see ComputeKernelRadius's own comment
const float KERNEL_REACH_IN_PARTICLE_RADII = 3.0; // how many particle-radii out the kernel should reach - see ComputeKernelRadius's own comment for why this value

// World-space units (meters, in this project's convention) - a depth
// difference smaller than this blends normally; bigger than this is
// treated as "a different surface" and excluded, which is what preserves
// silhouette edges instead of smearing across them. Widened from an
// original 0.08, tuned before FluidParticleRadius (the live-tunable
// visual sphere size, tested around 0.12) existed as a separate,
// adjustable control - 0.08 ended up tighter than a single particle's
// own radius, since two neighboring particles' surface depths can differ
// by close to a full radius even when touching, so it was suppressing
// legitimate blending between adjacent, connected particles, not just
// correctly preserving real silhouette edges. If FluidParticleRadius
// gets tuned much larger or smaller than ~0.12 going forward, this
// should move with it (roughly on the order of the visual radius itself,
// not a fixed fraction of it - hasn't been tested across a wide radius
// range yet).
const float DEPTH_SIGMA = 0.2;

// DEPTH_SIGMA above was tuned back when this pass used one FIXED 6px
// kernel radius for every particle regardless of distance. Now that
// ComputeKernelRadius scales the radius up to MAX_KERNEL_RADIUS for
// close/dense particles, the same fixed 0.2 sigma is often far tighter
// than the legitimate, same-surface depth variation a wide kernel
// naturally spans - two samples 20+ pixels apart on the SAME curved
// blob can differ by close to a full FluidParticleRadius, well outside
// a sigma of 0.2. Sitting that far out on the Gaussian's tail means the
// range weight ends up dominated by sub-pixel raster noise rather than
// real surface shape, and because this blur runs as two independent 1D
// passes (H then V - see this file's own top comment), that noise isn't
// symmetric between them: confirmed by building a numeric replica of
// this exact filter and measuring adjacent-pixel depth jitter after
// both passes - jitter dropped by roughly 10x (vertical) when the range
// weight's influence was removed entirely, and comparably when it was
// widened instead, versus no meaningful change from adjusting anything
// else (integer-vs-continuous radius, fixed-vs-adaptive radius, more
// blur iterations). That jitter is exactly what reads as fine
// cross-hatching once fluid_composite.frag turns it into a normal - the
// depth VALUES look smooth to the eye, but the pixel-to-pixel
// DIFFERENCE (which is what a normal is built from) was noisy enough to
// visibly light differently pixel to pixel, worst on isolated/small
// particles (mostly silhouette, so the whole visible face is affected)
// and on the big pool's own outer rim (its interior is masked by
// overlap from many particles; the rim isn't).
//
// Scaling sigma with the actual kernel radius in use (not a fixed
// value) keeps the same intent - reject real silhouette edges, blend
// real same-surface neighbors - but calibrated to how far THIS pixel's
// kernel actually reaches, instead of a constant tuned for the old
// fixed 6px case. DEPTH_SIGMA_RADIUS_SCALE is the empirically-found
// multiplier from that same replica (8.0 removed nearly all the
// measured jitter; much smaller values left most of it); if silhouettes
// start looking too soft/rounded at high particle density, this is the
// one number to pull back down - there's a real, direct tradeoff
// between how tightly this rejects a real edge and how much of this
// noise it lets back in.
const float DEPTH_SIGMA_RADIUS_SCALE = 8.0;

// How far (in texels, along this pass's blur direction) the kernel
// needs to reach to actually bridge two nearby particles into one
// surface, given how large FluidParticleRadius projects to on screen at
// this specific pixel's depth. Two TOUCHING spheres of that radius have
// centers 2*particleRadius apart in world space - projected to screen
// pixels via the standard perspective-projection size formula
// (worldSize / depth * pixelsPerWorldUnitAtUnitDistance), reaching just
// 2.0 particle-radii out is the minimum that bridges that exact gap.
// Widened to 3.0 so particles noticeably short of actually touching
// still read as one connected surface (the ask was for the liquid to
// start looking cohesive earlier/closer up, not just at the moment
// particles physically overlap) - the tradeoff is a slightly softer,
// less detailed surface even where particles ARE dense, since every
// pixel now averages in a wider neighborhood regardless. If that
// softening turns out to be too much, this is the one number to pull
// back down toward 2.0. Floored at the original fixed
// KERNEL_RADIUS_FLOOR (never narrower than what already worked for
// distant/small particles) and capped at MAX_KERNEL_RADIUS (bounds the
// cost for a particle sitting right up against the camera, where the
// projected size would otherwise grow without limit as depth approaches
// zero).
int ComputeKernelRadius(float centerDepth)
{
    float pixelRadius = (pc.particleRadius / max(centerDepth, 0.001)) * pc.pixelsPerWorldUnitAtUnitDistance;
    float reachPixels = pixelRadius * KERNEL_REACH_IN_PARTICLE_RADII;
    return int(clamp(reachPixels, float(KERNEL_RADIUS_FLOOR), float(MAX_KERNEL_RADIUS)));
}

void main()
{
    vec4 center = texture(sourceDepth, fragScreenUV);
    float centerDepth = center.r;
    if (centerDepth > SENTINEL_THRESHOLD)
    {
        // No fluid at this pixel - pass the sentinel (and whatever
        // garbage color rode along with it, never displayed) through
        // untouched rather than letting neighboring real depth/color
        // values bleed INTO empty space, which would grow every blob's
        // silhouette outward a little on every blur pass.
        outBlurred = center;
        return;
    }

    // Computed once from THIS pixel's own depth, not the original fixed
    // KERNEL_RADIUS/SPATIAL_SIGMA pair - see ComputeKernelRadius's own
    // comment. spatialSigma keeps the same 2:1 radius:sigma ratio the
    // original fixed values had (6 and 3.0), so the Gaussian falloff
    // SHAPE stays consistent as the radius scales, rather than becoming
    // unusually flat or unusually peaked at either extreme.
    int kernelRadius = ComputeKernelRadius(centerDepth);
    float spatialSigma = float(kernelRadius) / 2.0;

    // See DEPTH_SIGMA_RADIUS_SCALE's comment - widen the range-weight
    // sigma in proportion to how much wider than the original fixed 6px
    // this pixel's own kernel radius has grown.
    float depthSigma = DEPTH_SIGMA * max(float(kernelRadius) / float(KERNEL_RADIUS_FLOOR), 1.0)
                        * DEPTH_SIGMA_RADIUS_SCALE;

    float totalWeight = 0.0;
    float totalDepth = 0.0;
    vec3 totalColor = vec3(0.0);

    // Tap count is capped (~17) no matter how big the kernel gets. The
    // Gaussian's sigma is kernelRadius/2, so sampling it every
    // kernelRadius/8 pixels is still well inside what the filter can
    // resolve (stride <= sigma/4) - the result matches the dense version
    // to within noise, but a particle filling the screen no longer costs
    // 65 RGBA32F fetches per pixel per pass (which, times a retina
    // framebuffer, was a big part of the close-up lag).
    int stride = max(1, kernelRadius / 8);
    int tapsEachSide = kernelRadius / stride;

    for (int k = -tapsEachSide; k <= tapsEachSide; ++k)
    {
        int i = k * stride;
        vec2 offset = pc.direction * pc.texelSize * float(i);
        vec4 sampleValue = texture(sourceDepth, fragScreenUV + offset);
        float sampleDepth = sampleValue.r;
        if (sampleDepth > SENTINEL_THRESHOLD)
        {
            continue; // don't let empty background pull this surface's depth (or color) outward either
        }

        float fi = float(i);
        float spatialWeight = exp(-(fi * fi) / (2.0 * spatialSigma * spatialSigma));
        float depthDiff = sampleDepth - centerDepth;
        float rangeWeight = exp(-(depthDiff * depthDiff) / (2.0 * depthSigma * depthSigma));
        float weight = spatialWeight * rangeWeight;

        totalDepth += sampleDepth * weight;
        totalColor += sampleValue.gba * weight;
        totalWeight += weight;
    }

    outBlurred = totalWeight > 0.0
        ? vec4(totalDepth / totalWeight, totalColor / totalWeight)
        : center;
}
