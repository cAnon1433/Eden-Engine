#version 450

// OPTIONAL (Renderer::FluidTemporalEnabled, OFF by default). Pass 4 of the screen-space fluid reconstruction - temporal
// accumulation via REPROJECTION. Blends this frame's polished depth+color
// (fluid_blur.frag then fluid_blur_polish.frag's output) against a
// running history buffer, using each particle's actual screen-space
// motion (fluid_depth.vert/.frag's second output) to find where that same
// point was last frame, rather than assuming it was at the same pixel.
//
// This replaces an earlier same-pixel version (no reprojection at all),
// which was a deliberate, explicitly-flagged simplification made under
// deadline pressure - see this project's own history/commit notes. It
// had a real, structural problem: gating on "how much did depth change
// at this exact pixel" can't actually distinguish "the camera moved, this
// is a different point now" from "this particle moved, but reprojection
// would show it's still a good match" - both look identical to a
// same-pixel check. Tightening the gate to fix one failure mode
// (ghosting - stale edges lingering a frame or two) directly worsened the
// other (a wiggly line tracking every blob's rim, from the gate
// constantly rejecting-then-accepting right where depth changes fastest).
// No single threshold fixed both at once, because the same-pixel design
// was comparing the wrong things. Reprojection fixes the actual
// comparison: once "where was this on screen last frame" is answered
// properly via the motion vector instead of assumed to be "here," the
// depth-difference gate is comparing the SAME physical point across two
// frames, which is what it was always supposed to be doing.
//
// Two separate history images are still ping-ponged once per RENDERED
// FRAME (Renderer::m_FluidHistoryIndex) - NOT per frame-in-flight slot
// (m_CurrentFrame). See Renderer::DrawFrame's own comment at the call
// site for how the two indices interact; that part is unchanged from the
// same-pixel version.

layout(set = 0, binding = 0) uniform sampler2D newFrame;   // this frame's polished depth+color (fluid_blur_polish.frag's output)
layout(set = 1, binding = 0) uniform sampler2D motionTex;  // this frame's UNBLURRED per-particle motion vector (fluid_depth.frag's second output)
layout(set = 2, binding = 0) uniform sampler2D history;    // last frame's accumulated result

layout(location = 0) in vec2 fragScreenUV;
layout(location = 0) out vec4 outAccumulated;

const float SENTINEL_THRESHOLD = 1.0e6;
const float SENTINEL_VALUE = 1.0e9; // matches every other fluid pass's clear value - see fluid_blur.frag

// World-space units. How much the depth AT THE REPROJECTED POINT is
// allowed to differ from this frame's value before history is treated as
// stale and discarded rather than blended. Now that reprojection answers
// "where was this physical point last frame" correctly (see this file's
// top comment), this is back closer to its original tight value - it
// only needs to absorb genuine per-frame change (the liquid's own motion/
// deformation, chemistry reactions, melt/reform), not reprojection error.
// Untested against a real build; if this reads as too trigger-happy
// (losing smoothing on ordinary flowing liquid) widen it, if any
// ghosting/ reappears tighten it - but reprojection should mean this
// needs much less retuning than the same-pixel version did.
// Hard depth gate. Kept, but it is NOT what prevents ghosting - see the
// neighborhood clamp below. A depth-only gate can't reject a stale feature
// whose DEPTH is tiny but whose NORMAL is large (e.g. a faint ridge where
// one particle's rim used to overlap another's): fluid_composite.frag
// differentiates this buffer to get normals, so a sub-gate depth ridge in
// history still lights as a visible outline. That is the "outlines stuck
// on screen for a few frames, then fade" bug - the ridge just decays at
// HISTORY_WEIGHT^n.
const float REJECT_DEPTH_DELTA = 0.05;

// Peak history weight, reached only on a still, interior pixel.
const float HISTORY_WEIGHT = 0.6;

// Motion (in pixels) below which history is fully trusted / above which it
// is dropped entirely. Ghosts only ever appear when something moved, and
// when something moved smoothing isn't worth the artifacts.
const float MOTION_FULL_TRUST_PX = 0.25;
const float MOTION_NO_TRUST_PX = 1.5;

void main()
{
    vec4 newSample = texture(newFrame, fragScreenUV);
    float newDepth = newSample.r;
    if (newDepth >= SENTINEL_THRESHOLD)
    {
        outAccumulated = vec4(SENTINEL_VALUE, 0.0, 0.0, 0.0);
        return;
    }

    vec2 texel = 1.0 / vec2(textureSize(newFrame, 0));

    // 3x3 neighborhood of THIS frame's result: min/max for clamping, and a
    // silhouette test. Any empty neighbor means this pixel is on a rim,
    // which is exactly where stuck outlines show up - no history there.
    vec4 nMin = newSample;
    vec4 nMax = newSample;
    bool silhouette = false;
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            vec4 n = texture(newFrame, fragScreenUV + vec2(dx, dy) * texel);
            if (n.r >= SENTINEL_THRESHOLD)
            {
                silhouette = true;
                continue;
            }
            nMin = min(nMin, n);
            nMax = max(nMax, n);
        }
    }
    if (silhouette)
    {
        outAccumulated = newSample;
        return;
    }

    vec2 motionVector = texture(motionTex, fragScreenUV).rg;
    vec2 reprojectedUV = fragScreenUV - motionVector;
    bool inBounds = all(greaterThanEqual(reprojectedUV, vec2(0.0))) && all(lessThanEqual(reprojectedUV, vec2(1.0)));
    vec4 oldSample = inBounds ? texture(history, reprojectedUV) : vec4(SENTINEL_VALUE, 0.0, 0.0, 0.0);

    if (oldSample.r >= SENTINEL_THRESHOLD)
    {
        outAccumulated = newSample;
        return;
    }

    // Neighborhood clamp (the standard TAA ghost killer): history is
    // forced into the range the current frame's own neighborhood spans, so
    // a stale ridge/step that the current frame doesn't contain can't
    // survive no matter how well reprojection matched.
    oldSample = clamp(oldSample, nMin, nMax);

    if (abs(newDepth - oldSample.r) > REJECT_DEPTH_DELTA)
    {
        outAccumulated = newSample;
        return;
    }

    float motionPx = length(motionVector / texel);
    float weight = HISTORY_WEIGHT * (1.0 - smoothstep(MOTION_FULL_TRUST_PX, MOTION_NO_TRUST_PX, motionPx));

    outAccumulated = mix(newSample, oldSample, weight);
}
