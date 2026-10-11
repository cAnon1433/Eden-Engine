#version 450

// Pass 1 (fragment half) - see fluid_depth.vert's file comment for the
// overview. For each fragment inside this point sprite's circle,
// reconstructs the actual point on the particle's sphere SURFACE (not
// just its center) and outputs the world-space distance from the camera
// to that surface point PLUS this particle's own diffusing color (R =
// distance, GBA = color - see the color attachment format comment in
// Renderer::CreateFluidSurfaceResources), sampled by fluid_blur.frag/
// fluid_composite.frag later. Also writes a real per-fragment
// gl_FragDepth so overlapping particles correctly occlude each other via
// this pass's own LOCAL depth attachment (Renderer::m_FluidDepthPassDS -
// never sampled afterward, exists only to make nearest-particle-wins
// work within this one pass, same reason any depth buffer exists).

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
    mat4 prevViewProj; // see UniformBufferObject::prevViewProj's comment (RendererTypes.h) - needed here for this pass's motion-vector output below
} camera;

layout(location = 0) in vec3 worldCenter;
layout(location = 1) in float radius;
layout(location = 2) in vec2 localCoord;
layout(location = 3) in vec3 particleColor;
layout(location = 4) in vec3 positionDelta; // see fluid_depth.vert's outPositionDelta comment
layout(location = 5) flat in float nearestNdcDepth; // see fluid_depth.vert's outNearestNdcDepth comment

// The opaque scene's depth buffer (meshes, voxel terrain, raymarched objects),
// rendered BEFORE the fluid passes - see Renderer::DrawFrame. Lets liquid be
// hidden by solids at the source instead of only at the final composite:
// particles behind a wall/terrain/object cost almost nothing here, and can no
// longer leak into the blur/thickness passes (the "x-ray" ghosts).
layout(set = 2, binding = 0) uniform sampler2D sceneDepth;

layout(push_constant) uniform FluidDepthFragPushConstants
{
    float radius;   // (vertex stage's copy - declared so the block layout matches)
    float invTargetW; // 1 / this pass's render target width  (fluid targets may be scaled down)
    float invTargetH;
    float pad;
} pc;

layout(location = 0) out vec4 outWorldDistanceAndColor;
// Second MRT output (see Renderer::m_FluidDepthPrepassRenderPass) - this
// fragment's OWN motion vector, computed below from the actual
// reconstructed surface point (surfacePos), not just the particle center.
// Not blurred/polished afterward - fluid_temporal.frag samples this
// directly. Discarded along with everything else outside the circle
// below (the `discard` applies to every output of this invocation), so
// this is only ever written for a pixel that's actually part of a
// particle's silhouette.
layout(location = 1) out vec2 outMotion;
// Third MRT output: this fragment's exact ANALYTIC sphere normal (world
// space), unblurred. fluid_composite.frag uses it instead of the
// blurred-depth normal for isolated particles/droplets, where there is no
// neighboring surface for a screen-space normal to be reconstructed from
// (a lone droplet is almost all silhouette, and the depth profile is
// deliberately flattened near the rim by edgeFade - fine inside a pool where
// overlap hides it, wrong as a visible droplet shape).
layout(location = 2) out vec4 outNormal;

// Without this qualifier, writing gl_FragDepth below disables early
// fragment testing entirely on most hardware (Metal/MoltenVK - this
// project's primary target - documents this explicitly: a fragment
// function writing depth loses early-Z unless it declares which
// direction the depth can move). That means EVERY fragment covered by
// EVERY particle's billboard quad runs this full sphere-reconstruction
// math, even fragments a closer particle will completely overdraw -
// cost scales with particleCount * overdraw * radius^2, which is
// exactly "gets worse with more particles and a bigger visual radius,
// independent of anything carve-related" - the pattern that pointed at
// this in the first place.
//
// depth_less asserts the depth THIS shader writes is never farther
// (greater, in Vulkan's 0=near/1=far convention) than the depth that
// would have been interpolated from the flat quad's own vertices. That
// still holds with the per-particle viewDir reconstruction below (see
// its own comment): the surface point is always pulled toward the
// camera along viewDir, which points from the camera toward this
// particle, so dot(viewDir, cameraForward) is positive for anything
// actually inside the view frustum (nothing behind the camera gets
// this far) - meaning the pull always reduces depth along the camera's
// own forward axis too, never increases it, smoothly approaching zero
// pull as heightUnits does (see its own comment - still always >= 0,
// the edge-fade multiplier added there doesn't change that). That's
// precisely the guarantee depth_less requires, which is what lets the
// GPU still early-reject a fragment whose flat-quad depth alone already
// fails against what's already in the depth buffer, without running
// this shader at all.
layout(depth_less) out float gl_FragDepth;

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

    // Scene occlusion, cheap early-out: even this sphere's nearest possible
    // point is behind the opaque scene at this pixel -> nothing of it can be
    // visible here. One texture fetch instead of the full reconstruction.
    vec2 screenUV = gl_FragCoord.xy * vec2(pc.invTargetW, pc.invTargetH);
    float sceneZ = texture(sceneDepth, screenUV).r;
    if (nearestNdcDepth > sceneZ)
    {
        discard;
    }

    // Height of the sphere surface above the local.xy plane, in sphere-
    // radius units (0 at the silhouette edge, 1 at dead center) - the
    // standard point-sprite-as-sphere reconstruction, unchanged by the
    // quad-vs-point-sprite switch.
    //
    // edgeFade tames a real numerical instability in the raw sqrt(1-r2)
    // formula: its derivative with respect to r2 diverges to infinity as
    // r2 approaches 1 (verified directly - at r2=0.999999 it's already
    // past 470, climbing unbounded from there). That means a SUB-PIXEL
    // difference in r2 between two adjacent screen pixels near the rim
    // - from ordinary rasterization, or from the tiniest camera motion -
    // produces a large, visible swing in reconstructed depth right at
    // the silhouette. The bilateral blur's own edge-preservation logic
    // (see fluid_blur.frag) then correctly does its job and refuses to
    // smooth across what looks like a real depth discontinuity - except
    // here it isn't one, it's reconstruction noise, so what survives is
    // a visible spiky fringe around every particle's rim. Multiplying by
    // a smoothstep that itself eases to zero at r2=1 fixes this
    // properly, not just approximately: two functions each individually
    // approaching zero at the same point, one with an unbounded
    // derivative and one with a ZERO derivative there (smoothstep's
    // defining property), multiply to a derivative that also goes to
    // zero (checked numerically: the same r2=0.999999 point that read
    // 470+ unfaded reads under 0.0000005 faded) - not a cosmetic
    // reduction, the instability is actually gone. Only reshapes the
    // outer ~8% of the radius (fade starts at r2=0.85, i.e. about 92% of
    // the way out) - the bulk of the sphere, and its silhouette size, is
    // unchanged.
    float edgeFade = 1.0 - smoothstep(0.85, 1.0, r2);
    float heightUnits = sqrt(1.0 - r2) * edgeFade;

    // viewDir, right, up: the SAME per-particle billboard basis
    // fluid_depth.vert built the quad from (see that shader's own
    // comment for the full reasoning and the Gram-Schmidt derivation) -
    // recomputed here rather than passed as a varying, since it's a
    // deterministic function of worldCenter/cameraPosition/cameraRight,
    // all already available, and recomputing guarantees this shader
    // can never drift out of sync with the vertex shader's own version.
    // Using the global camera.cameraRight/cameraUp here instead (as this
    // used to) would reconstruct local.x/local.y against a DIFFERENT
    // plane than the one the quad was actually built in for any
    // off-center particle - not just individually wrong the way the
    // viewDir-for-height fix above addresses, but inconsistent with the
    // quad's own geometry, which is its own source of edge error.
    vec3 viewDir = normalize(worldCenter - camera.cameraPosition);
    vec3 right = normalize(camera.cameraRight - viewDir * dot(camera.cameraRight, viewDir));
    vec3 up = cross(right, viewDir);

    vec3 surfacePos = worldCenter
        + right * (local.x * radius)
        + up * (local.y * radius)
        - viewDir * (heightUnits * radius);

    vec4 clipHit = camera.proj * camera.view * vec4(surfacePos, 1.0);
    gl_FragDepth = clipHit.z / clipHit.w;

    // Exact per-fragment test (the early-out above is only conservative):
    // this surface point is behind the scene -> hidden (also what makes
    // liquid properly intersect the ground/terrain instead of floating on it).
    if (gl_FragDepth > sceneZ)
    {
        discard;
    }

    // Linear view-space depth - distance from the camera to surfacePos
    // measured ALONG camera.cameraForward, not raw Euclidean distance
    // (length(surfacePos - cameraPosition)) like this used to output.
    // Those two quantities are only equal for a pixel exactly at screen
    // center; everywhere else the camera's rays diverge (perspective),
    // so Euclidean distance overstates how far along any OTHER pixel's
    // own ray the surface actually is - fluid_composite.frag's
    // WorldPointAtDistance reconstructs world position by walking a
    // per-pixel ray by this stored amount, and that reconstruction is
    // only correct if the stored value means "distance along the
    // camera's forward axis," which is what its own formula assumes
    // (see that function's comment). Passing Euclidean distance there
    // silently reconstructed the wrong world position for every pixel
    // off screen-center, worse the further off-center - and since which
    // pixels are "off-center" for a given particle changes continuously
    // as the camera rotates, that mismatch is what actually reads as
    // particles swimming, on top of (not instead of) the surface-
    // reconstruction fix above.
    // Analytic normal from the UNFADED sphere height (edgeFade only exists to
    // tame the depth reconstruction; the true normal keeps going round).
    outNormal = vec4(normalize(right * local.x + up * local.y - viewDir * sqrt(max(1.0 - r2, 0.0))), 1.0);

    float linearDepth = dot(surfacePos - camera.cameraPosition, camera.cameraForward);
    outWorldDistanceAndColor = vec4(linearDepth, particleColor);

    // Motion vector for THIS surface point specifically, not just the
    // particle's center (see fluid_depth.vert's outPositionDelta comment
    // for why that distinction matters). previousSurfacePos is surfacePos
    // shifted by the same rigid-body positionDelta the center used - valid
    // because surfacePos is already worldCenter plus a fixed local offset
    // (right/up/viewDir * local-space coordinates), and a sphere's own
    // shape doesn't change as it translates, so shifting the already-
    // reconstructed surface point by the center's exact displacement is
    // equivalent to reconstructing it fresh around the previous center.
    // Reusing clipHit (already computed above, THIS frame's projection of
    // surfacePos) for the current UV instead of recomputing it is
    // deliberate too - keeps both terms built from the same projection
    // path.
    vec3 previousSurfacePos = surfacePos - positionDelta;
    vec4 previousClip = camera.prevViewProj * vec4(previousSurfacePos, 1.0);

    // Same UV convention as raymarch.vert (clipPos * 0.5 + 0.5, Y-down) -
    // matches what fluid_temporal.frag's fragScreenUV already uses, so
    // reprojectedUV = fragScreenUV - outMotion needs no sign flip there.
    vec2 currentUV = (clipHit.xy / clipHit.w) * 0.5 + 0.5;
    vec2 previousUV = (previousClip.xy / previousClip.w) * 0.5 + 0.5;
    outMotion = currentUV - previousUV;
}
