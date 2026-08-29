#version 450

// GPU-resident particle rendering: no vertex/instance buffer at all is
// bound for this draw call (see Renderer::RegisterParticleGPUSource /
// DrawFrame) - position comes straight from the compute pipeline's
// storage buffer, indexed by gl_InstanceIndex, with zero CPU readback.
// This is what makes the GPU port actually end-to-end resident: every
// other step (density/force/integrate/boundary) already ran without
// touching the CPU, and this is the step that would otherwise force a
// GPU->CPU->GPU round trip right before rendering, for no reason other
// than "that's how the old instanced-draw path worked."
//
// Reuses particle_point.frag unchanged - the fragment shader only cares
// about fragColor and gl_PointCoord, neither of which changes here.

layout(binding = 0) uniform CameraUBO
{
    mat4 view;
    mat4 proj;
    vec3 cameraPosition;
    vec3 lightDirection;
    vec3 lightColor;
    vec3 ambientColor;
    float particlePointSize;
} camera;

// Same buffer ParticleSystemGPU's compute pipelines write into (binding
// 0 there too, not a coincidence - this is set 1 here, camera stays set
// 0, so the two never collide despite sharing a binding NUMBER). Bound
// read-only from the vertex stage.
layout(std430, set = 1, binding = 0) readonly buffer PositionsBuffer
{
    vec4 positions[];
};

// Flat debug-viz color - see ParticleSystemGPU::BuildDrawColor. A single
// push constant rather than per-particle color: nothing in the SPH
// solver currently varies particle color (same as the CPU path's
// BuildDrawList, which also passes one shared colorOverride for every
// particle).
layout(push_constant) uniform ParticleColor
{
    vec4 color;
} particleColor;

layout(location = 0) out vec3 fragColor;

void main()
{
    vec3 worldPosition = positions[gl_InstanceIndex].xyz;

    // Matches particle_integrate.comp's void-kill sentinel (see that
    // shader's own comment) - a particle pinned there has left the
    // simulated world and is never coming back, so there's nothing
    // correct to draw for it. Forcing it outside the clip volume (w
    // component 0, so the x/y/z=2 values survive perspective divide as
    // genuinely out-of-bounds NDC coordinates) rather than just zeroing
    // gl_PointSize - a degenerate/clipped vertex is rejected by the
    // rasterizer before it costs any fragment work, which a zero-size
    // but still "on-screen" point isn't guaranteed to be for free on
    // every implementation. -1.0e7 threshold sits with a huge margin
    // above the actual -1.0e8 sentinel and just as huge a margin below
    // anything a real scene would ever place a particle at, so this
    // can't misfire on legitimate (if unusually fast-falling) particles.
    if (worldPosition.y < -1.0e7)
    {
        // Negative w fails the clip test (-w <= x <= w) unconditionally
        // for any real x/y/z, regardless of sign - clipping happens in
        // homogeneous space BEFORE the perspective divide, so unlike a
        // w=0 position (real divide-by-zero risk on some
        // implementations), this can't produce NaN/undefined rasterizer
        // behavior on any conformant implementation.
        gl_Position = vec4(0.0, 0.0, 0.0, -1.0);
        gl_PointSize = 0.0;
        fragColor = vec3(0.0);
        return;
    }

    gl_Position = camera.proj * camera.view * vec4(worldPosition, 1.0);
    gl_PointSize = camera.particlePointSize;
    fragColor = particleColor.color.rgb;
}
