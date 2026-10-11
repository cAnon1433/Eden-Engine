#version 450

// Pass 1 of the screen-space fluid reconstruction (see fluid_composite.frag
// for the overview and Renderer::InitFluidSurfacePass's comment for the
// full pipeline). Reads positions straight from the same GPU-resident
// storage buffer particle_point_gpu.vert already reads (see
// RegisterParticleGPUSource - this pipeline reuses that exact
// m_ParticleGPUStorageSet/m_ParticleGPUSetLayout rather than a second
// registration path), indexed by gl_InstanceIndex, same as that shader.
//
// Draws each particle as a camera-facing BILLBOARD QUAD (6 vertices,
// gl_VertexIndex-generated - no vertex buffer, same "no vertex buffer
// needed" trick raymarch.vert's fullscreen triangle already uses, just a
// 4-corner quad instead of a 3-corner fullscreen shape) rather than a
// GL_POINTS point sprite sized via gl_PointSize. This was a real bug in
// the first version of this pass: gl_PointSize is clamped by
// VkPhysicalDeviceLimits::pointSizeRange, and the computed size was
// evidently saturating that clamp across the entire visual-radius slider
// range, making every particle render at the same (wrong, oversized)
// pixel footprint regardless of the radius control - and, downstream of
// that, keeping neighboring particles' depths too far apart for
// fluid_blur.frag's bilateral weight to ever fuse them. Billboard quads
// are ordinary rasterized triangles sized through the normal perspective
// projection, exactly like every other draw call in this renderer - no
// device-specific size ceiling applies to them at all.

layout(binding = 0) uniform CameraUBO
{
    mat4 view;
    mat4 proj;
    vec3 cameraPosition;
    vec3 lightDirection;
    vec3 lightColor;
    vec3 ambientColor;
    float particlePointSize; // unused here
    vec3 cameraRight;
    vec3 cameraUp;
    vec3 cameraForward;
    mat4 prevViewProj; // see UniformBufferObject::prevViewProj's comment (RendererTypes.h)
} camera;

layout(std430, set = 1, binding = 0) readonly buffer PositionBuffer
{
    vec4 positions[];
};

// Per-particle color (Chemical ID base build - see
// ParticleSystemGPU::ChemicalTable's comment). Set 1 binding 1, added
// alongside PositionBuffer's existing binding 0 by
// Renderer::RegisterParticleGPUSource - this is the SAME diffusing color
// buffer particle_force.comp/particle_integrate.comp read and mutate
// each substep, not a static per-chemical lookup: two particles that
// started as different chemicals can end up visually similar colors
// after enough diffusion, which is the whole point (see
// fluid_composite.frag for where this actually gets displayed).
layout(std430, set = 1, binding = 1) readonly buffer ColorBuffer
{
    vec4 colors[];
};

// Set 1 binding 2, added alongside Position/ColorBuffer by
// Renderer::RegisterParticleGPUSource - Renderer's own snapshot of the
// position buffer AS OF THE END OF THE PREVIOUS RENDERED FRAME (see
// Renderer.h's m_ParticlePreviousPositions comment for why this replaced
// a velocity*deltaTime estimate: ParticleSystemGPU steps on a fixed-
// timestep accumulator, 0 to several physics steps per RENDERED frame,
// so a render frame's wall-clock deltaTime routinely does not match how
// far the simulation actually advanced - velocity*deltaTime could show
// motion on a frame where the particle didn't actually move at all).
// This buffer holds the real previous position, zero extrapolation.
layout(std430, set = 1, binding = 2) readonly buffer PreviousPositionBuffer
{
    vec4 previousPositions[];
};

layout(push_constant) uniform FluidDepthPushConstants
{
    float radius;
} pc;

layout(location = 0) out vec3 outWorldCenter;
layout(location = 1) out float outRadius;
layout(location = 2) out vec2 outLocalCoord;
layout(location = 3) out vec3 outColor;
// This particle's exact displacement since last frame (current minus
// previous position - NOT scaled by any deltaTime, since
// PreviousPositionBuffer already IS "wherever this particle actually
// was last render", however much simulation time that represents).
// fluid_depth.frag uses this to compute a per-FRAGMENT motion vector
// from the actual reconstructed surface point, not just this particle's
// center - see that shader's outMotion assignment for why per-fragment
// matters (a particle moving toward/away from the camera changes its
// screen footprint SIZE, which a single shared vector can't represent,
// worst exactly at the silhouette edge). Same value at all 6 vertices of
// one particle's quad, so interpolation introduces no error.
layout(location = 4) out vec3 outPositionDelta;

// NDC depth of the NEAREST point this particle's sphere could possibly
// occupy (its center pulled a full radius toward the camera) - flat, same
// for all 6 vertices. fluid_depth.frag compares it with the scene depth
// buffer: if even that nearest point is behind something opaque, the whole
// particle is hidden and the fragment bails out before any sphere math.
layout(location = 5) flat out float outNearestNdcDepth;

// Two triangles, CCW, quad corners in [-1,1] local billboard space -
// local.x follows this particle's own `right` (see main() below), local.y
// follows its own `up` (both "positive" in the intuitive sense - unlike
// gl_PointCoord, which is Y-down, this needs no sign flip when
// fluid_depth.frag reconstructs the same point).
const vec2 CORNERS[6] = vec2[6](
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
    vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0)
);

void main()
{
    vec3 worldPos = positions[gl_InstanceIndex].xyz;

    // Same void-kill sentinel check as particle_point_gpu.vert - a
    // particle parked far below the world (SimParamsGPU::voidKillY) must
    // stay excluded from the fluid surface too.
    if (worldPos.y < -1.0e8)
    {
        gl_Position = vec4(1e9, 1e9, 1e9, 1.0);
        outWorldCenter = worldPos;
        outRadius = 0.0;
        outLocalCoord = vec2(0.0);
        outColor = vec3(1.0);
        outPositionDelta = vec3(0.0);
        outNearestNdcDepth = 2.0; // behind everything - never matters, the quad is off-screen
        return;
    }

    vec2 corner = CORNERS[gl_VertexIndex];

    // Per-particle billboard basis, not the screen-wide camera.cameraRight/
    // cameraUp - a global basis means the quad's own plane is parallel
    // to the screen-center's view plane for EVERY particle, regardless
    // of where that particle actually sits on screen. For anything off
    // dead-center, that's not quite the plane a true camera-facing
    // billboard should use, which makes the reconstructed silhouette
    // subtly wrong - and specifically reads as swimming as the camera
    // rotates, since a given particle's off-center angle keeps changing
    // as you turn, rather than being a fixed, unnoticed error.
    //
    // Gram-Schmidt: take the EXISTING, already-stable camera.cameraRight
    // and project out whatever component of it points along viewDir,
    // leaving a vector that's exactly perpendicular to viewDir and as
    // close as possible (minimal rotation) to the screen's own right
    // axis - no new degenerate case beyond "a particle sits exactly
    // along the camera's own local X axis," which doesn't happen at
    // realistic FOVs. up is then derived the SAME way Camera::
    // UpdateVectors computes the camera's own Up from Right and Front
    // (see Camera.cpp: right = cross(forward, someUp), up =
    // cross(right, forward)) - just with viewDir standing in for
    // forward, so the handedness/sign convention matches exactly, not
    // just approximately. fluid_depth.frag computes this IDENTICAL
    // basis itself (from the same worldCenter/cameraPosition/
    // cameraRight, so it's guaranteed to agree) rather than receiving it
    // as a varying - see that shader's own comment.
    vec3 viewDir = normalize(worldPos - camera.cameraPosition);
    vec3 right = normalize(camera.cameraRight - viewDir * dot(camera.cameraRight, viewDir));
    vec3 up = cross(right, viewDir);

    vec3 cornerWorldPos = worldPos
        + right * (corner.x * pc.radius)
        + up * (corner.y * pc.radius);

    gl_Position = camera.proj * camera.view * vec4(cornerWorldPos, 1.0);
    outWorldCenter = worldPos;
    outRadius = pc.radius;
    outLocalCoord = corner;
    outColor = colors[gl_InstanceIndex].rgb;
    {
        vec4 nearestClip = camera.proj * camera.view * vec4(worldPos - viewDir * pc.radius, 1.0);
        // A particle straddling the camera plane has no meaningful NDC depth -
        // report "in front of everything" so it is never culled by mistake.
        outNearestNdcDepth = nearestClip.w > 1.0e-4 ? clamp(nearestClip.z / nearestClip.w, -1.0, 2.0) : -1.0;
    }
    outPositionDelta = worldPos - previousPositions[gl_InstanceIndex].xyz; // see outPositionDelta's own comment - the actual motion vector is computed per-fragment in fluid_depth.frag
}
