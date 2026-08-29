#version 450

// Sphere-traces every live "raymarch object" (small, frequently-mutated
// SDF volumes - melt/carve/reform blobs, and per RaymarchSystem's ECS
// conversion work, ordinary props too) directly against their
// GPU-resident density fields, reusing the exact trilinear-sample +
// tetrahedron-gradient math VoxelSystemGPU::SampleSignedDistance/
// SampleGradient already do on CPU for physics (see those functions'
// comments in VoxelSystemGPU.cpp) - this is the same math, ported to
// GLSL, reading VoxelSystemGPU's shared density buffer marching cubes
// also consumes, just sampled per-pixel here instead of per-vertex-once.
//
// Runs as a fullscreen pass AFTER rasterized terrain has already been
// drawn into the same depth buffer this frame (see Renderer::DrawFrame -
// this pipeline is recorded after m_GraphicsPipeline/m_VoxelPipeline,
// same render pass, depth test ON, depth write ON) - gl_FragDepth is
// written per-hit so terrain correctly occludes/is-occluded-by
// raymarched objects instead of one always winning regardless of depth.
//
// No #include - GLSL has no such thing without extensions this project
// doesn't use (see particle_integrate.comp's comment on the same
// constraint); the sampling functions below are hand-duplicated from
// VoxelSystemGPU.cpp's CPU versions and must be kept in sync by hand if
// that math ever changes.

layout(binding = 0) uniform CameraUBO
{
    mat4 view;
    mat4 proj;
    vec3 cameraPosition;
    vec3 lightDirection;
    vec3 lightColor;
    vec3 ambientColor;
    float particlePointSize; // unused by this shader - see triangle.frag's identical comment
} camera;

// Must match kRaymarchMaxObjects in RaymarchTypes.h - sizes the
// ObjectBuffer array below (a small fixed-count buffer, one entry per
// object's metadata) and RaymarchPushConstants' effective upper bound
// on pc.objectCount.
#define MAX_OBJECTS 64

struct RaymarchObjectGPU
{
    mat4 invModel;        // world -> object-local, for ray transform into the field's own space
    vec3 localMin;         // object-local AABB min, centered on zero - see RaymarchTypes.h's comment on the center-position convention
    float voxelSize;
    vec3 localMax;         // object-local AABB max, centered on zero
    // This object's starting index (float ELEMENTS, not bytes) within
    // DensityBuffer below - see RaymarchTypes.h's RaymarchObjectGPU
    // comment. Previously an unused pad float in this struct slot;
    // occupies the exact same std430 position, so nothing else in this
    // layout moved.
    uint densityOffset;
    ivec3 sampleDims;      // SampleDims() - see VoxelVolumeDesc::SampleDims
    float pad1;
    vec3 tintColor;
    float pad2;
    vec3 fieldOffset;      // add to a local AABB-space position to get the density field's own (0,0,0)-origin sample-space position - see RaymarchTypes.h
    float pad3;
};

layout(set = 1, binding = 0, std430) readonly buffer ObjectBuffer
{
    RaymarchObjectGPU objects[MAX_OBJECTS];
};

// ONE shared density buffer, holding every registered volume's samples
// back-to-back (VoxelSystemGPU::m_SharedDensityBuffer) - each object in
// ObjectBuffer above finds its own region within this single array via
// densityOffset (see SampleDensityTrilinear below). This replaced an
// earlier per-object-buffer design (one storage-buffer descriptor per
// raymarch object, which hit MoltenVK's
// maxPerStageDescriptorStorageBuffers=31 ceiling at a low object count)
// and, after that, a true-bindless attempt (unbounded densityBuffers[]
// array via descriptor indexing) that MoltenVK rejected outright on
// this project's actual dev hardware (vkGetDescriptorSetLayoutSupport
// returned supported=VK_FALSE for the required flag combination,
// independent of array size - see RaymarchTypes.h's kRaymarchMaxObjects
// comment for the full history). One shared buffer sidesteps the
// descriptor-count problem entirely: this binding is always exactly ONE
// ordinary storage-buffer descriptor, regardless of how many raymarch
// objects exist - portable by construction, not a MoltenVK-specific fix.
layout(set = 1, binding = 1, std430) readonly buffer DensityBuffer
{
    float density[];
};

layout(push_constant) uniform PushConstants
{
    int objectCount;
    float nearPlane;
    float farPlane;
} pc;

layout(location = 0) in vec2 fragScreenUV;
layout(location = 0) out vec4 outColor;

const float SHININESS = 32.0;
const float SPECULAR_STRENGTH = 0.5;
const int MAX_STEPS = 96;
const float HIT_EPSILON = 0.001;

int SampleIndexClamped(ivec3 s, ivec3 dims)
{
    ivec3 c = clamp(s, ivec3(0), dims - ivec3(1));
    return c.x + c.y * dims.x + c.z * dims.x * dims.y;
}

// Mirrors VoxelSystemGPU::SampleDensityTrilinear exactly (same 8-corner
// fetch + trilinear mix) - densityOffset is this object's own starting
// index within the single shared DensityBuffer (see that binding's
// comment above), added to every local sample index before indexing
// density[]. Ordinary array indexing - no descriptor indexing/
// nonuniformEXT needed, since this always reads the same one bound
// buffer regardless of which object is being sampled (only the OFFSET
// varies per object, not which buffer/descriptor).
float SampleDensityTrilinear(uint densityOffset, vec3 localPos, float voxelSize, ivec3 sampleDims)
{
    vec3 gridPos = localPos / voxelSize;
    vec3 maxGrid = vec3(sampleDims) - vec3(1.0 + 1e-4);
    vec3 clampedGrid = clamp(gridPos, vec3(0.0), max(maxGrid, vec3(0.0)));

    ivec3 i0 = ivec3(floor(clampedGrid));
    vec3 frac = clampedGrid - vec3(i0);

    float c000 = density[densityOffset + SampleIndexClamped(i0 + ivec3(0,0,0), sampleDims)];
    float c100 = density[densityOffset + SampleIndexClamped(i0 + ivec3(1,0,0), sampleDims)];
    float c010 = density[densityOffset + SampleIndexClamped(i0 + ivec3(0,1,0), sampleDims)];
    float c110 = density[densityOffset + SampleIndexClamped(i0 + ivec3(1,1,0), sampleDims)];
    float c001 = density[densityOffset + SampleIndexClamped(i0 + ivec3(0,0,1), sampleDims)];
    float c101 = density[densityOffset + SampleIndexClamped(i0 + ivec3(1,0,1), sampleDims)];
    float c011 = density[densityOffset + SampleIndexClamped(i0 + ivec3(0,1,1), sampleDims)];
    float c111 = density[densityOffset + SampleIndexClamped(i0 + ivec3(1,1,1), sampleDims)];

    float c00 = mix(c000, c100, frac.x);
    float c10 = mix(c010, c110, frac.x);
    float c01 = mix(c001, c101, frac.x);
    float c11 = mix(c011, c111, frac.x);
    float c0 = mix(c00, c10, frac.y);
    float c1 = mix(c01, c11, frac.y);
    return mix(c0, c1, frac.z);
}

// Mirrors VoxelSystemGPU::SampleSignedDistance - clamps into the sample
// grid, adds back exact box-distance for query points outside the
// field's own bounds (see that function's comment for why). localPos is
// in the OBJECT's centered AABB space (matches localMin/localMax and
// RayAabbLocal); fieldOffset shifts it into the density field's own
// (0,0,0)-origin sample space before clamping/sampling - see
// RaymarchObjectGPU's fieldOffset comment for why these are different
// spaces now.
float SampleSignedDistanceLocal(uint densityOffset, vec3 localPos, vec3 fieldOffset, vec3 localMax, float voxelSize, ivec3 sampleDims)
{
    vec3 fieldPos = localPos + fieldOffset;
    vec3 fieldExtent = localMax + fieldOffset; // field-space upper bound, i.e. localMax shifted the same way fieldPos was
    vec3 clampedField = clamp(fieldPos, vec3(0.0), fieldExtent);
    float outsideDistance = length(fieldPos - clampedField);
    return SampleDensityTrilinear(densityOffset, clampedField, voxelSize, sampleDims) + outsideDistance;
}

// Tetrahedron-tap gradient, same 4-sample technique as
// VoxelSystemGPU::SampleGradient (Inigo Quilez's calcNormal trick).
vec3 SampleGradientLocal(uint densityOffset, vec3 localPos, vec3 fieldOffset, vec3 localMax, float voxelSize, ivec3 sampleDims)
{
    float epsilon = max(voxelSize, 1e-4);
    const vec3 d0 = vec3(1.0, -1.0, -1.0);
    const vec3 d1 = vec3(-1.0, -1.0, 1.0);
    const vec3 d2 = vec3(-1.0, 1.0, -1.0);
    const vec3 d3 = vec3(1.0, 1.0, 1.0);

    vec3 gradient =
        d0 * SampleSignedDistanceLocal(densityOffset, localPos + d0 * epsilon, fieldOffset, localMax, voxelSize, sampleDims) +
        d1 * SampleSignedDistanceLocal(densityOffset, localPos + d1 * epsilon, fieldOffset, localMax, voxelSize, sampleDims) +
        d2 * SampleSignedDistanceLocal(densityOffset, localPos + d2 * epsilon, fieldOffset, localMax, voxelSize, sampleDims) +
        d3 * SampleSignedDistanceLocal(densityOffset, localPos + d3 * epsilon, fieldOffset, localMax, voxelSize, sampleDims);

    float len = length(gradient);
    return len > 0.0 ? gradient / len : vec3(0.0, 1.0, 0.0);
}

// Ray-AABB slab test in object-local space - the rejection step: pixels
// whose ray never enters an object's bounds skip sphere-tracing that
// object entirely. tMin/tMax are the entry/exit distances along the ray
// (tMin clamped to >= 0 - a ray whose origin is already inside the box
// should start marching at t=0, not at a negative "behind the camera"
// entry point).
bool RayAabbLocal(vec3 ro, vec3 rd, vec3 boxMin, vec3 boxMax, out float tMin, out float tMax)
{
    vec3 invD = 1.0 / rd;
    vec3 t0 = (boxMin - ro) * invD;
    vec3 t1 = (boxMax - ro) * invD;
    vec3 tSmall = min(t0, t1);
    vec3 tBig = max(t0, t1);

    tMin = max(max(tSmall.x, tSmall.y), tSmall.z);
    tMax = min(min(tBig.x, tBig.y), tBig.z);
    tMin = max(tMin, 0.0);
    return tMax >= tMin;
}

void main()
{
    // Reconstruct a world-space camera ray from this pixel's clip-space
    // position - standard inverse-view-projection unprojection. Cheap
    // enough per-pixel at Eden's current object counts (see
    // RaymarchSystem's comment on why this isn't precomputed on the CPU
    // and passed in as a per-corner ray direction instead - worth
    // revisiting only if profiling actually shows this matters).
    vec2 ndc = fragScreenUV * 2.0 - 1.0;
    mat4 invView = inverse(camera.view);
    mat4 invProj = inverse(camera.proj);

    vec4 nearPoint4 = invView * invProj * vec4(ndc, 0.0, 1.0);
    vec4 farPoint4 = invView * invProj * vec4(ndc, 1.0, 1.0);
    vec3 nearPoint = nearPoint4.xyz / nearPoint4.w;
    vec3 farPoint = farPoint4.xyz / farPoint4.w;

    vec3 rayOrigin = camera.cameraPosition;
    vec3 rayDir = normalize(farPoint - nearPoint);

    float closestT = 1e30;
    int hitObject = -1;
    vec3 hitLocalPos = vec3(0.0);
    vec3 hitWorldPos = vec3(0.0);

    // Loop over MAX_OBJECTS (the ObjectBuffer's fixed capacity) with an
    // early break past pc.objectCount, the number actually live this
    // frame. DensityBuffer itself is not per-object - every object
    // reads from the SAME single bound buffer, using its own
    // obj.densityOffset to find its region within it (see
    // SampleDensityTrilinear/DensityBuffer's binding comment above) -
    // ordinary array indexing, no descriptor-varying behavior at all.
    for (int i = 0; i < MAX_OBJECTS; ++i)
    {
        if (i >= pc.objectCount)
        {
            break;
        }

        RaymarchObjectGPU obj = objects[i];

        vec4 localOrigin4 = obj.invModel * vec4(rayOrigin, 1.0);
        vec4 localDirEnd4 = obj.invModel * vec4(rayOrigin + rayDir, 1.0);
        vec3 localOrigin = localOrigin4.xyz;
        vec3 localDir = normalize(localDirEnd4.xyz - localOrigin);

        float tMin, tMax;
        if (!RayAabbLocal(localOrigin, localDir, obj.localMin, obj.localMax, tMin, tMax))
        {
            continue; // ray never enters this object's bounds - skip sphere-tracing it
        }

        // Closest-hit early-exit: convert this object's local-space
        // entry point back to a world-space distance and skip
        // sphere-tracing it if it can't possibly beat the closest hit
        // already found. Deliberately NOT comparing tMin directly
        // (object-local t units aren't guaranteed to match world-space
        // units unless the transform is translation-only - a scaled or
        // rotated raymarch object would silently break that
        // assumption), so this pays one extra matrix multiply per
        // object instead - correct for every transform, not just the
        // unscaled/unrotated ones every current spawn site happens to
        // use.
        if (hitObject >= 0)
        {
            vec3 localEntryPoint = localOrigin + localDir * tMin;
            vec4 worldEntry4 = inverse(obj.invModel) * vec4(localEntryPoint, 1.0);
            float worldEntryT = length(worldEntry4.xyz - rayOrigin);
            if (worldEntryT >= closestT)
            {
                continue; // this object's nearest possible surface is already farther than our best hit
            }
        }

        float t = tMin;
        for (int step = 0; step < MAX_STEPS && t < tMax; ++step)
        {
            vec3 p = localOrigin + localDir * t;
            float d = SampleSignedDistanceLocal(obj.densityOffset, p, obj.fieldOffset, obj.localMax, obj.voxelSize, obj.sampleDims);

            if (d < HIT_EPSILON)
            {
                // Hit found at this local-space t - world position and
                // closest-hit comparison against OTHER objects happens
                // below, once, outside the step loop (see the comment
                // there for why: avoids paying a matrix multiply on
                // every step just to maybe discard the result).
                break;
            }

            t += max(d, HIT_EPSILON * 0.5);
        }

        // Second pass only runs when the march above actually found a
        // hit inside this object - recompute worldHit properly here
        // rather than inside the loop, keeping the hot loop free of the
        // extra matrix multiply on every step.
        if (t < tMax)
        {
            vec3 localHitPos = localOrigin + localDir * t;
            mat4 model = inverse(obj.invModel);
            vec4 worldHit4 = model * vec4(localHitPos, 1.0);
            vec3 worldHit = worldHit4.xyz;
            float worldT = length(worldHit - rayOrigin);

            if (worldT < closestT)
            {
                closestT = worldT;
                hitObject = i;
                hitLocalPos = localHitPos;
                hitWorldPos = worldHit;
            }
        }
    }

    if (hitObject < 0)
    {
        discard; // no object hit at this pixel - let whatever rasterized underneath (terrain, clear color) show through
    }

    RaymarchObjectGPU hitObj = objects[hitObject];
    vec3 localNormal = SampleGradientLocal(hitObj.densityOffset, hitLocalPos, hitObj.fieldOffset, hitObj.localMax, hitObj.voxelSize, hitObj.sampleDims);
    // Normal matrix from the object's forward model matrix (inverse of
    // invModel, computed once here rather than re-inverting per step -
    // see the hit-refinement pass above for the same model matrix,
    // computed separately there since this scope can't see a variable
    // local to that earlier block).
    mat4 hitModel = inverse(hitObj.invModel);
    mat3 normalMatrix = transpose(inverse(mat3(hitModel)));
    vec3 N = normalize(normalMatrix * localNormal);

    vec3 L = normalize(-camera.lightDirection);
    vec3 V = normalize(camera.cameraPosition - hitWorldPos);
    vec3 H = normalize(L + V);

    float diffuseFactor = max(dot(N, L), 0.0);
    float specularFactor = diffuseFactor > 0.0 ? pow(max(dot(N, H), 0.0), SHININESS) : 0.0;

    vec3 ambient = camera.ambientColor;
    vec3 diffuse = diffuseFactor * camera.lightColor;
    vec3 specular = specularFactor * SPECULAR_STRENGTH * camera.lightColor;
    vec3 lit = hitObj.tintColor * (ambient + diffuse) + specular;

    outColor = vec4(lit, 1.0);

    // Write real depth so rasterized terrain (drawn earlier in the same
    // render pass, same depth buffer) correctly occludes/is-occluded-by
    // this hit instead of one path always winning. Reproject the world
    // hit through camera.proj*camera.view rather than deriving depth
    // from closestT directly - matches exactly what the rasterizer's
    // own gl_Position pipeline produces, so the two paths agree at the
    // boundary between a raymarched object and rasterized geometry.
    vec4 clipHit = camera.proj * camera.view * vec4(hitWorldPos, 1.0);
    gl_FragDepth = clipHit.z / clipHit.w;
}
