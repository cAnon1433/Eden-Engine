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
    vec3 cameraRight;        // unused here; declared so the members after it land at the right offsets
    vec3 cameraUp;
    vec3 cameraForward;
    mat4 prevViewProj;
    mat4 invViewProj;        // inverse(proj * view), computed once per frame on the CPU - see main() for why that matters
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

    // clampedGrid <= sampleDims - 1 - 1e-4 and sampleDims >= 2 on every axis
    // (VoxelDims >= 1, SampleDims = VoxelDims + 1), so i0 <= dims - 2 and
    // i0 + 1 is always in range: the per-corner SampleIndexClamped() calls
    // this used to make (3 clamps each, 8 corners) were provably no-ops.
    // One base index plus constant strides gives the identical 8 reads.
    int strideY = sampleDims.x;
    int strideZ = sampleDims.x * sampleDims.y;
    uint base = densityOffset + uint(i0.x + i0.y * strideY + i0.z * strideZ);

    float c000 = density[base];
    float c100 = density[base + 1u];
    float c010 = density[base + uint(strideY)];
    float c110 = density[base + uint(strideY) + 1u];
    float c001 = density[base + uint(strideZ)];
    float c101 = density[base + uint(strideZ) + 1u];
    float c011 = density[base + uint(strideZ + strideY)];
    float c111 = density[base + uint(strideZ + strideY) + 1u];

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
    // Per-PIXEL cost matters enormously in this shader - it runs over the
    // whole screen. The original main() did, for every pixel:
    //   * inverse(camera.view) and inverse(camera.proj)  (two 4x4 inversions)
    //   * inverse(obj.invModel) per object (and again at the hit, plus
    //     transpose(inverse(mat3(...))) for the normal)
    // All of that is gone; none of it changes a single output pixel:
    //   * the inverse view-projection comes from the UBO (CPU, once a frame);
    //   * the local-space ray direction is deliberately NOT normalized, which
    //     makes the ray parameter t equal WORLD distance along the (normalized)
    //     world ray - so "world hit distance", the closest-hit comparison and
    //     the hit position need no model matrix at all. Sphere-tracing steps
    //     convert a local-space distance d to world t by dividing by the local
    //     direction's length;
    //   * for an affine model matrix, inverse(mat3(model)) == mat3(invModel),
    //     so the normal matrix is just transpose(mat3(invModel)).
    vec2 ndc = fragScreenUV * 2.0 - 1.0;
    vec4 farPoint4 = camera.invViewProj * vec4(ndc, 1.0, 1.0);
    vec3 farPoint = farPoint4.xyz / farPoint4.w;

    vec3 rayOrigin = camera.cameraPosition;
    vec3 rayDir = normalize(farPoint - rayOrigin);

    float closestT = 1e30;
    int hitObject = -1;
    vec3 hitLocalPos = vec3(0.0);

    // Loop over MAX_OBJECTS (the ObjectBuffer's fixed capacity) with an
    // early break past pc.objectCount, the number actually live this frame.
    for (int i = 0; i < MAX_OBJECTS; ++i)
    {
        if (i >= pc.objectCount)
        {
            break;
        }

        RaymarchObjectGPU obj = objects[i];

        vec3 localOrigin = (obj.invModel * vec4(rayOrigin, 1.0)).xyz;
        vec3 localDir = mat3(obj.invModel) * rayDir;  // NOT normalized - see the comment at the top of main()
        float localDirLength = length(localDir);

        float tMin, tMax; // world distances along the ray (thanks to the unnormalized localDir)
        if (!RayAabbLocal(localOrigin, localDir, obj.localMin, obj.localMax, tMin, tMax))
        {
            continue; // ray never enters this object's bounds - skip sphere-tracing it
        }

        // Closest-hit early-exit: this object's nearest possible surface is
        // already farther than the best hit found so far.
        if (hitObject >= 0 && tMin >= closestT)
        {
            continue;
        }

        float invLocalDirLength = 1.0 / localDirLength;
        float t = tMin;
        for (int step = 0; step < MAX_STEPS && t < tMax; ++step)
        {
            vec3 p = localOrigin + localDir * t;
            float d = SampleSignedDistanceLocal(obj.densityOffset, p, obj.fieldOffset, obj.localMax, obj.voxelSize, obj.sampleDims);

            if (d < HIT_EPSILON)
            {
                break;
            }

            // d is a distance in the object's local space; the ray moves
            // localDirLength local units per unit of world t.
            t += max(d, HIT_EPSILON * 0.5) * invLocalDirLength;
        }

        if (t < tMax && t < closestT)
        {
            closestT = t;
            hitObject = i;
            hitLocalPos = localOrigin + localDir * t;
        }
    }

    if (hitObject < 0)
    {
        discard; // no object hit at this pixel - let whatever rasterized underneath (terrain, clear color) show through
    }

    vec3 hitWorldPos = rayOrigin + rayDir * closestT;

    RaymarchObjectGPU hitObj = objects[hitObject];
    vec3 localNormal = SampleGradientLocal(hitObj.densityOffset, hitLocalPos, hitObj.fieldOffset, hitObj.localMax, hitObj.voxelSize, hitObj.sampleDims);
    mat3 normalMatrix = transpose(mat3(hitObj.invModel)); // == transpose(inverse(mat3(model))) for an affine model
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
    // this hit instead of one path always winning.
    vec4 clipHit = camera.proj * camera.view * vec4(hitWorldPos, 1.0);
    gl_FragDepth = clipHit.z / clipHit.w;
}
