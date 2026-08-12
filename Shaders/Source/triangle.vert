#version 450

layout(binding = 0) uniform CameraUBO
{
    mat4 view;
    mat4 proj;
    vec3 cameraPosition;
    vec3 lightDirection;
    vec3 lightColor;
    vec3 ambientColor;
    float particlePointSize; // unused by this shader - see RendererTypes.h's UniformBufferObject comment; kept declared here so this block's std140 layout stays byte-identical across every shader that binds it
} camera;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

// Per-instance attributes (binding 1, VK_VERTEX_INPUT_RATE_INSTANCE) - one
// set of these per INSTANCE, not per vertex, supplied by the instance
// buffer instead of a push constant. A mat4 needs four consecutive
// locations (2-5), one vec4 "column" each - that's a hardware vertex input
// limitation, not a stylistic choice. See InstanceData in RendererTypes.h.
layout(location = 2) in vec4 inModelCol0;
layout(location = 3) in vec4 inModelCol1;
layout(location = 4) in vec4 inModelCol2;
layout(location = 5) in vec4 inModelCol3;
layout(location = 6) in vec4 inColorOverride; // rgb = override color, a = 0.0 (off) or 1.0 (on)

// locations 7/8, not "next in sequence" - see Vertex::GetAttributeDescriptions
// in RendererTypes.h for why there are deliberate gaps between Vertex's own
// locations (0, 1) and these.
layout(location = 7) in vec2 inTexCoord;
layout(location = 8) in vec3 inNormal;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec3 fragNormalWorld;
layout(location = 3) out vec3 fragPositionWorld;

void main()
{
    mat4 model = mat4(inModelCol0, inModelCol1, inModelCol2, inModelCol3);

    vec4 worldPosition = model * vec4(inPosition, 1.0);
    gl_Position = camera.proj * camera.view * worldPosition;

    // inColorOverride.a is always exactly 0.0 or 1.0 (set from
    // ColorComponent presence in RenderSystem), so this mix() is a hard
    // switch, not a blend - cheaper and branch-free compared to an if.
    fragColor = mix(inColor, inColorOverride.rgb, inColorOverride.a);
    fragTexCoord = inTexCoord;
    fragPositionWorld = worldPosition.xyz;

    // Normal matrix = inverse-transpose of the model matrix's upper-left
    // 3x3 - the correct transform for normals under non-uniform scale
    // (plain model-matrix rotation is only correct when scale is uniform;
    // a squashed/stretched object needs this instead, or normals end up
    // tilted away from the actual surface). Computed here, per-vertex, on
    // the GPU rather than once per-instance on the CPU and added to
    // InstanceData - simpler and one less thing to keep in sync, at the
    // cost of redoing the same inverse() for every vertex of an instance
    // instead of once. Fine at Eden's current scale; worth revisiting
    // (precompute on CPU, add to InstanceData) if vertex-heavy meshes with
    // many instances ever make this measurably expensive.
    mat3 normalMatrix = transpose(inverse(mat3(model)));
    fragNormalWorld = normalize(normalMatrix * inNormal);
}
