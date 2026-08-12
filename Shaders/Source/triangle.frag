#version 450

layout(binding = 0) uniform CameraUBO
{
    mat4 view;
    mat4 proj;
    vec3 cameraPosition;
    vec3 lightDirection;
    vec3 lightColor;
    vec3 ambientColor;
    float particlePointSize; // unused by this shader - see triangle.vert's identical comment
} camera;

layout(set = 1, binding = 0) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragNormalWorld;
layout(location = 3) in vec3 fragPositionWorld;

layout(location = 0) out vec4 outColor;

// No per-material control yet (that's a MaterialComponent-shaped feature,
// not built yet) - flat constants shared by everything Eden draws, same
// spirit as the single hardcoded light in Renderer::DrawFrame.
const float SHININESS = 32.0;
const float SPECULAR_STRENGTH = 0.5;

void main()
{
    vec3 albedoRgb = fragColor * texture(texSampler, fragTexCoord).rgb;

    vec3 N = normalize(fragNormalWorld);
    // camera.lightDirection is the direction the light TRAVELS (e.g.
    // pointing down from the sun toward the ground) - the vector FROM the
    // surface TOWARD the light for the diffuse dot product is the
    // opposite of that.
    vec3 L = normalize(-camera.lightDirection);
    vec3 V = normalize(camera.cameraPosition - fragPositionWorld);
    vec3 H = normalize(L + V); // Blinn-Phong halfway vector - cheaper and more stable than reflect()-based specular

    float diffuseFactor = max(dot(N, L), 0.0);
    float specularFactor = diffuseFactor > 0.0
        ? pow(max(dot(N, H), 0.0), SHININESS)
        : 0.0; // no specular highlight on faces already facing away from the light

    vec3 ambient = camera.ambientColor;
    vec3 diffuse = diffuseFactor * camera.lightColor;
    vec3 specular = specularFactor * SPECULAR_STRENGTH * camera.lightColor;

    vec3 lit = albedoRgb * (ambient + diffuse) + specular;
    outColor = vec4(lit, 1.0);
}
