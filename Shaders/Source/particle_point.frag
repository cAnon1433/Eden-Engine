#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

void main()
{
    // gl_PointCoord is [0,1] across the point sprite's square footprint -
    // shift to [-0.5, 0.5] and discard anything outside a unit circle, so
    // particles render as round dots instead of Vulkan's default square
    // point sprites. Comparing squared distance against 0.25 (0.5*0.5)
    // avoids a sqrt for what's ultimately just a radius check.
    vec2 centered = gl_PointCoord - vec2(0.5);
    if (dot(centered, centered) > 0.25)
    {
        discard;
    }

    // Flat color, no lighting - this is a debug/simulation visualization
    // for the particle system (see ParticleSystem.h's own framing of the
    // instanced-cube path this replaces), not lit scene geometry.
    outColor = vec4(fragColor, 1.0);
}
