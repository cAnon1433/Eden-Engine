#pragma once

namespace Eden
{
    // Marker + data component: any entity with both this and a
    // TransformComponent gets spun by SpinSystem. An entity with a
    // TransformComponent but no RotationSpeedComponent just sits still -
    // that's ECS composition doing its job, no inheritance/flags needed.
    struct RotationSpeedComponent
    {
        float degreesPerSecond = 45.0f;
    };
}
