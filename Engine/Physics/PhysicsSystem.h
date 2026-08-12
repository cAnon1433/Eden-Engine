#pragma once

#include "../ECS/Registry.h"
#include "../ECS/Components/TransformComponent.h"
#include "RigidBodyComponent.h"

namespace Eden
{
    // Deliberately NOT a System (see ECS/System.h). System::Update runs at
    // whatever variable deltaTime the current frame happened to take -
    // physics must NOT do that. Integrating the same forces over a bigger
    // dt on a slow frame produces bigger, less stable position changes,
    // and the simulation stops being reproducible frame-to-frame. Call
    // Step() directly from main.cpp's fixed-timestep accumulator instead
    // of registering this via Registry::RegisterSystem/UpdateSystems -
    // see the accumulator loop added around registry.UpdateSystems() in
    // main.cpp.
    class PhysicsSystem
    {
    public:
        // Applied to every Dynamic body with useGravity == true, every
        // Step. Y-up, matching Eden's existing camera/transform convention
        // (TransformComponent::rotationDegrees.y is yaw).
        glm::vec3 gravity{ 0.0f, -9.81f, 0.0f };

        // Advances every entity with both TransformComponent and
        // RigidBodyComponent by exactly `fixedDeltaTime` seconds, using
        // semi-implicit (symplectic) Euler: velocity is updated from
        // acceleration FIRST, then position is updated from the NEW
        // velocity, not the old one. Unconditionally more stable than
        // explicit Euler for identical cost - the standard first-pass
        // choice (see "Eden Physics - Planning Notes", step 1).
        //
        // Static bodies are skipped entirely (never integrated). Kinematic
        // bodies are moved by their existing velocity but never receive
        // gravity, accumulated force/torque, or damping - something else
        // is responsible for setting their velocity. Dynamic bodies get
        // the full treatment: gravity, accumulated force/torque, damping,
        // then integration.
        void Step(Registry& registry, float fixedDeltaTime);
    };
}
