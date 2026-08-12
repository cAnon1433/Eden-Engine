#pragma once

namespace Eden
{
    class Registry;

    // A System reads and/or mutates component data every frame. Register
    // one via Registry::RegisterSystem<T>(); Registry::UpdateSystems(dt)
    // calls every registered system's Update() in registration order.
    //
    // Rendering is deliberately NOT a System in this sense - see
    // Systems/RenderSystem.h, which reads Transform+Mesh and produces a
    // draw list rather than mutating ECS state, and is called directly
    // from the main loop instead of through UpdateSystems.
    class System
    {
    public:
        virtual ~System() = default;
        virtual void Update(Registry& registry, float deltaTime) = 0;
    };
}
