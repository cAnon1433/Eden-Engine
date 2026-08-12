#pragma once

#include "../Registry.h"
#include "../Components/TransformComponent.h"
#include "../Components/RotationSpeedComponent.h"

namespace Eden
{
    class SpinSystem : public System
    {
    public:
        void Update(Registry& registry, float deltaTime) override
        {
            // See Registry::GetStorage's comment - resolved once here
            // instead of twice per entity, per frame.
            auto& transforms = registry.GetStorage<TransformComponent>();
            auto& spins = registry.GetStorage<RotationSpeedComponent>();

            for (Entity entity : registry.View<TransformComponent, RotationSpeedComponent>())
            {
                auto& transform = transforms.Get(entity);
                auto& spin = spins.Get(entity);
                transform.rotationDegrees.y += spin.degreesPerSecond * deltaTime;
            }
        }
    };
}
