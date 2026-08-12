#include "PhysicsSystem.h"

namespace Eden
{
    void PhysicsSystem::Step(Registry& registry, float fixedDeltaTime)
    {
        // See Registry::GetStorage's comment - resolved once here, not
        // once per entity. Same pattern as every other hot-path system
        // (SpinSystem, LifetimeSystem, RenderSystem::BuildDrawList).
        auto& transforms = registry.GetStorage<TransformComponent>();
        auto& bodies = registry.GetStorage<RigidBodyComponent>();

        for (Entity entity : registry.View<TransformComponent, RigidBodyComponent>())
        {
            auto& transform = transforms.Get(entity);
            auto& body = bodies.Get(entity);

            if (body.type == BodyType::Static)
            {
                // Never integrated. Zero out any velocity that might have
                // been set (by mistake, or before a body was reassigned
                // to Static) so nothing silently starts drifting the
                // moment something flips this back to Dynamic without
                // also clearing velocity.
                body.linearVelocity = glm::vec3(0.0f);
                body.angularVelocity = glm::vec3(0.0f);
                body.accumulatedForce = glm::vec3(0.0f);
                body.accumulatedTorque = glm::vec3(0.0f);
                continue;
            }

            if (body.type == BodyType::Dynamic)
            {
                if (!body.isSleeping)
                {
                    if (body.useGravity)
                    {
                        body.linearVelocity += gravity * fixedDeltaTime;
                    }

                    // F = ma -> a = F * inverseMass. Nothing pushes into
                    // accumulatedForce/Torque yet as of this step (no
                    // collision response exists), but wiring the accumulator
                    // into integration now means the impulse-resolution work
                    // later (Planning Notes, step 4) doesn't need to touch
                    // this function again - it just needs somewhere to write.
                    body.linearVelocity += body.accumulatedForce * body.inverseMass * fixedDeltaTime;
                    body.angularVelocity += body.accumulatedTorque * body.inverseMass * fixedDeltaTime;

                    // Crude proportional damping: v *= (1 - damping * dt).
                    // Clamped at 0 so a large damping*dt (e.g. a huge stalled
                    // frame despite the accumulator's step cap) can't flip
                    // velocity's sign and inject energy instead of removing
                    // it.
                    float linearFactor = glm::max(0.0f, 1.0f - body.linearDamping * fixedDeltaTime);
                    float angularFactor = glm::max(0.0f, 1.0f - body.angularDamping * fixedDeltaTime);
                    body.linearVelocity *= linearFactor;
                    body.angularVelocity *= angularFactor;

                    body.accumulatedForce = glm::vec3(0.0f);
                    body.accumulatedTorque = glm::vec3(0.0f);

                    // NOTE: sleep timer/threshold bookkeeping is NOT done
                    // here even though this is where velocity gets its
                    // final integration for the step - it has to happen
                    // AFTER CollisionSystem::Step resolves contacts (see
                    // CollisionSystem.cpp), not here. A resting body's
                    // velocity right at this point already has this
                    // step's gravity added back in and hasn't been
                    // cancelled by contact resolution yet, so checking
                    // "is it slow enough" here would almost never see a
                    // resting body as slow - it'd see the pre-resolution
                    // velocity spike every single step and never
                    // accumulate sleep time. This was a real bug in an
                    // earlier version of this function - documenting it
                    // so it doesn't get reintroduced.
                }
                // else: sleeping - frozen. Velocity is already zero (set
                // the moment it fell asleep, above), so gravity/forces/
                // damping are skipped entirely and the integration below
                // is a harmless no-op until CollisionSystem wakes it.
            }

            // Kinematic bodies skip straight to here - gravity, force
            // accumulation, and damping above are all Dynamic-only.
            // Kinematic velocity is set by whatever's driving it
            // (animation, AI, a moving-platform script); PhysicsSystem's
            // only job for them is to move the transform by it.
            transform.position += body.linearVelocity * fixedDeltaTime;
            transform.rotationDegrees += body.angularVelocity * fixedDeltaTime;
        }
    }
}
