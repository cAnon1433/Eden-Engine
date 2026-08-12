#pragma once

#include <glm/glm.hpp>

namespace Eden
{
    enum class ColliderShape
    {
        Sphere,
        Box,
        Capsule,
        Plane
    };

    // A collision shape attached to an entity. The actual analytic
    // distance/gradient math per shape lives in SDF.h, kept separate from
    // this component so pure math functions don't have to live inside a
    // struct definition - this is "SDF for collision only" per the
    // Planning Notes' open design question, not visual/deformable
    // geometry; the renderer never reads this.
    //
    // ROTATION: colliders ARE rotation-aware (an earlier version of this
    // deliberately wasn't - that limitation is gone, see SDF::WorldToLocal
    // and CollisionSystem.cpp's narrow-phase tests). Box-vs-Box uses a
    // real oriented-box SAT test; Capsule's axis and Box's support-point
    // calculations both use the entity's actual rotation. The one
    // remaining approximation is Box-vs-Capsule, which is still a
    // simplified closest-point method (documented at TestBoxCapsule in
    // CollisionSystem.cpp), now just a rotation-aware version of that
    // same approximation rather than an additionally-wrong one.
    struct ColliderComponent
    {
        ColliderShape shape = ColliderShape::Sphere;

        // Only the fields relevant to `shape` are meaningful. Plain
        // side-by-side fields rather than a union/std::variant - cheap to
        // store, trivial to inspect/edit in ImGui, matches the
        // keep-it-simple-until-proven-otherwise approach used elsewhere
        // (e.g. Registry::View returning vector<Entity> instead of a
        // custom iterator).
        float radius = 0.5f;           // Sphere, Capsule
        glm::vec3 halfExtents{ 0.5f }; // Box

        // Capsule stands along local +Y (matching Eden's Y-up
        // convention). halfHeight is half the distance between the two
        // hemisphere centers; radius is added on top at each end, so
        // total capsule height = 2 * (halfHeight + radius).
        float halfHeight = 0.5f;

        // Plane: infinite, passes through (TransformComponent::position +
        // rotated localOffset), oriented by rotation * this normal (so a
        // Plane collider rotates with its entity same as every other
        // shape - set this to (0,1,0) and rotate the entity to tilt the
        // plane, rather than baking the tilt into this field). Expected
        // normalized - SDF::Plane doesn't renormalize it for you.
        glm::vec3 planeNormal{ 0.0f, 1.0f, 0.0f };

        // Offsets the collider from TransformComponent::position, in the
        // entity's own local frame - rotates WITH the entity, same as a
        // normal parent-child offset would. Useful when a mesh's visual
        // origin isn't at its geometric center.
        glm::vec3 localOffset{ 0.0f };
    };
}
