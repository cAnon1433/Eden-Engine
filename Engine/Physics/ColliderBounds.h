#pragma once

#include "ColliderComponent.h"
#include "SDF.h"

#include <glm/glm.hpp>
#include <cmath>

namespace Eden
{
    // Conservative WORLD-space AABB half-extents for a collider - shared
    // by CollisionSystem.cpp's broad phase (grid-bucketing colliders for
    // narrow-phase pair generation) and ParticleSystemGPU.cpp's collider
    // cell mask (see that class's own comment on why GPU particle
    // collision needs the same kind of spatial index). Originally lived
    // only in CollisionSystem.cpp as a private helper; pulled out here
    // when the particle system needed the identical computation rather
    // than duplicate it - two independent implementations of "world AABB
    // of a rotated OBB/capsule" is exactly the kind of thing that quietly
    // drifts out of sync over time, and unlike the CPU/GPU SDF math
    // duplication elsewhere in this project (unavoidable - GLSL can't
    // #include C++ headers), there's no language barrier forcing this
    // one to be duplicated too.
    //
    // Doesn't need to be tight, just conservative - never smaller than
    // the shape's true rotated bounds. Sphere is rotation-invariant (a
    // rotated sphere is the same sphere) so its local half-extents ARE
    // its world ones; Box and Capsule need the actual rotation to bound
    // correctly - a rotated box's world AABB is bigger than its own
    // half-extents, and using the un-rotated bound here would let a
    // broad phase miss cells/cells a tilted shape actually overlaps,
    // silently dropping candidate pairs.
    //
    // `scale` applied FIRST (to collider.radius/halfExtents/halfHeight),
    // before rotation - matching every other scale fix in this project,
    // and matching TransformComponent::GetModelMatrix's own
    // translate*rotate*scale order. Without this, a scaled-up collider's
    // broad-phase AABB would stay authored-size, which could make a
    // broad phase reject a pair the (correctly scaled) narrow phase would
    // actually consider overlapping.
    inline glm::vec3 ColliderWorldAabbHalfExtents(const ColliderComponent& collider, const glm::vec3& rotationDegrees, const glm::vec3& scale)
    {
        switch (collider.shape)
        {
            case ColliderShape::Sphere:
                // Isotropic shape under anisotropic scale - same
                // averaging approximation used for capsule radius below
                // (exact under uniform scale).
                return glm::vec3(collider.radius * (scale.x + scale.y + scale.z) / 3.0f);

            case ColliderShape::Box:
            {
                // Standard OBB -> AABB formula: the world half-extent
                // along each world axis is the sum, over the box's own
                // local axes, of |that axis's component along the world
                // axis| * the box's half-extent along that local axis.
                // Equivalent to transforming the half-extent vector by
                // the absolute value of the rotation matrix.
                glm::mat3 rotation = SDF::RotationMatrixFromDegrees(rotationDegrees);
                glm::vec3 scaledHalf = collider.halfExtents * scale;
                glm::vec3 worldHalf(0.0f);
                for (int worldAxis = 0; worldAxis < 3; ++worldAxis)
                {
                    worldHalf[worldAxis] =
                        std::abs(rotation[0][worldAxis]) * scaledHalf.x +
                        std::abs(rotation[1][worldAxis]) * scaledHalf.y +
                        std::abs(rotation[2][worldAxis]) * scaledHalf.z;
                }
                return worldHalf;
            }

            case ColliderShape::Capsule:
            {
                // A capsule's world AABB is its rotated axis (scaled by
                // halfHeight) plus radius padding in every direction -
                // bounds the two hemisphere caps wherever the axis
                // actually points, not just straight up. halfHeight
                // scales along local Y (matching TestCapsulePlane);
                // radius uses the same isotropic X/Z-average
                // approximation as above.
                glm::mat3 rotation = SDF::RotationMatrixFromDegrees(rotationDegrees);
                glm::vec3 axis = rotation * glm::vec3(0.0f, 1.0f, 0.0f);
                float scaledHalfHeight = collider.halfHeight * scale.y;
                float scaledRadius = collider.radius * 0.5f * (scale.x + scale.z);
                return glm::abs(axis) * scaledHalfHeight + glm::vec3(scaledRadius);
            }

            case ColliderShape::Plane:
                return glm::vec3(0.0f); // unused - planes bypass grid-based broad phases entirely, see callers

            case ColliderShape::Voxel:
                // collider.halfExtents holds the volume's own local AABB
                // half-extents (see ColliderComponent's comment),
                // computed once at registration since these volumes
                // never rotate or resize - no rotation step needed here
                // unlike Box, this is already world-aligned.
                return collider.halfExtents;
        }
        return glm::vec3(0.5f);
    }
}
