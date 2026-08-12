#pragma once

#include "ColliderComponent.h"
#include "../ECS/Components/TransformComponent.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace Eden::SDF
{
    // Builds the same rotation (Y then X then Z, matching
    // TransformComponent::GetModelMatrix exactly) as a plain 3x3 matrix -
    // no translation/scale, just orientation. Used everywhere a collider
    // needs to know which way its entity is actually facing: WorldToLocal
    // below, and the OBB/rotated-axis narrow-phase tests in
    // CollisionSystem.cpp. Keeping this in one place means physics and
    // rendering can never quietly disagree about which way "rotated" means.
    inline glm::mat3 RotationMatrixFromDegrees(const glm::vec3& rotationDegrees)
    {
        glm::mat4 r(1.0f);
        r = glm::rotate(r, glm::radians(rotationDegrees.y), glm::vec3(0.0f, 1.0f, 0.0f));
        r = glm::rotate(r, glm::radians(rotationDegrees.x), glm::vec3(1.0f, 0.0f, 0.0f));
        r = glm::rotate(r, glm::radians(rotationDegrees.z), glm::vec3(0.0f, 0.0f, 1.0f));
        return glm::mat3(r);
    }
    // Every Distance function below takes a point already in the SHAPE'S
    // LOCAL SPACE (relative to the shape's own center) - see WorldToLocal.
    // Standard SDF sign convention: negative = inside, zero = on the
    // surface, positive = outside. Distance is always exact (these are
    // real distance fields, not just "inside/outside" tests) since narrow
    // phase needs actual penetration depth, not just a boolean.

    inline float Sphere(const glm::vec3& localPoint, float radius)
    {
        return glm::length(localPoint) - radius;
    }

    inline float Box(const glm::vec3& localPoint, const glm::vec3& halfExtents)
    {
        glm::vec3 q = glm::abs(localPoint) - halfExtents;
        float outsideDistance = glm::length(glm::max(q, glm::vec3(0.0f)));
        float insideDistance = glm::min(glm::max(q.x, glm::max(q.y, q.z)), 0.0f);
        return outsideDistance + insideDistance;
    }

    // Capsule along local +Y: two radius-`radius` hemispheres centered at
    // (0, +halfHeight, 0) and (0, -halfHeight, 0), joined by a cylinder.
    inline float Capsule(const glm::vec3& localPoint, float halfHeight, float radius)
    {
        glm::vec3 p = localPoint;
        p.y -= glm::clamp(p.y, -halfHeight, halfHeight);
        return glm::length(p) - radius;
    }

    // Infinite plane through the shape's local origin, oriented by
    // `normal` (expected normalized).
    inline float Plane(const glm::vec3& localPoint, const glm::vec3& normal)
    {
        return glm::dot(localPoint, normal);
    }

    // Dispatches to the right Distance function above for `collider`.
    inline float Distance(const ColliderComponent& collider, const glm::vec3& localPoint)
    {
        switch (collider.shape)
        {
            case ColliderShape::Sphere:  return Sphere(localPoint, collider.radius);
            case ColliderShape::Box:     return Box(localPoint, collider.halfExtents);
            case ColliderShape::Capsule: return Capsule(localPoint, collider.halfHeight, collider.radius);
            case ColliderShape::Plane:   return Plane(localPoint, collider.planeNormal);
        }
        return 0.0f; // unreachable - silences -Wreturn-type on the enum switch above
    }

    // Central-difference numerical gradient. Deliberate simplification:
    // Sphere and Plane get exact analytic normals below (trivial closed
    // forms), but Box and Capsule fall back to this instead of hand-
    // deriving their own closed-form gradients (the box one in
    // particular has real case-splitting at edges/corners). This is
    // slightly more expensive and only approximate right at sharp
    // features (a box corner), but it's called once per narrow-phase
    // contact, not per-vertex or per-entity-per-frame, so the cost is
    // negligible - flag if profiling ever says otherwise.
    template<typename DistanceFn>
    glm::vec3 NumericalGradient(const glm::vec3& localPoint, DistanceFn distanceFn, float epsilon = 0.0001f)
    {
        glm::vec3 dx(epsilon, 0.0f, 0.0f);
        glm::vec3 dy(0.0f, epsilon, 0.0f);
        glm::vec3 dz(0.0f, 0.0f, epsilon);

        glm::vec3 gradient
        {
            distanceFn(localPoint + dx) - distanceFn(localPoint - dx),
            distanceFn(localPoint + dy) - distanceFn(localPoint - dy),
            distanceFn(localPoint + dz) - distanceFn(localPoint - dz)
        };

        float length = glm::length(gradient);
        return length > 0.0f ? gradient / length : glm::vec3(0.0f, 1.0f, 0.0f);
    }

    // Surface normal at `localPoint` (need not be exactly ON the surface -
    // the gradient of a signed distance field is defined, and meaningful,
    // everywhere, which is exactly why narrow-phase collision uses SDFs
    // instead of pure boolean intersection tests).
    //
    // Returned in `collider`'s LOCAL (unrotated) space, same as
    // `localPoint` - callers that need this in world space (basically
    // everyone doing actual physics response) must rotate it back with
    // the same RotationMatrixFromDegrees(transform.rotationDegrees) used
    // to produce localPoint via WorldToLocal below. Distance() doesn't
    // have this problem (a scalar distance is the same regardless of
    // orientation); only this directional one does.
    inline glm::vec3 Normal(const ColliderComponent& collider, const glm::vec3& localPoint)
    {
        switch (collider.shape)
        {
            case ColliderShape::Sphere:
            {
                float length = glm::length(localPoint);
                return length > 0.0f ? localPoint / length : glm::vec3(0.0f, 1.0f, 0.0f);
            }
            case ColliderShape::Plane:
                return collider.planeNormal;
            case ColliderShape::Box:
                return NumericalGradient(localPoint, [&collider](const glm::vec3& p) { return Box(p, collider.halfExtents); });
            case ColliderShape::Capsule:
                return NumericalGradient(localPoint, [&collider](const glm::vec3& p) { return Capsule(p, collider.halfHeight, collider.radius); });
        }
        return glm::vec3(0.0f, 1.0f, 0.0f); // unreachable
    }

    // Converts a world-space point into `collider`'s local space, given
    // the TransformComponent it's attached to - fully rotation-aware (an
    // earlier version deliberately skipped rotationDegrees; that
    // limitation is gone). `localOffset` is interpreted in the entity's
    // own local frame and rotates WITH it, same as any normal parent-
    // child offset would - so the collider's world center is `position +
    // rotation * localOffset`, not just `position + localOffset`.
    //
    // Deliberately does NOT touch transform.scale - see ScaledCollider()
    // below for why, and for how scale actually gets applied. An earlier
    // version of this function instead divided the returned point by
    // scale directly, which looked correct (it does give an exact
    // surface/zero-crossing location) but was actually wrong: it also
    // silently shrinks Distance()'s returned MAGNITUDE by roughly
    // 1/scale, and every caller compares that magnitude against WORLD-
    // unit thresholds (boundaryRadius, sweep step sizes, kSurfaceEpsilon)
    // - so those thresholds become far too generous relative to the
    // shrunk distance. Caught by direct testing before it shipped (a
    // particle meant to rest exactly at a scaled box's true surface
    // settled about 0.15 world units too far away instead) - the fix
    // below keeps this function computing exact world-unit distances
    // throughout, same as it always did.
    inline glm::vec3 WorldToLocal(const glm::vec3& worldPoint, const TransformComponent& transform, const ColliderComponent& collider)
    {
        glm::mat3 rotation = RotationMatrixFromDegrees(transform.rotationDegrees);
        glm::vec3 worldColliderCenter = transform.position + rotation * collider.localOffset;
        // Inverse of an orthonormal rotation matrix is its transpose -
        // cheaper than a general matrix inverse and always exact here.
        return glm::transpose(rotation) * (worldPoint - worldColliderCenter);
    }

    // Returns a COPY of `collider` with its own shape parameters
    // (halfExtents/radius/halfHeight) pre-multiplied by `scale`, for
    // feeding into Distance()/Normal() alongside a point produced by the
    // scale-FREE WorldToLocal above - this is what actually makes
    // collision respect TransformComponent::scale, without disturbing
    // WorldToLocal's world-unit-exact distance guarantee (see its
    // comment above). This mirrors exactly how CollisionSystem.cpp's own
    // narrow-phase tests (TestBoxBox, TestCapsulePlane, etc.) already
    // apply scale - to the shape's own dimensions, not to the query
    // point - so this is the SAME fix, just packaged for the generic-SDF
    // callers (TestSphereVsSDFShape, and every particle boundary check)
    // that go through Distance()/Normal() directly instead of hand-
    // rolling their own shape-specific math.
    //
    // halfExtents scales per-axis (exact for any scale, since Box's SDF
    // is already axis-aligned and per-axis half-extents are exactly what
    // it expects). radius and halfHeight use the same isotropic-average
    // approximation used throughout CollisionSystem.cpp for round shapes
    // under anisotropic scale (exact under uniform scale, a reasonable
    // approximation otherwise - a non-uniformly-scaled sphere is
    // genuinely an ellipsoid, which no single radius can represent
    // exactly). Guards zero/degenerate scale components by treating them
    // as 1, same reasoning as everywhere else scale is multiplied by in
    // this codebase - a 0 shouldn't collapse a collider to nothing or
    // produce NaN downstream.
    inline ColliderComponent ScaledCollider(const ColliderComponent& collider, const glm::vec3& scale)
    {
        glm::vec3 safeScale(
            scale.x != 0.0f ? scale.x : 1.0f,
            scale.y != 0.0f ? scale.y : 1.0f,
            scale.z != 0.0f ? scale.z : 1.0f);

        ColliderComponent result = collider;
        result.halfExtents = collider.halfExtents * safeScale;
        result.radius = collider.radius * (safeScale.x + safeScale.y + safeScale.z) / 3.0f;
        result.halfHeight = collider.halfHeight * safeScale.y;
        return result;
    }
}
