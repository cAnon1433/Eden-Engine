#pragma once

#include "Camera.h"

#include <glm/glm.hpp>
#include <array>

namespace Eden
{
    // Six-plane view frustum. Used for coarse visibility culling only -
    // like CollisionSystem's broad phase, this doesn't need to be exact,
    // just conservative (never culls something that's actually at least
    // partially visible).
    //
    // Planes are stored as (a, b, c, d) with the INSIDE half-space being
    // ax + by + cz + d >= 0 - a point/sphere is inside the frustum only
    // if that holds for all six planes simultaneously.
    class Frustum
    {
    public:
        // `viewProjection` should be projection * view (see
        // Renderer::GetViewProjectionMatrix) - column-vector convention,
        // matching glm's default and every other matrix in this engine.
        //
        // Only the four SIDE planes (left/right/top/bottom) come from
        // `viewProjection` here, via the standard Gribb/Hartmann row-
        // combination technique - those four are convention-independent
        // (x/w and y/w land in [-1, 1] in both OpenGL- and Vulkan-style
        // clip space, so the row3+/-row0 and row3+/-row1 combinations are
        // correct either way). Near/far deliberately do NOT come from
        // the matrix: that pair DOES depend on whether the projection
        // targets OpenGL's [-1, 1] NDC depth range or Vulkan's native
        // [0, 1] one, and this codebase doesn't visibly force either
        // (no GLM_FORCE_DEPTH_ZERO_TO_ONE, no vertex-shader remap - see
        // Camera::GetProjectionMatrix's own comment, which only mentions
        // flipping Y, not touching Z), so which one is actually in
        // effect isn't something to silently assume. `nearPlane`/
        // `farPlane` are Camera's own already-correct scalar distances
        // instead - a directly-constructed plane from a known distance
        // along a known forward vector needs no assumption about matrix
        // conventions at all.
        static Frustum FromViewProjection(const glm::mat4& viewProjection, const glm::vec3& cameraPosition,
                                           const glm::vec3& cameraForward, float nearPlane, float farPlane)
        {
            Frustum frustum;

            // Row extraction: glm stores matrices column-major, so "row
            // i" of the mathematical matrix is `vp[0][i], vp[1][i],
            // vp[2][i], vp[3][i]`.
            auto row = [&viewProjection](int i) { return glm::vec4(viewProjection[0][i], viewProjection[1][i], viewProjection[2][i], viewProjection[3][i]); };
            glm::vec4 r0 = row(0), r1 = row(1), r3 = row(3);

            frustum.m_Planes[0] = Normalize(r3 + r0); // left
            frustum.m_Planes[1] = Normalize(r3 - r0); // right
            frustum.m_Planes[2] = Normalize(r3 + r1); // bottom
            frustum.m_Planes[3] = Normalize(r3 - r1); // top

            // Constructed directly from a point-on-plane + normal, not
            // extracted from the matrix - see the long comment above for
            // why. Normal points INWARD (toward the frustum interior),
            // matching this class's ax+by+cz+d >= 0 = inside convention.
            glm::vec3 nearPoint = cameraPosition + cameraForward * nearPlane;
            glm::vec3 farPoint = cameraPosition + cameraForward * farPlane;
            frustum.m_Planes[4] = PlaneFromPointNormal(nearPoint, cameraForward);   // near: inward = forward
            frustum.m_Planes[5] = PlaneFromPointNormal(farPoint, -cameraForward);   // far: inward = backward

            return frustum;
        }

        // True if the sphere is at least partially inside the frustum -
        // false ONLY if it's provably entirely outside at least one
        // plane. `radius` should already be a WORLD-space radius (an
        // entity's mesh-local bounding radius times its own
        // TransformComponent::scale - see RenderSystem::BuildDrawList),
        // not the mesh's raw local-space one.
        bool ContainsSphere(const glm::vec3& center, float radius) const
        {
            for (const glm::vec4& plane : m_Planes)
            {
                float signedDistance = plane.x * center.x + plane.y * center.y + plane.z * center.z + plane.w;
                if (signedDistance < -radius)
                {
                    return false; // entirely on the outside of this one plane - definitely not visible
                }
            }
            return true;
        }

    private:
        static glm::vec4 Normalize(const glm::vec4& plane)
        {
            float length = glm::length(glm::vec3(plane));
            return length > 0.0f ? plane / length : plane;
        }

        static glm::vec4 PlaneFromPointNormal(const glm::vec3& point, const glm::vec3& normal)
        {
            glm::vec3 n = glm::normalize(normal);
            return glm::vec4(n, -glm::dot(n, point));
        }

        std::array<glm::vec4, 6> m_Planes;
    };
}
