#include "CollisionSystem.h"

#include "ColliderComponent.h"
#include "ColliderBounds.h"
#include "RigidBodyComponent.h"
#include "SDF.h"
#include "../ECS/Components/TransformComponent.h"
#include "../Voxel/VoxelSystemGPU.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <vector>

namespace Eden
{
    namespace
    {
        // --- Narrow phase -----------------------------------------------
        //
        // Every Test* function below returns a ContactInfo whose `normal`
        // points AWAY FROM its first shape argument, TOWARD its second -
        // i.e. "A -> B", the direction you'd push B to separate it from
        // A. ResolveContact() (further down) relies on this convention:
        // it pushes B along +normal and A along -normal, both scaled by
        // inverse mass.

        struct ContactInfo
        {
            bool colliding = false;
            glm::vec3 normal{ 0.0f, 1.0f, 0.0f }; // A -> B
            float penetration = 0.0f;

            // Approximate world-space contact point, used to compute the
            // lever arm (point - center of mass) that torque comes from -
            // see ResolveContact. Each pair test below produces its own
            // best estimate; Box-Box in particular is a genuine
            // approximation (a single point, not a real clipped contact
            // manifold), but it's enough to give a corner-landing box a
            // real lever arm to tip over on, which is the actual goal -
            // this isn't aiming for production-grade multi-point stacking
            // stability.
            glm::vec3 point{ 0.0f };
        };

        ContactInfo TestSphereSphere(const glm::vec3& posA, float radiusA, const glm::vec3& posB, float radiusB)
        {
            glm::vec3 delta = posB - posA;
            float distance = glm::length(delta);
            float penetration = (radiusA + radiusB) - distance;
            if (penetration <= 0.0f)
            {
                return {};
            }

            glm::vec3 normal = distance > 1e-6f ? delta / distance : glm::vec3(0.0f, 1.0f, 0.0f);
            glm::vec3 point = posA + normal * radiusA; // A's surface point facing B
            return { true, normal, penetration, point };
        }

        // Exact for any shape with a true signed distance field (every
        // shape SDF.h defines qualifies): a sphere overlaps `other` iff
        // the SDF distance from the sphere's center to `other`'s surface
        // is less than the sphere's radius. Treats the sphere as "A" and
        // `other` as "B" - the returned normal points sphere -> other.
        // Fully rotation-aware "for free": SDF::WorldToLocal already
        // accounts for `other`'s rotation, so this needed no change
        // beyond rotating the returned LOCAL normal back into world space.
        ContactInfo TestSphereVsSDFShape(const glm::vec3& sphereWorldPos, float sphereRadius,
                                          const ColliderComponent& other, const TransformComponent& otherTransform)
        {
            // `other`'s own scale is applied to a scaled COPY of its
            // shape parameters (see SDF::ScaledCollider), not to the
            // query point - WorldToLocal stays scale-free and exact in
            // world units; see both functions' comments in SDF.h for why
            // that split matters.
            ColliderComponent scaledOther = SDF::ScaledCollider(other, otherTransform.scale);
            glm::vec3 local = SDF::WorldToLocal(sphereWorldPos, otherTransform, other);
            float distanceToSurface = SDF::Distance(scaledOther, local);
            float penetration = sphereRadius - distanceToSurface;
            if (penetration <= 0.0f)
            {
                return {};
            }

            // SDF::Normal is in `other`'s LOCAL space and points away
            // from its surface (outward) - i.e. from other toward the
            // sphere. Rotate it into world space, then negate to get
            // sphere -> other.
            glm::mat3 otherRotation = SDF::RotationMatrixFromDegrees(otherTransform.rotationDegrees);
            glm::vec3 normal = -(otherRotation * SDF::Normal(scaledOther, local));
            glm::vec3 point = sphereWorldPos + normal * sphereRadius; // sphere's surface point facing other
            return { true, normal, penetration, point };
        }

        // Sphere-vs-Voxel: same shape as TestSphereVsSDFShape just above
        // (a sphere overlaps iff the SDF distance from its center to the
        // other surface is less than its radius) but reads
        // VoxelSystemGPU's live density field instead of an analytic
        // SDF.h formula - see SampleSignedDistance/SampleGradient's
        // comments. Voxel volumes never rotate (ColliderComponent::
        // voxelVolume's comment), so unlike TestSphereVsSDFShape there's
        // no SDF::WorldToLocal/rotation step - SampleSignedDistance
        // already takes a world-space point directly.
        ContactInfo TestSphereVsVoxel(const glm::vec3& sphereWorldPos, float sphereRadius,
                                       const ColliderComponent& voxelCollider, const VoxelSystemGPU* voxelSystem)
        {
            if (voxelSystem == nullptr || voxelCollider.voxelVolume == InvalidVoxelVolumeHandle)
            {
                // No voxel system was passed to CollisionSystem::Step
                // (see its header comment - voxelSystem is optional), or
                // this collider's handle was never set. No contact
                // rather than a crash.
                return {};
            }

            float distanceToSurface = voxelSystem->SampleSignedDistance(voxelCollider.voxelVolume, sphereWorldPos);
            float penetration = sphereRadius - distanceToSurface;
            if (penetration <= 0.0f)
            {
                return {};
            }

            // Defensive clamp, specific to Voxel: a reformed/carved
            // surface is genuinely irregular (union-of-spheres blobs,
            // concave seams, coarse voxel resolution meeting much finer
            // SmoothMin detail) in a way the 4 analytic shapes never
            // are, so a single SampleSignedDistance read can occasionally
            // report much deeper penetration than the body actually
            // needs correcting by this step - CCD (above) now prevents
            // the worst case (tunneling deep into the interior before
            // detection), but doesn't guarantee every read near a sharp
            // concave feature is exact. Capping at the sphere's own
            // radius means a real contact still fully resolves within a
            // couple of steps (positionalCorrectionPercent already only
            // partially corrects per step - see CollisionSystem.h), just
            // never in one explosive, physically-nonsensical shove. This
            // is what was producing the reported repeated-upward-
            // teleport on a reformed surface.
            penetration = glm::min(penetration, sphereRadius);

            // SampleGradient points away from solid material (outward,
            // same convention as SDF::Normal) - i.e. voxel -> sphere.
            // Negate for sphere -> voxel, matching every other Test*
            // function's normal convention.
            glm::vec3 normal = -voxelSystem->SampleGradient(voxelCollider.voxelVolume, sphereWorldPos);
            glm::vec3 point = sphereWorldPos + normal * sphereRadius; // sphere's surface point facing the volume
            return { true, normal, penetration, point };
        }

        // Box/Capsule-vs-Voxel share the same core idea, which is why
        // both are built on this one helper: SampleSignedDistance only
        // answers "what's the distance/gradient AT THIS ONE POINT" -
        // exact for a sphere (one point + a radius IS the whole shape),
        // but a box or capsule is an extended volume, so no single
        // sample is enough on its own. This tests several representative
        // WORLD-SPACE points against the field (each treated as its own
        // tiny sphere-vs-voxel query, effectively) and returns whichever
        // one penetrates deepest - the same "several point-samples,
        // trust the worst one" shape TestSphereVsVoxel's own penetration
        // clamp comment already gestures at being necessary for
        // irregular reformed/carved surfaces, just applied here to
        // MULTIPLE points instead of one.
        //
        // This is an approximation, not exact narrow-phase (a true
        // box-vs-SDF test would need to find the actual closest surface
        // point on the box to the field's isosurface, which has no
        // closed form against an arbitrary density field) - it can miss
        // a genuine contact that happens to fall between sample points,
        // or report a contact normal that's slightly off from a
        // corner/edge's true contact geometry. Acceptable for the same
        // reason TestSphereVsVoxel's own penetration clamp is: this
        // project's voxel/reform surfaces are irregular by nature
        // (union-of-spheres blobs, carved terrain), so "close enough,
        // corrects over a couple of steps" already describes how every
        // Voxel narrow-phase result behaves here, sphere included - this
        // isn't a new tradeoff, just extending an existing one to boxes
        // and capsules.
        //
        // sampleRadius is the EFFECTIVE per-point contact radius at that
        // sample - for Box this is 0 (a face-center point has no
        // inherent radius, unlike a capsule's swept-sphere axis points),
        // for Capsule this is the capsule's own radius (each axis sample
        // really is a tiny sphere query, since a capsule genuinely IS a
        // swept sphere).
        ContactInfo TestPointsVsVoxel(const std::vector<glm::vec3>& worldPoints, float sampleRadius,
                                       const ColliderComponent& voxelCollider, const VoxelSystemGPU* voxelSystem)
        {
            if (voxelSystem == nullptr || voxelCollider.voxelVolume == InvalidVoxelVolumeHandle)
            {
                return {}; // see TestSphereVsVoxel's identical check for why
            }

            bool found = false;
            float deepestPenetration = 0.0f;
            glm::vec3 deepestNormal{ 0.0f, 1.0f, 0.0f };
            glm::vec3 deepestPoint{ 0.0f };

            for (const glm::vec3& worldPoint : worldPoints)
            {
                float distanceToSurface = voxelSystem->SampleSignedDistance(voxelCollider.voxelVolume, worldPoint);
                float penetration = sampleRadius - distanceToSurface;
                if (penetration <= 0.0f)
                {
                    continue; // this sample point isn't in contact, check the rest
                }

                if (!found || penetration > deepestPenetration)
                {
                    found = true;
                    // Same defensive clamp TestSphereVsVoxel applies,
                    // same reasoning (irregular surface, occasional
                    // over-reported penetration) - capped per-sample at
                    // a generous-but-bounded value rather than
                    // sampleRadius itself, since sampleRadius is 0 for
                    // Box's face-center points (see this function's own
                    // comment) and a 0-radius cap would zero out every
                    // Box contact's correction entirely. 0.5 world units
                    // matches the scale TestSphereVsVoxel's own clamp
                    // operates at for this project's typical object
                    // sizes (see that function's sphereRadius-based
                    // clamp for comparison) without depending on a
                    // per-shape radius that may not exist.
                    deepestPenetration = glm::min(penetration, 0.5f);
                    deepestNormal = -voxelSystem->SampleGradient(voxelCollider.voxelVolume, worldPoint); // outward, see TestSphereVsVoxel's normal comment
                    deepestPoint = worldPoint + deepestNormal * sampleRadius;
                }
            }

            if (!found)
            {
                return {};
            }
            return { true, deepestNormal, deepestPenetration, deepestPoint };
        }

        // 7 sample points: the box's own center plus its 6 face centers
        // (center +/- halfExtents along each LOCAL axis, then rotated
        // into world space) - covers the case a single center sample
        // would miss (a box resting flat on a voxel surface, where the
        // box's CENTER is well clear of the surface but its bottom face
        // is already penetrating). Corners are deliberately NOT
        // included (would be 8 more points, doubling the sample count)
        // - face centers already catch the common "box resting on/
        // pressed into a mostly-flat voxel surface" case this project's
        // actual scenes produce (terrain, reform blobs), and TestPointsVsVoxel's
        // own comment already documents that exact corner-vs-concave-
        // seam contacts are an accepted approximation, not a guarantee,
        // for every Voxel narrow-phase test in this file.
        ContactInfo TestBoxVsVoxel(const ColliderComponent& boxCollider, const TransformComponent& boxTransform,
                                    const ColliderComponent& voxelCollider, const VoxelSystemGPU* voxelSystem)
        {
            glm::mat3 rotation = SDF::RotationMatrixFromDegrees(boxTransform.rotationDegrees);
            glm::vec3 center = boxTransform.position + rotation * boxCollider.localOffset;
            // Scaled here at the call site, NOT assumed pre-scaled -
            // TestCollision's callers pass raw ECS ColliderComponents
            // (see TestCollision's own broad-phase call site), so every
            // Test* function in this file is individually responsible
            // for applying transform.scale itself. Matches TestBoxBox's
            // identical halfA/halfB scaling exactly (see that function's
            // comment for why scale is applied along the LOCAL axis
            // before rotation).
            glm::vec3 half = boxCollider.halfExtents * boxTransform.scale;

            std::vector<glm::vec3> samplePoints;
            samplePoints.reserve(7);
            samplePoints.push_back(center);
            samplePoints.push_back(center + rotation[0] * half.x);
            samplePoints.push_back(center - rotation[0] * half.x);
            samplePoints.push_back(center + rotation[1] * half.y);
            samplePoints.push_back(center - rotation[1] * half.y);
            samplePoints.push_back(center + rotation[2] * half.z);
            samplePoints.push_back(center - rotation[2] * half.z);

            return TestPointsVsVoxel(samplePoints, 0.0f, voxelCollider, voxelSystem);
        }

        // 3 sample points along the capsule's own local +Y axis (see
        // SDF::Capsule's comment for that axis convention): both
        // hemisphere centers (+/-halfHeight) plus the midpoint - each
        // treated as its own sphere-vs-voxel query at the capsule's
        // radius (a capsule genuinely IS two spheres swept along a
        // segment, so this is a much closer approximation than Box's
        // face-center sampling, not just "fewer points because capsules
        // are simpler"). The midpoint sample catches a capsule lying
        // flat against a voxel surface where both ends happen to clear
        // it but the middle doesn't (rare given typical capsule
        // proportions, but the cost of one extra sample is negligible
        // next to correctness here).
        ContactInfo TestCapsuleVsVoxel(const ColliderComponent& capsuleCollider, const TransformComponent& capsuleTransform,
                                        const ColliderComponent& voxelCollider, const VoxelSystemGPU* voxelSystem)
        {
            glm::mat3 rotation = SDF::RotationMatrixFromDegrees(capsuleTransform.rotationDegrees);
            glm::vec3 center = capsuleTransform.position + rotation * capsuleCollider.localOffset;
            glm::vec3 axis = rotation[1]; // local +Y, see SDF::Capsule's comment
            // Same scale convention TestCapsuleCapsule uses (see that
            // function's comment): halfHeight scales along local Y,
            // radius is isotropically averaged across local X/Z rather
            // than per-axis (a capsule's cross-section is rotationally
            // symmetric, so there's no well-defined "X radius" vs "Z
            // radius" the way a box has independent extents per axis).
            // NOT pre-scaled by the caller - see TestBoxVsVoxel's
            // identical correction/comment for why.
            float halfHeight = capsuleCollider.halfHeight * capsuleTransform.scale.y;
            float radius = capsuleCollider.radius * 0.5f * (capsuleTransform.scale.x + capsuleTransform.scale.z);

            std::vector<glm::vec3> samplePoints = {
                center + axis * halfHeight,
                center,
                center - axis * halfHeight,
            };

            return TestPointsVsVoxel(samplePoints, radius, voxelCollider, voxelSystem);
        }

        // Oriented-box vs oriented-box, via the standard Separating Axis
        // Theorem: two convex shapes are NOT overlapping iff some axis
        // exists along which their projections don't overlap, and for two
        // boxes it's enough to test each box's 3 face normals plus the 9
        // cross products of one box's edges with the other's (15 axes
        // total). If every axis shows overlap, the boxes are colliding,
        // and the axis with the SMALLEST overlap approximates the true
        // contact normal/penetration depth - the same "least penetration
        // axis" idea an earlier axis-aligned version of this used, now
        // over all 15 candidate axes instead of just 3.
        ContactInfo TestBoxBox(const ColliderComponent& colliderA, const TransformComponent& transformA,
                                const ColliderComponent& colliderB, const TransformComponent& transformB)
        {
            glm::mat3 rotationA = SDF::RotationMatrixFromDegrees(transformA.rotationDegrees);
            glm::mat3 rotationB = SDF::RotationMatrixFromDegrees(transformB.rotationDegrees);
            glm::vec3 posA = transformA.position + rotationA * colliderA.localOffset;
            glm::vec3 posB = transformB.position + rotationB * colliderB.localOffset;

            glm::vec3 axesA[3] = { rotationA[0], rotationA[1], rotationA[2] };
            glm::vec3 axesB[3] = { rotationB[0], rotationB[1], rotationB[2] };
            // Scaled by the entity's own TransformComponent::scale here,
            // at definition - halfA/halfB feed BOTH the SAT projection
            // test below AND the contact-point heuristic further down
            // (ClosestPointOnBox calls), so fixing it once here means
            // neither has to separately remember to account for scale.
            // Scale is applied along each LOCAL axis before rotation
            // (matching TransformComponent::GetModelMatrix's translate *
            // rotate * scale order), which is exactly what multiplying
            // component-wise here does, since axesA[i]/axesB[i] are
            // already the corresponding local axis directions in world
            // space.
            float halfA[3] = { colliderA.halfExtents.x * transformA.scale.x, colliderA.halfExtents.y * transformA.scale.y, colliderA.halfExtents.z * transformA.scale.z };
            float halfB[3] = { colliderB.halfExtents.x * transformB.scale.x, colliderB.halfExtents.y * transformB.scale.y, colliderB.halfExtents.z * transformB.scale.z };

            glm::vec3 centerDelta = posB - posA;

            float bestOverlap = std::numeric_limits<float>::max();
            glm::vec3 bestAxis(0.0f, 1.0f, 0.0f);
            constexpr float kParallelEpsilon = 1e-5f;

            // Returns false the moment a genuine separating axis is
            // found (boxes cannot be overlapping - caller should bail
            // out immediately); otherwise tracks the minimum-overlap
            // axis seen so far in bestAxis/bestOverlap.
            auto TestAxis = [&](glm::vec3 axis) -> bool
            {
                float length = glm::length(axis);
                if (length < kParallelEpsilon)
                {
                    // Degenerate axis (near-parallel edges in the cross-
                    // product case) - not a meaningful separating axis,
                    // skip rather than divide by near-zero.
                    return true;
                }
                axis /= length;

                float radiusA = halfA[0] * std::abs(glm::dot(axesA[0], axis))
                              + halfA[1] * std::abs(glm::dot(axesA[1], axis))
                              + halfA[2] * std::abs(glm::dot(axesA[2], axis));
                float radiusB = halfB[0] * std::abs(glm::dot(axesB[0], axis))
                              + halfB[1] * std::abs(glm::dot(axesB[1], axis))
                              + halfB[2] * std::abs(glm::dot(axesB[2], axis));

                float distance = glm::dot(centerDelta, axis);
                float overlap = (radiusA + radiusB) - std::abs(distance);

                if (overlap <= 0.0f)
                {
                    return false; // separating axis found
                }

                if (overlap < bestOverlap)
                {
                    bestOverlap = overlap;
                    // Orient so the axis points from A toward B, matching
                    // this file's A -> B normal convention.
                    bestAxis = (distance >= 0.0f) ? axis : -axis;
                }
                return true;
            };

            for (int i = 0; i < 3; ++i) { if (!TestAxis(axesA[i])) return {}; }
            for (int i = 0; i < 3; ++i) { if (!TestAxis(axesB[i])) return {}; }
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    if (!TestAxis(glm::cross(axesA[i], axesB[j]))) return {};
                }
            }

            // Closest point on a box (given its own center, axes, half-
            // extents) to an external point - the standard OBB closest-
            // point query. When the point lies outside all three of the
            // box's face-slabs (the common case for a corner contact),
            // this correctly returns that corner.
            auto ClosestPointOnBox = [](const glm::vec3& boxCenter, const glm::vec3 boxAxes[3], const float boxHalf[3], const glm::vec3& toPoint) -> glm::vec3
            {
                glm::vec3 delta = toPoint - boxCenter;
                glm::vec3 result = boxCenter;
                for (int i = 0; i < 3; ++i)
                {
                    float projected = glm::clamp(glm::dot(delta, boxAxes[i]), -boxHalf[i], boxHalf[i]);
                    result += projected * boxAxes[i];
                }
                return result;
            };

            // Approximate contact point: the closest point on the LARGER
            // of the two boxes' own (possibly rotated) volume to the
            // SMALLER box's center. Two earlier, wrong attempts are worth
            // recording, since both failure modes are real and this is
            // what actually needs to avoid both:
            //
            // 1st - midpoint of each box's along-axis support point, with
            // off-axis components left as each box's raw, unclipped
            // center: drifted arbitrarily far during sustained sliding
            // (nothing bounded it to either surface). Caused a box on a
            // slowly-moving platform to spin up and slide backwards at
            // ~27x the platform's speed.
            //
            // 2nd - center of the world-AABB overlap region: fixed the
            // drift, but lost the lever arm for a genuine corner contact
            // - a box landing on another box tilted so one corner points
            // up ended up with ~zero torque and balanced on the corner
            // forever, silently breaking the actual feature this rewrite
            // was for.
            //
            // 3rd - closest point on EACH box to the OTHER's center,
            // averaged: correctly found the corner (fixed the 2nd
            // failure), but reintroduced the 1st - the SMALLER/sliding
            // box's own closest-point-to-other calculation still shifts
            // toward ITS corner as IT tilts, recreating the same
            // feedback loop from a different direction.
            //
            // 4th - closest point on whichever box is LARGER: broke the
            // sliding feedback loop, but "larger" isn't actually the
            // right thing to key off - a corner test with the FALLING
            // box larger than the tilted platform it lands on picked the
            // falling box's own (flat, unrotated) geometry and completely
            // missed the platform's corner, giving zero torque again.
            // Confirmed directly: reversing the size relationship in the
            // exact corner-tumble test produced 0.00 degrees of rotation.
            //
            // This version keys off something that's actually meaningful
            // instead of size: which box's FACE the winning separating
            // axis (bestAxis) aligns with. If bestAxis closely matches
            // one of A's own local axes, A is presenting a flat face as
            // the contact surface - which means B's corner/edge is what's
            // poking into it, so B's own geometry is what should locate
            // the true contact point (and vice versa). This is exactly
            // the geometric relationship a corner-on-face contact has,
            // regardless of which box happens to be bigger.
            //
            // BUT this alone still recreates the sliding feedback loop
            // through a different path: it's instantaneously correct,
            // and that's the problem - the moment a resting box picks up
            // even a fraction of a degree of tilt from friction, the
            // alignment shifts to favor using THAT box's own (now
            // slightly tilted) geometry to find "its corner," which is
            // exactly the mechanism that created the loop in the 1st and
            // 3rd attempts, just reached from a different direction.
            // Confirmed directly: this exact heuristic, unguarded,
            // reproduced the ~27x sliding-platform blowup again.
            //
            // Fix: require a MEANINGFUL margin before trusting the
            // "which box is more tilted" signal at all. A genuine corner
            // landing (a box dropped rotated 45 degrees) blows past this
            // margin immediately - the alignment gap is large from the
            // very first contact. A box picking up gradual tilt from
            // sustained sliding friction stays near a tie for a long time
            // (starting at a PERFECT tie, since it begins perfectly
            // flat), so it keeps resolving through the default (stable)
            // branch instead of ever using its own drifting geometry -
            // which means the feedback loop never gets a foothold to
            // begin compounding in the first place.
            float alignmentWithA = glm::max(glm::max(std::abs(glm::dot(bestAxis, axesA[0])), std::abs(glm::dot(bestAxis, axesA[1]))), std::abs(glm::dot(bestAxis, axesA[2])));
            float alignmentWithB = glm::max(glm::max(std::abs(glm::dot(bestAxis, axesB[0])), std::abs(glm::dot(bestAxis, axesB[1]))), std::abs(glm::dot(bestAxis, axesB[2])));

            constexpr float kAlignmentMargin = 0.1f; // ~a few degrees of tilt difference before the corner-detection branch is trusted
            glm::vec3 contactPoint;
            if (alignmentWithA - alignmentWithB > kAlignmentMargin)
            {
                // A's face is CLEARLY the better-aligned contact surface
                // -> B's geometry finds B's corner/edge poking into it.
                contactPoint = ClosestPointOnBox(posB, axesB, halfB, posA);
            }
            else if (alignmentWithB - alignmentWithA > kAlignmentMargin)
            {
                contactPoint = ClosestPointOnBox(posA, axesA, halfA, posB);
            }
            else
            {
                // Near-tie (near-flat contact, including the common
                // perfectly-flat resting case) - default to A's geometry.
                // Arbitrary as a tie-break, but stable: it doesn't matter
                // which side "A" happens to be for a genuinely symmetric
                // contact, and staying on this branch is exactly what
                // keeps a slowly-tilting body's own drift from feeding
                // back into its own contact point.
                contactPoint = ClosestPointOnBox(posA, axesA, halfA, posB);
            }

            return { true, bestAxis, bestOverlap, contactPoint };
        }

        // Support-point test: the box vertex furthest in the
        // -planeNormal direction is the one most likely to have poked
        // through the plane. Exact for an oriented box against an
        // infinite plane - uses the box's ACTUAL (possibly rotated)
        // local axes to find that vertex, not world axes.
        ContactInfo TestBoxPlane(const ColliderComponent& boxCollider, const TransformComponent& boxTransform,
                                  const ColliderComponent& planeCollider, const TransformComponent& planeTransform)
        {
            glm::mat3 boxRotation = SDF::RotationMatrixFromDegrees(boxTransform.rotationDegrees);
            glm::mat3 planeRotation = SDF::RotationMatrixFromDegrees(planeTransform.rotationDegrees);

            glm::vec3 boxPos = boxTransform.position + boxRotation * boxCollider.localOffset;
            glm::vec3 planePoint = planeTransform.position + planeRotation * planeCollider.localOffset;
            glm::vec3 planeNormal = planeRotation * planeCollider.planeNormal;

            glm::vec3 axes[3] = { boxRotation[0], boxRotation[1], boxRotation[2] };
            // Same scaling as TestBoxBox's halfA/halfB - local axis i's
            // extent is scaled by the entity's own scale component i.
            float halfExtents[3] = { boxCollider.halfExtents.x * boxTransform.scale.x, boxCollider.halfExtents.y * boxTransform.scale.y, boxCollider.halfExtents.z * boxTransform.scale.z };

            glm::vec3 support = boxPos;
            for (int i = 0; i < 3; ++i)
            {
                float sign = (glm::dot(axes[i], planeNormal) >= 0.0f) ? 1.0f : -1.0f;
                support -= sign * halfExtents[i] * axes[i];
            }

            float distance = glm::dot(support - planePoint, planeNormal);
            if (distance >= 0.0f)
            {
                return {};
            }

            // A = box, B = plane. The box sits on the +planeNormal side,
            // so "away from the box, into the plane" is -planeNormal.
            // `support` IS the contact point here - it's already the
            // exact vertex poking through the plane.
            return { true, -planeNormal, -distance, support };
        }

        // Exact: the plane's SDF is linear, so the closest point on a
        // straight segment to the plane is always one of its two
        // endpoints - no search along the segment needed. Uses the
        // capsule's ACTUAL rotated axis, not an assumed world +Y.
        ContactInfo TestCapsulePlane(const ColliderComponent& capsuleCollider, const TransformComponent& capsuleTransform,
                                     const ColliderComponent& planeCollider, const TransformComponent& planeTransform)
        {
            glm::mat3 capsuleRotation = SDF::RotationMatrixFromDegrees(capsuleTransform.rotationDegrees);
            glm::mat3 planeRotation = SDF::RotationMatrixFromDegrees(planeTransform.rotationDegrees);

            glm::vec3 capsulePos = capsuleTransform.position + capsuleRotation * capsuleCollider.localOffset;
            glm::vec3 planePoint = planeTransform.position + planeRotation * planeCollider.localOffset;
            glm::vec3 planeNormal = planeRotation * planeCollider.planeNormal;

            glm::vec3 axis = capsuleRotation * glm::vec3(0.0f, 1.0f, 0.0f);
            // Scaled here rather than deep in the math below: halfHeight
            // runs along the capsule's own local +Y, so it scales by
            // transform.scale.y. radius is the capsule's cross-section
            // in local X/Z, which is isotropic by construction (a
            // capsule can't represent an elliptical cross-section) -
            // averaging scale.x/scale.z is the standard approximation
            // for a non-uniformly-scaled round shape (exact whenever
            // scale.x == scale.z, which includes the common uniform-
            // scale case).
            float scaledHalfHeight = capsuleCollider.halfHeight * capsuleTransform.scale.y;
            float scaledRadius = capsuleCollider.radius * 0.5f * (capsuleTransform.scale.x + capsuleTransform.scale.z);
            glm::vec3 top = capsulePos + axis * scaledHalfHeight;
            glm::vec3 bottom = capsulePos - axis * scaledHalfHeight;

            float distanceTop = glm::dot(top - planePoint, planeNormal);
            float distanceBottom = glm::dot(bottom - planePoint, planeNormal);
            bool topIsCloser = distanceTop < distanceBottom;
            float closestDistance = topIsCloser ? distanceTop : distanceBottom;
            glm::vec3 closestEnd = topIsCloser ? top : bottom;

            float penetration = scaledRadius - closestDistance;
            if (penetration <= 0.0f)
            {
                return {};
            }

            glm::vec3 point = closestEnd - planeNormal * scaledRadius; // capsule's surface point facing the plane
            return { true, -planeNormal, penetration, point };
        }

        // Standard parametric closest-point-between-two-segments
        // computation, clamped to each segment. Degenerate (zero-length
        // segment) cases are handled explicitly rather than left to
        // divide by zero.
        void ClosestPointsSegmentSegment(const glm::vec3& p1, const glm::vec3& q1,
                                          const glm::vec3& p2, const glm::vec3& q2,
                                          glm::vec3& outC1, glm::vec3& outC2)
        {
            constexpr float kEpsilon = 1e-8f;

            glm::vec3 d1 = q1 - p1;
            glm::vec3 d2 = q2 - p2;
            glm::vec3 r = p1 - p2;

            float a = glm::dot(d1, d1);
            float e = glm::dot(d2, d2);
            float f = glm::dot(d2, r);

            float s, t;

            if (a <= kEpsilon && e <= kEpsilon)
            {
                outC1 = p1;
                outC2 = p2;
                return;
            }

            if (a <= kEpsilon)
            {
                s = 0.0f;
                t = glm::clamp(f / e, 0.0f, 1.0f);
            }
            else
            {
                float c = glm::dot(d1, r);
                if (e <= kEpsilon)
                {
                    t = 0.0f;
                    s = glm::clamp(-c / a, 0.0f, 1.0f);
                }
                else
                {
                    float b = glm::dot(d1, d2);
                    float denom = a * e - b * b;

                    s = (denom > kEpsilon) ? glm::clamp((b * f - c * e) / denom, 0.0f, 1.0f) : 0.0f;
                    t = (b * s + f) / e;

                    if (t < 0.0f)
                    {
                        t = 0.0f;
                        s = glm::clamp(-c / a, 0.0f, 1.0f);
                    }
                    else if (t > 1.0f)
                    {
                        t = 1.0f;
                        s = glm::clamp((b - c) / a, 0.0f, 1.0f);
                    }
                }
            }

            outC1 = p1 + d1 * s;
            outC2 = p2 + d2 * t;
        }

        // Uses each capsule's ACTUAL rotated axis to build its segment
        // endpoints, then reduces to the same closest-point-between-
        // segments + sphere-sphere test as before.
        ContactInfo TestCapsuleCapsule(const ColliderComponent& colliderA, const TransformComponent& transformA,
                                       const ColliderComponent& colliderB, const TransformComponent& transformB)
        {
            glm::mat3 rotationA = SDF::RotationMatrixFromDegrees(transformA.rotationDegrees);
            glm::mat3 rotationB = SDF::RotationMatrixFromDegrees(transformB.rotationDegrees);

            glm::vec3 posA = transformA.position + rotationA * colliderA.localOffset;
            glm::vec3 posB = transformB.position + rotationB * colliderB.localOffset;

            glm::vec3 axisA = rotationA * glm::vec3(0.0f, 1.0f, 0.0f);
            glm::vec3 axisB = rotationB * glm::vec3(0.0f, 1.0f, 0.0f);

            // Same scale handling as TestCapsulePlane: halfHeight along
            // local Y, radius isotropically averaged across local X/Z.
            float halfHeightA = colliderA.halfHeight * transformA.scale.y;
            float halfHeightB = colliderB.halfHeight * transformB.scale.y;
            float radiusA = colliderA.radius * 0.5f * (transformA.scale.x + transformA.scale.z);
            float radiusB = colliderB.radius * 0.5f * (transformB.scale.x + transformB.scale.z);

            glm::vec3 topA = posA + axisA * halfHeightA;
            glm::vec3 bottomA = posA - axisA * halfHeightA;
            glm::vec3 topB = posB + axisB * halfHeightB;
            glm::vec3 bottomB = posB - axisB * halfHeightB;

            glm::vec3 closestA, closestB;
            ClosestPointsSegmentSegment(bottomA, topA, bottomB, topB, closestA, closestB);

            return TestSphereSphere(closestA, radiusA, closestB, radiusB);
        }

        // APPROXIMATE, on purpose: finds the point on the capsule's
        // (actual, rotated) axis nearest to the box's CENTER (not its
        // surface), then tests that point-plus-radius against the box's
        // exact SDF - which is itself now fully rotation-aware via
        // SDF::WorldToLocal. A true closest-point-on-axis-to-box-surface
        // query would be more accurate (and isn't free), especially for a
        // long capsule laid near a box edge/corner. Good enough for a
        // first pass; revisit if box/capsule pairs need to be reliable at
        // high speed or in tight stacks.
        ContactInfo TestBoxCapsule(const ColliderComponent& boxCollider, const TransformComponent& boxTransform,
                                    const ColliderComponent& capsuleCollider, const TransformComponent& capsuleTransform)
        {
            glm::mat3 boxRotation = SDF::RotationMatrixFromDegrees(boxTransform.rotationDegrees);
            glm::mat3 capsuleRotation = SDF::RotationMatrixFromDegrees(capsuleTransform.rotationDegrees);

            glm::vec3 boxCenter = boxTransform.position + boxRotation * boxCollider.localOffset;
            glm::vec3 capsuleCenter = capsuleTransform.position + capsuleRotation * capsuleCollider.localOffset;
            glm::vec3 capsuleAxis = capsuleRotation * glm::vec3(0.0f, 1.0f, 0.0f);

            // Same scale handling as TestCapsulePlane/TestCapsuleCapsule -
            // boxCollider's own scale is handled automatically inside
            // TestSphereVsSDFShape below via SDF::WorldToLocal.
            float capsuleHalfHeight = capsuleCollider.halfHeight * capsuleTransform.scale.y;
            float capsuleRadius = capsuleCollider.radius * 0.5f * (capsuleTransform.scale.x + capsuleTransform.scale.z);

            float t = glm::clamp(glm::dot(boxCenter - capsuleCenter, capsuleAxis), -capsuleHalfHeight, capsuleHalfHeight);
            glm::vec3 nearestOnAxis = capsuleCenter + capsuleAxis * t;

            ContactInfo result = TestSphereVsSDFShape(nearestOnAxis, capsuleRadius, boxCollider, boxTransform);
            if (!result.colliding)
            {
                return {};
            }

            // TestSphereVsSDFShape treated the capsule's axis point as
            // "A" and the box as "B" (capsule -> box). Canonical order
            // here is (Box = A, Capsule = B) - flip it.
            result.normal = -result.normal;
            return result;
        }

        constexpr int ShapeOrder(ColliderShape shape) { return static_cast<int>(shape); }

        // Dispatches to the right pair test above. Canonicalizes shape
        // order first (Sphere < Box < Capsule < Plane) so only one
        // ordering of each pair needs an implementation - if the caller's
        // (A, B) is the "wrong" way round for the table below, this swaps
        // and negates the resulting normal instead of duplicating every
        // test with arguments reversed.
        ContactInfo TestCollision(const ColliderComponent& colliderA, const TransformComponent& transformA,
                                   const ColliderComponent& colliderB, const TransformComponent& transformB,
                                   const VoxelSystemGPU* voxelSystem = nullptr)
        {
            if (ShapeOrder(colliderA.shape) > ShapeOrder(colliderB.shape))
            {
                ContactInfo result = TestCollision(colliderB, transformB, colliderA, transformA, voxelSystem);
                result.normal = -result.normal;
                return result;
            }

            switch (colliderA.shape)
            {
                case ColliderShape::Sphere:
                {
                    glm::mat3 rotationA = SDF::RotationMatrixFromDegrees(transformA.rotationDegrees);
                    glm::vec3 posA = transformA.position + rotationA * colliderA.localOffset;
                    // Isotropic shape under anisotropic scale - same
                    // averaging approximation used throughout this file
                    // (exact under uniform scale).
                    float radiusA = colliderA.radius * (transformA.scale.x + transformA.scale.y + transformA.scale.z) / 3.0f;

                    // Sphere-vs-anything (including Sphere-Sphere, which
                    // TestSphereVsSDFShape could also handle via
                    // SDF::Sphere, but the direct formula avoids the
                    // WorldToLocal/gradient round-trip for the single
                    // most common pair - and a sphere's rotation never
                    // matters anyway).
                    switch (colliderB.shape)
                    {
                        case ColliderShape::Sphere:
                        {
                            glm::mat3 rotationB = SDF::RotationMatrixFromDegrees(transformB.rotationDegrees);
                            glm::vec3 posB = transformB.position + rotationB * colliderB.localOffset;
                            float radiusB = colliderB.radius * (transformB.scale.x + transformB.scale.y + transformB.scale.z) / 3.0f;
                            return TestSphereSphere(posA, radiusA, posB, radiusB);
                        }
                        case ColliderShape::Box:
                        case ColliderShape::Capsule:
                        case ColliderShape::Plane:
                            // colliderB's own scale is handled inside
                            // TestSphereVsSDFShape via SDF::WorldToLocal
                            // (see SDF.h) - only the sphere's own radiusA
                            // needs to be scaled here, at the call site.
                            return TestSphereVsSDFShape(posA, radiusA, colliderB, transformB);

                        case ColliderShape::Voxel:
                            return TestSphereVsVoxel(posA, radiusA, colliderB, voxelSystem);
                    }
                    break;
                }

                case ColliderShape::Box:
                    switch (colliderB.shape)
                    {
                        case ColliderShape::Box:     return TestBoxBox(colliderA, transformA, colliderB, transformB);
                        case ColliderShape::Capsule: return TestBoxCapsule(colliderA, transformA, colliderB, transformB);
                        case ColliderShape::Plane:   return TestBoxPlane(colliderA, transformA, colliderB, transformB);
                        case ColliderShape::Voxel:   return TestBoxVsVoxel(colliderA, transformA, colliderB, voxelSystem);
                        default: break; // Sphere never reaches here post-canonicalization
                    }
                    break;

                case ColliderShape::Capsule:
                    switch (colliderB.shape)
                    {
                        case ColliderShape::Capsule: return TestCapsuleCapsule(colliderA, transformA, colliderB, transformB);
                        case ColliderShape::Plane:   return TestCapsulePlane(colliderA, transformA, colliderB, transformB);
                        case ColliderShape::Voxel:   return TestCapsuleVsVoxel(colliderA, transformA, colliderB, voxelSystem);
                        default: break;
                    }
                    break;

                case ColliderShape::Plane:
                    // Only reachable for Plane-Plane once canonicalized -
                    // degenerate, deliberately unhandled (see
                    // CollisionSystem.h).
                    break;

                case ColliderShape::Voxel:
                    // Only reachable for Voxel-Voxel once canonicalized -
                    // two volumes of static world geometry never need to
                    // collide with each other. Sphere/Box/Capsule-vs-
                    // Voxel are all handled above, in each of THOSE
                    // shapes' own switch cases (TestSphereVsVoxel/
                    // TestBoxVsVoxel/TestCapsuleVsVoxel) - Voxel always
                    // canonicalizes to the B side (highest ShapeOrder),
                    // so this outer Voxel case genuinely only ever sees
                    // Voxel-Voxel pairs, not a leftover gap.
                    break;
            }

            return {};
        }

        // --- Resolution ---------------------------------------------------

        struct Contact
        {
            Entity a;
            Entity b;
            glm::vec3 normal; // a -> b
            float penetration;
            glm::vec3 point; // approximate world-space contact location, see ContactInfo
        };

        // --- Rotational inertia --------------------------------------------
        //
        // Standard per-shape solid-body moment-of-inertia formulas, about
        // the shape's own center, in its own LOCAL (unrotated) axes -
        // exactly analogous to SDF.h's local-space distance functions.
        // Returns the INVERSE (1/I per axis) since that's what every use
        // site actually wants, same reasoning as inverseMass. A
        // zero-or-negative inverseMass (Static/Kinematic, or a sleeping
        // body being treated as an anchor) means "immovable AND
        // non-rotating for this resolution" - returns a zero vector,
        // which naturally zeroes out every angular term downstream
        // without needing a separate special case.
        // `scale` applied here for the same reason as everywhere else in
        // this file: an entity scaled up in the inspector should get
        // proportionally larger inertia, or a light, correctly-shaped
        // rigid body would suddenly tumble like it's still authored-size
        // once you'd made it visually (and now collision-wise, after the
        // fixes above) several times bigger. Same isotropic-radius
        // averaging approximation as AabbHalfExtents/TestCapsulePlane
        // for Sphere/Capsule under anisotropic scale.
        glm::vec3 ComputeLocalInverseInertia(const ColliderComponent& collider, float inverseMass, const glm::vec3& scale)
        {
            if (inverseMass <= 0.0f)
            {
                return glm::vec3(0.0f);
            }

            float mass = 1.0f / inverseMass;
            glm::vec3 localInertia(1.0f);

            switch (collider.shape)
            {
                case ColliderShape::Sphere:
                {
                    float r = collider.radius * (scale.x + scale.y + scale.z) / 3.0f;
                    float i = (2.0f / 5.0f) * mass * r * r;
                    localInertia = glm::vec3(i);
                    break;
                }
                case ColliderShape::Box:
                {
                    glm::vec3 h = collider.halfExtents * scale;
                    // Solid cuboid about its center, full side lengths
                    // 2h: I_axis = (m/3)*(sum of the OTHER two half-
                    // extents squared) - standard result, equivalent to
                    // the usual (m/12)*(b^2+c^2) form written in terms of
                    // half-extents instead of full side lengths.
                    localInertia.x = (mass / 3.0f) * (h.y * h.y + h.z * h.z);
                    localInertia.y = (mass / 3.0f) * (h.x * h.x + h.z * h.z);
                    localInertia.z = (mass / 3.0f) * (h.x * h.x + h.y * h.y);
                    break;
                }
                case ColliderShape::Capsule:
                {
                    // Approximated as a solid CYLINDER of the same radius
                    // and overall length (2*(halfHeight+radius)) - a true
                    // capsule's exact inertia (cylinder + two hemisphere
                    // caps, each with their own parallel-axis contribution)
                    // is more involved. Consistent with the narrow phase's
                    // existing Box-Capsule approximation - good enough for
                    // believable tumbling, not claiming exactness.
                    float r = collider.radius * 0.5f * (scale.x + scale.z);
                    float halfHeight = collider.halfHeight * scale.y;
                    float length = 2.0f * (halfHeight + r);
                    float spinInertia = 0.5f * mass * r * r; // about the capsule's own long axis (local Y)
                    float perpInertia = (mass / 12.0f) * (3.0f * r * r + length * length); // about axes perpendicular to it
                    localInertia = glm::vec3(perpInertia, spinInertia, perpInertia);
                    break;
                }
                case ColliderShape::Plane:
                    // A Dynamic Plane isn't a physically meaningful
                    // configuration (infinite flat geometry with finite
                    // mass) and isn't really supported anywhere else in
                    // this system either - small arbitrary fallback so
                    // this doesn't produce a zero-divide if one is ever
                    // created.
                    localInertia = glm::vec3(mass * 0.1f);
                    break;

                case ColliderShape::Voxel:
                    // Unreachable in practice - Voxel colliders are
                    // always Static (see ColliderShape::Voxel's
                    // comment), and the inverseMass <= 0.0f check above
                    // already returns before this switch for any
                    // non-positive-mass body. Same arbitrary-fallback
                    // reasoning as Plane in case that assumption is ever
                    // violated.
                    localInertia = glm::vec3(mass * 0.1f);
                    break;
            }

            return glm::vec3(
                localInertia.x > 0.0f ? 1.0f / localInertia.x : 0.0f,
                localInertia.y > 0.0f ? 1.0f / localInertia.y : 0.0f,
                localInertia.z > 0.0f ? 1.0f / localInertia.z : 0.0f);
        }

        // Rotates a diagonal local inverse-inertia tensor into world
        // space: I_world^-1 = R * diag(I_local^-1) * R^T. Needs to be
        // redone every time the body's orientation changes (i.e. every
        // resolution call, not cached) since it depends on current
        // rotation.
        glm::mat3 WorldInverseInertiaTensor(const glm::vec3& localInverseInertia, const glm::mat3& rotation)
        {
            glm::mat3 diagonal(0.0f);
            diagonal[0][0] = localInverseInertia.x;
            diagonal[1][1] = localInverseInertia.y;
            diagonal[2][2] = localInverseInertia.z;
            return rotation * diagonal * glm::transpose(rotation);
        }

        // Simple union-find (path compression, no union-by-rank - contact
        // graphs here are small and rebuilt fresh every step, doesn't
        // need to be maximally optimal) used to group entities into
        // connected "islands" via this step's contacts, so sleep can wake
        // an entire touching stack at once instead of one hop per step.
        // See the wake-propagation pass in Step().
        struct UnionFind
        {
            std::unordered_map<Entity, Entity> parent;

            Entity Find(Entity e)
            {
                auto it = parent.find(e);
                if (it == parent.end())
                {
                    parent[e] = e;
                    return e;
                }
                if (it->second == e)
                {
                    return e;
                }
                Entity root = Find(it->second);
                parent[e] = root; // path compression
                return root;
            }

            void Union(Entity a, Entity b)
            {
                Entity rootA = Find(a);
                Entity rootB = Find(b);
                if (rootA != rootB)
                {
                    parent[rootA] = rootB;
                }
            }
        };

        void ResolveContact(ComponentStorage<TransformComponent>& transforms,
                             ComponentStorage<RigidBodyComponent>& bodies,
                             ComponentStorage<ColliderComponent>& colliders,
                             const Contact& contact, float restitution, float friction,
                             float correctionPercent, float correctionSlop, bool forceWake,
                             bool applyPositionalCorrection, float maxAngularSpeed)
        {
            auto& transformA = transforms.Get(contact.a);
            auto& transformB = transforms.Get(contact.b);
            const ColliderComponent& colliderA = colliders.Get(contact.a);
            const ColliderComponent& colliderB = colliders.Get(contact.b);

            // A collider with no RigidBodyComponent is treated as
            // immovable static geometry (infinite mass) - e.g. level
            // geometry that was never meant to simulate. Local dummies,
            // not shared, so writes to one never alias the other.
            RigidBodyComponent staticDummyA;
            staticDummyA.type = BodyType::Static;
            staticDummyA.inverseMass = 0.0f;
            RigidBodyComponent staticDummyB;
            staticDummyB.type = BodyType::Static;
            staticDummyB.inverseMass = 0.0f;

            bool hasBodyA = bodies.Has(contact.a);
            bool hasBodyB = bodies.Has(contact.b);
            RigidBodyComponent& bodyA = hasBodyA ? bodies.Get(contact.a) : staticDummyA;
            RigidBodyComponent& bodyB = hasBodyB ? bodies.Get(contact.b) : staticDummyB;

            // Wake decision happens FIRST, using whatever velocity these
            // bodies actually have right now (a still-sleeping body's
            // velocity is exactly 0, so this naturally reads as "no
            // disturbance" unless the OTHER side is moving fast into it
            // or forceWake is set). See kWakeVelocityThreshold's comment
            // below for why 0.5 and not something smaller.
            //
            // Sleeping bodies: normally only wake for a genuinely new
            // impact (real velocity into the contact), not for continuing
            // to rest against something it's already settled on -
            // otherwise a sleeping body touching static geometry would
            // wake up again every single step just from the resting
            // contact itself, which defeats the point of sleeping.
            // `forceWake` bypasses that: it's set by Step() when either
            // side of this contact was TELEPORTED this step (an
            // inspector drag, a script setting position directly) rather
            // than moved by velocity - a case the velocity check below
            // can't see at all, since a teleport has no velocity
            // associated with it. Step() also runs a separate sleep-
            // island pass (see the union-find block there) that can wake
            // a whole connected stack from one disturbance in a single
            // step; this per-pair check is what that pass ultimately
            // relies on for its own "is this a genuine disturbance"
            // decision.
            //
            // kWakeVelocityThreshold must clear ordinary per-step gravity
            // accumulation (fixedDeltaTime * |gravity|, ~0.164 at the
            // default 60Hz/9.81 settings) or it's not measuring a real
            // disturbance at all - an earlier value of 0.15 sat right on
            // top of that noise floor, so ANY awake body resting near a
            // sleeping one would spuriously "impact" it every step from
            // gravity alone, before that step's own resolution had a
            // chance to cancel it back out.
            constexpr float kWakeVelocityThreshold = 0.5f;
            if (bodyA.isSleeping || bodyB.isSleeping)
            {
                glm::vec3 preWakeRelativeVelocity = bodyB.linearVelocity - bodyA.linearVelocity;
                float preWakeVelocityAlongNormal = glm::dot(preWakeRelativeVelocity, contact.normal);

                if (forceWake || glm::abs(preWakeVelocityAlongNormal) > kWakeVelocityThreshold)
                {
                    bodyA.isSleeping = false;
                    bodyA.sleepTimer = 0.0f;
                    bodyB.isSleeping = false;
                    bodyB.sleepTimer = 0.0f;
                }
                // else: stays asleep - NOT an early return. A body that
                // stays asleep still needs to act as a support for
                // whatever's resting on it; the fix for that is below,
                // in how invMass gets computed, not by skipping this
                // contact entirely. An earlier version of this function
                // DID return early here, which seemed like the obviously
                // safe choice ("don't touch a sleeping body's state") but
                // was a real bug: it meant a contact against a sleeping
                // body was never resolved AT ALL until something woke it,
                // so anything resting on a sleeping body in a stack had
                // nothing holding it up and free-fell straight through.
                // Caught by watching a 3-box stack's top two boxes gain
                // huge velocity and sink into the bottom one the instant
                // it fell asleep.
            }

            // Static/Kinematic bodies are infinite mass for resolution
            // purposes regardless of whatever's stored in inverseMass -
            // PhysicsSystem::Step already enforces this for Static, but
            // Kinematic bodies keep a real velocity (they're externally
            // driven), so it has to be re-enforced here rather than
            // trusted from the component. A body that's STILL sleeping
            // after the wake check above is treated the same way - a
            // temporary immovable anchor for this contact, exactly like
            // Static, until something actually wakes it.
            float invMassA = (bodyA.type == BodyType::Dynamic && !bodyA.isSleeping) ? bodyA.inverseMass : 0.0f;
            float invMassB = (bodyB.type == BodyType::Dynamic && !bodyB.isSleeping) ? bodyB.inverseMass : 0.0f;
            float invMassSum = invMassA + invMassB;

            if (invMassSum <= 0.0f)
            {
                // Both sides immovable (two Static bodies, a Static/
                // Kinematic pair, or - now correctly handled the same
                // way - two sleeping bodies touching each other) -
                // geometrically overlapping but nothing here can or
                // should move to fix it.
                return;
            }

            // Rotational inertia, in WORLD space (must be recomputed
            // every call - it depends on current orientation, not
            // cached). A body with invMassA/B == 0 (Static/Kinematic/
            // sleeping/no RigidBodyComponent) automatically gets a zero
            // inverse-inertia tensor too - see ComputeLocalInverseInertia
            // - so it never picks up angular velocity here, same as it
            // never picks up linear velocity.
            glm::mat3 rotationA = SDF::RotationMatrixFromDegrees(transformA.rotationDegrees);
            glm::mat3 rotationB = SDF::RotationMatrixFromDegrees(transformB.rotationDegrees);
            glm::mat3 invInertiaA = WorldInverseInertiaTensor(ComputeLocalInverseInertia(colliderA, invMassA, transformA.scale), rotationA);
            glm::mat3 invInertiaB = WorldInverseInertiaTensor(ComputeLocalInverseInertia(colliderB, invMassB, transformB.scale), rotationB);

            // Lever arms from each body's center of mass (its collider's
            // actual world center, matching how the narrow phase itself
            // computes position - see localOffset's doc comment) to the
            // contact point. This is what makes an off-center hit produce
            // torque instead of only a straight-line push - a box landing
            // corner-first gets a lever arm far from its center, and the
            // resulting angular impulse is what makes it topple instead
            // of balancing on the corner forever, which is the actual
            // point of this whole rewrite: an earlier version of this
            // function only ever applied LINEAR impulses, so orientation
            // was never affected by collisions at all.
            glm::vec3 centerA = transformA.position + rotationA * colliderA.localOffset;
            glm::vec3 centerB = transformB.position + rotationB * colliderB.localOffset;
            glm::vec3 rA = contact.point - centerA;
            glm::vec3 rB = contact.point - centerB;

            // bodyA.angularVelocity / bodyB.angularVelocity are stored in
            // DEGREES per second - that's what TransformComponent's
            // rotationDegrees and PhysicsSystem's direct-add integration
            // use throughout Eden, and what the UI labels. But every
            // formula below (moment of inertia, r x impulse, v = omega x
            // r) is standard rigid-body physics, which is only
            // dimensionally correct in RADIANS per second - that's not a
            // style choice, it falls directly out of how the cross-
            // product relationship between angular and linear velocity is
            // derived. Working copies here convert once at the start and
            // convert back once at the end, so the physics math in
            // between doesn't need to know or care about the degrees-
            // based storage convention.
            //
            // This is a real bug that shipped: without this conversion,
            // every collision-generated spin was ~57x weaker than
            // physically correct (1 radian = ~57.3 degrees) - the radian-
            // based impulse math was writing its result directly into a
            // degrees-based field with no conversion at all. Reported
            // directly: setting angularVelocity to something like 5
            // produced almost no visible rotation, because 5 was being
            // treated as 5 degrees/sec instead of the ~286 degrees/sec
            // that "5 radians/sec" actually is.
            glm::vec3 angularVelocityA = glm::radians(bodyA.angularVelocity);
            glm::vec3 angularVelocityB = glm::radians(bodyB.angularVelocity);

            // Velocity of each body's MATERIAL POINT at the contact
            // location - includes the contribution from spin
            // (angularVelocity x r), not just the two centers' straight-
            // line velocities. This is what actually determines whether
            // the surfaces are approaching or separating right at the
            // contact, which matters once bodies can spin.
            glm::vec3 velocityAtA = bodyA.linearVelocity + glm::cross(angularVelocityA, rA);
            glm::vec3 velocityAtB = bodyB.linearVelocity + glm::cross(angularVelocityB, rB);
            glm::vec3 relativeVelocity = velocityAtB - velocityAtA;
            float velocityAlongNormal = glm::dot(relativeVelocity, contact.normal);

            if (velocityAlongNormal < 0.0f)
            {
                // Effective mass along the normal, now including each
                // body's angular resistance to the impulse: a hit far
                // from the center of mass spins the body more and
                // linearly accelerates it less than the same impulse
                // applied at the center would. This term is the standard
                // rigid-body-contact formula for that trade-off.
                glm::vec3 angularTermA = glm::cross(invInertiaA * glm::cross(rA, contact.normal), rA);
                glm::vec3 angularTermB = glm::cross(invInertiaB * glm::cross(rB, contact.normal), rB);
                float denom = invMassSum + glm::dot(angularTermA + angularTermB, contact.normal);
                if (denom <= 0.0f)
                {
                    denom = invMassSum; // guard against a degenerate inertia configuration; invMassSum > 0 is already guaranteed here
                }

                float j = -(1.0f + restitution) * velocityAlongNormal / denom;
                glm::vec3 impulse = j * contact.normal;

                bodyA.linearVelocity -= impulse * invMassA;
                bodyB.linearVelocity += impulse * invMassB;
                angularVelocityA -= invInertiaA * glm::cross(rA, impulse);
                angularVelocityB += invInertiaB * glm::cross(rB, impulse);

                // Coulomb friction: same idea as before (clamp the
                // tangential impulse to friction * the normal impulse
                // magnitude), now measured and applied at the contact
                // point's actual relative velocity - including spin - and
                // producing torque of its own, same as the normal impulse
                // does. A tangential force applied off-center is exactly
                // what makes something start rolling/spinning from
                // friction, not just sliding.
                glm::vec3 velocityAtAAfter = bodyA.linearVelocity + glm::cross(angularVelocityA, rA);
                glm::vec3 velocityAtBAfter = bodyB.linearVelocity + glm::cross(angularVelocityB, rB);
                glm::vec3 relativeVelocityAfterNormal = velocityAtBAfter - velocityAtAAfter;
                glm::vec3 tangentVelocity = relativeVelocityAfterNormal - contact.normal * glm::dot(relativeVelocityAfterNormal, contact.normal);
                float tangentSpeed = glm::length(tangentVelocity);

                if (tangentSpeed > 1e-5f)
                {
                    glm::vec3 tangent = tangentVelocity / tangentSpeed;

                    glm::vec3 angularTermAt = glm::cross(invInertiaA * glm::cross(rA, tangent), rA);
                    glm::vec3 angularTermBt = glm::cross(invInertiaB * glm::cross(rB, tangent), rB);
                    float denomT = invMassSum + glm::dot(angularTermAt + angularTermBt, tangent);
                    if (denomT <= 0.0f)
                    {
                        denomT = invMassSum;
                    }

                    float jt = -glm::dot(relativeVelocityAfterNormal, tangent) / denomT;
                    float maxFriction = j * friction;
                    jt = glm::clamp(jt, -maxFriction, maxFriction);

                    glm::vec3 frictionImpulse = jt * tangent;
                    bodyA.linearVelocity -= frictionImpulse * invMassA;
                    bodyB.linearVelocity += frictionImpulse * invMassB;
                    angularVelocityA -= invInertiaA * glm::cross(rA, frictionImpulse);
                    angularVelocityB += invInertiaB * glm::cross(rB, frictionImpulse);
                }

                // See maxAngularSpeed's doc comment in CollisionSystem.h -
                // this bounds a real solver-stability failure mode
                // (friction torque feedback), not just a defensive
                // formality. Operates on the radian-space value, matching
                // the "radians/sec" the field is documented in.
                float angularSpeedA = glm::length(angularVelocityA);
                if (angularSpeedA > maxAngularSpeed)
                {
                    angularVelocityA *= (maxAngularSpeed / angularSpeedA);
                }
                float angularSpeedB = glm::length(angularVelocityB);
                if (angularSpeedB > maxAngularSpeed)
                {
                    angularVelocityB *= (maxAngularSpeed / angularSpeedB);
                }
            }
            // else: already separating along the normal - no velocity
            // impulse needed, but the bodies can still be geometrically
            // overlapping (e.g. one step's worth of penetration from
            // last frame), so positional correction below still applies.

            // Convert back to degrees/sec ONCE, here, regardless of which
            // branch above ran (velocityAlongNormal >= 0 still needs this
            // - angularVelocityA/B were read from the stored degrees
            // value at the top, so even an unmodified pass-through must
            // be written back consistently, not just skipped).
            bodyA.angularVelocity = glm::degrees(angularVelocityA);
            bodyB.angularVelocity = glm::degrees(angularVelocityB);

            if (applyPositionalCorrection)
            {
                // Only run this on ONE of the resolutionIterations
                // passes (the caller decides which, typically the last)
                // - contact.penetration is a fixed snapshot from
                // detection, not something that gets recomputed as
                // bodies move. Velocity impulses above are safe to
                // re-run every iteration because they read CURRENT
                // velocity each time and naturally converge; running
                // this same fixed correction multiple times per step
                // would instead multiply it, moving bodies several times
                // further than intended - a real bug this comment exists
                // to stop from being reintroduced (caught by testing a
                // 3-box stack, which visibly flew apart when this ran on
                // every iteration).
                //
                // Deliberately LINEAR only, not angular - positional
                // correction just resolves interpenetration by translating
                // the bodies apart; it doesn't try to correct orientation.
                // The velocity-based angular impulse above is what
                // actually produces rotation, and it does so gradually
                // over subsequent steps, which is the normal/expected way
                // this kind of solver handles it.
                float correctionMagnitude = std::max(contact.penetration - correctionSlop, 0.0f) * correctionPercent / invMassSum;
                glm::vec3 correction = correctionMagnitude * contact.normal;
                transformA.position -= correction * invMassA;
                transformB.position += correction * invMassB;
            }
        }
        // SDF-based conservative advancement ("sphere tracing" applied to
        // movement, same core idea as ray marching): instead of sampling
        // a FIXED number of points along the body's path - which can
        // still skip a thin obstacle if travel distance per sample
        // exceeds the obstacle's thickness, no matter how many samples
        // you use - this uses the actual SDF distance value to know
        // exactly how far it's safe to move before checking again. That
        // distance is a real lower bound on how close anything else is,
        // so advancing by it can never skip past a surface, regardless
        // of speed. This is why building this on real SDFs earlier
        // (rather than say, precomputed convex-hull collision) pays off
        // here specifically.
        //
        // Still an approximation in one place: the MOVING body itself is
        // treated as a bounding sphere of `approxRadius` for this pass
        // (not its true Box/Capsule shape) - standard simplification,
        // conservative (the true shape fits inside that sphere in the
        // dimension that matters, its smallest half-extent), and it's
        // what the discrete resolution pass right after this corrects
        // for anyway once the body's stopped near the real surface.
        void ConservativeAdvanceDynamicBodies(const std::vector<Entity>& entities,
                                               ComponentStorage<TransformComponent>& transforms,
                                               ComponentStorage<ColliderComponent>& colliders,
                                               ComponentStorage<RigidBodyComponent>& bodies,
                                               float fixedDeltaTime, int maxIterations,
                                               const VoxelSystemGPU* voxelSystem)
        {
            for (Entity entity : entities)
            {
                if (!bodies.Has(entity) || bodies.Get(entity).type != BodyType::Dynamic)
                {
                    continue;
                }

                RigidBodyComponent& body = bodies.Get(entity);
                ColliderComponent& colliderE = colliders.Get(entity);

                if (colliderE.shape == ColliderShape::Plane)
                {
                    // A moving infinite plane isn't physically meaningful
                    // to sweep-test - skip.
                    continue;
                }

                glm::vec3 displacement = body.linearVelocity * fixedDeltaTime;
                float travelDistance = glm::length(displacement);

                float approxRadius = 0.5f;
                switch (colliderE.shape)
                {
                    case ColliderShape::Sphere:  approxRadius = colliderE.radius; break;
                    case ColliderShape::Capsule: approxRadius = colliderE.radius; break;
                    case ColliderShape::Box:     approxRadius = glm::min(colliderE.halfExtents.x, glm::min(colliderE.halfExtents.y, colliderE.halfExtents.z)); break;
                    default: break;
                }

                if (approxRadius <= 0.0f || travelDistance <= approxRadius)
                {
                    // Moved less than roughly its own size this step -
                    // the normal discrete check at the end position is
                    // reliable enough on its own, no marching needed.
                    continue;
                }

                TransformComponent& transformE = transforms.Get(entity);
                glm::vec3 endPos = transformE.position;
                glm::vec3 startPos = endPos - displacement;
                glm::vec3 travelDir = displacement / travelDistance;

                glm::vec3 position = startPos;
                float remaining = travelDistance;
                constexpr float kSurfaceEpsilon = 0.001f; // "close enough to touching" - stop marching

                bool hitSomething = false;

                for (int iteration = 0; iteration < maxIterations && remaining > 0.0f; ++iteration)
                {
                    // The largest step we could possibly need is however
                    // far is left to travel this frame - everything
                    // below only ever shrinks it further.
                    float safeStep = remaining;

                    for (Entity other : entities)
                    {
                        if (other == entity)
                        {
                            continue;
                        }

                        const ColliderComponent& colliderOther = colliders.Get(other);
                        const TransformComponent& transformOther = transforms.Get(other);

                        float distanceToOtherSurface;
                        if (colliderOther.shape == ColliderShape::Voxel)
                        {
                            // SDF::Distance (below) has no Voxel case -
                            // route through the live density field
                            // directly instead. This is the fix for a
                            // real bug: silently skipping Voxel here (an
                            // earlier version of this loop did) let a
                            // fast-moving sphere tunnel past its own
                            // radius INTO a voxel volume's interior
                            // before the discrete pass ever saw it - the
                            // discrete pass then read a large, garbage
                            // penetration (sphere center already deep
                            // inside solid) and ResolveContact corrected
                            // it in one violent shove, every step,
                            // because the sphere immediately fell back
                            // in afterward. Marching against the real
                            // surface here is what actually stops that.
                            distanceToOtherSurface = (voxelSystem != nullptr && colliderOther.voxelVolume != InvalidVoxelVolumeHandle)
                                ? voxelSystem->SampleSignedDistance(colliderOther.voxelVolume, position) - approxRadius
                                : safeStep; // no voxel system available - can't march against it, don't let it shrink safeStep
                        }
                        else
                        {
                            // SDF::Distance works uniformly for Sphere, Box,
                            // Capsule, AND Plane - a real payoff of building
                            // narrow phase on true distance fields instead of
                            // one-off shape-pair code: this loop doesn't need
                            // to know or care which shape `other` is.
                            // `other`'s own scale applied via a scaled
                            // COPY of its shape parameters, same reasoning as
                            // TestSphereVsSDFShape - see SDF::ScaledCollider.
                            ColliderComponent scaledOther = SDF::ScaledCollider(colliderOther, transformOther.scale);
                            glm::vec3 local = SDF::WorldToLocal(position, transformOther, colliderOther);
                            distanceToOtherSurface = SDF::Distance(scaledOther, local) - approxRadius;
                        }

                        if (distanceToOtherSurface < safeStep)
                        {
                            safeStep = distanceToOtherSurface;
                        }
                    }

                    if (safeStep <= kSurfaceEpsilon)
                    {
                        // Already touching (or would be within epsilon) -
                        // stop marching right here. Don't advance any
                        // further; this position is what the discrete
                        // pass below needs to see to detect and resolve
                        // the contact.
                        hitSomething = true;
                        break;
                    }

                    position += travelDir * safeStep;
                    remaining -= safeStep;
                }

                transformE.position = position;

                if (hitSomething)
                {
                    // Hard stop, not a proper bounce: kill whatever part
                    // of this step's velocity was still driving the body
                    // further along its travel direction. The discrete
                    // pass right after this function returns does the
                    // actual impulse/friction/positional-correction
                    // response now that the body is sitting somewhere the
                    // overlap can actually be detected.
                    float speedIntoObstacle = glm::dot(body.linearVelocity, travelDir);
                    if (speedIntoObstacle > 0.0f)
                    {
                        body.linearVelocity -= travelDir * speedIntoObstacle;
                    }
                }
                // else: consumed the whole displacement without getting
                // near anything - position is already endPos (up to
                // floating point error from the iterative sum), nothing
                // else to do.
            }
        }

        // --- Broad phase ---------------------------------------------------

        struct CellKey
        {
            int x, y, z;
            bool operator==(const CellKey& other) const { return x == other.x && y == other.y && z == other.z; }
        };

        struct CellKeyHash
        {
            size_t operator()(const CellKey& key) const
            {
                // Not cryptographic - just needs to spread grid cells
                // reasonably across buckets. Standard hash-combine.
                size_t h = std::hash<int>()(key.x);
                h ^= std::hash<int>()(key.y) + 0x9e3779b9u + (h << 6) + (h >> 2);
                h ^= std::hash<int>()(key.z) + 0x9e3779b9u + (h << 6) + (h >> 2);
                return h;
            }
        };

        CellKey CellOf(const glm::vec3& p, float cellSize)
        {
            return CellKey{
                static_cast<int>(std::floor(p.x / cellSize)),
                static_cast<int>(std::floor(p.y / cellSize)),
                static_cast<int>(std::floor(p.z / cellSize))
            };
        }

        // Rough AABB half-extents per shape in WORLD space, used only to
        // decide which grid cells a collider touches - see
        // ColliderWorldAabbHalfExtents (ColliderBounds.h) for the actual
        // implementation, shared with ParticleSystemGPU's collider cell
        // mask now rather than duplicated here.

        // Uniform-grid broad phase: buckets every non-Plane collider's
        // AABB into grid cells (a collider spanning multiple cells is
        // inserted into all of them), then only generates a candidate
        // pair for two entities that share at least one cell - this is
        // the "cheap to build, cheap to rebuild every frame" approach
        // the Planning Notes call for, over something heavier like a
        // BVH. Plane colliders bypass the grid entirely (an "infinite"
        // shape doesn't bucket meaningfully) and are instead paired
        // against every other entity directly - reasonable since planes
        // are expected to be rare, static level geometry, not something
        // you'll have thousands of.
        // Adaptive broad-phase cell size: takes the average world-space
        // diameter of every non-Plane collider present this step and
        // uses roughly double that as the cell size - the sweet spot for
        // a uniform grid is a typical object spanning about one cell (too
        // small and objects span many cells; too large and cells hold
        // many unrelated entities, drifting back toward brute force).
        // Falls back to a fixed default when there's nothing measurable
        // yet (first frame, or a scene that's Plane-only).
        float ComputeAdaptiveCellSize(const std::vector<Entity>& entities,
                                       ComponentStorage<ColliderComponent>& colliders,
                                       ComponentStorage<TransformComponent>& transforms)
        {
            float totalDiameter = 0.0f;
            int count = 0;

            for (Entity entity : entities)
            {
                const ColliderComponent& collider = colliders.Get(entity);
                if (collider.shape == ColliderShape::Plane)
                {
                    continue; // "infinite" - not a meaningful size sample
                }

                glm::vec3 half = ColliderWorldAabbHalfExtents(collider, transforms.Get(entity).rotationDegrees, transforms.Get(entity).scale);
                totalDiameter += 2.0f * glm::max(half.x, glm::max(half.y, half.z));
                ++count;
            }

            if (count == 0)
            {
                return 4.0f;
            }

            float averageDiameter = totalDiameter / static_cast<float>(count);
            return glm::clamp(averageDiameter * 2.0f, 0.25f, 500.0f);
        }

        std::vector<std::pair<Entity, Entity>> BuildCandidatePairs(
            const std::vector<Entity>& entities,
            ComponentStorage<TransformComponent>& transforms,
            ComponentStorage<ColliderComponent>& colliders,
            float cellSize)
        {
            std::unordered_map<CellKey, std::vector<Entity>, CellKeyHash> grid;
            std::vector<Entity> planeEntities;

            for (Entity entity : entities)
            {
                const ColliderComponent& collider = colliders.Get(entity);
                if (collider.shape == ColliderShape::Plane)
                {
                    planeEntities.push_back(entity);
                    continue;
                }

                const TransformComponent& entityTransform = transforms.Get(entity);
                glm::mat3 rotation = SDF::RotationMatrixFromDegrees(entityTransform.rotationDegrees);
                glm::vec3 center = entityTransform.position + rotation * collider.localOffset;
                glm::vec3 half = ColliderWorldAabbHalfExtents(collider, entityTransform.rotationDegrees, entityTransform.scale);

                CellKey minCell = CellOf(center - half, cellSize);
                CellKey maxCell = CellOf(center + half, cellSize);

                for (int cx = minCell.x; cx <= maxCell.x; ++cx)
                {
                    for (int cy = minCell.y; cy <= maxCell.y; ++cy)
                    {
                        for (int cz = minCell.z; cz <= maxCell.z; ++cz)
                        {
                            grid[CellKey{ cx, cy, cz }].push_back(entity);
                        }
                    }
                }
            }

            // Dedupe with a std::set of canonical (min,max) pairs - an
            // entity spanning several cells would otherwise generate the
            // same pair once per shared cell. A hash set would be faster
            // at large candidate counts, but this stays simple and the
            // whole point of the grid is that this set should be small
            // relative to entities.size()^2.
            std::set<std::pair<Entity, Entity>> uniquePairs;

            for (const auto& [key, cellEntities] : grid)
            {
                for (size_t i = 0; i < cellEntities.size(); ++i)
                {
                    for (size_t j = i + 1; j < cellEntities.size(); ++j)
                    {
                        Entity a = cellEntities[i];
                        Entity b = cellEntities[j];
                        uniquePairs.insert(a < b ? std::make_pair(a, b) : std::make_pair(b, a));
                    }
                }
            }

            for (Entity plane : planeEntities)
            {
                for (Entity other : entities)
                {
                    if (other == plane)
                    {
                        continue;
                    }
                    uniquePairs.insert(plane < other ? std::make_pair(plane, other) : std::make_pair(other, plane));
                }
            }

            return std::vector<std::pair<Entity, Entity>>(uniquePairs.begin(), uniquePairs.end());
        }

        // Complements ConservativeAdvanceDynamicBodies, which sweeps a
        // DYNAMIC body's own velocity-driven movement. This handles the
        // mirror case: a Static/Kinematic body (or a bodyless collider,
        // treated as static geometry - see ResolveContact) that got
        // TELEPORTED a meaningful distance/rotation this step - an
        // inspector drag, a script setting its position/rotation
        // directly, anything that isn't PhysicsSystem integrating a
        // velocity (see the external-move detection in Step()).
        //
        // Two different responses depending on how big the move was
        // relative to the mover's own size, because one approach can't
        // handle both cases well:
        //
        // SMALL movement (<= the mover's own approximate size this
        // step - ordinary dragging, shaking, a slow residual from
        // something else): resolve using the mover's FINAL position and
        // the SAME friction-aware impulse math every other contact uses,
        // by briefly borrowing its velocity field. This is what makes
        // something on a gently-shaken platform respond realistically -
        // slip if shaken hard enough, not be glued to it no matter what,
        // which an earlier version of this (always doing a blind 1:1
        // position carry) got wrong.
        //
        // LARGE movement (bigger than the mover's own size in one step -
        // a big single-frame jump): fall back to sampling along the
        // mover's path and, if `other` was found anywhere in that path,
        // carrying it by the mover's FULL raw displacement. This is
        // deliberately NOT the friction-based approach above - a fast,
        // large mover can fully swallow a much thinner resting object,
        // and resolving THAT via the mover's final-position contact
        // normal picks whichever axis has the SHALLOWEST overlap (a real
        // SAT property), which for a swallowed thin object is very often
        // "push it out the bottom" rather than "put it back on top." Full
        // displacement carry sidesteps that failure mode entirely -
        // caught by this exact regression test when this branch was
        // first written, then broken again and caught again when a
        // later change tried to unify both cases into one approach.
        void SweepMoversAgainstDynamics(const std::vector<Entity>& entities,
                                         ComponentStorage<TransformComponent>& transforms,
                                         ComponentStorage<ColliderComponent>& colliders,
                                         ComponentStorage<RigidBodyComponent>& bodies,
                                         const std::unordered_map<Entity, glm::vec3>& moverDisplacements,
                                         float fixedDeltaTime, int maxSamples,
                                         float restitution, float friction,
                                         float correctionPercent, float correctionSlop,
                                         float maxBorrowedSpeed, float maxAngularSpeed)
        {
            for (const auto& [mover, displacement] : moverDisplacements)
            {
                float travelDistance = glm::length(displacement);
                if (travelDistance < 0.0001f)
                {
                    // No positional movement this step - this entry
                    // exists purely for a rotation-only disturbance (see
                    // Step()'s external-move detection). Nothing to
                    // sweep; forceWake in the normal resolution pass
                    // later in Step() is what actually handles it.
                    continue;
                }

                const ColliderComponent& moverCollider = colliders.Get(mover);
                if (moverCollider.shape == ColliderShape::Plane)
                {
                    // An infinite plane's own "path" isn't meaningful to
                    // sweep - its already-infinite extent means overlap
                    // is already correctly caught at its final position
                    // regardless of how far it moved.
                    continue;
                }

                const TransformComponent& moverTransform = transforms.Get(mover);
                glm::vec3 moverHalf = ColliderWorldAabbHalfExtents(moverCollider, moverTransform.rotationDegrees, moverTransform.scale);
                float moverApproxRadius = glm::min(moverHalf.x, glm::min(moverHalf.y, moverHalf.z));
                if (moverApproxRadius <= 0.0f)
                {
                    moverApproxRadius = 0.5f;
                }

                bool isSmallMovement = travelDistance <= moverApproxRadius;

                glm::vec3 endPos = moverTransform.position; // current (post-move) position
                glm::vec3 startPos = endPos - displacement;

                TransformComponent moverSampleTransform;
                moverSampleTransform.rotationDegrees = moverTransform.rotationDegrees;

                for (Entity other : entities)
                {
                    if (other == mover || !bodies.Has(other) || bodies.Get(other).type != BodyType::Dynamic)
                    {
                        continue;
                    }

                    const ColliderComponent& otherCollider = colliders.Get(other);
                    const TransformComponent& otherTransform = transforms.Get(other);

                    bool disturbed = false;

                    if (isSmallMovement)
                    {
                        moverSampleTransform.position = endPos;
                        ContactInfo finalContact = TestCollision(moverCollider, moverSampleTransform, otherCollider, otherTransform);

                        if (finalContact.colliding)
                        {
                            if (bodies.Has(mover))
                            {
                                // Borrow the mover's velocity field for
                                // exactly one resolution call, representing
                                // how fast it actually moved this step even
                                // though it got here by teleport, not
                                // integration - then restore whatever was
                                // there before, so nothing about the mover's
                                // real state changes. This routes the
                                // interaction through the exact same
                                // friction/mass-aware math a normal moving
                                // Kinematic body already gets, so how much
                                // of the mover's motion transfers to `other`
                                // depends on friction and mass, not a rigid
                                // 1:1 copy.
                                RigidBodyComponent& moverBody = bodies.Get(mover);
                                glm::vec3 savedVelocity = moverBody.linearVelocity;
                                bool savedSleeping = moverBody.isSleeping;

                                // Cap the implied velocity - see
                                // maxBorrowedSpeed's doc comment in
                                // CollisionSystem.h. Without this, even a
                                // "micro" drag could imply several m/s once
                                // divided by the small fixed timestep,
                                // launching whatever was resting on top.
                                glm::vec3 borrowedVelocity = displacement / fixedDeltaTime;
                                float borrowedSpeed = glm::length(borrowedVelocity);
                                if (borrowedSpeed > maxBorrowedSpeed && borrowedSpeed > 0.0f)
                                {
                                    borrowedVelocity *= (maxBorrowedSpeed / borrowedSpeed);
                                }
                                moverBody.linearVelocity = borrowedVelocity;
                                moverBody.isSleeping = false;

                                Contact synthetic{ mover, other, finalContact.normal, finalContact.penetration, finalContact.point };
                                ResolveContact(transforms, bodies, colliders, synthetic, restitution, friction,
                                                correctionPercent, correctionSlop, /*forceWake*/true,
                                                /*applyPositionalCorrection*/true, maxAngularSpeed);

                                moverBody.linearVelocity = savedVelocity;
                                moverBody.isSleeping = savedSleeping;
                            }
                            else
                            {
                                // No RigidBodyComponent on the mover -
                                // nothing to borrow a velocity from. Push
                                // `other` directly using the real contact
                                // geometry - still direction-correct, just
                                // not friction/mass-weighted.
                                transforms.Get(other).position += finalContact.normal * finalContact.penetration;
                            }
                            disturbed = true;
                        }
                    }
                    else
                    {
                        int samples = glm::clamp(static_cast<int>(glm::ceil(travelDistance / moverApproxRadius)), 1, maxSamples);

                        for (int i = 0; i <= samples; ++i)
                        {
                            float t = static_cast<float>(i) / static_cast<float>(samples);
                            moverSampleTransform.position = glm::mix(startPos, endPos, t);

                            if (TestCollision(moverCollider, moverSampleTransform, otherCollider, otherTransform).colliding)
                            {
                                transforms.Get(other).position += displacement;
                                disturbed = true;
                                break;
                            }
                        }
                    }

                    if (!disturbed)
                    {
                        continue;
                    }

                    RigidBodyComponent& otherBody = bodies.Get(other);
                    if (otherBody.isSleeping)
                    {
                        otherBody.isSleeping = false;
                        otherBody.sleepTimer = 0.0f;
                    }
                }
            }
        }
    }

    void CollisionSystem::Step(Registry& registry, float fixedDeltaTime, const VoxelSystemGPU* voxelSystem)
    {
        auto& transforms = registry.GetStorage<TransformComponent>();
        auto& colliders = registry.GetStorage<ColliderComponent>();
        auto& bodies = registry.GetStorage<RigidBodyComponent>();

        std::vector<Entity> entities = registry.View<TransformComponent, ColliderComponent>();

        // --- External-move detection (see CollisionSystem.h) ---------
        //
        // Compares each entity's position AND rotation to what they were
        // at the END of the PREVIOUS Step() call (m_lastKnownPositions/
        // Rotations are only updated once, at the very bottom of this
        // function, after everything - sweep, resolution, sleep
        // bookkeeping - is finished). That ordering matters: comparing
        // against a snapshot taken at the START of the previous step
        // instead (before ITS OWN positional correction ran) would
        // misread normal end-of-step settling movement as an "external"
        // move and immediately re-wake a body the instant it fell
        // asleep - a real bug an earlier version of this function had.
        //
        // For a Kinematic body specifically - the one BodyType
        // PhysicsSystem actually integrates position/rotation for from a
        // real velocity every step - only the part of this step's change
        // NOT explained by that velocity counts as "external." Without
        // this, a Kinematic platform moving under its own set velocity
        // would look like a fresh teleport every single step (its
        // position genuinely does change every step), triggering the
        // mover-sweep carry ON TOP OF the friction-based push the normal
        // resolution pass already applies for a body with real velocity
        // - doubling its effect on anything resting on it. This was a
        // real, reported bug: a slowly-moving Kinematic platform made
        // resting objects visibly slide by an amplified amount. Static
        // bodies never legitimately move via velocity (PhysicsSystem
        // never integrates them), so for them the full observed change
        // is always "external," same as before.
        std::unordered_map<Entity, glm::vec3> teleportedMovers; // entity -> external ("extra," not velocity-explained) position displacement this step
        {
            constexpr float kTeleportEpsilon = 0.0001f;
            constexpr float kRotationEpsilonDegrees = 0.05f;

            for (Entity entity : entities)
            {
                const TransformComponent& currentTransform = transforms.Get(entity);
                auto posIt = m_lastKnownPositions.find(entity);
                auto rotIt = m_lastKnownRotations.find(entity);

                if (posIt == m_lastKnownPositions.end())
                {
                    continue; // first time seeing this entity - nothing to compare against yet
                }

                glm::vec3 observedPositionDelta = currentTransform.position - posIt->second;
                glm::vec3 observedRotationDelta = (rotIt != m_lastKnownRotations.end())
                    ? (currentTransform.rotationDegrees - rotIt->second)
                    : glm::vec3(0.0f);

                glm::vec3 expectedPositionDelta(0.0f);
                glm::vec3 expectedRotationDelta(0.0f);
                if (bodies.Has(entity) && bodies.Get(entity).type == BodyType::Kinematic)
                {
                    const RigidBodyComponent& kinematicBody = bodies.Get(entity);
                    expectedPositionDelta = kinematicBody.linearVelocity * fixedDeltaTime;
                    expectedRotationDelta = kinematicBody.angularVelocity * fixedDeltaTime;
                }

                glm::vec3 extraPositionDelta = observedPositionDelta - expectedPositionDelta;
                glm::vec3 extraRotationDelta = observedRotationDelta - expectedRotationDelta;

                bool positionDisturbed = glm::length(extraPositionDelta) > kTeleportEpsilon;
                bool rotationDisturbed = glm::length(extraRotationDelta) > kRotationEpsilonDegrees;

                if (!positionDisturbed && !rotationDisturbed)
                {
                    continue;
                }

                if (!bodies.Has(entity))
                {
                    // A collider with no RigidBodyComponent is treated as
                    // static geometry (see ResolveContact) - if it moved
                    // or rotated, anything sleeping against it needs to
                    // know.
                    teleportedMovers[entity] = extraPositionDelta;
                    continue;
                }

                RigidBodyComponent& body = bodies.Get(entity);
                if (body.type == BodyType::Static || body.type == BodyType::Kinematic)
                {
                    // Present in the map even for a rotation-only change
                    // with zero position delta - SweepMoversAgainstDynamics
                    // skips the positional sweep for those (nothing to
                    // sweep), but the entry still makes forceWake fire
                    // for anything touching this entity in the normal
                    // resolution pass later in Step(), which is what
                    // actually pushes a resting object out of a newly-
                    // rotated shape.
                    teleportedMovers[entity] = extraPositionDelta;
                }
                else if (body.isSleeping)
                {
                    // A sleeping Dynamic body doesn't move OR rotate on
                    // its own - if either changed anyway, something
                    // external did it (inspector drag, script). Wake it
                    // directly rather than waiting for a contact to do
                    // it.
                    body.isSleeping = false;
                    body.sleepTimer = 0.0f;
                }
            }
        }

        // Handle fast-dragged/rotated Static/Kinematic bodies BEFORE
        // anything else this step - see SweepMoversAgainstDynamics above.
        // This has to run before both the Dynamic-body CCD sweep and the
        // normal discrete pass, so anything it carries/pushes is already
        // in its corrected position for the rest of this step to see.
        SweepMoversAgainstDynamics(entities, transforms, colliders, bodies, teleportedMovers,
                                    fixedDeltaTime, maxSweepSubsteps,
                                    restitution, friction, positionalCorrectionPercent, positionalCorrectionSlop,
                                    maxMoverBorrowedSpeed, maxAngularSpeed);

        if (enableContinuousCollisionSweep)
        {
            ConservativeAdvanceDynamicBodies(entities, transforms, colliders, bodies, fixedDeltaTime, maxSweepSubsteps, voxelSystem);
        }

        if (adaptiveBroadPhaseCellSize)
        {
            float target = ComputeAdaptiveCellSize(entities, colliders, transforms);
            // EMA-smoothed rather than snapping straight to the new
            // target every step - a grid whose cell size changes
            // abruptly frame to frame would reshuffle which entities
            // share a cell for no physical reason, adding noise for
            // nothing.
            constexpr float kSmoothing = 0.1f;
            m_adaptiveCellSize += (target - m_adaptiveCellSize) * kSmoothing;
            broadPhaseCellSize = m_adaptiveCellSize; // mirror into the public field so the UI shows the live value
        }

        // Real broad phase now - see BuildCandidatePairs above. Only
        // entities that share a grid cell (or involve a Plane) get a
        // narrow-phase test at all.
        std::vector<Contact> contacts;
        for (const auto& [a, b] : BuildCandidatePairs(entities, transforms, colliders, broadPhaseCellSize))
        {
            // Two immovable bodies (no RigidBodyComponent - static
            // level geometry - or Static/Kinematic) can never resolve
            // against each other; skip the pair test entirely rather
            // than detect a contact ResolveContact would just no-op
            // on every single step.
            bool aDynamic = bodies.Has(a) && bodies.Get(a).type == BodyType::Dynamic;
            bool bDynamic = bodies.Has(b) && bodies.Get(b).type == BodyType::Dynamic;
            if (!aDynamic && !bDynamic)
            {
                continue;
            }

            ContactInfo info = TestCollision(colliders.Get(a), transforms.Get(a), colliders.Get(b), transforms.Get(b), voxelSystem);
            if (info.colliding)
            {
                contacts.push_back({ a, b, info.normal, info.penetration, info.point });
            }
        }

        // --- Sleep islands ---------------------------------------------
        //
        // An earlier version of this only woke the two entities directly
        // involved in whatever triggered a wake - correct for a single
        // pair, but a tall resting stack would wake one level per step as
        // the disturbance propagated upward through successive contacts,
        // introducing a real (if usually brief) lag. This groups every
        // entity touching in a contact this step into connected
        // components via union-find, then wakes every Dynamic member of
        // any component that contains at least one "influencer" - all at
        // once, before resolution runs, so a whole stack responds in the
        // same step instead of trickling awake over several.
        {
            UnionFind unionFind;
            for (const Contact& contact : contacts)
            {
                unionFind.Union(contact.a, contact.b);
            }

            // An entity "influences" its whole connected group to wake if
            // it's a teleported mover this step, or involved in a
            // contact with real velocity behind it (a genuine impact,
            // not just continued resting contact) - the same threshold
            // ResolveContact's own per-pair check uses.
            //
            // Deliberately NOT included: "any already-awake Dynamic
            // neighbor." That seems reasonable but isn't - multiple
            // bodies settling together don't all cross the sleep
            // threshold at the exact same simulation step, so a group
            // mid-settle always has at least one member technically
            // still awake until the very last one finishes. Treating
            // "awake" alone as a wake-trigger meant the moment the FIRST
            // body in a stack fell asleep, its still-settling neighbors
            // (awake, barely moving, on their way to sleep themselves)
            // would immediately wake it right back up - a self-defeating
            // loop that meant nothing in a multi-body group could ever
            // actually stay asleep. Caught by testing a 3-box stack,
            // which never stopped oscillating between asleep and awake
            // every other step until this was removed.
            // Same threshold and reasoning as ResolveContact's per-pair
            // check - must clear ordinary per-step gravity accumulation
            // or it's not a real disturbance (see that comment for the
            // full explanation of why 0.15 was wrong).
            constexpr float kWakeVelocityThreshold = 0.5f;

            std::vector<Entity> influencers;
            for (const Contact& contact : contacts)
            {
                if (teleportedMovers.count(contact.a) > 0 || teleportedMovers.count(contact.b) > 0)
                {
                    influencers.push_back(contact.a);
                    continue;
                }

                RigidBodyComponent staticDummyA;
                staticDummyA.type = BodyType::Static;
                staticDummyA.inverseMass = 0.0f;
                RigidBodyComponent staticDummyB;
                staticDummyB.type = BodyType::Static;
                staticDummyB.inverseMass = 0.0f;

                RigidBodyComponent& bodyA = bodies.Has(contact.a) ? bodies.Get(contact.a) : staticDummyA;
                RigidBodyComponent& bodyB = bodies.Has(contact.b) ? bodies.Get(contact.b) : staticDummyB;

                float velocityAlongNormal = glm::dot(bodyB.linearVelocity - bodyA.linearVelocity, contact.normal);
                if (glm::abs(velocityAlongNormal) > kWakeVelocityThreshold)
                {
                    influencers.push_back(contact.a);
                }
            }

            std::unordered_map<Entity, bool> rootShouldWake;
            for (Entity influencer : influencers)
            {
                rootShouldWake[unionFind.Find(influencer)] = true;
            }

            if (!rootShouldWake.empty())
            {
                for (const Contact& contact : contacts)
                {
                    if (!rootShouldWake.count(unionFind.Find(contact.a)))
                    {
                        continue;
                    }

                    for (Entity e : { contact.a, contact.b })
                    {
                        if (bodies.Has(e))
                        {
                            RigidBodyComponent& body = bodies.Get(e);
                            if (body.type == BodyType::Dynamic && body.isSleeping)
                            {
                                body.isSleeping = false;
                                body.sleepTimer = 0.0f;
                            }
                        }
                    }
                }
            }
        }

        // See resolutionIterations' doc comment in CollisionSystem.h -
        // re-running the same contact list lets corrections propagate
        // through a whole chain/stack in one step instead of leaving a
        // residual velocity loop behind. forceWake is only meaningful
        // once (it's about whether a body should wake up at all, not
        // something that needs re-deciding each pass), so it's computed
        // before the loop and reused every iteration.
        std::vector<bool> forceWakeByContact(contacts.size());
        for (size_t i = 0; i < contacts.size(); ++i)
        {
            forceWakeByContact[i] = teleportedMovers.count(contacts[i].a) > 0 || teleportedMovers.count(contacts[i].b) > 0;
        }

        for (int iteration = 0; iteration < resolutionIterations; ++iteration)
        {
            bool isLastIteration = (iteration == resolutionIterations - 1);
            for (size_t i = 0; i < contacts.size(); ++i)
            {
                ResolveContact(transforms, bodies, colliders, contacts[i], restitution, friction,
                                positionalCorrectionPercent, positionalCorrectionSlop, forceWakeByContact[i],
                                isLastIteration, maxAngularSpeed);
            }
        }

        // Sleep bookkeeping runs LAST, after every contact this step has
        // been resolved - this is the only point where a resting body's
        // velocity reflects its actual settled state (gravity applied,
        // then cancelled back out by contact resolution). Checking any
        // earlier - e.g. right after PhysicsSystem's integration, before
        // resolution - sees the pre-resolution velocity spike every
        // step and a resting body would never accumulate sleep time.
        // See PhysicsSystem.cpp's note on this same point.
        constexpr float kSleepLinearThreshold = 0.05f; // units/sec
        constexpr float kSleepAngularThreshold = 3.0f; // degrees/sec
        constexpr float kSleepTimeToSleep = 0.5f;      // seconds

        // A body actively touching a moving Kinematic surface (a
        // conveyor, a slow-moving platform) is never truly "at rest" -
        // it's continuously being disturbed by friction, even if that
        // friction is gentle enough that the resulting velocity each
        // step stays under the sleep threshold. Without this, such a
        // body could fall asleep mid-drag; once asleep it's treated as
        // an immovable anchor for resolution purposes (see
        // ResolveContact), so it would stop responding to the platform
        // entirely and appear to detach/stick in place while the
        // platform keeps sliding underneath it - a real, caught
        // regression from adding proper friction/torque resolution
        // (which made per-step velocity transfer gentler and more
        // realistic, but also meant it could now legitimately dip under
        // the sleep threshold while still genuinely being dragged).
        std::unordered_set<Entity> touchingMovingKinematic;
        constexpr float kMovingKinematicThreshold = 0.001f; // ignore essentially-stationary Kinematic bodies (float noise)
        for (const Contact& contact : contacts)
        {
            bool aMoving = bodies.Has(contact.a) && bodies.Get(contact.a).type == BodyType::Kinematic
                          && glm::length(bodies.Get(contact.a).linearVelocity) > kMovingKinematicThreshold;
            bool bMoving = bodies.Has(contact.b) && bodies.Get(contact.b).type == BodyType::Kinematic
                          && glm::length(bodies.Get(contact.b).linearVelocity) > kMovingKinematicThreshold;
            if (aMoving)
            {
                touchingMovingKinematic.insert(contact.b);
            }
            if (bMoving)
            {
                touchingMovingKinematic.insert(contact.a);
            }
        }

        for (Entity entity : entities)
        {
            if (!bodies.Has(entity))
            {
                continue;
            }

            RigidBodyComponent& body = bodies.Get(entity);
            if (body.type != BodyType::Dynamic || body.isSleeping)
            {
                continue;
            }

            if (touchingMovingKinematic.count(entity) > 0)
            {
                body.sleepTimer = 0.0f;
                continue;
            }

            float speed = glm::length(body.linearVelocity);
            float angularSpeed = glm::length(body.angularVelocity);

            if (speed < kSleepLinearThreshold && angularSpeed < kSleepAngularThreshold)
            {
                body.sleepTimer += fixedDeltaTime;
                if (body.sleepTimer >= kSleepTimeToSleep)
                {
                    body.isSleeping = true;
                    body.linearVelocity = glm::vec3(0.0f);
                    body.angularVelocity = glm::vec3(0.0f);
                }
            }
            else
            {
                body.sleepTimer = 0.0f;
            }
        }

        // Snapshot final positions AND rotations LAST, after sweep/
        // resolution/sleep bookkeeping have all finished moving things -
        // see the comment at the top of this function for why the
        // timing here matters.
        for (Entity entity : entities)
        {
            const TransformComponent& finalTransform = transforms.Get(entity);
            m_lastKnownPositions[entity] = finalTransform.position;
            m_lastKnownRotations[entity] = finalTransform.rotationDegrees;
        }
    }
}
