#include "ParticleSystem.h"
#include "SPHKernels.h"
#include "../ECS/Components/TransformComponent.h"
#include "../Physics/ColliderComponent.h"
#include "../Physics/SDF.h"

#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <mutex>

namespace Eden
{
    void ParticleSystem::Emit(const glm::vec3& position, const glm::vec3& velocity)
    {
        m_Particles.AddParticle(position, velocity);
    }

    void ParticleSystem::EmitBox(const glm::vec3& min, const glm::vec3& max, float spacing, const glm::vec3& initialVelocity)
    {
        // 0.6 * h keeps particles dense enough to have several real
        // neighbors from the very first Step() call - see the comment on
        // this function in ParticleSystem.h for why that matters (a
        // too-sparse initial fill reads as near-zero density/pressure
        // for the first several steps no matter how restDensity is set).
        float actualSpacing = spacing > 0.0f ? spacing : smoothingRadius * 0.6f;

        m_Particles.Reserve(m_Particles.Count() +
            static_cast<size_t>(((max.x - min.x) / actualSpacing + 1) *
                                 ((max.y - min.y) / actualSpacing + 1) *
                                 ((max.z - min.z) / actualSpacing + 1)));

        for (float x = min.x; x <= max.x; x += actualSpacing)
        {
            for (float y = min.y; y <= max.y; y += actualSpacing)
            {
                for (float z = min.z; z <= max.z; z += actualSpacing)
                {
                    m_Particles.AddParticle(glm::vec3(x, y, z), initialVelocity);
                }
            }
        }
    }

    void ParticleSystem::Clear()
    {
        m_Particles.Clear();
    }

    void ParticleSystem::Step(Registry& registry, float fixedDeltaTime)
    {
        if (m_Particles.Count() == 0)
        {
            return;
        }

        // See `substeps` in ParticleSystem.h - WCSPH with Tait's equation
        // needs a smaller step than the rigid-body solver's 1/60s tick to
        // stay stable, so this divides the incoming fixed timestep down
        // rather than integrating once at the full tick.
        float subDt = fixedDeltaTime / static_cast<float>(std::max(1, substeps));

        // Resolved once per Step() call, reused across every substep -
        // see ResolveBoundaries' declaration comment in ParticleSystem.h.
        auto colliderEntities = registry.View<TransformComponent, ColliderComponent>();

        ParticleSpatialHash grid;

        for (int step = 0; step < substeps; ++step)
        {
            // Rebuilt every substep - particles move every substep, so
            // any cached grid from the previous one is immediately
            // stale. Cell size == smoothingRadius, matching
            // ParticleSpatialHash's own sizing assumption (see its
            // class comment).
            grid.Build(m_Particles, smoothingRadius);
            BuildNeighborLists(grid);

            ComputeDensityPressure();
            ComputeForces();
            Integrate(subDt);
            ResolveBoundaries(registry, colliderEntities, subDt);
        }
    }

    void ParticleSystem::BuildNeighborLists(const ParticleSpatialHash& grid)
    {
        size_t count = m_Particles.Count();
        m_NeighborOffsets.resize(count + 1);

        if (count == 0)
        {
            m_NeighborOffsets[0] = 0;
            m_NeighborIndices.clear();
            return;
        }

        // Per-chunk scratch, collected under a small mutex - one lock
        // per WORKER (e.g. 8), not per particle, so contention here is
        // negligible even though it's "just" a mutex.
        struct ChunkResult
        {
            size_t begin;
            std::vector<uint32_t> offsetsWithinChunk; // relative to this chunk's own `indices`
            std::vector<uint32_t> indices;
        };

        std::vector<ChunkResult> chunkResults;
        std::mutex resultsMutex;

        m_ThreadPool.ParallelFor(count, [&](size_t begin, size_t end)
        {
            ChunkResult result;
            result.begin = begin;
            result.offsetsWithinChunk.resize(end - begin);

            for (size_t i = begin; i < end; ++i)
            {
                result.offsetsWithinChunk[i - begin] = static_cast<uint32_t>(result.indices.size());

                grid.ForEachNeighbor(m_Particles.positions[i], [&](uint32_t j)
                {
                    result.indices.push_back(j);
                });
            }

            std::lock_guard<std::mutex> lock(resultsMutex);
            chunkResults.push_back(std::move(result));
        });

        // Sorted by begin so the merged output matches what a single-
        // threaded pass would have produced (particle order preserved) -
        // not required for correctness (SPH's neighbor sums don't care
        // what order they're summed in, up to floating-point rounding),
        // but keeps behavior reproducible/debuggable rather than
        // depending on whichever worker happened to finish first.
        std::sort(chunkResults.begin(), chunkResults.end(),
            [](const ChunkResult& a, const ChunkResult& b) { return a.begin < b.begin; });

        size_t totalIndices = 0;
        for (auto& chunk : chunkResults)
        {
            totalIndices += chunk.indices.size();
        }

        m_NeighborIndices.clear();
        m_NeighborIndices.reserve(totalIndices);

        for (auto& chunk : chunkResults)
        {
            size_t base = m_NeighborIndices.size();
            for (size_t i = 0; i < chunk.offsetsWithinChunk.size(); ++i)
            {
                m_NeighborOffsets[chunk.begin + i] = static_cast<uint32_t>(base + chunk.offsetsWithinChunk[i]);
            }
            m_NeighborIndices.insert(m_NeighborIndices.end(), chunk.indices.begin(), chunk.indices.end());
        }

        m_NeighborOffsets[count] = static_cast<uint32_t>(m_NeighborIndices.size());
    }

    void ParticleSystem::ComputeDensityPressure()
    {
        size_t count = m_Particles.Count();

        // Read-only access to neighbor positions, write-only to this
        // particle's own density/pressure slot - safe across disjoint
        // [begin,end) chunks, see ParticleThreadPool::ParallelFor's
        // contract.
        m_ThreadPool.ParallelFor(count, [this](size_t begin, size_t end)
        {
            for (size_t i = begin; i < end; ++i)
            {
                float density = 0.0f;
                const glm::vec3& pi = m_Particles.positions[i];

                uint32_t rangeBegin = m_NeighborOffsets[i];
                uint32_t rangeEnd = m_NeighborOffsets[i + 1];
                for (uint32_t n = rangeBegin; n < rangeEnd; ++n)
                {
                    uint32_t j = m_NeighborIndices[n];
                    const glm::vec3& pj = m_Particles.positions[j];
                    float r = glm::length(pi - pj);
                    density += particleMass * SPH::Poly6(r, smoothingRadius);
                }

                // A particle always sees itself as a neighbor (r = 0,
                // Poly6(0, h) is nonzero) - this is correct and expected,
                // not double counting: self-contribution to density is
                // part of the standard SPH formulation, not a bug to
                // guard against.
                m_Particles.densities[i] = density;

                // Tait equation of state (see `stiffness`/`gamma`
                // comment in ParticleSystem.h) - clamped to non-negative
                // to avoid the tensile instability that negative
                // pressure below rest density would otherwise cause.
                float ratio = density / restDensity;
                float pressure = stiffness * (glm::pow(ratio, gamma) - 1.0f);
                m_Particles.pressures[i] = std::max(0.0f, pressure);
            }
        });
    }

    void ParticleSystem::ComputeForces()
    {
        size_t count = m_Particles.Count();

        // Reads neighbor density/pressure (written by
        // ComputeDensityPressure, which has already fully completed by
        // the time this runs - Step() calls them sequentially), writes
        // only this particle's own forces[i] slot - safe across
        // disjoint chunks for the same reason as ComputeDensityPressure
        // above.
        m_ThreadPool.ParallelFor(count, [this](size_t begin, size_t end)
        {
            for (size_t i = begin; i < end; ++i)
            {
                glm::vec3 pressureForce(0.0f);
                glm::vec3 viscosityForce(0.0f);

                const glm::vec3& pi = m_Particles.positions[i];
                const glm::vec3& vi = m_Particles.velocities[i];
                float pressureI = m_Particles.pressures[i];

                uint32_t rangeBegin = m_NeighborOffsets[i];
                uint32_t rangeEnd = m_NeighborOffsets[i + 1];
                for (uint32_t n = rangeBegin; n < rangeEnd; ++n)
                {
                    uint32_t j = m_NeighborIndices[n];
                    if (j == i)
                    {
                        continue;
                    }

                    const glm::vec3& pj = m_Particles.positions[j];
                    glm::vec3 rVec = pi - pj;
                    float r = glm::length(rVec);

                    if (r <= 0.0f || r > smoothingRadius)
                    {
                        continue;
                    }

                    float rhoJ = m_Particles.densities[j];
                    if (rhoJ <= 0.0f)
                    {
                        continue; // shouldn't happen (self-contribution guarantees rho > 0), guarding div-by-zero anyway
                    }

                    // Symmetric pressure force (Muller/Charypar/Gross
                    // 2003, eq. 10) - using (p_i + p_j) / (2 * rho_j)
                    // rather than the alternate p_i/rho_i^2 + p_j/rho_j^2
                    // form some references use; both are valid
                    // derivations of the same physical force, this is
                    // the more commonly implemented one and matches the
                    // reference this module was built against.
                    float pressureJ = m_Particles.pressures[j];
                    glm::vec3 gradW = SPH::SpikyGradient(rVec, r, smoothingRadius);
                    pressureForce -= particleMass * (pressureI + pressureJ) / (2.0f * rhoJ) * gradW;

                    // Viscosity force (eq. 14) - diffuses velocity
                    // toward neighbor velocities.
                    const glm::vec3& vj = m_Particles.velocities[j];
                    float lapW = SPH::ViscosityLaplacian(r, smoothingRadius);
                    viscosityForce += particleMass * (vj - vi) / rhoJ * lapW;
                }

                viscosityForce *= viscosityCoefficient;

                // Gravity deliberately NOT added here - applied directly
                // as an acceleration in Integrate(), matching
                // PhysicsSystem::Step's identical convention for rigid
                // bodies (see RigidBodyComponent.h's comment on
                // accumulatedForce). forces[i] here is pressure +
                // viscosity only.
                m_Particles.forces[i] = pressureForce + viscosityForce;
            }
        });
    }

    void ParticleSystem::Integrate(float dt)
    {
        size_t count = m_Particles.Count();

        for (size_t i = 0; i < count; ++i)
        {
            float density = m_Particles.densities[i];
            glm::vec3 acceleration = gravity;

            if (density > 0.0f)
            {
                // Lagrangian SPH acceleration: force-per-unit-volume
                // divided by density gives acceleration, same as F = ma
                // rearranged with density standing in for a per-particle
                // "how much stuff is actually here" term.
                acceleration += m_Particles.forces[i] / density;
            }

            m_Particles.velocities[i] += acceleration * dt;
            m_Particles.positions[i] += m_Particles.velocities[i] * dt;
        }
    }

    void ParticleSystem::ResolveBoundaries(Registry& registry, const std::vector<Entity>& colliderEntities, float dt)
    {
        if (colliderEntities.empty())
        {
            return;
        }

        // Precomputed once per call (colliders are few, typically single
        // digits) rather than once per particle - avoids every one of
        // potentially thousands of particles recomputing the same
        // handful of rotation matrices, and also gives ParallelFor below
        // a clean per-particle-only write pattern to parallelize (see
        // its comment on ComputeDensityPressure/ComputeForces for the
        // same reasoning).
        struct ColliderInfo
        {
            const TransformComponent* transform;
            ColliderComponent scaledCollider; // shape parameters pre-scaled by transform->scale, see SDF::ScaledCollider
            glm::mat3 rotation;
        };

        std::vector<ColliderInfo> colliders;
        colliders.reserve(colliderEntities.size());
        for (Entity entity : colliderEntities)
        {
            const auto& transform = registry.GetComponent<TransformComponent>(entity);
            const auto& collider = registry.GetComponent<ColliderComponent>(entity);
            colliders.push_back(ColliderInfo{ &transform, SDF::ScaledCollider(collider, transform.scale), SDF::RotationMatrixFromDegrees(transform.rotationDegrees) });
        }

        constexpr float kSurfaceEpsilon = 0.001f; // "close enough to touching" - stop marching. Same value/reasoning as Physics/CollisionSystem.cpp's ConservativeAdvanceDynamicBodies.

        // O(particles * colliders), no broad phase - fine for the
        // handful of static/kinematic/dynamic colliders a hand-placed
        // scene has today. If this ever needs to scale to a large
        // static level built from many small colliders, that's the
        // signal to reuse the same spatial-hash-based rejection
        // CollisionSystem's broad phase already does, not a case to
        // pre-optimize for now.
        m_ThreadPool.ParallelFor(m_Particles.Count(), [this, &colliders, dt, kSurfaceEpsilon](size_t begin, size_t end)
        {
            for (size_t i = begin; i < end; ++i)
            {
                // --- Tunnelling fix: SDF conservative-advancement sweep ---
                // Direct port of the same technique Physics/CollisionSystem.cpp
                // uses for fast rigid bodies (ConservativeAdvanceDynamicBodies) -
                // sphere-trace the particle's displacement this substep against
                // every collider's SDF, taking the largest step that's still
                // provably safe (the minimum distance-to-surface across all
                // colliders, minus the particle's own boundaryRadius) each
                // iteration, until either the full displacement is covered or a
                // surface is reached. Analytic SDFs are exactly what make this
                // cheap and shape-agnostic - the same reason CollisionSystem's
                // version doesn't need per-shape sweep code either.
                //
                // Only runs when this substep's displacement exceeds the
                // particle's own boundaryRadius - i.e. the particle moved more
                // than its own "size" this substep, which is the only
                // circumstance where a thin collider could be skipped entirely
                // between the start and end of a single discrete step. Below
                // that threshold the existing cheap discrete check (below) is
                // just as correct and doesn't pay for a march - same threshold
                // logic CollisionSystem uses (travelDistance <= approxRadius
                // skips the sweep there too).
                glm::vec3 displacement = m_Particles.velocities[i] * dt;
                float travelDistance = glm::length(displacement);

                if (boundaryRadius > 0.0f && travelDistance > boundaryRadius)
                {
                    glm::vec3 endPos = m_Particles.positions[i];
                    glm::vec3 startPos = endPos - displacement;
                    glm::vec3 travelDir = displacement / travelDistance;

                    glm::vec3 position = startPos;
                    float remaining = travelDistance;
                    bool hitSomething = false;

                    for (int iteration = 0; iteration < maxSweepIterations && remaining > 0.0f; ++iteration)
                    {
                        float safeStep = remaining;

                        for (const ColliderInfo& c : colliders)
                        {
                            glm::vec3 local = SDF::WorldToLocal(position, *c.transform, c.scaledCollider);
                            float distanceToSurface = SDF::Distance(c.scaledCollider, local) - boundaryRadius;

                            if (distanceToSurface < safeStep)
                            {
                                safeStep = distanceToSurface;
                            }
                        }

                        if (safeStep <= kSurfaceEpsilon)
                        {
                            hitSomething = true;
                            break;
                        }

                        position += travelDir * safeStep;
                        remaining -= safeStep;
                    }

                    m_Particles.positions[i] = position;

                    if (hitSomething)
                    {
                        // Same hard-stop reasoning as the rigid-body version:
                        // kill the component of velocity still driving the
                        // particle further along its travel direction, then
                        // let the discrete pass below (which still runs, using
                        // this now-swept position) apply the normal boundary
                        // damping/restitution response.
                        float speedIntoObstacle = glm::dot(m_Particles.velocities[i], travelDir);
                        if (speedIntoObstacle > 0.0f)
                        {
                            m_Particles.velocities[i] -= travelDir * speedIntoObstacle;
                        }
                    }
                }

                // --- Discrete penetration check (unchanged from before) ---
                // Always runs, even after a sweep above - the sweep stops
                // AT the surface (within kSurfaceEpsilon), it doesn't apply
                // the boundary damping/restitution response itself. This is
                // also the ONLY check that runs for the common case (slow-
                // moving particles), so it can't be skipped.
                for (const ColliderInfo& c : colliders)
                {
                    glm::vec3 localPoint = SDF::WorldToLocal(m_Particles.positions[i], *c.transform, c.scaledCollider);
                    float distance = SDF::Distance(c.scaledCollider, localPoint);

                    if (distance >= boundaryRadius)
                    {
                        continue; // not penetrating this collider
                    }

                    glm::vec3 localNormal = SDF::Normal(c.scaledCollider, localPoint);
                    glm::vec3 worldNormal = glm::normalize(c.rotation * localNormal);

                    float penetration = boundaryRadius - distance;
                    m_Particles.positions[i] += worldNormal * penetration;

                    float normalVelocity = glm::dot(m_Particles.velocities[i], worldNormal);
                    if (normalVelocity < 0.0f)
                    {
                        // Removes the into-surface component of velocity
                        // entirely (boundaryRestitution == 0 by default -
                        // absorbed, not bounced, see ParticleSystem.h).
                        // This is one-way: nothing here writes back to
                        // `entity`'s RigidBodyComponent, so the collider
                        // itself feels no reaction force from the fluid
                        // hitting it yet.
                        m_Particles.velocities[i] -= (1.0f + boundaryRestitution) * normalVelocity * worldNormal;
                    }
                }
            }
        });
    }

    std::vector<DrawCommand> ParticleSystem::BuildDrawList(MeshHandle pointMesh, const glm::vec4& colorOverride) const
    {
        std::vector<DrawCommand> drawList;
        drawList.reserve(m_Particles.Count());

        for (size_t i = 0; i < m_Particles.Count(); ++i)
        {
            if (drawList.size() >= MAX_INSTANCES_PER_FRAME)
            {
                // Same cap RenderSystem::BuildDrawList respects - callers
                // combine this list with the ECS one before handing it to
                // Renderer::DrawFrame, so both need to respect the same
                // shared instance-buffer ceiling. Silent here rather than
                // warning a second time; RenderSystem::BuildDrawList
                // already owns that warning for the combined list.
                break;
            }

            // Translation only - no scale. The point mesh's one vertex
            // sits at the local origin, and scaling the origin is still
            // the origin (see particle_point.vert's comment), so baking
            // a scale factor into this model matrix would be dead code,
            // not a bug exactly, just pointless. Apparent size is
            // controlled globally via Renderer::ParticlePointSize instead.
            DrawCommand cmd;
            cmd.mesh = pointMesh;
            cmd.model = glm::translate(glm::mat4(1.0f), m_Particles.positions[i]);
            cmd.colorOverride = colorOverride;
            drawList.push_back(cmd);
        }

        return drawList;
    }
}
