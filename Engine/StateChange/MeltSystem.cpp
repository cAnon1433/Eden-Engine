#include "MeltSystem.h"

namespace Eden
{
    uint32_t MeltSystem::Melt(VoxelSystemGPU& voxels, VoxelVolumeHandle volume, ParticleSystemGPU& particles,
                               const glm::vec3& initialVelocity, float spacing)
    {
        VoxelSystemGPU::VolumeBounds bounds = voxels.GetVolumeBounds(volume);

        // -1.0f (the default) means "use the same rest-spacing
        // convention EmitBox falls back to" - a melted solid packing at
        // the SPH sim's own natural particle density. An explicit,
        // smaller spacing packs denser than that (see this function's
        // header comment on why that's a real, separate knob from the
        // sim's own tuning).
        float actualSpacing = spacing > 0.0f ? spacing : particles.smoothingRadius * 0.6f;
        if (actualSpacing <= 0.0f)
        {
            return 0; // degenerate tuning - nothing sane to spawn at
        }

        std::vector<glm::vec3> spawnPositions;
        for (float x = bounds.worldMin.x; x <= bounds.worldMax.x; x += actualSpacing)
        {
            for (float y = bounds.worldMin.y; y <= bounds.worldMax.y; y += actualSpacing)
            {
                for (float z = bounds.worldMin.z; z <= bounds.worldMax.z; z += actualSpacing)
                {
                    glm::vec3 worldPos(x, y, z);
                    // Exact field query, not the cheap chunkSolid flag -
                    // a volume that's already been partially carved
                    // before melting should only spawn particles where
                    // material actually remains.
                    if (voxels.IsWorldPointSolidExact(volume, worldPos))
                    {
                        spawnPositions.push_back(worldPos);
                    }
                }
            }
        }

        uint32_t countBefore = particles.ParticleCount();
        // Melted particles start fully "hot" (1.0) - ordinary Emit/
        // EmitBox particles default to GPU_HEAT_NOT_MELTED and never
        // decay, so this is what actually enrolls these particles in
        // the heat/cooldown -> reform pipeline.
        particles.EmitPoints(spawnPositions, initialVelocity, 1.0f);
        uint32_t spawned = particles.ParticleCount() - countBefore;

        // The solid is gone regardless of whether every sample point
        // found room in the particle system - a partially-truncated
        // melt (GPU_MAX_PARTICLES reached) still shouldn't leave the
        // original solid half-rendered.
        voxels.ClearVolume(volume);

        // ClearVolume only writes densityCPU and marks chunks dirty -
        // same split as SeedSphere/Carve, neither of which marches
        // themselves either (see their header comments: caller marches
        // after). Melt() is the "do the whole conversion" entry point
        // though, so it owns the re-march itself rather than leaving a
        // silent "cleared but still drawing stale triangles" gap for
        // every caller to remember.
        voxels.MarchDirtyChunks(volume);

        return spawned;
    }
}
