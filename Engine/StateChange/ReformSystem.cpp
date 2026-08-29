#include "ReformSystem.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace Eden
{
    std::vector<VoxelVolumeHandle> ReformSystem::Reform(ParticleSystemGPU& particles, VoxelSystemGPU& voxels,
                                                          float clusterRadius, float particleRadius,
                                                          uint32_t minClusterSize, float smoothRadius)
    {
        std::vector<VoxelVolumeHandle> newVolumes;

        std::vector<ReformCluster> clusters = ClusterSystem::FindReformClusters(particles, clusterRadius, minClusterSize);
        if (clusters.empty())
        {
            return newVolumes;
        }

        // Resolution every reformed volume uses - tied to particleRadius
        // (finer than the particles themselves would just be wasted
        // voxels; coarser would visibly chunk the blob), and, critically,
        // FIXED - never coarsened to fit a budget. An earlier version of
        // this function computed one shared bounding box across every
        // cluster found and coarsened voxelSize as needed to keep that
        // whole box under a sample cap; a single distant outlier cluster
        // (a stray cooled particle far from the main puddle) could
        // balloon that shared box and drag the WHOLE grid's resolution
        // down with it, including the main puddle - particles ended up
        // smaller than a single voxel cell, so marching cubes only
        // caught isolated points instead of smooth blobs (tiny slivers
        // scattered across the whole area, not the puddle shape). Fixed
        // resolution + grouping only nearby clusters together (below)
        // fixes that at the source instead of capping around it.
        float voxelSize = std::max(0.02f, particleRadius * 0.6f);

        // Room past the outermost particle for that particle's own
        // radius to actually show up in the field, plus smoothRadius
        // (SmoothMin's blend can dip past a single sphere's own surface
        // - see VoxelSystemGPU.cpp's comment on SmoothMin) so the blend
        // itself doesn't get clipped at the volume's own edge, plus one
        // extra voxel so the isosurface at a cluster's true boundary
        // doesn't get clipped either.
        float margin = particleRadius + smoothRadius + voxelSize;

        // VRAM cap: RegisterVolume allocates a worst-case marching-cubes
        // vertex buffer PER CHUNK unconditionally, and chunk count grows
        // with the cube of samples-per-axis - this bounds how large any
        // ONE volume's chunk grid can get. Used below as the grouping
        // constraint (a group only forms if it fits), not as a
        // resolution-coarsening knob anymore.
        constexpr int32_t kMaxSamplesPerAxis = 96;

        // Per-cluster world-space bounds (with margin already applied),
        // used both for grouping decisions and for each finalized
        // group's actual volume extent.
        std::vector<glm::vec3> clusterMin(clusters.size());
        std::vector<glm::vec3> clusterMax(clusters.size());
        for (size_t i = 0; i < clusters.size(); ++i)
        {
            glm::vec3 lo(std::numeric_limits<float>::max());
            glm::vec3 hi(std::numeric_limits<float>::lowest());
            for (const glm::vec3& pos : clusters[i].positions)
            {
                lo = glm::min(lo, pos);
                hi = glm::max(hi, pos);
            }
            clusterMin[i] = lo - glm::vec3(margin);
            clusterMax[i] = hi + glm::vec3(margin);
        }

        // Greedy grouping: try to add each cluster to the first existing
        // group whose combined extent (with this cluster folded in)
        // still fits kMaxSamplesPerAxis at the FIXED voxelSize above; if
        // none fits, start a new group. Not globally optimal bin-
        // packing, but simple, and correct in the case that actually
        // matters - a puddle's own clusters are all close together and
        // merge into one group/volume (one set of GPU round trips),
        // while a genuinely distant straggler naturally can't fit any
        // existing group and gets its own, still-full-resolution volume
        // instead of degrading everyone else's.
        std::vector<std::vector<size_t>> groups;
        std::vector<glm::vec3> groupMin, groupMax;

        for (size_t ci = 0; ci < clusters.size(); ++ci)
        {
            bool placed = false;
            for (size_t g = 0; g < groups.size(); ++g)
            {
                glm::vec3 candidateMin = glm::min(groupMin[g], clusterMin[ci]);
                glm::vec3 candidateMax = glm::max(groupMax[g], clusterMax[ci]);
                glm::vec3 candidateExtent = candidateMax - candidateMin;
                float maxComponent = std::max({ candidateExtent.x, candidateExtent.y, candidateExtent.z });
                int32_t neededSamples = static_cast<int32_t>(std::ceil(maxComponent / voxelSize)) + 1;

                if (neededSamples <= kMaxSamplesPerAxis)
                {
                    groups[g].push_back(ci);
                    groupMin[g] = candidateMin;
                    groupMax[g] = candidateMax;
                    placed = true;
                    break;
                }
            }

            if (!placed)
            {
                groups.push_back({ ci });
                groupMin.push_back(clusterMin[ci]);
                groupMax.push_back(clusterMax[ci]);
            }
        }

        std::vector<uint32_t> allIndicesToKill;

        for (size_t g = 0; g < groups.size(); ++g)
        {
            glm::vec3 worldMin = groupMin[g];
            glm::vec3 worldMax = groupMax[g];
            glm::vec3 extent = worldMax - worldMin;

            float chunkWorldSize = voxelSize * static_cast<float>(kVoxelChunkSize);
            glm::ivec3 chunkDims(
                std::max(1, static_cast<int32_t>(std::ceil(extent.x / chunkWorldSize))),
                std::max(1, static_cast<int32_t>(std::ceil(extent.y / chunkWorldSize))),
                std::max(1, static_cast<int32_t>(std::ceil(extent.z / chunkWorldSize))));

            VoxelVolumeDesc desc;
            desc.origin = worldMin;
            desc.voxelSize = voxelSize;
            desc.chunkDims = chunkDims;

            VoxelVolumeHandle handle = voxels.RegisterVolume(desc);

            std::vector<glm::vec3> localPositions;
            for (size_t ci : groups[g])
            {
                for (const glm::vec3& pos : clusters[ci].positions)
                {
                    localPositions.push_back(pos - worldMin);
                }
                allIndicesToKill.insert(allIndicesToKill.end(), clusters[ci].particleIndices.begin(),
                                         clusters[ci].particleIndices.end());
            }

            voxels.SeedFromParticles(handle, localPositions, particleRadius, smoothRadius);
            voxels.MarchDirtyChunks(handle);

            // World placement is already baked into desc.origin (worldMin
            // above) - same convention the main.cpp smoke-test volume
            // uses, identity transform on top of a world-space origin.
            voxels.SetTransform(handle, glm::mat4(1.0f));

            newVolumes.push_back(handle);
        }

        particles.KillParticles(allIndicesToKill);

        return newVolumes;
    }
}
