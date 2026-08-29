#include "ClusterSystem.h"
#include "../Particles/GPU/ParticleGPUTypes.h"

#include <unordered_map>
#include <deque>

namespace Eden
{
    namespace
    {
        // Same "hash the integer cell coordinate" idea the GPU hash grid
        // uses (particle_build_grid.comp's HashCell), just as a 64-bit
        // CPU key instead of a fixed-size hash table - the eligible-
        // particle set here is a small fraction of GPU_MAX_PARTICLES, so
        // an unordered_map's overhead doesn't matter the way it would
        // per-frame on the GPU path.
        int64_t CellKey(const glm::ivec3& cell)
        {
            constexpr int64_t kOffset = 1 << 20; // keeps coords positive before packing
            int64_t x = static_cast<int64_t>(cell.x) + kOffset;
            int64_t y = static_cast<int64_t>(cell.y) + kOffset;
            int64_t z = static_cast<int64_t>(cell.z) + kOffset;
            return (x << 42) | (y << 21) | z;
        }
    }

    std::vector<ReformCluster> ClusterSystem::FindReformClusters(ParticleSystemGPU& particles, float clusterRadius,
                                                                   uint32_t minClusterSize)
    {
        std::vector<ReformCluster> result;

        std::vector<glm::vec4> allPositions, allVelocities;
        std::vector<float> allHeat;
        particles.ReadBackAll(allPositions, allVelocities, allHeat);

        if (allPositions.empty() || clusterRadius <= 0.0f)
        {
            return result;
        }

        // Eligible = actually part of the melt system (not
        // GPU_HEAT_NOT_MELTED) and cooled enough to be a reform
        // candidate.
        std::vector<uint32_t> eligible;
        eligible.reserve(allPositions.size());
        for (uint32_t i = 0; i < allPositions.size(); ++i)
        {
            float heat = allHeat[i];
            if (heat >= 0.0f && heat <= GPU_HEAT_REFORM_THRESHOLD)
            {
                eligible.push_back(i);
            }
        }

        if (eligible.empty())
        {
            return result;
        }

        // Bucket eligible particles into a uniform grid, cell size =
        // clusterRadius (same "one particle can only need to check its
        // 27-cell neighborhood" reasoning as the GPU SPH grid).
        std::unordered_map<int64_t, std::vector<uint32_t>> grid;
        grid.reserve(eligible.size());
        for (uint32_t index : eligible)
        {
            glm::vec3 pos = glm::vec3(allPositions[index]);
            glm::ivec3 cell = glm::ivec3(glm::floor(pos / clusterRadius));
            grid[CellKey(cell)].push_back(index);
        }

        float clusterRadiusSq = clusterRadius * clusterRadius;
        std::vector<bool> visited(allPositions.size(), false);

        for (uint32_t seedIndex : eligible)
        {
            if (visited[seedIndex])
            {
                continue;
            }

            // BFS flood fill over the grid, same shape as a connected-
            // components search over a voxel/graph structure - not
            // recursive, this can easily be thousands of particles deep.
            std::vector<uint32_t> component;
            std::deque<uint32_t> queue;
            queue.push_back(seedIndex);
            visited[seedIndex] = true;

            while (!queue.empty())
            {
                uint32_t current = queue.front();
                queue.pop_front();
                component.push_back(current);

                glm::vec3 currentPos = glm::vec3(allPositions[current]);
                glm::ivec3 currentCell = glm::ivec3(glm::floor(currentPos / clusterRadius));

                for (int dz = -1; dz <= 1; ++dz)
                for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                {
                    auto it = grid.find(CellKey(currentCell + glm::ivec3(dx, dy, dz)));
                    if (it == grid.end())
                    {
                        continue;
                    }

                    for (uint32_t candidate : it->second)
                    {
                        if (visited[candidate])
                        {
                            continue;
                        }

                        glm::vec3 candidatePos = glm::vec3(allPositions[candidate]);
                        if (glm::dot(candidatePos - currentPos, candidatePos - currentPos) <= clusterRadiusSq)
                        {
                            visited[candidate] = true;
                            queue.push_back(candidate);
                        }
                    }
                }
            }

            if (component.size() >= minClusterSize)
            {
                ReformCluster cluster;
                cluster.positions.reserve(component.size());
                cluster.particleIndices.reserve(component.size());
                for (uint32_t index : component)
                {
                    cluster.positions.push_back(glm::vec3(allPositions[index]));
                    cluster.particleIndices.push_back(index);
                }
                result.push_back(std::move(cluster));
            }
        }

        return result;
    }
}
