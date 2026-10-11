#include "ReformSystem.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <utility>

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
        //
        // kMaxMergeGap is the second, previously-missing half of that
        // "genuinely distant" check: the resolution-fit test above only
        // asks "would the combined box still fit the sample budget", NOT
        // "are these two things actually near each other" - with a fine
        // voxelSize (particleRadius*0.6, deliberately not coarsened -
        // see that constant's own comment), kMaxSamplesPerAxis*voxelSize
        // can easily span several meters, comfortably wide enough to
        // scoop up droplets nowhere near the main pool purely because
        // the fine resolution "affords" the box, not because they
        // belong together - this was a real, observed bug: hardening
        // one pool was also hardening scattered droplets well outside
        // it. Reusing clusterRadius (the same "particles this close
        // count as one contiguous body" threshold ClusterSystem itself
        // already uses one level down) as the basis for "clusters this
        // close count as one hardened body" keeps both merge decisions
        // grounded in the same physical notion of "close enough",
        // instead of one being physical and the other an accidental
        // side effect of voxel resolution. The x3 multiplier is
        // deliberate headroom, not the same strict adjacency test -
        // cluster BOUNDING BOXES already have real extent (unlike the
        // point-particles clusterRadius normally compares), so the gap
        // between two cluster boxes can legitimately be a bit larger
        // than a single clusterRadius while still clearly reading as
        // "one puddle, several fingers" rather than "two unrelated
        // puddles."
        float maxMergeGap = clusterRadius * 3.0f;

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

                // AABB-to-AABB gap (0 if the boxes already overlap/touch)
                // between this cluster and the group's CURRENT bounds -
                // the actual proximity test the resolution-fit check
                // above was missing. Computed against groupMin/groupMax
                // (not candidateMin/Max) deliberately: this asks "is the
                // new cluster near the group AS IT STANDS", not "would
                // it be near the group after we've already grown the box
                // to include it", which would trivially always pass.
                glm::vec3 gap = glm::max(glm::vec3(0.0f), glm::max(groupMin[g] - clusterMax[ci], clusterMin[ci] - groupMax[g]));
                float gapDistance = glm::length(gap);

                if (neededSamples <= kMaxSamplesPerAxis && gapDistance <= maxMergeGap)
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

            // Chunk compaction - compute this group's REAL density field
            // and classify which chunks actually need geometry BEFORE
            // registering anything, instead of always reserving
            // worst-case vertex space for every chunk in desc regardless
            // of whether it ever produces a triangle. See
            // VoxelSystemGPU::BuildChunkRemapTable/RegisterVolume's
            // chunkRemapOverride comment for the full mechanism (a GPU-
            // side remap table voxel_march.comp consults so it can
            // safely write into a buffer sized for only the chunks that
            // matter) - this is what makes a genuinely large, spread-out
            // pool (mostly empty/solid interior, thin surface shell)
            // affordable instead of hitting the same worst-case cost a
            // small, dense one would.
            std::vector<float> density(desc.NumSamples());
            VoxelSystemGPU::SplatParticlesIntoDensityField(density, desc, localPositions, particleRadius, smoothRadius);

            std::vector<uint32_t> chunkRemap;
            uint32_t compactChunkCount = VoxelSystemGPU::BuildChunkRemapTable(density, desc, chunkRemap);

            if (compactChunkCount == 0)
            {
                // No chunk in this group's bounding box actually crosses
                // the isosurface at all - nothing would ever be visible.
                // Not a safety-cap situation, just genuinely nothing to
                // build (e.g. particles too sparse relative to voxelSize/
                // smoothRadius to ever connect into a real surface) -
                // leave the particles liquid and move on quietly, same
                // as the safety-cap skip below does, just without a
                // warning since this isn't a size problem.
                continue;
            }

            // Hard safety check BEFORE committing to RegisterVolume - see
            // VoxelSystemGPU::EstimateVertexBufferBytes' own comment for
            // the crash this closes: a real reform produced a desc whose
            // vertex buffer needed a single 5.26GB allocation, which
            // exceeded the actual device limit (4GB - maxMemoryAllocationSize,
            // reported directly in the resulting Vulkan error) and took
            // the whole VkDevice down.
            //
            // Now checked against the REAL compact cost (compactChunkCount),
            // not the old worst-case-for-every-chunk estimate - this is
            // the actual point of chunk compaction: a pool that used to
            // get rejected purely because its BOUNDING BOX was large now
            // only gets rejected if its ACTUAL surface geometry is large,
            // which is a much rarer, much more honest reason to say no.
            // 2GB is calibrated against the ACTUAL reported device
            // ceiling from the crash that motivated this check (4GB) -
            // half of a real, hardware-reported number, not a guess. This
            // is still a real simplification worth flagging: fixed at
            // build time, not queried from the running device's own
            // maxMemoryAllocationSize (that would need
            // vkGetPhysicalDeviceProperties2 with a
            // VkPhysicalDeviceVulkan11Properties/Maintenance3 chain, which
            // this project doesn't set up anywhere yet).
            constexpr VkDeviceSize kMaxVertexBufferBytes = 2ull * 1024ull * 1024ull * 1024ull;
            VkDeviceSize estimatedBytes = VoxelSystemGPU::EstimateVertexBufferBytes(desc, compactChunkCount);
            if (estimatedBytes > kMaxVertexBufferBytes)
            {
                // Rate-limited to one message per DISTINCT (chunkDims,
                // compactChunkCount) pair, not once per call - see this
                // check's own original comment for why (Reform() runs
                // periodically; an unchanging oversized pool would
                // otherwise spam this every cycle forever).
                static std::vector<std::pair<glm::ivec3, uint32_t>> s_LastWarned;
                auto key = std::make_pair(chunkDims, compactChunkCount);
                bool alreadyWarned = std::find(s_LastWarned.begin(), s_LastWarned.end(), key) != s_LastWarned.end();
                if (!alreadyWarned)
                {
                    std::printf("[Reform] skipped a group (chunkDims=(%d,%d,%d), %u/%u chunks actually need "
                                "geometry, estimated vertex buffer %.1f MB) - exceeds the %.0f MB per-volume "
                                "safety cap. Its particles remain liquid rather than risk the out-of-device-memory "
                                "crash this check exists to prevent. (This message won't repeat for the same size "
                                "until it changes.)\n",
                                chunkDims.x, chunkDims.y, chunkDims.z, compactChunkCount, desc.NumChunks(),
                                static_cast<double>(estimatedBytes) / (1024.0 * 1024.0),
                                static_cast<double>(kMaxVertexBufferBytes) / (1024.0 * 1024.0));
                    s_LastWarned.push_back(key);
                }
                continue;
            }

            VoxelVolumeHandle handle = voxels.RegisterVolume(desc, chunkRemap, compactChunkCount);
            voxels.SetDensityFieldDirect(handle, std::move(density));
            voxels.MarchDirtyChunks(handle);

            // Chemical ID round-trip (see VoxelSystemGPU::SetChemicalId's
            // comment) - a GROUP can span multiple clusters (grouping
            // above is purely a GPU-efficiency decision, see this
            // function's own comment on why), so this weights each
            // cluster's already-computed averageColor/dominantChemicalId
            // by that cluster's particle count, same "simple majority
            // over particles" reasoning ClusterSystem itself uses one
            // level down. Not a real alloy/mixture model, same
            // disclaimer as ReformCluster's own comment.
            glm::vec3 groupColorSum(0.0f);
            uint32_t groupParticleCount = 0;
            std::array<uint32_t, GPU_MAX_CHEMICALS> groupChemicalVotes{};
            for (size_t ci : groups[g])
            {
                uint32_t clusterSize = static_cast<uint32_t>(clusters[ci].positions.size());
                groupColorSum += glm::vec3(clusters[ci].averageColor) * static_cast<float>(clusterSize);
                groupParticleCount += clusterSize;
                groupChemicalVotes[clusters[ci].dominantChemicalId] += clusterSize;
            }
            glm::vec4 groupColor = groupParticleCount > 0
                ? glm::vec4(groupColorSum / static_cast<float>(groupParticleCount), 1.0f)
                : glm::vec4(1.0f);
            uint32_t groupChemical = 0;
            uint32_t bestVotes = 0;
            for (uint32_t c = 0; c < GPU_MAX_CHEMICALS; ++c)
            {
                if (groupChemicalVotes[c] > bestVotes)
                {
                    bestVotes = groupChemicalVotes[c];
                    groupChemical = c;
                }
            }
            voxels.SetChemicalId(handle, groupChemical);

            // World placement is already baked into desc.origin (worldMin
            // above) - same convention the main.cpp smoke-test volume
            // uses, identity transform on top of a world-space origin.
            // groupColor (not a fixed/absent tint) is what makes "the
            // hardened form keeps properties like colour" actually true -
            // this volume now visibly renders as whatever color its
            // source particles had actually diffused to, not the
            // chemical's flat default or the old "no override" white.
            voxels.SetTransform(handle, glm::mat4(1.0f), groupColor);

            newVolumes.push_back(handle);
        }

        particles.KillParticles(allIndicesToKill);

        return newVolumes;
    }
}
