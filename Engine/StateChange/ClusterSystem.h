#pragma once

#include "../Particles/GPU/ParticleSystemGPU.h"
#include <glm/glm.hpp>
#include <vector>

namespace Eden
{
    // A reform-eligible group of cooled, spatially-clustered particles -
    // world-space positions plus each particle's index into
    // ParticleSystemGPU's buffers at the moment FindReformClusters read
    // them back (valid to pass straight to ParticleSystemGPU::KillParticles
    // as long as no Step()/Emit*/KillParticles call happens in between -
    // ReformSystem's whole point is to consume a cluster in one
    // uninterrupted call).
    struct ReformCluster
    {
        std::vector<glm::vec3> positions;
        std::vector<uint32_t> particleIndices;
    };

    // Reform-detection half of the melt/reform pipeline (see the
    // project's Eden Particle State Transitions planning notes -
    // "Clustering/reform detection" build-order step). The notes'
    // stated preference was reusing ParticleSystemGPU's own GPU hash
    // grid for this; in practice that grid only exists mid-Step()
    // (built and torn down within RecordPendingSteps, never persisted),
    // and reform already needs a full CPU-side readback regardless (see
    // ParticleSystemGPU::ReadBackAll's comment - VoxelSystemGPU's
    // CPU-authoritative density field has to be built from real CPU
    // positions somewhere in this pipeline). So this reclusters on the
    // CPU, after that one unavoidable readback, with its own simple
    // grid rather than trying to reach into the GPU compute grid's
    // transient state.
    class ClusterSystem
    {
    public:
        // Reads back every active particle's position + heat, keeps
        // only particles with 0 <= heat <= GPU_HEAT_REFORM_THRESHOLD
        // (see ParticleGPUTypes.h - excludes GPU_HEAT_NOT_MELTED
        // particles automatically, they're never reform candidates),
        // then flood-fills those into connected components using a
        // uniform grid keyed at clusterRadius (two cooled particles
        // within clusterRadius of each other join the same cluster).
        // Returns every component with at least minClusterSize
        // particles - deliberately no upper limit or "best cluster"
        // picking for this pass (see planning notes: single-material,
        // no cross-cluster joining logic needed yet).
        static std::vector<ReformCluster> FindReformClusters(ParticleSystemGPU& particles, float clusterRadius,
                                                               uint32_t minClusterSize = 8);
    };
}
