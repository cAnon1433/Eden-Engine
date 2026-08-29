#pragma once

#include "ClusterSystem.h"
#include "../Voxel/VoxelSystemGPU.h"
#include "../Particles/GPU/ParticleSystemGPU.h"

namespace Eden
{
    // Reform half of the melt/reform pipeline - the counterpart to
    // MeltSystem::Melt. Finds cooled, clustered particles (via
    // ClusterSystem), and for each cluster: registers a brand-new voxel
    // volume sized to the cluster's own current extent, seeds it with
    // VoxelSystemGPU::SeedFromParticles (the blobby-SDF mode - see the
    // planning notes' Architecture decision: reform target is the
    // blob's actual shape, not the original pre-melt mesh), marches it,
    // and kills the consumed particles.
    //
    // Deliberately returns the new handles rather than tracking them
    // itself - same "raw handles, not ECS entities" shape MeltSystem
    // and the rest of this pipeline already use (see MeltSystem.h's
    // class comment); main.cpp's voxelSources list is what actually
    // needs to know about them to draw.
    class ReformSystem
    {
    public:
        // clusterRadius should match the SPH smoothingRadius scale
        // particles naturally pack at (see ClusterSystem::FindReformClusters -
        // too small and a resting puddle never counts as "clustered",
        // too large and unrelated nearby puddles merge into one blob).
        // minClusterSize filters out noise; if particles aren't settling
        // into one cohesive pool before they cool (see
        // ParticleSystemGPU's viscosityCoefficient/boundaryFriction -
        // that's what actually controls settling, and heatColdMultiplier
        // for why cooling itself can be uneven across a resting pool),
        // you'll get many small, scattered clusters instead of one big
        // one. That's a settling/cooling problem to fix at the source,
        // not something to paper over by raising minClusterSize.
        // particleRadius feeds SeedFromParticles directly - each
        // particle's sphere radius in the reformed density field, and
        // therefore what actually determines whether neighboring
        // particles' geometry visually fuses into one blob or stays as
        // separate isolated balls. This is NOT the same thing as
        // ParticleSystemGPU::boundaryRadius (the tiny hard-collision
        // radius) - particles rest at the SPH's natural equilibrium
        // spacing (smoothingRadius * 0.6), which is much larger than
        // 2*boundaryRadius, so seeding with boundaryRadius means resting
        // neighbors' spheres never overlap and every particle marches
        // out alone regardless of clustering. Pass something scaled off
        // smoothingRadius * 0.6 instead (a fraction below 1.0, so
        // neighbors' spheres genuinely overlap rather than just
        // touching) - see main.cpp's reformBlobRadius for the value this
        // was actually tuned with.
        //
        // Every cluster FindReformClusters returns is greedily grouped
        // by spatial proximity into as few volumes as possible without
        // ever coarsening resolution below particleRadius's own scale
        // (see the .cpp for why - an earlier version forced everything
        // into ONE shared bounding box and coarsened to fit, which meant
        // a single distant outlier cluster could degrade the WHOLE
        // grid's resolution, including a perfectly good main puddle,
        // down to slivers). Nearby clusters (the common case - a
        // puddle's own clusters are all close together) still end up in
        // one volume, one set of GPU round trips; a genuinely distant
        // straggler gets its own separate, still-full-resolution volume
        // instead.
        //
        // smoothRadius passes straight through to SeedFromParticles -
        // see that function's comment. 0.0 keeps the old hard-union
        // look (individual spheres stay visible); something around
        // particleRadius rounds the seams into one continuous blob.
        static std::vector<VoxelVolumeHandle> Reform(ParticleSystemGPU& particles, VoxelSystemGPU& voxels,
                                                       float clusterRadius, float particleRadius,
                                                       uint32_t minClusterSize = 8, float smoothRadius = 0.0f);
    };
}
