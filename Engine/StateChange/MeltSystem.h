#pragma once

#include "../Voxel/VoxelSystemGPU.h"
#include "../Particles/GPU/ParticleSystemGPU.h"

namespace Eden
{
    // Solid -> liquid half of the particle state-change pipeline (see
    // the project's Eden Particle State Transitions planning notes).
    // Reform (liquid -> solid) is separate, later work - this only
    // converts a voxel volume's currently-solid region into SPH
    // particles and empties the volume's rendered geometry.
    //
    // Deliberately a stateless static call, not a System: unlike
    // LifetimeSystem/SpinSystem, neither VoxelSystemGPU volumes nor
    // ParticleSystemGPU particles are ECS entities today (see
    // VoxelVolumeComponent's header comment - it's defined but unused;
    // main.cpp drives both through raw handles). Melt() matches that
    // shape rather than forcing ECS registration as a prerequisite.
    // Revisit if/when volumes become real entities.
    class MeltSystem
    {
    public:
        // Samples `volume`'s density field on a grid spaced at `spacing`
        // (default -1.0f, meaning "use particles.smoothingRadius * 0.6",
        // the same rest-spacing convention EmitBox falls back to), emits
        // one particle per solid sample (density < kVoxelIsoLevel) into
        // `particles`, then calls voxels.ClearVolume() so the original
        // solid no longer renders. Synchronous/blocking (ClearVolume's
        // MarchDirtyChunks call waits on the GPU queue, same as every
        // other Seed/Carve/March call in this system) - fine for an
        // event-driven melt trigger, not meant to run every frame.
        //
        // Pass a smaller explicit spacing for a denser fill - useful
        // when the default rest-spacing reads as visually sparse for a
        // given volume size, independent of whether the SPH sim itself
        // stays stable at that density (check GPU_MAX_PARTICLES headroom
        // and viscosity/substeps if you push this a lot smaller).
        //
        // Returns the number of particles actually spawned (may be less
        // than the solid sample count if ParticleSystemGPU is near
        // GPU_MAX_PARTICLES - see EmitPoints' truncation warning).
        static uint32_t Melt(VoxelSystemGPU& voxels, VoxelVolumeHandle volume, ParticleSystemGPU& particles,
                              const glm::vec3& initialVelocity = glm::vec3(0.0f), float spacing = -1.0f);
    };
}
