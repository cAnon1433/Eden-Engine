#pragma once

#include "../ECS/Registry.h"
#include "../Renderer/Renderer.h"
#include "../Physics/PhysicsSystem.h"
#include "../Physics/CollisionSystem.h"
#include "../Particles/ParticleSystem.h"
#include "../Voxel/VoxelSystemGPU.h"

namespace Eden::UI
{
    // Draws Eden's in-engine mesh editor: create cubes (untextured,
    // textured, or loaded from an OBJ file), create raymarch/SDF boxes,
    // and destroy existing entities of either kind, without touching
    // main.cpp/code between runs.
    //
    // Call this from inside the buildUI callback passed to
    // Renderer::DrawFrame - it issues ImGui:: calls directly (ImGui::Begin/
    // End and friends), it does NOT call ImGui::NewFrame()/Render() itself,
    // those are Renderer's job (see Renderer::DrawFrame). Needs Registry
    // (to create/destroy entities), Renderer (to upload new mesh/texture
    // GPU resources), and VoxelSystemGPU (to register/seed the density
    // fields raymarch boxes render from) - that's why this lives in its
    // own Engine/UI module rather than inside ECS, Renderer, or Voxel,
    // none of which should depend on each other.
    void DrawMeshPanel(Registry& registry, Renderer& renderer, VoxelSystemGPU& voxelSystem);

    // Draws a small global settings window exposing PhysicsSystem/
    // CollisionSystem tunables (gravity, restitution, friction,
    // positional correction, CCD sweep, broad-phase cell size) at
    // runtime - these were previously only settable in code. Same
    // calling convention as DrawMeshPanel: call from inside the buildUI
    // callback passed to Renderer::DrawFrame.
    void DrawPhysicsSettingsPanel(PhysicsSystem& physicsSystem, CollisionSystem& collisionSystem);

    // spawnOrigin is where the next Spawn press calls EmitBox centered on
    // - exposed as drag fields so a spawn location can be repositioned
    // without a rebuild. particlePointSize is Renderer::ParticlePointSize
    // by reference (screen-space pixel diameter for the point-sprite
    // render mode - see Shaders/Source/particle_point.vert) - lives on
    // Renderer, not ParticleSystem, since it's a rendering concern (how
    // big a dot looks on screen) entirely separate from solver state, but
    // it's exposed from this same panel since that's where a person
    // tuning "how the fluid looks/behaves" would naturally look for it.
    void DrawParticleSettingsPanel(ParticleSystem& particleSystem, glm::vec3& spawnOrigin, float& particlePointSize);

    // Draws a small window exposing Renderer's visibility-culling
    // tunables: on/off for occlusion culling (EnableOcclusionCulling)
    // and the minimum occluder footprint (MinOccluderFootprintCells) -
    // see Renderer.h's comments on both for what they actually do.
    // Frustum culling has no runtime toggle exposed (it's cheap and has
    // no real tradeoff to tune, unlike occlusion culling's AABB-vs-exact-
    // silhouette accuracy tradeoff - see SoftwareOcclusionBuffer.h).
    void DrawRendererSettingsPanel(Renderer& renderer);
}
