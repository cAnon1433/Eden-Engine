#pragma once

#include "../ECS/Registry.h"
#include "../Renderer/Renderer.h"
#include "../Physics/PhysicsSystem.h"
#include "../Physics/CollisionSystem.h"
#include "../Particles/ParticleSystem.h"

namespace Eden::UI
{
    // Draws Eden's in-engine mesh editor: create cubes (untextured,
    // textured, or loaded from an OBJ file) and destroy existing ones,
    // without touching main.cpp/code between runs.
    //
    // Call this from inside the buildUI callback passed to
    // Renderer::DrawFrame - it issues ImGui:: calls directly (ImGui::Begin/
    // End and friends), it does NOT call ImGui::NewFrame()/Render() itself,
    // those are Renderer's job (see Renderer::DrawFrame). Needs both
    // Registry (to create/destroy entities) and Renderer (to upload new
    // mesh/texture GPU resources) - that's why this lives in its own
    // Engine/UI module rather than inside ECS or Renderer, neither of
    // which should depend on the other.
    void DrawMeshPanel(Registry& registry, Renderer& renderer);

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
}
