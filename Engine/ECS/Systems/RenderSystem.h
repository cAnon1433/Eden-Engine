#pragma once

#include "../Registry.h"
#include "../Components/TransformComponent.h"
#include "../Components/MeshComponent.h"
#include "../Components/ColorComponent.h"
#include "../Components/VisibilityComponent.h"
#include "../../Renderer/Vulkan/RendererTypes.h"
#include "../../Renderer/Renderer.h"
#include "../../Renderer/Frustum.h"

#include <vector>
#include <iostream>

namespace Eden::RenderSystem
{
    // Deliberately a free function, not an Eden::System - it doesn't
    // mutate ECS state, it reads Transform+Mesh(+Color+Visibility) pairs
    // and turns them into a per-frame draw list for Renderer::DrawFrame.
    // Call this directly from the main loop after Registry::UpdateSystems(),
    // not through UpdateSystems itself.
    //
    // `renderer` is used for two things: building this frame's Frustum
    // (Renderer::GetViewFrustum - camera-derived, so it has to be built
    // fresh each call, not cached) and looking up each mesh's bounding
    // radius for the culling test below. Frustum culling is a coarse,
    // conservative visibility test - entities entirely outside the
    // camera's view volume are skipped before a DrawCommand is ever
    // built for them, so the cost this actually removes is CPU-side
    // (model matrix fetch, instance buffer upload, and ultimately the
    // GPU draw call itself), not just "hidden geometry happens to get
    // Z-rejected later". Matters most once something ELSE is also
    // competing for frame budget (physics, SPH particles) - a scene that
    // renders 166,375 instanced cubes fine in isolation can still fall
    // behind once most of those cubes are off-screen at any given camera
    // angle and get built into the draw list anyway.
    inline std::vector<DrawCommand> BuildDrawList(Registry& registry, Renderer& renderer)
    {
        std::vector<DrawCommand> drawList;

        // Resolved ONCE per BuildDrawList call, not once per entity - see
        // Registry::GetStorage's comment. At tens of thousands of entities
        // this is the difference between ~6 hash-map lookups and ~150,000
        // of them, every single frame.
        auto& transforms = registry.GetStorage<TransformComponent>();
        auto& meshes = registry.GetStorage<MeshComponent>();
        auto& visibilities = registry.GetStorage<VisibilityComponent>();
        auto& colors = registry.GetStorage<ColorComponent>();

        // Built once per call (camera-dependent, so it can't be cached
        // across frames) and reused for every entity below - same
        // "resolve once, not per-entity" discipline as the storage
        // references above.
        Frustum frustum = renderer.GetViewFrustum();

        // Deliberately NOT registry.View<TransformComponent, MeshComponent>()
        // here, even though that's the pattern every other system uses.
        // View() eagerly walks EVERY matching entity to build its result
        // vector before this function gets to process any of it - which
        // means a scene with more entities than MAX_INSTANCES_PER_FRAME
        // still pays full O(entity count) up front, just to build a list
        // whose tail Renderer::DrawFrame then throws away. Iterating
        // transforms.Entities() directly and breaking the instant the cap
        // is hit keeps this bounded at O(min(entity count, cap)) instead.
        for (Entity entity : transforms.Entities())
        {
            if (drawList.size() >= MAX_INSTANCES_PER_FRAME)
            {
                // Renderer::DrawFrame's own MAX_INSTANCES_PER_FRAME clamp
                // still exists as a defense-in-depth backstop (in case
                // something else ever feeds it a list built a different
                // way), but THIS is what actually stops the work now -
                // there's no reason to resolve a mesh handle, fetch a
                // model matrix, or check for a color override on an
                // entity that can't fit in this frame's fixed-size
                // instance buffer regardless.
                // Logged once, not every frame - printing to stderr 60
                // times a second while over the cap is real I/O cost on
                // top of the entity-count problem this is warning about,
                // which defeats the point of a perf warning.
                static bool s_WarnedAboveCap = false;
                if (!s_WarnedAboveCap)
                {
                    std::cerr << "Eden: more than " << MAX_INSTANCES_PER_FRAME
                               << " drawable entities exist, capping this frame's draw list there "
                               << "(see MAX_INSTANCES_PER_FRAME in RendererTypes.h)\n";
                    s_WarnedAboveCap = true;
                }
                break;
            }

            if (!meshes.Has(entity))
            {
                continue;
            }

            // VisibilityComponent is opt-in: no component at all means
            // visible. Only entities that explicitly set visible = false
            // get skipped here.
            if (visibilities.Has(entity) && !visibilities.Get(entity).visible)
            {
                continue;
            }

            const auto& meshComp = meshes.Get(entity);
            if (meshComp.handle == InvalidMeshHandle)
            {
                continue;
            }

            const TransformComponent& transform = transforms.Get(entity);

            // Frustum cull BEFORE building the DrawCommand (model matrix
            // fetch, color lookup) below - the entire point is to skip
            // that work for off-screen entities, not just skip the draw
            // call while still paying CPU cost to prepare it.
            //
            // World-space bounding radius = mesh's own local-space
            // bounding radius (Renderer::GetMeshBoundingRadius - computed
            // once at mesh creation, see Mesh::GetBoundingRadius) times
            // the LARGEST scale component, not an average or per-axis
            // scale - a bounding SPHERE can't represent anisotropic
            // scale exactly, and erring toward "too big" here only costs
            // an occasional unnecessary draw, while erring toward "too
            // small" would mean wrongly culling something actually
            // visible. Same conservative-over-exact reasoning
            // CollisionSystem's AabbHalfExtents/ScaledCollider used for
            // round shapes under anisotropic scale, just picking the max
            // instead of an average since culling specifically needs
            // "never too small", not "reasonably close".
            float meshRadius = renderer.GetMeshBoundingRadius(meshComp.handle);
            float maxScale = glm::max(transform.scale.x, glm::max(transform.scale.y, transform.scale.z));
            float worldRadius = meshRadius * maxScale;

            if (!frustum.ContainsSphere(transform.position, worldRadius))
            {
                continue;
            }

            DrawCommand drawCmd;
            drawCmd.mesh = meshComp.handle;
            drawCmd.model = transform.GetModelMatrix();

            if (colors.Has(entity))
            {
                drawCmd.colorOverride = glm::vec4(colors.Get(entity).color, 1.0f);
            }

            drawList.push_back(drawCmd);
        }

        return drawList;
    }
}
