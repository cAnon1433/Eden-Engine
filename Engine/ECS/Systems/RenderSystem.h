#pragma once

#include "../Registry.h"
#include "../Components/TransformComponent.h"
#include "../Components/MeshComponent.h"
#include "../Components/ColorComponent.h"
#include "../Components/VisibilityComponent.h"
#include "../../Renderer/Vulkan/RendererTypes.h"
#include "../../Renderer/Renderer.h"
#include "../../Renderer/Frustum.h"
#include "../../Renderer/SoftwareOcclusionBuffer.h"

#include <vector>
#include <algorithm>
#include <iostream>

namespace Eden::RenderSystem
{
    namespace detail
    {
        // Everything needed to either build this entity's DrawCommand or
        // decide it's occluded, collected up front so the occlusion pass
        // below doesn't need to re-touch ECS storage per candidate -
        // same "resolve once" discipline as the storage references in
        // BuildDrawList itself.
        struct Candidate
        {
            MeshHandle mesh;
            glm::mat4 model;
            bool hasColorOverride;
            glm::vec4 colorOverride;

            bool projected; // false if ProjectAabb couldn't place this behind-camera-adjacent box - see its own comment
            SoftwareOcclusionBuffer::ScreenRect rect;
            float nearDepth;
        };
    }

    // Deliberately a free function, not an Eden::System - it doesn't
    // mutate ECS state, it reads Transform+Mesh(+Color+Visibility) pairs
    // and turns them into a per-frame draw list for Renderer::DrawFrame.
    // Call this directly from the main loop after Registry::UpdateSystems(),
    // not through UpdateSystems itself.
    //
    // `renderer` is used for three things: building this frame's Frustum
    // (Renderer::GetViewFrustum - camera-derived, so it has to be built
    // fresh each call, not cached), looking up each mesh's bounding
    // radius, and running the software occlusion pass (see
    // SoftwareOcclusionBuffer.h) via Renderer::GetOcclusionBuffer.
    // Frustum culling and occlusion culling are both coarse, conservative
    // visibility tests done BEFORE a DrawCommand is ever built, so the
    // cost they remove is CPU-side (model matrix fetch, instance buffer
    // upload, GPU draw call) - not just "hidden geometry gets Z-rejected
    // later, who cares". Matters most once something ELSE is also
    // competing for frame budget (physics, SPH particles), and once a
    // scene actually has large objects blocking view of others - a
    // scattered field of same-size objects with nothing substantial
    // blocking anything gets little from the occlusion pass specifically
    // (frustum culling still helps regardless of scene shape).
    inline std::vector<DrawCommand> BuildDrawList(Registry& registry, Renderer& renderer)
    {
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
        glm::mat4 viewProjection = renderer.GetViewProjectionMatrix();
        SoftwareOcclusionBuffer& occlusionBuffer = renderer.GetOcclusionBuffer();
        bool occlusionEnabled = renderer.EnableOcclusionCulling;

        std::vector<detail::Candidate> candidates;

        // --- Phase 1: gather frustum-visible candidates ------------------
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
            if (candidates.size() >= MAX_INSTANCES_PER_FRAME)
            {
                // Renderer::DrawFrame's own MAX_INSTANCES_PER_FRAME clamp
                // still exists as a defense-in-depth backstop (in case
                // something else ever feeds it a list built a different
                // way), but THIS is what actually stops the work now -
                // capping CANDIDATE collection, not just the final list,
                // since occlusion culling can only ever shrink the final
                // count further - there's no scenario where collecting
                // MORE than the frame could ever draw helps.
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

            // Frustum cull BEFORE building anything else below - the
            // entire point is to skip that work for off-screen entities,
            // not just skip the draw call while still paying CPU cost to
            // prepare it.
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

            detail::Candidate candidate;
            candidate.mesh = meshComp.handle;
            candidate.model = transform.GetModelMatrix();
            candidate.hasColorOverride = colors.Has(entity);
            if (candidate.hasColorOverride)
            {
                candidate.colorOverride = glm::vec4(colors.Get(entity).color, 1.0f);
            }

            if (occlusionEnabled)
            {
                // Sphere-derived world AABB - same conservative
                // sphere-to-bounds approach as the frustum test above,
                // reused here rather than inventing a tighter true AABB
                // (a real per-mesh AABB would cull slightly more
                // aggressively for non-cube meshes, but isn't computed
                // anywhere yet - see Mesh::GetBoundingRadius, which only
                // tracks a bounding sphere. Worth revisiting if occlusion
                // culling's effectiveness on non-cube meshes turns out to
                // matter, not needed to get a correct first version).
                glm::vec3 worldMin = transform.position - glm::vec3(worldRadius);
                glm::vec3 worldMax = transform.position + glm::vec3(worldRadius);
                candidate.projected = occlusionBuffer.ProjectAabb(worldMin, worldMax, viewProjection,
                    renderer.GetCamera().Position, renderer.GetCamera().Front, candidate.rect);
                candidate.nearDepth = candidate.projected ? candidate.rect.nearDepth : 0.0f;
            }
            else
            {
                candidate.projected = false;
                candidate.nearDepth = 0.0f;
            }

            candidates.push_back(candidate);
        }

        std::vector<DrawCommand> drawList;
        drawList.reserve(candidates.size());

        if (!occlusionEnabled)
        {
            for (const detail::Candidate& c : candidates)
            {
                DrawCommand drawCmd;
                drawCmd.mesh = c.mesh;
                drawCmd.model = c.model;
                if (c.hasColorOverride)
                {
                    drawCmd.colorOverride = c.colorOverride;
                }
                drawList.push_back(drawCmd);
            }
            return drawList;
        }

        // --- Phase 2: occlusion pass, nearest-to-farthest -----------------
        // Sorted so each candidate is only ever tested against occluders
        // that are genuinely NEARER than it - processing in this order
        // is what makes the single interleaved test-then-rasterize pass
        // below correct rather than order-dependent-and-buggy: testing a
        // candidate against occluders collected from farther-away
        // objects would be physically backwards (something farther away
        // can't occlude something nearer), and testing a candidate
        // against ITS OWN just-rasterized depth would wrongly cull it
        // against itself. Near-to-far order sidesteps both: by the time
        // candidate N is tested, the buffer only contains depth from
        // candidates 0..N-1, all strictly nearer.
        std::sort(candidates.begin(), candidates.end(),
            [](const detail::Candidate& a, const detail::Candidate& b)
            {
                // Un-projected candidates (behind-camera edge case, see
                // ProjectAabb) sort last - order among them doesn't
                // matter, they skip the occlusion test entirely below.
                if (a.projected != b.projected)
                {
                    return a.projected; // projected ones first
                }
                return a.nearDepth < b.nearDepth;
            });

        occlusionBuffer.Clear();
        int minOccluderCells = renderer.MinOccluderFootprintCells;

        for (const detail::Candidate& c : candidates)
        {
            bool visible = true;

            if (c.projected)
            {
                visible = occlusionBuffer.IsVisible(c.rect);

                if (visible)
                {
                    int width = c.rect.maxX - c.rect.minX + 1;
                    int height = c.rect.maxY - c.rect.minY + 1;
                    if (width * height >= minOccluderCells)
                    {
                        occlusionBuffer.RasterizeOccluder(c.rect);
                    }
                }
            }
            // !c.projected: couldn't be placed in the occlusion buffer at
            // all (behind-camera-adjacent edge case) - treated as
            // visible here, same "let frustum culling handle genuinely
            // off-screen cases" reasoning as ProjectAabb's own comment.

            if (!visible)
            {
                continue;
            }

            DrawCommand drawCmd;
            drawCmd.mesh = c.mesh;
            drawCmd.model = c.model;
            if (c.hasColorOverride)
            {
                drawCmd.colorOverride = c.colorOverride;
            }
            drawList.push_back(drawCmd);
        }

        return drawList;
    }
}
