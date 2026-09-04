#include "Engine/Renderer/Renderer.h"
#include "Engine/ECS/Registry.h"
#include "Engine/ECS/Components/TransformComponent.h"
#include "Engine/ECS/Components/MeshComponent.h"
#include "Engine/ECS/Components/RotationSpeedComponent.h"
#include "Engine/ECS/Components/LifetimeComponent.h"
#include "Engine/ECS/Components/ColorComponent.h"
#include "Engine/ECS/Systems/SpinSystem.h"
#include "Engine/ECS/Systems/RenderSystem.h"
#include "Engine/ECS/Systems/LifetimeSystem.h"
#include "Engine/UI/EdenUI.h"
#include "Engine/Physics/RigidBodyComponent.h"
#include "Engine/Physics/ColliderComponent.h"
#include "Engine/Physics/PhysicsSystem.h"
#include "Engine/Physics/CollisionSystem.h"
#include "Engine/Particles/ParticleSystem.h"
#include "Engine/Particles/GPU/ParticleSystemGPU.h"
#include "Engine/Voxel/VoxelSystemGPU.h"
#include "Engine/Renderer/Raymarch/RaymarchSystem.h"
#include "Engine/StateChange/MeltSystem.h"
#include "Engine/StateChange/ReformSystem.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <imgui.h>

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <optional>
#include <limits>

namespace
{
    constexpr uint32_t WINDOW_WIDTH = 1280;
    constexpr uint32_t WINDOW_HEIGHT = 720;

    // Physics runs at a fixed 60Hz regardless of render framerate - see
    // Engine/Physics/PhysicsSystem.h for why. MAX_PHYSICS_STEPS_PER_FRAME
    // guards against the "spiral of death": if a frame stalls hard (a
    // breakpoint, a slow asset load), the accumulator could otherwise
    // queue up hundreds of catch-up steps and try to run them all in one
    // frame, which takes even longer, which queues up more steps next
    // frame. Capping it means physics visibly slows down/skips instead of
    // permanently falling further behind and locking up the app.
    constexpr float FIXED_TIMESTEP = 1.0f / 60.0f;
    constexpr int MAX_PHYSICS_STEPS_PER_FRAME = 5;

    bool g_FirstMouse = true;
    double g_LastMouseX = 0.0;
    double g_LastMouseY = 0.0;

    // Starts locked (GLFW_CURSOR_DISABLED, matching the original always-
    // captured FPS-style mouse-look). TAB toggles this - while unlocked,
    // camera look/movement are skipped entirely so the mouse and keyboard
    // are free for clicking ImGui buttons and typing in its text fields.
    // Without this, there'd be no way to interact with Engine/UI/EdenUI.h
    // at all: a locked, invisible, window-centered cursor can't click
    // anything.
    bool g_CursorLocked = true;

    // Edge-tracked (not held-state) so a carve fires once per click, not
    // once per frame the button happens to be down - see
    // ProcessVoxelCarveInput's own comment on why continuous "drill"
    // carving isn't what this does yet.
    bool g_CarveMouseWasDown = false;

    // Edge-tracked the same way g_CarveMouseWasDown is - M triggers one
    // melt per press, not one per frame the key happens to be held.
    bool g_MeltKeyWasDown = false;

    // Edge-tracked the same way g_MeltKeyWasDown is - H triggers one
    // reform pass per press.
    bool g_ReformKeyWasDown = false;

    // TEMP DEBUG - remove after diagnosing. J dumps a vertical column of
    // live SampleSignedDistance readings through the original test
    // sphere's own center, spanning its full registered grid height -
    // shows exactly where physics currently thinks the surface is,
    // independent of any specific collision moment.
    bool g_ProbeKeyWasDown = false;

    void FramebufferResizeCallback(GLFWwindow* window, int /*width*/, int /*height*/)
    {
        auto* renderer = reinterpret_cast<Eden::Renderer*>(glfwGetWindowUserPointer(window));
        renderer->NotifyFramebufferResized();
    }

    void KeyCallback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/)
    {
        if (key == GLFW_KEY_TAB && action == GLFW_PRESS)
        {
            g_CursorLocked = !g_CursorLocked;
            glfwSetInputMode(window, GLFW_CURSOR, g_CursorLocked ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
            // Re-locking after the cursor moved freely in UI mode would
            // otherwise cause a big, jarring camera snap on the next
            // CursorPosCallback (computed as a large jump from wherever
            // the cursor last was to the window center) - forcing
            // g_FirstMouse back to true makes that next callback just
            // re-anchor silently instead, same as the very first frame.
            g_FirstMouse = true;
        }
    }

    void CursorPosCallback(GLFWwindow* window, double xpos, double ypos)
    {
        if (!g_CursorLocked)
        {
            // In UI mode: don't touch g_LastMouseX/Y either, so the big
            // jump that would otherwise be computed is simply never
            // computed - re-locking (see KeyCallback) resets g_FirstMouse
            // instead of relying on this branch to have kept them in sync.
            return;
        }

        auto* renderer = reinterpret_cast<Eden::Renderer*>(glfwGetWindowUserPointer(window));

        if (g_FirstMouse)
        {
            g_LastMouseX = xpos;
            g_LastMouseY = ypos;
            g_FirstMouse = false;
        }

        float xOffset = static_cast<float>(xpos - g_LastMouseX);
        float yOffset = static_cast<float>(g_LastMouseY - ypos);

        g_LastMouseX = xpos;
        g_LastMouseY = ypos;

        renderer->GetCamera().ProcessMouseMovement(xOffset, yOffset);
    }

    void ProcessKeyboardInput(GLFWwindow* window, Eden::Renderer& renderer, float deltaTime)
    {
        using Eden::CameraMovement;
        auto& camera = renderer.GetCamera();

        // Movement is tied to the same lock as mouse-look (see
        // KeyCallback/CursorPosCallback) - otherwise typing "w" or "s"
        // into one of EdenUI's text fields would also drive the camera
        // forward/backward while you're trying to type a file path.
        if (g_CursorLocked)
        {
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
                camera.ProcessKeyboard(CameraMovement::Forward, deltaTime);
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
                camera.ProcessKeyboard(CameraMovement::Backward, deltaTime);
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
                camera.ProcessKeyboard(CameraMovement::Left, deltaTime);
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
                camera.ProcessKeyboard(CameraMovement::Right, deltaTime);
            if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
                camera.ProcessKeyboard(CameraMovement::Up, deltaTime);
            if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
                camera.ProcessKeyboard(CameraMovement::Down, deltaTime);
        }

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    // Left-click carves at wherever the (locked, FPS-style) camera is
    // actually looking - a real ray march against VoxelSystemGPU's
    // CPU-authoritative density field (RaycastSurfaceSegments), not the
    // original mathematical seed sphere an earlier version of this
    // function used.
    //
    // Plain click: a single bite-sized carve at the nearest surface
    // point only - easiest to reason about while testing (one click,
    // one visible dent). Hold LEFT SHIFT while clicking: drill a full
    // tunnel through every solid segment the ray crosses (outer shell,
    // a hollow from an earlier carve, a far wall, however many there
    // are) - the "punch all the way through" behavior. Both call the
    // same Carve()/MarchDirtyChunks(); this is purely about how much of
    // the ray gets carved per click.
    //
    // Both are synchronous, blocking calls (vkQueueWaitIdle inside
    // MarchDirtyChunks; Carve() itself is pure CPU + a fast partial
    // upload), fine for occasional clicks but would visibly stall the
    // frame if carving became continuous/held-button - not a concern
    // for a first interactive test, worth revisiting (a deferred-
    // dispatch pattern like ParticleSystemGPU's RecordPendingSteps) if
    // "drill while holding" ever becomes the actual desired feel.
    // Cheap ray-vs-AABB slab test, world-space - used to skip candidates
    // in FindTargetedVolume whose bounds the ray doesn't actually come
    // near, BEFORE calling RaycastSurfaceSegments. Real bug this fixes,
    // not just a performance nicety: SampleDensityTrilinear CLAMPS any
    // out-of-bounds query to the volume's nearest edge sample rather than
    // reporting "not near this volume" - harmless when there was one
    // terrain volume (or small, localized test shapes) in the candidate
    // list, but with terrain now split into a 3x3 chunk grid
    // (RegisterTerrainChunks below), every click raycasts against 9
    // volumes, most of which the camera isn't anywhere near. For those,
    // every sample along the ray got clamped to that chunk's boundary
    // column, which can read as a false "solid" hit right at the ray
    // origin - and FindTargetedVolume keeps whichever candidate's hit is
    // CLOSEST, so a spurious near-origin false-positive on some far-off
    // chunk could beat the real hit on the ground actually being looked
    // at. Returns false (skip) only when the ray provably never comes
    // within maxDistance of the box - true positives always still reach
    // the real raycast.
    bool RayIntersectsAABB(const glm::vec3& rayOrigin, const glm::vec3& rayDir, float maxDistance,
                            const glm::vec3& aabbMin, const glm::vec3& aabbMax)
    {
        float tMin = 0.0f;
        float tMax = maxDistance;

        for (int axis = 0; axis < 3; ++axis)
        {
            float origin = rayOrigin[axis];
            float dir = rayDir[axis];
            float boxMin = aabbMin[axis];
            float boxMax = aabbMax[axis];

            if (std::abs(dir) < 1e-8f)
            {
                // Parallel to this axis' slab - miss unless already
                // inside it on this axis.
                if (origin < boxMin || origin > boxMax)
                {
                    return false;
                }
                continue;
            }

            float invDir = 1.0f / dir;
            float t0 = (boxMin - origin) * invDir;
            float t1 = (boxMax - origin) * invDir;
            if (t0 > t1) std::swap(t0, t1);

            tMin = std::max(tMin, t0);
            tMax = std::min(tMax, t1);
            if (tMin > tMax)
            {
                return false;
            }
        }

        return true;
    }

    // Raycasts every candidate volume and returns whichever one the
    // camera is actually looking at (closest hit's entry point across
    // all candidates), plus that hit's surface segments - or nullopt if
    // none are hit within range. Shared by carve and melt: neither
    // should be hardcoded to a single volume anymore now that
    // ReformSystem can create new ones at runtime - both need "whatever
    // hardened mesh I'm aiming at right now", the original test sphere
    // being just one entry in that candidate list, not special-cased.
    std::optional<Eden::VoxelVolumeHandle> FindTargetedVolume(Eden::VoxelSystemGPU& voxelSystem, const Eden::Camera& camera,
                                                                const std::vector<Eden::VoxelVolumeHandle>& candidates,
                                                                float maxDistance,
                                                                std::vector<std::pair<glm::vec3, glm::vec3>>& outSegments)
    {
        std::optional<Eden::VoxelVolumeHandle> best;
        float bestDistanceSq = std::numeric_limits<float>::max();
        glm::vec3 rayDir = glm::normalize(camera.Front);

        for (Eden::VoxelVolumeHandle handle : candidates)
        {
            // AABB pre-check - see RayIntersectsAABB's own comment for
            // why this is a correctness fix, not just an optimization.
            Eden::VoxelSystemGPU::VolumeBounds bounds = voxelSystem.GetVolumeBounds(handle);
            if (!RayIntersectsAABB(camera.Position, rayDir, maxDistance, bounds.worldMin, bounds.worldMax))
            {
                continue;
            }

            std::vector<std::pair<glm::vec3, glm::vec3>> segments;
            if (!voxelSystem.RaycastSurfaceSegments(handle, camera.Position, camera.Front, maxDistance, segments))
            {
                continue; // nothing solid along this ray for this volume - e.g. an already-melted-empty one
            }

            glm::vec3 toEntry = segments.front().first - camera.Position;
            float distanceSq = glm::dot(toEntry, toEntry);
            if (distanceSq < bestDistanceSq)
            {
                bestDistanceSq = distanceSq;
                best = handle;
                outSegments = std::move(segments);
            }
        }

        return best;
    }

    // Sleep is a pure per-pair velocity check (see CollisionSystem.cpp's
    // ResolveContact) - a sleeping body resting on a Voxel volume is
    // NEVER told when that volume's own shape changes underneath it,
    // because carving/melting doesn't touch the sleeping body's own
    // transform or velocity at all, so nothing about it looks like a
    // "genuine disturbance" from the wake check's point of view. Left
    // alone, a ball asleep on the original sphere stays frozen exactly
    // there even after a carve hollows out what it's resting on - not a
    // stale collision READ (SampleSignedDistance is confirmed live and
    // correct every frame), just a body that's asleep and therefore
    // never re-evaluates whether to fall. Real physics engines require
    // exactly this: waking nearby sleepers is the caller's job whenever
    // static/kinematic geometry is edited out from under them. Called
    // right after Carve() and right after Melt() below.
    void WakeSleepingBodiesNearVolume(Eden::Registry& registry, Eden::VoxelSystemGPU& voxelSystem,
                                       Eden::VoxelVolumeHandle handle)
    {
        Eden::VoxelSystemGPU::VolumeBounds bounds = voxelSystem.GetVolumeBounds(handle);

        // Expanded past the volume's own AABB: a resting body's CENTER
        // sits outside the volume by roughly its own radius, and this
        // function has no idea what shape/size that body is - a flat,
        // generous margin (much larger than any test object in this
        // scene) is simpler and safer than trying to know every
        // collider's exact size here.
        constexpr float kWakeMargin = 1.5f;
        glm::vec3 min = bounds.worldMin - glm::vec3(kWakeMargin);
        glm::vec3 max = bounds.worldMax + glm::vec3(kWakeMargin);

        for (Eden::Entity e : registry.View<Eden::TransformComponent, Eden::RigidBodyComponent>())
        {
            Eden::RigidBodyComponent& body = registry.GetComponent<Eden::RigidBodyComponent>(e);
            if (body.type != Eden::BodyType::Dynamic || !body.isSleeping)
            {
                continue;
            }

            const glm::vec3& pos = registry.GetComponent<Eden::TransformComponent>(e).position;
            if (pos.x >= min.x && pos.x <= max.x &&
                pos.y >= min.y && pos.y <= max.y &&
                pos.z >= min.z && pos.z <= max.z)
            {
                body.isSleeping = false;
                body.sleepTimer = 0.0f;
            }
        }
    }

    // Cross-tile carve helper. Terrain is 9 separate VoxelVolumeHandles
    // (see the terrain registration loop's own comment on why), and
    // Carve() only ever touches ONE handle. Previously that meant a
    // carve brush reaching into a neighboring tile's territory left that
    // tile completely untouched - a PERMANENT stale seam at every tile
    // boundary, no matter how correct the within-one-volume chunk sync
    // is (that's a different bug, already fixed - see Carve()'s chunkMin
    // BUGFIX comment. Confirmed by testing: internal chunk boundaries
    // are clean now, only tile boundaries still show a seam - this is
    // the cross-volume version of that exact same "shared boundary
    // sample" problem, one level up).
    //
    // Fix: carve every terrain tile whose bounds the (padded) brush
    // could plausibly reach, not just whichever one FindTargetedVolume
    // picked as the primary target - not "detect a crossing," just
    // unconditionally check all of them, since Carve() already clamps
    // gracefully (see its own "BOUNDARY CLAMPED" logging) when a request
    // mostly falls outside a volume's bounds, so carving a tile that
    // turns out not to actually be touched is a safe, cheap no-op, not a
    // correctness risk.
    //
    // Only applies when the primary target IS a terrain tile - the
    // small test sphere and any ReformSystem blob are isolated objects
    // with no "neighbor" concept, so this is skipped entirely for those
    // (checked via terrainTileHandles, not by shape/size, since that's
    // the actual thing that determines whether "neighbor" means
    // anything here).
    std::vector<Eden::VoxelVolumeHandle> CarveAcrossTerrainTiles(Eden::VoxelSystemGPU& voxelSystem,
                                                                   const std::vector<Eden::VoxelVolumeHandle>& terrainTileHandles,
                                                                   Eden::VoxelVolumeHandle primaryTarget,
                                                                   const glm::vec3& worldPos, float radius)
    {
        std::vector<Eden::VoxelVolumeHandle> touched;
        touched.push_back(primaryTarget);
        voxelSystem.Carve(primaryTarget, worldPos, radius);

        auto sameHandle = [](Eden::VoxelVolumeHandle a, Eden::VoxelVolumeHandle b)
        {
            return a == b; // VoxelVolumeHandle is a packed generational uint64_t (see VoxelField.h) - plain equality already compares both index and generation
        };

        bool primaryIsTerrainTile = false;
        for (Eden::VoxelVolumeHandle tile : terrainTileHandles)
        {
            if (sameHandle(tile, primaryTarget))
            {
                primaryIsTerrainTile = true;
                break;
            }
        }
        if (!primaryIsTerrainTile)
        {
            return touched;
        }

        // Same margin Carve() itself already uses internally (1 voxel of
        // padding beyond the brush radius) - a neighbor only needs
        // carving at all if the brush could plausibly reach even one
        // sample into its territory. This is generous on purpose, not
        // just "1 voxel": a straight face-adjacent neighbor only ever
        // has ONE axis out of bounds, so the distance-to-AABB check
        // below reduces to that single axis and a small margin is
        // plenty - but a DIAGONAL corner neighbor has TWO axes out of
        // bounds simultaneously, and the Euclidean distance combines
        // them (sqrt(dx^2+dz^2)), which climbs past a tight margin fast
        // even when the carve is genuinely close to the corner. Confirmed
        // real, not theoretical: a logged corner carve measured 1.42
        // units from the diagonal tile's nearest point while the old
        // margin was 1.3 - missed by a hair, every single time near that
        // corner, leaving it permanently uncarved while its two edge-
        // adjacent neighbors kept getting carved. Being this generous
        // costs nothing - Carve() already no-ops cleanly on a tile the
        // sphere doesn't actually reach (see its own "BOUNDARY CLAMPED"
        // logging), so a few extra harmless calls near a corner is a
        // fine trade for not silently skipping the diagonal tile.
        float margin = radius + 2.0f;

        for (Eden::VoxelVolumeHandle tile : terrainTileHandles)
        {
            if (sameHandle(tile, primaryTarget) || !voxelSystem.IsValid(tile))
            {
                continue;
            }

            Eden::VoxelSystemGPU::VolumeBounds bounds = voxelSystem.GetVolumeBounds(tile);
            glm::vec3 closestPoint = glm::clamp(worldPos, bounds.worldMin, bounds.worldMax);
            float distance = glm::length(worldPos - closestPoint);
            if (distance <= margin)
            {
                voxelSystem.Carve(tile, worldPos, radius);
                touched.push_back(tile);
            }
        }

        // TEMP DEBUG - corner-crack investigation. When this carve
        // touched 3+ tiles (a corner, not just a shared edge - an edge
        // only ever touches 2), sample the density at the exact carve
        // position from EVERY nearby terrain tile - not just the ones
        // this function decided to touch - and print them side by
        // side. All tiles seed from the same global noise function, so
        // pre-carve these should already closely agree; if one of them
        // still shows a meaningfully different (uncarved-looking) value
        // right after a carve that was supposed to reach it, that's
        // direct, undeniable proof of exactly which tile got missed and
        // by how much - not a proxy metric like [AmbiguousScan]. Remove
        // once the corner bug is actually understood.
        if (touched.size() >= 3)
        {
            std::printf("[CornerDensityCheck] worldPos=(%.3f,%.3f,%.3f) touchedCount=%zu\n",
                        worldPos.x, worldPos.y, worldPos.z, touched.size());
            for (Eden::VoxelVolumeHandle tile : terrainTileHandles)
            {
                if (!voxelSystem.IsValid(tile))
                {
                    continue;
                }
                Eden::VoxelSystemGPU::VolumeBounds bounds = voxelSystem.GetVolumeBounds(tile);
                glm::vec3 closest = glm::clamp(worldPos, bounds.worldMin, bounds.worldMax);
                float distToTile = glm::length(worldPos - closest);
                if (distToTile <= 3.0f) // only print tiles actually anywhere near this corner
                {
                    float density = voxelSystem.SampleSignedDistance(tile, worldPos);
                    bool wasTouched = sameHandle(tile, primaryTarget);
                    for (Eden::VoxelVolumeHandle t : touched)
                    {
                        if (sameHandle(t, tile))
                        {
                            wasTouched = true;
                        }
                    }
                    std::printf("[CornerDensityCheck]   tile density=%.4f touched=%s distToBounds=%.3f\n",
                                density, wasTouched ? "YES" : "no", distToTile);
                }
            }
        }

        return touched;
    }

    void ProcessVoxelCarveInput(GLFWwindow* window, Eden::Renderer& renderer, Eden::Registry& registry,
                                 Eden::VoxelSystemGPU& voxelSystem, const std::vector<Eden::VoxelVolumeHandle>& candidates,
                                 const std::vector<Eden::VoxelVolumeHandle>& terrainTileHandles)
    {
        if (!g_CursorLocked)
        {
            // A click made while the cursor is free (interacting with
            // EdenUI) shouldn't register as a carve the moment TAB locks
            // the camera back - reset the edge-tracker so re-locking
            // doesn't fire a carve from a stale "was down" state.
            g_CarveMouseWasDown = false;
            return;
        }

        bool mouseDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        if (mouseDown && !g_CarveMouseWasDown)
        {
            Eden::Camera& camera = renderer.GetCamera();

            // Generous search range along the view ray - big enough to
            // comfortably reach across the whole scene (container +
            // reformed volumes can end up meters apart) without
            // searching forever into empty space beyond everything.
            constexpr float kMaxCarveSearchDistance = 6.0f;

            std::vector<std::pair<glm::vec3, glm::vec3>> segments;
            std::optional<Eden::VoxelVolumeHandle> target =
                FindTargetedVolume(voxelSystem, camera, candidates, kMaxCarveSearchDistance, segments);

            if (target.has_value())
            {
                Eden::VoxelVolumeHandle voxelVolume = *target;

                constexpr float kCarveRadius = 0.3f; // world units - back down from 1.5. That bump was never a real fix, just a workaround that traded away precision - RecomputeExactDistances (called below) is what actually fixes stale-margin false collisions now, so radius can go back to a size that's actually useful for detail carving (a bite-sized chunk, not the whole sphere).
                float carveStep = kCarveRadius * 1.2f; // slight overlap so a tunnel has no gaps

                // Every volume actually touched this gesture (primary
                // target plus any terrain-tile neighbor CarveAcrossTerrainTiles
                // reached into) - deduplicated so a volume that gets
                // carved multiple times in one drag (common - the drill-
                // through loop below steps in small increments) only
                // gets ONE march/recompute/wake pass afterward, not one
                // per carve ball.
                std::vector<Eden::VoxelVolumeHandle> touchedVolumes;
                auto recordTouched = [&touchedVolumes](const std::vector<Eden::VoxelVolumeHandle>& newlyTouched)
                {
                    for (Eden::VoxelVolumeHandle h : newlyTouched)
                    {
                        bool alreadyRecorded = false;
                        for (Eden::VoxelVolumeHandle existing : touchedVolumes)
                        {
                            if (existing == h)
                            {
                                alreadyRecorded = true;
                                break;
                            }
                        }
                        if (!alreadyRecorded)
                        {
                            touchedVolumes.push_back(h);
                        }
                    }
                };

                bool drillThrough = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS;
                if (drillThrough)
                {
                    // Carve a chain of overlapping spheres along EVERY
                    // solid segment the ray crosses, not just the first
                    // one - see the function comment.
                    for (const auto& [entry, exit] : segments)
                    {
                        float segmentLength = glm::length(exit - entry);
                        glm::vec3 segmentDir = segmentLength > 1e-5f ? (exit - entry) / segmentLength : glm::vec3(0.0f);
                        for (float d = 0.0f; d <= segmentLength; d += carveStep)
                        {
                            recordTouched(CarveAcrossTerrainTiles(voxelSystem, terrainTileHandles, voxelVolume,
                                                                   entry + segmentDir * d, kCarveRadius));
                        }
                    }
                }
                else
                {
                    // Single bite at the FIRST (nearest) segment's entry
                    // point only - the easy-to-test default.
                    recordTouched(CarveAcrossTerrainTiles(voxelSystem, terrainTileHandles, voxelVolume,
                                                           segments.front().first, kCarveRadius));
                }

                // One march per touched volume for the whole click's
                // worth of carving, not one per carve ball or per
                // segment - see Carve()'s own comment on why it's split
                // from MarchDirtyChunks.
                for (Eden::VoxelVolumeHandle touched : touchedVolumes)
                {
                    voxelSystem.MarchDirtyChunks(touched);

                    // The actual fix for stale-margin false collisions (see
                    // RecomputeExactDistances' own comment for the full
                    // mechanism) - run once per completed carve gesture,
                    // same batching reasoning as MarchDirtyChunks above, not
                    // once per carve ball. Correctness no longer depends on
                    // carve radius or coverage after this runs.
                    voxelSystem.RecomputeExactDistances(touched);

                    // TEMP DEBUG - testing whether marching cubes' classic
                    // face-ambiguity case is actually present at carve
                    // sites before committing to a table rewrite over it.
                    // Remove once the crack bug is understood.
                    voxelSystem.DebugScanAmbiguousCells(touched);

                    // See WakeSleepingBodiesNearVolume's comment - a body
                    // asleep on this volume needs to be told its shape just
                    // changed, or it never falls into what was just carved.
                    WakeSleepingBodiesNearVolume(registry, voxelSystem, touched);
                }
            }
        }
        g_CarveMouseWasDown = mouseDown;
    }

    // M key: melt whatever hardened volume the camera is aimed at
    // (original test sphere OR any volume ReformSystem has since
    // created - see FindTargetedVolume) - converts that volume's whole
    // current solid region into SPH particles via MeltSystem::Melt, then
    // clears it so it stops rendering. Melting is NOT one-time: a
    // volume that came from a previous H reform is exactly as meltable
    // as the original sphere, since ReformSystem builds real
    // CPU-authoritative VoxelVolumeGPU volumes (same SDF machinery, same
    // Melt() entry point) - there's nothing "final" about having gone
    // through reform once. One-shot per press, same edge-tracking shape
    // as the carve mouse button above.
    void ProcessMeltInput(GLFWwindow* window, Eden::Renderer& renderer, Eden::Registry& registry,
                           Eden::VoxelSystemGPU& voxelSystem, const std::vector<Eden::VoxelVolumeHandle>& candidates,
                           Eden::ParticleSystemGPU& particleSystem)
    {
        if (!g_CursorLocked)
        {
            g_MeltKeyWasDown = false;
            return;
        }

        bool keyDown = glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS;
        if (keyDown && !g_MeltKeyWasDown)
        {
            Eden::Camera& camera = renderer.GetCamera();
            constexpr float kMaxMeltSearchDistance = 6.0f; // same range as carving - see ProcessVoxelCarveInput

            std::vector<std::pair<glm::vec3, glm::vec3>> segments; // unused here - Melt() converts the whole volume, not just where aimed; FindTargetedVolume just needs SOME hit to pick a target
            std::optional<Eden::VoxelVolumeHandle> target =
                FindTargetedVolume(voxelSystem, camera, candidates, kMaxMeltSearchDistance, segments);

            if (target.has_value())
            {
                // Melt() defaults to particles.smoothingRadius * 0.6, the
                // SPH sim's own equilibrium rest-spacing - see MeltSystem.h.
                // A previous version of this test passed a tighter explicit
                // spacing to get more particles into the same sphere; that
                // was wrong - packing denser than the equilibrium spacing
                // starts particles ALREADY over-density, and WCSPH's
                // pressure term is steep (density^gamma, gamma=7), so they
                // immediately blow apart. The real way to get more
                // particles for the same visual volume is a finer overall
                // SPH resolution (smaller smoothingRadius, see
                // ParticleSystemGPU.h's default), not a denser melt fill.
                uint32_t spawned = Eden::MeltSystem::Melt(voxelSystem, *target, particleSystem);
                std::cout << "Eden: melted voxel volume into " << spawned << " particles\n";

                // Melt() clears the ENTIRE volume (ClearVolume) - an even
                // bigger shape change than a single carve bite, same
                // reasoning as WakeSleepingBodiesNearVolume's comment.
                WakeSleepingBodiesNearVolume(registry, voxelSystem, *target);
            }
        }
        g_MeltKeyWasDown = keyDown;
    }

    // H key: harden/reform smoke test - runs ReformSystem::Reform over
    // the whole particle system once per press. Manual trigger, same
    // reasoning as ProcessMeltInput: Reform() does a full blocking
    // GPU->CPU particle readback (see ParticleSystemGPU::ReadBackAll),
    // so it's explicitly NOT something to run every frame - a real game
    // would drive this periodically or off the heat system once that
    // exists, not off a keypress, but a manual trigger is what's
    // testable right now.
    // Gives a registered VoxelSystemGPU volume real collision by making
    // it an actual ECS entity - Static RigidBodyComponent (these never
    // move, see VoxelVolumeComponent's own comment) + a
    // ColliderShape::Voxel collider referencing the handle, sized from
    // the volume's own known world bounds. Called once for the initial
    // test sphere and once per new handle ReformSystem::Reform returns -
    // see both call sites below. Melting a volume (ClearVolume) doesn't
    // need this touched again: density everywhere reads back as
    // "outside" afterward, so SampleSignedDistance naturally reports no
    // solid surface anywhere and the collider just stops producing
    // contacts on its own, no entity bookkeeping required.
    Eden::Entity RegisterVoxelPhysicsEntity(Eden::Registry& registry, Eden::VoxelSystemGPU& voxelSystem,
                                             Eden::VoxelVolumeHandle handle)
    {
        Eden::VoxelSystemGPU::VolumeBounds bounds = voxelSystem.GetVolumeBounds(handle);
        glm::vec3 worldCenter = (bounds.worldMin + bounds.worldMax) * 0.5f;
        glm::vec3 halfExtents = (bounds.worldMax - bounds.worldMin) * 0.5f;

        Eden::Entity e = registry.CreateEntity();

        Eden::TransformComponent transform;
        transform.position = worldCenter;
        registry.AddComponent(e, transform);

        registry.AddComponent(e, Eden::VoxelVolumeComponent{ handle });

        Eden::RigidBodyComponent body;
        body.type = Eden::BodyType::Static;
        body.inverseMass = 0.0f;
        registry.AddComponent(e, body);

        Eden::ColliderComponent collider;
        collider.shape = Eden::ColliderShape::Voxel;
        collider.voxelVolume = handle;
        // localOffset stays zero - transform.position IS the volume's
        // world center (see worldCenter above), matching every other
        // collider's "position = center" convention, unlike
        // VoxelVolumeDesc::origin which is a min corner.
        collider.halfExtents = halfExtents;
        registry.AddComponent(e, collider);

        return e;
    }

    // --- Terrain Phase 3: chunk-level resident/sleep state -------------
    // Planning-notes addendum's "scenes" section flags this as an
    // explicit requirement, distinct from the existing per-RIGID-BODY
    // sleep/wake system (union-find islands, WakeSleepingBodiesNearVolume
    // above) - that one is triggered by physical rest/disturbance of a
    // single body. This is a per-REGION concept layered on top: is a
    // given terrain chunk within simulation range of the player at all,
    // independent of whether anything resting on it happens to be
    // physically at rest.
    //
    // What "sleep" actually gates for a terrain chunk today: removing
    // ColliderComponent (excludes it from CollisionSystem::Step's
    // registry.View<TransformComponent, ColliderComponent>() AND
    // ParticleSystemGPU's identical collider-gathering view - both
    // re-query the registry fresh every step, so this is a REAL
    // per-frame cost reduction, not cosmetic), plus exclusion from the
    // render draw-source list and the carve/melt candidate list (no
    // point raycasting or drawing a chunk nobody's near). It does NOT
    // gate any kind of reduced tick rate, because nothing currently
    // ticks on a resting terrain chunk every frame in the first place -
    // the addendum's own open question ("what ticks at reduced rate vs
    // not at all") doesn't have material to apply to yet for inert
    // static geometry. Revisit once something (heat decay, energy
    // propagation) actually runs per-frame against terrain.
    //
    // Known, deliberately unaddressed gap: a dynamic body resting on a
    // chunk that goes to sleep will fall through - the addendum
    // explicitly flags "how chunk wake/sleep transitions avoid a
    // visible pop/stutter" as real, open, undecided design work, and
    // this doesn't attempt to solve it. Acceptable for now because
    // sleep only triggers well outside the area being actively played
    // in for this small test map.
    struct TerrainChunkState
    {
        Eden::VoxelVolumeHandle handle;
        Eden::Entity entity;
        glm::vec3 worldCenterXZ; // Y intentionally unused - see UpdateTerrainChunkResidency's own comment on why this is an XZ-only distance check
        bool resident = true;   // every chunk starts resident - matches Phase 2's "everything registered and marched at startup" behavior
    };

    // Hysteresis (wake < sleep) instead of one shared threshold - a
    // chunk exactly at the boundary distance would otherwise flicker
    // resident/sleeping every frame as the player's position jitters by
    // fractions of a unit. Doesn't solve the addendum's "visible pop"
    // concern (there's still a hard cut when a chunk does cross), just
    // stops it from thrashing at the edge.
    //
    // These two numbers are tuned to this SPECIFIC 60x60m test map, not
    // a real target for a large world: kTerrainWakeRadius (30) is set
    // just past the farthest chunk-center-to-map-center distance
    // (~28.3m, the corner chunks) so every chunk starts resident at
    // spawn with no immediate pop-out, and the whole mechanism only
    // becomes observable by deliberately walking past the terrain's own
    // edge into the void beyond it. A real large world would want both
    // numbers much smaller relative to total world size.
    constexpr float kTerrainWakeRadius = 30.0f;
    constexpr float kTerrainSleepRadius = 38.0f;

    void UpdateTerrainChunkResidency(Eden::Registry& registry, Eden::VoxelSystemGPU& voxelSystem,
                                      std::vector<TerrainChunkState>& chunks, const glm::vec3& playerPosition)
    {
        for (TerrainChunkState& chunk : chunks)
        {
            if (!voxelSystem.IsValid(chunk.handle) || !registry.IsAlive(chunk.entity))
            {
                continue; // editor-destroyed or otherwise gone - not this function's job to handle
            }

            // XZ-only distance - vertical offset between the player's
            // eye height and a chunk's center is a near-constant
            // "standing height above ground" term that shouldn't count
            // toward residency the way horizontal distance should, same
            // reasoning real open-world streaming systems use.
            glm::vec2 delta(playerPosition.x - chunk.worldCenterXZ.x, playerPosition.z - chunk.worldCenterXZ.z);
            float distance = glm::length(delta);

            if (chunk.resident && distance > kTerrainSleepRadius)
            {
                chunk.resident = false;
                registry.RemoveComponent<Eden::ColliderComponent>(chunk.entity);
                std::printf("[TerrainLOD] chunk sleeping (distance %.1fm)\n", distance);
            }
            else if (!chunk.resident && distance < kTerrainWakeRadius)
            {
                chunk.resident = true;

                Eden::VoxelSystemGPU::VolumeBounds bounds = voxelSystem.GetVolumeBounds(chunk.handle);
                Eden::ColliderComponent collider;
                collider.shape = Eden::ColliderShape::Voxel;
                collider.voxelVolume = chunk.handle;
                collider.halfExtents = (bounds.worldMax - bounds.worldMin) * 0.5f;
                registry.AddComponent(chunk.entity, collider);

                // Same reasoning as the carve/melt call sites above -
                // static geometry just reappeared under whatever's
                // nearby; a sleeping dynamic body resting near it won't
                // re-evaluate contact on its own until nudged.
                WakeSleepingBodiesNearVolume(registry, voxelSystem, chunk.handle);
                std::printf("[TerrainLOD] chunk waking (distance %.1fm)\n", distance);
            }
        }
    }


    // Spawns one raymarch-rendered box: registers a small VoxelSystemGPU
    // volume sized to halfExtents, seeds it via SeedBox, and builds one
    // ECS entity carrying TransformComponent (world CENTER - see
    // RaymarchSystem.h's namespace comment) + VoxelVolumeComponent +
    // RaymarchVolumeComponent. This is the raymarch-path equivalent of
    // "registry.AddComponent(e, MeshComponent{cubeMesh})" for a box -
    // written once here since the mesh-to-raymarch conversion below
    // calls this at every former MeshComponent{cubeMesh} spawn site.
    //
    // withCollision=true additionally attaches RigidBodyComponent
    // (Static) + ColliderComponent, following RegisterVoxelPhysicsEntity's
    // exact pattern - collider.shape is set to ColliderShape::Voxel
    // regardless of the fact that this is a BOX-seeded volume, since
    // that's the shape whose narrow phase actually queries this
    // entity's own density field (SampleSignedDistance/SampleGradient)
    // rather than an analytic box formula - a voxel-box collider's
    // COLLISION SHAPE is still correct either way (it's a real
    // box-shaped density field, not an approximation), this is about
    // which narrow-phase code PATH gets used, not shape accuracy.
    // CollisionSystem now implements Sphere/Box/Capsule-vs-Voxel (see
    // CollisionSystem.cpp's TestSphereVsVoxel/TestBoxVsVoxel/
    // TestCapsuleVsVoxel), so any of those three collider shapes can
    // collide against this correctly.
    // Collision strategy for a raymarch-rendered box - see
    // SpawnRaymarchBox's own comment on why this isn't just a bool.
    enum class RaymarchBoxCollision
    {
        None,
        // DEFAULT CHOICE for anything raymarched/cube-marched in this
        // project going forward. ColliderShape::Voxel queries the SAME
        // density field this box renders from (SampleSignedDistance/
        // SampleGradient) - real narrow-phase support exists in both
        // CollisionSystem (general rigid-body physics: Sphere/Box/
        // Capsule-vs-Voxel, see CollisionSystem.cpp) AND
        // ParticleSystemGPU's GPU boundary-collision code path (see
        // ParticleGPUTypes.h's ColliderGPU::voxelParams/voxelSampleDims
        // and particle_integrate.comp's VoxelShapeDistance), so this is
        // safe for anything that needs to collide against ordinary
        // rigid bodies OR SPH particles. Critically, unlike AnalyticBox
        // below, a Voxel collider has no shape of its own - it just
        // holds a handle and queries whatever densityCPU currently
        // contains, so collision automatically tracks this box if it's
        // ever carved/melted/reformed into something else later. Prefer
        // this over AnalyticBox unless there's a specific reason not to
        // (see that option's own comment).
        Voxel,
        // Opt-in only, for STATIC geometry that's known to never
        // deform - ColliderShape::Box, analytic, sized to the SAME
        // halfExtents this function seeds the density field with. Gets
        // real box-vs-box SAT narrow phase (exact, not the multi-point
        // approximation TestBoxVsVoxel uses against a density field -
        // see that function's own comment on why it's an approximation)
        // at the cost of NOT tracking live shape changes the way Voxel
        // does - if this box is ever carved or melted, its collider
        // stays a fixed box shape while its rendered geometry changes
        // out from under it. Use this only when that tradeoff is
        // clearly worth it (performance-sensitive static geometry,
        // known to never deform) - Voxel above is the right default for
        // everything else.
        AnalyticBox
    };

    Eden::Entity SpawnRaymarchBox(Eden::Registry& registry, Eden::VoxelSystemGPU& voxelSystem,
                                   const glm::vec3& worldCenter, const glm::vec3& halfExtents,
                                   const glm::vec3& tintColor, RaymarchBoxCollision collision)
    {
        // Volume sized with a small margin beyond halfExtents so
        // SeedBox's box surface sits comfortably inside the field
        // rather than exactly at its boundary - same reasoning as the
        // marching-cubes test sphere's own kVoxelSeedRadius margin
        // comment above (Carve()/collision queries near a field's exact
        // edge get clamped and lose accuracy right at the boundary).
        constexpr float kFieldMargin = 0.2f;
        glm::vec3 fieldHalfExtents = halfExtents + glm::vec3(kFieldMargin);

        Eden::VoxelVolumeDesc desc;
        desc.origin = worldCenter - fieldHalfExtents; // VoxelVolumeDesc::origin is a min corner, unrelated to this entity's own TransformComponent convention
        desc.voxelSize = 0.08f; // finer than the test sphere's 0.15 - boxes read as boxes (not visibly faceted) at a coarser voxel size, but this stays consistent with SDF's exact-surface intent
        desc.chunkDims = glm::ivec3(2, 2, 2);
        Eden::VoxelVolumeHandle handle = voxelSystem.RegisterVolume(desc);

        glm::vec3 fieldExtent = glm::vec3(desc.VoxelDims()) * desc.voxelSize;
        glm::vec3 fieldLocalCenter = fieldExtent * 0.5f;
        voxelSystem.SeedBox(handle, fieldLocalCenter, halfExtents);
        // No MarchDirtyChunks - see the raymarch smoke-test sphere's
        // identical comment above for why a raymarch-only volume never
        // needs marching-cubes geometry generated.

        Eden::Entity e = registry.CreateEntity();

        Eden::TransformComponent transform;
        transform.position = worldCenter;
        registry.AddComponent(e, transform);

        registry.AddComponent(e, Eden::VoxelVolumeComponent{ handle });
        registry.AddComponent(e, Eden::RaymarchVolumeComponent{ tintColor });

        if (collision != RaymarchBoxCollision::None)
        {
            Eden::RigidBodyComponent body;
            body.type = Eden::BodyType::Static;
            body.inverseMass = 0.0f;
            registry.AddComponent(e, body);

            Eden::ColliderComponent collider;
            collider.halfExtents = halfExtents;
            if (collision == RaymarchBoxCollision::Voxel)
            {
                collider.shape = Eden::ColliderShape::Voxel;
                collider.voxelVolume = handle;
            }
            else // AnalyticBox
            {
                collider.shape = Eden::ColliderShape::Box;
                // No collider.voxelVolume set - ColliderShape::Box never
                // reads it, same as every other Box collider in this
                // file (see the physics smoke-test cube's own collider
                // setup elsewhere in this function).
            }
            registry.AddComponent(e, collider);
        }

        return e;
    }

    // TEMP DEBUG - remove after diagnosing. Scans a grid of vertical
    // columns across handle's FULL horizontal footprint (not just dead
    // center - the earlier single-column probe came back clean, but a
    // localized bad spot off-center wouldn't have shown up in that one
    // line). Only prints samples that are near a surface or inside solid
    // (dist < 0.3) plus each column's topmost/bottommost sample for
    // context, to keep output readable instead of dumping every sample
    // in the whole grid.
    void ProbeVoxelGrid(Eden::VoxelSystemGPU& voxelSystem, Eden::VoxelVolumeHandle handle)
    {
        Eden::VoxelSystemGPU::VolumeBounds bounds = voxelSystem.GetVolumeBounds(handle);
        float voxelSize = bounds.voxelSize;
        constexpr int kColumnsPerAxis = 7;

        std::cout << "[ProbeGrid] volume=" << handle << " worldMin=(" << bounds.worldMin.x << "," << bounds.worldMin.y << "," << bounds.worldMin.z
                  << ") worldMax=(" << bounds.worldMax.x << "," << bounds.worldMax.y << "," << bounds.worldMax.z
                  << ") " << kColumnsPerAxis << "x" << kColumnsPerAxis << " columns, only near-surface/solid rows shown\n";

        for (int cx = 0; cx < kColumnsPerAxis; ++cx)
        for (int cz = 0; cz < kColumnsPerAxis; ++cz)
        {
            float tx = (cx + 0.5f) / kColumnsPerAxis;
            float tz = (cz + 0.5f) / kColumnsPerAxis;
            float x = glm::mix(bounds.worldMin.x, bounds.worldMax.x, tx);
            float z = glm::mix(bounds.worldMin.z, bounds.worldMax.z, tz);

            float minDistThisColumn = 1e9f;
            float minDistY = 0.0f;
            bool printedHeader = false;

            for (float y = bounds.worldMin.y - voxelSize; y <= bounds.worldMax.y + voxelSize; y += voxelSize * 0.5f)
            {
                glm::vec3 p(x, y, z);
                float dist = voxelSystem.SampleSignedDistance(handle, p);
                bool solidExact = voxelSystem.IsWorldPointSolidExact(handle, p);

                if (dist < minDistThisColumn)
                {
                    minDistThisColumn = dist;
                    minDistY = y;
                }

                bool isEdge = (y <= bounds.worldMin.y - voxelSize + 1e-4f) || (y >= bounds.worldMax.y + voxelSize - voxelSize * 0.5f);
                if (dist < 0.3f || solidExact || isEdge)
                {
                    if (!printedHeader)
                    {
                        std::cout << "  column x=" << x << " z=" << z << ":\n";
                        printedHeader = true;
                    }
                    std::cout << "    y=" << y << "  dist=" << dist << "  IsWorldPointSolidExact=" << (solidExact ? "true" : "false") << "\n";
                }
            }

            std::cout << "  column x=" << x << " z=" << z << " -> min dist=" << minDistThisColumn << " at y=" << minDistY << "\n";
        }
    }

    void ProcessReformInput(GLFWwindow* window, Eden::Registry& registry, Eden::ParticleSystemGPU& particleSystem,
                             Eden::VoxelSystemGPU& voxelSystem, std::vector<Eden::VoxelVolumeHandle>& reformedVolumes)
    {
        if (!g_CursorLocked)
        {
            g_ReformKeyWasDown = false;
            return;
        }

        bool keyDown = glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS;
        if (keyDown && !g_ReformKeyWasDown)
        {
            // clusterRadius deliberately much wider than smoothingRadius
            // - a cohesion/surface-tension force now exists
            // (SimParamsGPU::cohesion, see particle_force.comp) but
            // defaults to 0/off, so out of the box this fluid still
            // doesn't clump into one blob when it settles - it spreads
            // into a puddle with gaps between locally-packed groups.
            // clusterRadius only controls which particles get batched
            // together for reform - it does NOT control whether the
            // resulting geometry visually fuses (see reformBlobRadius
            // below for that). If cohesion ever gets tuned up high
            // enough to change resting-pool shape meaningfully,
            // revisit whether clusterRadius still needs to be this
            // wide.
            //
            // reformBlobRadius (passed as SeedFromParticles' sphere
            // radius) is deliberately NOT particleSystem.boundaryRadius.
            // boundaryRadius (0.0333) is the tiny hard-collision radius;
            // particles actually REST at the SPH's natural equilibrium
            // spacing, smoothingRadius * 0.6 (~0.12) - far more than
            // 2*boundaryRadius apart. Seeding with boundaryRadius meant
            // neighboring particles' spheres in the density field never
            // overlapped, so every particle marched out as its own
            // isolated ball regardless of how well clustering grouped
            // them - clustering was never the bug. Using a fraction of
            // the actual rest spacing instead guarantees resting
            // neighbors' spheres overlap and fuse into one continuous
            // blob.
            float reformBlobRadius = particleSystem.smoothingRadius * 0.6f * 0.7f;

            // Smooths the seams between overlapping particle spheres
            // (see VoxelSystemGPU::SeedFromParticles' smoothRadius
            // parameter) - without this, the union is a hard min and
            // every particle's dome stays visible even where two
            // spheres overlap ("bubble bath" look). Roughly half
            // reformBlobRadius is a starting point, not a derived value
            // - raise it for a smoother/rounder result, lower it if the
            // blob starts losing too much surface detail/looking like a
            // shapeless blob.
            float reformSmoothRadius = reformBlobRadius * 0.5f;

            std::vector<Eden::VoxelVolumeHandle> newVolumes = Eden::ReformSystem::Reform(
                particleSystem, voxelSystem, particleSystem.smoothingRadius * 8.0f, reformBlobRadius, 8, reformSmoothRadius);

            reformedVolumes.insert(reformedVolumes.end(), newVolumes.begin(), newVolumes.end());

            // Give each freshly-reformed volume real collision - see
            // RegisterVoxelPhysicsEntity's comment.
            for (Eden::VoxelVolumeHandle newVolume : newVolumes)
            {
                RegisterVoxelPhysicsEntity(registry, voxelSystem, newVolume);
            }

            std::cout << "Eden: reform pass found " << newVolumes.size() << " cluster(s), "
                      << reformedVolumes.size() << " reformed volume(s) total\n";
        }
        g_ReformKeyWasDown = keyDown;
    }
}

int main()
{
    if (!glfwInit())
    {
        std::cerr << "Eden: failed to initialize GLFW\n";
        return EXIT_FAILURE;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Eden", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Eden: failed to create window\n";
        glfwTerminate();
        return EXIT_FAILURE;
    }

    Eden::Renderer renderer;

    glfwSetWindowUserPointer(window, &renderer);
    glfwSetFramebufferSizeCallback(window, FramebufferResizeCallback);
    glfwSetCursorPosCallback(window, CursorPosCallback);
    glfwSetKeyCallback(window, KeyCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    try
    {
        renderer.Init(window);

        // Declared here (moved up from its original spot near the
        // marching-cubes smoke test below) because the raymarch grid
        // conversion needs a live VoxelSystemGPU before that point now -
        // every SpawnRaymarchBox call, starting with the stress-test
        // grid right below, registers its own volume against this same
        // system. Init() only needs what Renderer already has ready
        // immediately after Init() returns.
        Eden::VoxelSystemGPU voxelSystem;
        voxelSystem.Init(renderer.GetDevice(), renderer.GetPhysicalDevice(), renderer.GetAllocator(),
                          renderer.GetCommandPool(), renderer.GetGraphicsQueue());

        Eden::Registry registry;
        registry.RegisterSystem<Eden::SpinSystem>();
        // Ticks every LifetimeComponent down and destroys the entity at
        // zero. Without this line, LifetimeComponent is just inert data -
        // nothing decrements it, nothing ever destroys the entity.
        registry.RegisterSystem<Eden::LifetimeSystem>();

        // ONE GPU upload, shared by every entity below via MeshHandle. This
        // is the direct fix for the MEGA CUBE problem: 1000 entities used
        // to mean 1000 separate vertex/index buffer uploads (and 1000
        // vkQueueWaitIdle stalls at startup). Now it's one upload no matter
        // how many entities reference it.
        Eden::MeshHandle cubeMesh = renderer.CreateCubeMesh(1.0f);

        // A grid of raymarched boxes - each one its own tiny VoxelSystemGPU
        // volume (see SpawnRaymarchBox above), NOT a shared mesh/instancing
        // scheme the way the old MeshComponent grid was. Shrunk again, from
        // 3x3x3 (27) to 2x2x2 (8), after kRaymarchMaxObjects itself had to
        // come DOWN to 24 (from 48) once real hardware testing hit an actual
        // MoltenVK validation failure - the density-buffer array binding's
        // descriptorCount counts against the fragment stage's
        // maxPerStageDescriptorStorageBuffers limit (31 on the tested Apple
        // Silicon GPU), which a 3x3x3 grid alone (27) would already crowd
        // before counting the spinning box, physics test cubes, container
        // floor/walls, and the original raymarch smoke-test sphere - all of
        // which also carry RaymarchVolumeComponent and compete for the same
        // 24-slot cap. See kRaymarchMaxObjects's own comment in
        // RaymarchTypes.h for the real numbers from that failure.
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                for (int k = 0; k < 2; ++k)
                {
                    glm::vec3 center = { -4.0f + j * 2.0f, -4.0f + k * 2.0f, i * 2.0f };
                    SpawnRaymarchBox(registry, voxelSystem, center, glm::vec3(0.5f),
                                      glm::vec3(1.0f, 0.5f, 0.2f), RaymarchBoxCollision::None);
                }
            }
        }

        // One extra box with a RotationSpeedComponent - proves SpinSystem
        // still works against a raymarch entity (it only needs
        // TransformComponent + RotationSpeedComponent, mesh-agnostic -
        // see SpinSystem::Update). No collider on this one either way,
        // so this conversion has zero physics-behavior impact.
        {
            Eden::Entity e = SpawnRaymarchBox(registry, voxelSystem, glm::vec3(0.0f, 3.0f, -2.0f), glm::vec3(0.5f),
                                               glm::vec3(1.0f, 0.5f, 0.2f), RaymarchBoxCollision::None);
            registry.AddComponent(e, Eden::RotationSpeedComponent{ 60.0f });
        }


        // Physics smoke test: a Dynamic cube with gravity enabled, high
        // up - should fall and (once it reaches the grid below) land or
        // pass through depending on collision.
        //
        // FORMERLY a known regression (documented here across several
        // sessions, kept for the historical record): converting this to
        // raymarch meant its collider became ColliderShape::Voxel
        // instead of the original ColliderShape::Box, and for a long
        // stretch CollisionSystem only implemented Sphere-vs-Voxel
        // narrow phase - so this Dynamic Voxel-shaped body had no
        // working pair against the Box-shaped floor grid below and fell
        // straight through. Left broken deliberately at the time (per
        // Hawkins' call to convert rendering first, fix collision
        // later), not silently patched over.
        //
        // RESOLVED: TestBoxVsVoxel now exists (CollisionSystem.cpp), so
        // this Box(floor)-vs-Voxel(this cube) pair - Box canonicalizes
        // to the A side, ShapeOrder(Box)=1 < ShapeOrder(Voxel)=4 - has
        // real narrow phase again. This cube should now land correctly.
        {
            Eden::Entity e = SpawnRaymarchBox(registry, voxelSystem, glm::vec3(0.0f, 12.0f, -2.0f), glm::vec3(0.5f),
                                               glm::vec3(0.2f, 0.6f, 1.0f), RaymarchBoxCollision::Voxel);

            Eden::RigidBodyComponent body;
            body.type = Eden::BodyType::Dynamic;
            body.inverseMass = 1.0f;
            registry.AddComponent(e, body); // overwrites the Static body SpawnRaymarchBox's withCollision path added
        }

        // A Static body sitting in the same spot as part of the grid,
        // just to confirm PhysicsSystem::Step correctly leaves Static
        // entities alone even when they carry a RigidBodyComponent. Same
        // KNOWN REGRESSION as the Dynamic cube above applies here too -
        // this is Static so PhysicsSystem::Step itself is unaffected
        // (Static bodies were always left alone regardless of collider
        // shape), but anything that WOULD have collided against this via
        // Box-vs-Box narrow phase now has nothing to collide against.
        {
            SpawnRaymarchBox(registry, voxelSystem, glm::vec3(0.0f, -4.0f, -2.0f), glm::vec3(0.5f),
                              glm::vec3(0.6f, 0.6f, 0.6f), RaymarchBoxCollision::Voxel);
        }

        Eden::PhysicsSystem physicsSystem;
        Eden::CollisionSystem collisionSystem;
        float physicsAccumulator = 0.0f;

        // --- Particle system (SPH fluid) smoke test -----------------
        // GPU-resident: neighbor search, density/pressure, forces,
        // integration, and boundary collision all run as compute
        // shaders (Engine/Particles/GPU/ParticleSystemGPU) - see that
        // class's header comment for how it differs from the CPU
        // ParticleSystem it replaces here (still present/compilable,
        // just no longer driven from this main loop). Rendered as round
        // point sprites via Renderer's GPU-particle pipeline
        // (RegisterParticleGPUSource, Shaders/Source/
        // particle_point_gpu.vert) - position is read directly from the
        // compute pipeline's storage buffer, no CPU readback.

        Eden::ParticleSystemGPU particleSystem;
        particleSystem.Init(renderer.GetDevice(), renderer.GetPhysicalDevice(), renderer.GetAllocator(),
                             renderer.GetCommandPool(), renderer.GetGraphicsQueue(),
                             voxelSystem.GetDensityBuffer());
        renderer.RegisterParticleGPUSource(particleSystem.GetPositionBuffer());

        glm::vec3 particleSpawnOrigin{ 0.0f, 6.0f, -2.0f };
        // One box of particles dropped above the static floor/grid so
        // there's something to look at immediately on launch - proves
        // Step() is running and boundary collision against the existing
        // ColliderComponents works, without needing to press Spawn in
        // the UI first. Remove or replace once that's no longer
        // interesting on its own (same reasoning as the physics smoke
        // test cube above it).
        particleSystem.EmitBox(particleSpawnOrigin - glm::vec3(0.5f), particleSpawnOrigin + glm::vec3(0.5f));

        // --- Terrain (Phase 2: grid of independent chunk volumes) -------
        // Phase 1 (previous update) proved noise generation + marching
        // cubes at ~60m scale as ONE VoxelVolumeHandle. This splits that
        // into a real grid of independently-registered handles - the
        // actual precondition Phase 3 (resident-radius load/unload,
        // planning-notes addendum's chunk-LOD/sleep section) needs to
        // exist against. Streaming/LOD itself is still NOT built here -
        // every chunk below is registered and marched unconditionally at
        // startup, same as Phase 1 was. This only changes WHAT the world
        // is made of (many small volumes instead of one big one), not
        // when they're resident.
        //
        // 3x3 grid, 20m per chunk = 60x60m total footprint, matching
        // Phase 1's coverage exactly. voxelSize dropped slightly (0.75 ->
        // 0.625) so 20m divides evenly into whole voxels (20 / (8*0.625)
        // = 4 exactly) - chunkDims (4,1,4) per chunk: 4*8=32 voxels
        // horizontal (20m), 1*8=8 voxels vertical (5m, -4..+1 around
        // baseHeight=0 - slightly less carve depth than Phase 1's 6m,
        // still comfortable). 9 chunks * 33x9x33 = 9,801 samples each =
        // 88,209 samples total - still comfortably inside
        // kSharedDensityBufferCapacityElements alongside the other
        // smoke-test volumes.
        //
        // Real, immediate payoff from splitting now rather than later:
        // RecomputeExactDistances (see Phase 1's own flagged concern)
        // is a whole-VOLUME Dijkstra, and a carve only ever targets ONE
        // volume (FindTargetedVolume picks whichever chunk the raycast
        // actually hit). So a carve here recomputes over ~9,801 samples
        // instead of the old single volume's ~59,049 - about 6x cheaper
        // per carve, for free, just from chunking existing.
        //
        // Noise seams: each chunk samples SeedHeightfieldNoise's fBm at
        // its own absolute worldPos (VoxelVolumeDesc::origin + local
        // offset) - since every chunk uses the IDENTICAL baseHeight/
        // amplitude/frequency/octaves/seed, the underlying noise field
        // is one continuous function of world space regardless of how
        // many volumes sample it, so adjacent chunks' edges line up with
        // no visible seam. This only holds as long as every chunk here
        // shares those five values - don't let a future per-biome
        // variation pass drift them apart without handling the boundary.
        constexpr float kTerrainVoxelSize = 0.625f;
        constexpr glm::ivec3 kTerrainChunkDims{ 4, 1, 4 }; // per world-chunk: 20m x 5m x 20m
        constexpr int kTerrainGridSize = 3;                // 3x3 world-chunks
        constexpr float kTerrainChunkWorldSize = 20.0f;    // kTerrainChunkDims.x * 8 * kTerrainVoxelSize
        constexpr float kTerrainGridWorldMin = -30.0f;     // -(kTerrainGridSize * kTerrainChunkWorldSize) / 2
        constexpr float kTerrainOriginY = -4.0f;

        constexpr float kTerrainBaseHeight = 0.0f;
        constexpr float kTerrainAmplitude = 0.5f;
        constexpr float kTerrainFrequency = 0.05f;
        constexpr int kTerrainOctaves = 3;
        constexpr uint32_t kTerrainSeed = 1;
        constexpr glm::vec4 kTerrainTint{ 0.32f, 0.4f, 0.22f, 1.0f };

        std::vector<TerrainChunkState> terrainChunks;
        terrainChunks.reserve(kTerrainGridSize * kTerrainGridSize);
        for (int cz = 0; cz < kTerrainGridSize; ++cz)
        for (int cx = 0; cx < kTerrainGridSize; ++cx)
        {
            Eden::VoxelVolumeDesc chunkDesc;
            chunkDesc.origin = glm::vec3(kTerrainGridWorldMin + cx * kTerrainChunkWorldSize, kTerrainOriginY,
                                          kTerrainGridWorldMin + cz * kTerrainChunkWorldSize);
            chunkDesc.voxelSize = kTerrainVoxelSize;
            chunkDesc.chunkDims = kTerrainChunkDims;
            Eden::VoxelVolumeHandle chunk = voxelSystem.RegisterVolume(chunkDesc);

            voxelSystem.SeedHeightfieldNoise(chunk, kTerrainBaseHeight, kTerrainAmplitude,
                                              kTerrainFrequency, kTerrainOctaves, kTerrainSeed);
            // MarchDirtyChunks deliberately NOT called here yet - see the
            // neighbor-wiring pass right after this loop for why: this
            // tile's ghost-sample neighbors (ADJACENT tiles, some of
            // which haven't been registered yet at this point in the
            // loop) need to be wired up BEFORE the first march, or that
            // march bakes in the old clamped-at-edge gradients along
            // every seam and nothing after startup ever corrects it
            // unless that specific chunk happens to get carved later.
            voxelSystem.SetTransform(chunk, glm::mat4(1.0f), kTerrainTint);

            // Real collision, same as every other voxel volume - Voxel is
            // the project-wide default (queries the live density field),
            // so carving/melting this later automatically stays correct
            // without any collider bookkeeping. particleSystem's existing
            // GPU boundary-collision path already handles
            // ColliderShape::Voxel (see ParticleGPUTypes.h's voxelParams),
            // so SPH particles get this for free too.
            Eden::Entity chunkEntity = RegisterVoxelPhysicsEntity(registry, voxelSystem, chunk);

            // Center for TerrainChunkState's residency check - reusing
            // GetVolumeBounds rather than recomputing from chunkDesc so
            // this always matches RegisterVoxelPhysicsEntity's own idea
            // of "center" exactly.
            Eden::VoxelSystemGPU::VolumeBounds chunkBounds = voxelSystem.GetVolumeBounds(chunk);
            glm::vec3 chunkCenter = (chunkBounds.worldMin + chunkBounds.worldMax) * 0.5f;

            terrainChunks.push_back(TerrainChunkState{ chunk, chunkEntity, chunkCenter, true });
        }

        // Ghost-sample neighbor wiring, then the FIRST march of every
        // tile - both deferred until all 9 tiles exist. terrainChunks is
        // in the same cz-major/cx-minor order as the registration loop
        // above (index = cz*kTerrainGridSize+cx), so a tile's grid
        // neighbors are just +-1 steps in that same indexing, no need to
        // search. Increasing cx/cz is increasing world X/Z (see
        // chunkDesc.origin above), so cx-1 is the -X neighbor, cx+1 the
        // +X neighbor, and the same pattern for cz/Z. A grid-edge tile's
        // missing side(s) correctly get InvalidVoxelVolumeHandle - the
        // terrain's own outer boundary has no neighbor to ghost-sample
        // from, same as it always did.
        for (int cz = 0; cz < kTerrainGridSize; ++cz)
        for (int cx = 0; cx < kTerrainGridSize; ++cx)
        {
            int index = cz * kTerrainGridSize + cx;
            Eden::VoxelVolumeHandle negX = (cx > 0) ? terrainChunks[index - 1].handle : Eden::InvalidVoxelVolumeHandle;
            Eden::VoxelVolumeHandle posX = (cx < kTerrainGridSize - 1) ? terrainChunks[index + 1].handle : Eden::InvalidVoxelVolumeHandle;
            Eden::VoxelVolumeHandle negZ = (cz > 0) ? terrainChunks[index - kTerrainGridSize].handle : Eden::InvalidVoxelVolumeHandle;
            Eden::VoxelVolumeHandle posZ = (cz < kTerrainGridSize - 1) ? terrainChunks[index + kTerrainGridSize].handle : Eden::InvalidVoxelVolumeHandle;
            voxelSystem.SetVolumeNeighbors(terrainChunks[index].handle, negX, posX, negZ, posZ);
        }
        for (const TerrainChunkState& chunk : terrainChunks)
        {
            voxelSystem.MarchDirtyChunks(chunk.handle);
        }

        // --- Voxel/marching-cubes deformable volume smoke test --------
        // First real user of Engine/Voxel/VoxelSystemGPU - registers one
        // small volume (2x2x2 chunks = 16^3 voxels, see
        // VoxelVolumeDesc's comment on why that's the default), seeds it
        // as a sphere via the same SDF convention Physics/SDF.h uses,
        // marches it once, and draws it through Renderer's new indirect
        // draw path (VoxelDrawSource). Static for now - see
        // VoxelVolumeComponent's comment on why a moving deformable
        // volume isn't wired up yet; corrosion/carving (removing density
        // at runtime and re-marching just the affected chunks) is the
        // next milestone on top of this, not implemented here.
        // voxelSystem itself is declared earlier now (see comment by
        // renderer.Init(window) above) - only its first real volume
        // registration happens here, unchanged from before.

        Eden::VoxelVolumeDesc voxelDesc;
        voxelDesc.origin = glm::vec3(3.0f, 0.0f, 0.0f); // off to the side of the particle/physics smoke tests
        voxelDesc.voxelSize = 0.15f;
        voxelDesc.chunkDims = glm::ivec3(2, 2, 2);
        Eden::VoxelVolumeHandle voxelVolume = voxelSystem.RegisterVolume(voxelDesc);

        // Sphere centered in the middle of the volume, radius sized to
        // comfortably clear the volume's own boundary (SampleDims() is
        // 17 voxels/axis at chunkDims=2, kVoxelChunkSize=8 - a radius of
        // 1.0 world units at voxelSize=0.15 is well inside that).
        //
        // BUGFIX: "well inside that" was true for the sphere's OWN
        // surface, but not for Carve()'s reach - Carve() pads its own
        // AABB by 1 voxel beyond the carve radius (kCarveRadius=0.3, so
        // ~0.45 world units beyond the sphere surface at the carve
        // point), and a click near the sphere's pole carves at a point
        // very close to the volume's own boundary. The old 1.0 radius
        // against a 2.4-unit box left only 0.2 units of margin at the
        // poles - well under what a carve there needs - so Carve()'s
        // padded sample/chunk range got clamped at the volume edge,
        // and RecomputeExactDistances' Dijkstra walk (which treats
        // out-of-bounds neighbors as simply absent, not "keep going")
        // never got a chance to correct the resulting boundary samples
        // against their true nearest-solid distance. The visible result
        // was a ragged, unclosed tear right at the carve site whenever
        // it landed near a pole. Shrunk to 0.8 so a carve anywhere on
        // the sphere's surface, including the poles, stays comfortably
        // clear of the volume boundary even after padding.
        glm::vec3 voxelExtent = glm::vec3(voxelDesc.VoxelDims()) * voxelDesc.voxelSize;
        constexpr float kVoxelSeedRadius = 0.8f;
        glm::vec3 voxelLocalCenter = voxelExtent * 0.5f;
        voxelSystem.SeedSphere(voxelVolume, voxelLocalCenter, kVoxelSeedRadius);
        voxelSystem.MarchDirtyChunks(voxelVolume);
        voxelSystem.SetTransform(voxelVolume, glm::mat4(1.0f));

        // Real collision for the test sphere - see
        // RegisterVoxelPhysicsEntity's comment above.
        RegisterVoxelPhysicsEntity(registry, voxelSystem, voxelVolume);

        // --- Raymarch smoke test ----------------------------------------
        // First real user of the raymarch path (Raymarch/RaymarchSystem.h)
        // - a second, separate voxel volume from the marching-cubes test
        // sphere above, deliberately NOT reusing it, so this can be
        // proven working in isolation without risking the existing
        // smoke test. Per the agreed build order: get one raymarched
        // sphere on screen and correct before converting any other
        // spawn site (floor/walls/physics cubes) to this path - see
        // RaymarchVolumeComponent's own comment for what stays out of
        // scope until then.
        //
        // No RegisterVoxelPhysicsEntity call here - this volume is
        // draw-only for now (see RaymarchVolumeComponent's comment on
        // why draw path and collision are separate entities/concerns);
        // it has no collider and nothing will physically interact with
        // it yet.
        Eden::VoxelVolumeDesc raymarchDesc;
        raymarchDesc.origin = glm::vec3(-3.0f, 1.5f, 0.0f); // off to the opposite side from the marching-cubes sphere, roughly eye level
        raymarchDesc.voxelSize = 0.15f;
        raymarchDesc.chunkDims = glm::ivec3(2, 2, 2);
        Eden::VoxelVolumeHandle raymarchVolume = voxelSystem.RegisterVolume(raymarchDesc);

        glm::vec3 raymarchExtent = glm::vec3(raymarchDesc.VoxelDims()) * raymarchDesc.voxelSize;
        glm::vec3 raymarchLocalCenter = raymarchExtent * 0.5f;
        voxelSystem.SeedSphere(raymarchVolume, raymarchLocalCenter, kVoxelSeedRadius);
        // Deliberately NOT calling MarchDirtyChunks - this volume is
        // never meant to produce marching-cubes geometry; raymarch.frag
        // samples its density field directly instead (see that shader's
        // top-of-file comment). Calling MarchDirtyChunks would just be
        // wasted GPU work building a mesh nothing draws.

        Eden::Entity raymarchEntity = registry.CreateEntity();
        Eden::TransformComponent raymarchTransform;
        // position = this volume's WORLD-SPACE MIN CORNER, matching
        // VoxelVolumeDesc::origin exactly - see RaymarchSystem.h's
        // namespace comment for why this is a DIFFERENT convention from
        // RegisterVoxelPhysicsEntity's world-CENTER convention on the
        // physics-collider entity (which this raymarch entity doesn't
        // have a counterpart of yet).
        raymarchTransform.position = raymarchDesc.origin;
        registry.AddComponent(raymarchEntity, raymarchTransform);
        registry.AddComponent(raymarchEntity, Eden::VoxelVolumeComponent{ raymarchVolume });
        registry.AddComponent(raymarchEntity, Eden::RaymarchVolumeComponent{ glm::vec3(0.9f, 0.3f, 0.2f) }); // warm red-orange, visually distinct from the white marching-cubes sphere

        // --- Melt-test catch container ---------------------------------
        // A simple open-top box (floor + 4 walls) sitting under the
        // voxel sphere above, so pressing M has somewhere for the
        // particles to land and be visible instead of free-falling out
        // of view. Static ColliderComponent entities like the physics
        // smoke-test cubes above - ParticleSystemGPU::Step() picks up
        // ANY TransformComponent+ColliderComponent entity via its own
        // registry.View() call (see Step()'s collider-gathering code),
        // so these don't need anything melt/particle-specific to be
        // seen by the SPH sim's boundary collision.
        {
            glm::vec3 voxelWorldCenter = voxelDesc.origin + voxelLocalCenter;

            // Sized/centered from the voxel sphere's own known footprint
            // (center +/- kVoxelSeedRadius) rather than hardcoded
            // numbers, so this still lines up if the volume above moves
            // or the seed radius changes.
            float floorHalfX = kVoxelSeedRadius + 0.6f;
            float floorHalfZ = kVoxelSeedRadius + 0.6f;
            float floorHalfY = 0.15f;
            float floorTopY = voxelWorldCenter.y - kVoxelSeedRadius - 0.2f; // a bit below the sphere's bottom
            glm::vec3 floorCenter{ voxelWorldCenter.x, floorTopY - floorHalfY, voxelWorldCenter.z };

            // Anything that falls below the floor's actual bottom face
            // has escaped the container entirely (fallen off an edge,
            // or otherwise slipped past collision) and is never coming
            // back - see SimParamsGPU::voidKillY's own comment
            // (ParticleGPUTypes.h) and particle_integrate.comp's use of
            // it. Margin below the floor's bottom (not just its top)
            // so this can't misfire on a particle legitimately resting
            // ON the floor. particleSystem already exists (Init'd
            // earlier in this function) - this only has to run once,
            // the floor doesn't move.
            particleSystem.voidKillY = (floorCenter.y - floorHalfY) - 5.0f;

            constexpr float kWallThickness = 0.1f;
            constexpr float kWallHalfHeight = 0.6f;
            glm::vec3 containerColor{ 0.35f, 0.3f, 0.28f };

            // Voxel collision (not AnalyticBox) - this container's whole
            // purpose is catching SPH particles from the melt test
            // (ParticleSystemGPU::Step), and that path now has full
            // Voxel support (see ParticleGPUTypes.h's ColliderGPU::
            // voxelParams/voxelSampleDims and particle_integrate.comp's
            // VoxelShapeDistance) - Voxel is the project-wide default
            // for anything raymarched/cube-marched going forward (see
            // RaymarchBoxCollision::Voxel's own comment), since unlike
            // AnalyticBox it queries this volume's LIVE density field,
            // so collision automatically tracks whatever this box's
            // actual shape becomes if it's ever carved/melted/reformed
            // later - AnalyticBox stays available for static geometry
            // that's known to never deform, where the extra narrow-
            // phase precision of a real box SAT test is worth more than
            // that live-tracking property.
            auto addRaymarchStaticBox = [&](const glm::vec3& center, const glm::vec3& halfExtents)
            {
                SpawnRaymarchBox(registry, voxelSystem, center, halfExtents, containerColor,
                                  RaymarchBoxCollision::Voxel);
            };

            // Floor
            addRaymarchStaticBox(floorCenter, glm::vec3(floorHalfX, floorHalfY, floorHalfZ));

            // Walls - each extended by kWallThickness along its own run
            // axis so the four corners overlap and close, rather than
            // leaving a particle-sized gap.
            float wallCenterY = floorTopY + kWallHalfHeight;
            addRaymarchStaticBox({ voxelWorldCenter.x + floorHalfX, wallCenterY, voxelWorldCenter.z },
                         { kWallThickness, kWallHalfHeight, floorHalfZ + kWallThickness }); // +X wall
            addRaymarchStaticBox({ voxelWorldCenter.x - floorHalfX, wallCenterY, voxelWorldCenter.z },
                         { kWallThickness, kWallHalfHeight, floorHalfZ + kWallThickness }); // -X wall
            addRaymarchStaticBox({ voxelWorldCenter.x, wallCenterY, voxelWorldCenter.z + floorHalfZ },
                         { floorHalfX + kWallThickness, kWallHalfHeight, kWallThickness }); // +Z wall
            addRaymarchStaticBox({ voxelWorldCenter.x, wallCenterY, voxelWorldCenter.z - floorHalfZ },
                         { floorHalfX + kWallThickness, kWallHalfHeight, kWallThickness }); // -Z wall
        }

        double lastFrameTime = glfwGetTime();

        // Grows as ProcessReformInput (H key) reforms new volumes -
        // see that function's comment. Kept alongside voxelVolume/
        // voxelSystem rather than folded into it since these are
        // dynamically created, not the one static smoke-test volume.
        std::vector<Eden::VoxelVolumeHandle> reformedVolumes;

        while (!glfwWindowShouldClose(window))
        {
            double currentFrameTime = glfwGetTime();
            float deltaTime = static_cast<float>(currentFrameTime - lastFrameTime);
            lastFrameTime = currentFrameTime;

            glfwPollEvents();
            ProcessKeyboardInput(window, renderer, deltaTime);
            // reformedVolumes is plain main-loop state, not ECS - it has
            // no way to find out a handle it's holding got destroyed by
            // the editor's entity-list "Destroy" button (EdenUI.cpp),
            // which now really frees the volume (see
            // VoxelSystemGPU::UnregisterVolume). Left unpruned, a stale
            // handle here would reach RaycastSurfaceSegments/
            // GetVertexBuffer below and throw (GetVolume's destroyed-
            // handle guard) instead of the silent use-after-free this
            // used to be. Pruning here, once a frame, is the fix -
            // cheap (IsValid does no throwing/bounds-checking work
            // beyond a size compare) and keeps every consumer below
            // simple.
            reformedVolumes.erase(
                std::remove_if(reformedVolumes.begin(), reformedVolumes.end(),
                                [&voxelSystem](Eden::VoxelVolumeHandle h) { return !voxelSystem.IsValid(h); }),
                reformedVolumes.end());

            // Terrain Phase 3 residency gate - see UpdateTerrainChunkResidency's
            // own comment for what this does and doesn't cover. Camera
            // position stands in for "player position" the same way
            // FindTargetedVolume already treats it - there's no separate
            // player-controller entity yet.
            UpdateTerrainChunkResidency(registry, voxelSystem, terrainChunks, renderer.GetCamera().Position);

            // Every hardened volume that currently exists - the
            // original test sphere plus anything ReformSystem has
            // created since - so carving can target ANY of them (see
            // FindTargetedVolume), not just the original. Terrain chunks
            // ARE carvable (that's the whole point), so they're included
            // here. Rebuilt each frame since reformedVolumes can grow
            // between frames (H key).
            std::vector<Eden::VoxelVolumeHandle> allVoxelVolumes;
            allVoxelVolumes.reserve(1 + terrainChunks.size() + reformedVolumes.size());
            // UNFILTERED list of every terrain tile, resident or not -
            // needed by CarveAcrossTerrainTiles' neighbor scan (not the
            // full TerrainChunkState, which isn't visible at that
            // function's point in the file). Deliberately NOT filtered
            // by residency the way allVoxelVolumes below is: sleeping
            // only removes a tile's ColliderComponent (physics), it does
            // NOT touch that tile's density field or mesh at all -
            // Carve()/MarchDirtyChunks work identically regardless of
            // residency. Filtering this list was a real, confirmed bug:
            // a corner carve reaching into a tile that happened to still
            // be asleep (hadn't woken up yet from the player's own
            // recent movement - see [TerrainLOD] sleeping/waking in the
            // console log) silently skipped that tile entirely, not
            // because of margin math but because it wasn't even a
            // candidate. Depending on approach direction this could make
            // some corners work and others not in a way that looked
            // coordinate-dependent without actually being about
            // coordinates - it was about which tiles happened to be
            // awake at that exact moment.
            std::vector<Eden::VoxelVolumeHandle> allTerrainHandles;
            allTerrainHandles.reserve(terrainChunks.size());
            for (const TerrainChunkState& chunk : terrainChunks)
            {
                if (voxelSystem.IsValid(chunk.handle))
                {
                    allTerrainHandles.push_back(chunk.handle);
                }
                if (chunk.resident && voxelSystem.IsValid(chunk.handle)) // only resident chunks are carve TARGETS (aiming/FindTargetedVolume) - sleeping ones aren't being looked at anyway
                {
                    allVoxelVolumes.push_back(chunk.handle);
                }
            }
            if (voxelSystem.IsValid(voxelVolume)) // same reasoning - the original test sphere is also editor-destroyable
            {
                allVoxelVolumes.push_back(voxelVolume);
            }
            allVoxelVolumes.insert(allVoxelVolumes.end(), reformedVolumes.begin(), reformedVolumes.end());

            // Separate, DELIBERATELY narrower list for melt - terrain is
            // NOT included. Melt's whole-volume->particle conversion
            // (see MeltSystem) spawns particles for the volume's entire
            // solid mass at once; that's fine for the small test sphere
            // it was built against, but a 20x5x20m terrain chunk holds
            // orders of magnitude more solid volume than anything it's
            // been tried on before - melting one dumps a huge number of
            // overlapping SPH particles into the same space in a single
            // frame, and the resulting pressure spike from that much
            // initial overlap is what read as "a huge explosion." This
            // isn't a case of tuning melt to be gentler; it's terrain
            // being fundamentally the wrong scale for a mechanic that
            // was designed around small, discrete objects. Excluding it
            // from the candidate list here means FindTargetedVolume
            // simply never considers a terrain chunk when melting - a
            // real scope decision, not a workaround pending a real fix.
            std::vector<Eden::VoxelVolumeHandle> meltableVolumes;
            meltableVolumes.reserve(1 + reformedVolumes.size());
            if (voxelSystem.IsValid(voxelVolume))
            {
                meltableVolumes.push_back(voxelVolume);
            }
            meltableVolumes.insert(meltableVolumes.end(), reformedVolumes.begin(), reformedVolumes.end());

            ProcessVoxelCarveInput(window, renderer, registry, voxelSystem, allVoxelVolumes, allTerrainHandles);
            ProcessMeltInput(window, renderer, registry, voxelSystem, meltableVolumes, particleSystem);
            ProcessReformInput(window, registry, particleSystem, voxelSystem, reformedVolumes);

            // TEMP DEBUG - remove after diagnosing.
            if (g_CursorLocked)
            {
                bool probeKeyDown = glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS;
                if (probeKeyDown && !g_ProbeKeyWasDown)
                {
                    ProbeVoxelGrid(voxelSystem, voxelVolume);
                }
                g_ProbeKeyWasDown = probeKeyDown;
            }

            // Fixed-timestep accumulator: physics steps happen zero or
            // more times per rendered frame, each advancing the
            // simulation by exactly FIXED_TIMESTEP seconds regardless of
            // how long the frame actually took. See PhysicsSystem.h for
            // why this can't just use the frame's variable deltaTime the
            // way SpinSystem/LifetimeSystem do.
            physicsAccumulator += deltaTime;
            int physicsStepsThisFrame = 0;
            while (physicsAccumulator >= FIXED_TIMESTEP && physicsStepsThisFrame < MAX_PHYSICS_STEPS_PER_FRAME)
            {
                physicsSystem.Step(registry, FIXED_TIMESTEP);
                collisionSystem.Step(registry, FIXED_TIMESTEP, &voxelSystem);
                // Runs after rigid bodies have already moved/resolved
                // for this tick, so particles collide against where
                // solids ended up this step rather than where they
                // started it (see ParticleSystem::Step's comment).
                particleSystem.Step(registry, FIXED_TIMESTEP, &voxelSystem);
                physicsAccumulator -= FIXED_TIMESTEP;
                ++physicsStepsThisFrame;
            }

            registry.UpdateSystems(deltaTime);
            auto drawList = Eden::RenderSystem::BuildDrawList(registry, renderer);

            // Built once, up front, so both DrawFrame arguments below
            // come from the SAME registry iteration and stay
            // index-matched by construction - see RaymarchFrameData's
            // comment in RaymarchSystem.h for why this replaced an
            // earlier two-lambda version that iterated registry.View()
            // twice and trusted both passes to agree on order.
            auto raymarchFrame = Eden::RaymarchSystem::BuildObjectList(registry, voxelSystem);

            renderer.DrawFrame(drawList, [&]()
            {
                Eden::UI::DrawMeshPanel(registry, renderer, voxelSystem);
                Eden::UI::DrawPhysicsSettingsPanel(physicsSystem, collisionSystem);
                Eden::UI::DrawRendererSettingsPanel(renderer);

                // GPU particle system has no ImGui panel of its own yet
                // (DrawParticleSettingsPanel in EdenUI targets the CPU
                // ParticleSystem's fields) - a minimal inline readout
                // instead of a full panel, since building out the GPU
                // path's own tuning UI is follow-up work, not required
                // to prove the compute pipeline itself.
                ImGui::Begin("Particles (GPU)");
                ImGui::Text("Particle count: %u / %u", particleSystem.ParticleCount(), particleSystem.Capacity());
                ImGui::SliderFloat("Point size", &renderer.ParticlePointSize, 1.0f, 32.0f);
                ImGui::ColorEdit3("Color", &renderer.ParticleGPUColor.x);

                // Live SPH tunables - added alongside the cohesion force
                // (see SimParamsGPU::cohesion's comment) so tuning
                // "bouncy droplets vs cohesive liquid" and "explodes/
                // jitters on impact" doesn't require a rebuild per
                // attempt. BuildSimParams() reads these fields fresh
                // every Step() call, so changes here take effect next
                // physics substep, live.
                if (ImGui::CollapsingHeader("SPH tuning", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    ImGui::SliderFloat("Stiffness", &particleSystem.stiffness, 500.0f, 8000.0f);
                    ImGui::SliderFloat("Viscosity", &particleSystem.viscosityCoefficient, 0.0f, 5.0f);
                    ImGui::SliderFloat("Cohesion", &particleSystem.cohesion, 0.0f, 8000.0f);
                    ImGui::SliderFloat("Boundary friction", &particleSystem.boundaryFriction, 0.0f, 1.0f);
                    ImGui::SliderFloat("Max acceleration", &particleSystem.maxAcceleration, 50.0f, 2000.0f);
                    ImGui::SliderInt("Substeps", &particleSystem.substeps, 1, 8);
                }

                // Screen-space fluid surface (see Renderer::FluidSurfaceEnabled's
                // comment) - the actual "looks like connected liquid, not
                // a cloud of dots" fix. Toggle-able rather than always-on
                // so this can be A/B'd against the old raw-point path, or
                // turned off as a fallback if the new render pass turns
                // out to have a problem this session's build/test pass
                // hasn't surfaced yet.
                if (ImGui::CollapsingHeader("Fluid surface render", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    ImGui::Checkbox("Enabled", &renderer.FluidSurfaceEnabled);
                    ImGui::SliderFloat("Visual radius", &renderer.FluidParticleRadius, 0.02f, 0.4f);
                    ImGui::ColorEdit3("Tint", &renderer.FluidTintColor.x);
                }

                if (ImGui::Button("Emit box"))
                {
                    particleSystem.EmitBox(particleSpawnOrigin - glm::vec3(0.5f), particleSpawnOrigin + glm::vec3(0.5f));
                }
                ImGui::SameLine();
                if (ImGui::Button("Clear"))
                {
                    particleSystem.Clear();
                }
                ImGui::End();
            },
            [&](VkCommandBuffer cmd)
            {
                particleSystem.RecordPendingSteps(cmd);
            },
            particleSystem.ParticleCount(),
            [&]()
            {
                // Original smoke-test volume + every volume Reform() has
                // created so far - rebuilt each frame since
                // reformedVolumes can grow between frames (H key).
                // reformedVolumes was already pruned of editor-destroyed
                // handles earlier this frame; voxelVolume itself is
                // guarded here the same way (see that earlier prune's
                // comment for why an unguarded GetVertexBuffer(handle)
                // on a destroyed volume now throws instead of silently
                // reading freed memory).
                std::vector<Eden::VoxelDrawSource> sources;
                sources.reserve(1 + terrainChunks.size() + reformedVolumes.size());
                for (const TerrainChunkState& chunk : terrainChunks) // sleeping chunks skip the draw call too - real per-frame draw-count reduction, see UpdateTerrainChunkResidency
                {
                    if (chunk.resident && voxelSystem.IsValid(chunk.handle))
                    {
                        sources.push_back({ voxelSystem.GetVertexBuffer(chunk.handle), voxelSystem.GetIndirectBuffer(chunk.handle),
                                             voxelSystem.GetInstanceBuffer(chunk.handle), voxelSystem.GetChunkCount(chunk.handle) });
                    }
                }
                if (voxelSystem.IsValid(voxelVolume))
                {
                    sources.push_back({ voxelSystem.GetVertexBuffer(voxelVolume), voxelSystem.GetIndirectBuffer(voxelVolume),
                                         voxelSystem.GetInstanceBuffer(voxelVolume), voxelSystem.GetChunkCount(voxelVolume) });
                }
                for (Eden::VoxelVolumeHandle handle : reformedVolumes)
                {
                    sources.push_back({ voxelSystem.GetVertexBuffer(handle), voxelSystem.GetIndirectBuffer(handle),
                                         voxelSystem.GetInstanceBuffer(handle), voxelSystem.GetChunkCount(handle) });
                }
                return sources;
            }(),
            raymarchFrame.objects,
            raymarchFrame.densityBuffer);
        }

        // WaitIdle() BEFORE any teardown touches a single buffer/
        // pipeline - not optional, and the direct cause of a real
        // shutdown-time crash (VK_ERROR_OUT_OF_DEVICE_MEMORY / lost
        // device / a VMA "allocations not freed" assertion on exit).
        // Renderer::Shutdown() below also calls WaitIdle() internally,
        // but only as its own first step - by then voxelSystem.Shutdown()
        // and particleSystem.Shutdown() had ALREADY destroyed buffers
        // and pipelines that the last submitted frame's command buffer
        // could still be actively executing against on the GPU, since
        // nothing before this point ever waited for it to finish. Same
        // underlying hazard class as UnregisterVolume's own
        // vkDeviceWaitIdle fix (see that function's comment) - GPU work
        // referencing a resource must be known-complete before that
        // resource is destroyed, and "the CPU-side Shutdown() call
        // happened" does not mean that's true.
        renderer.WaitIdle();

        voxelSystem.Shutdown();
        particleSystem.Shutdown();
        renderer.Shutdown();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Eden: fatal error: " << e.what() << "\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return EXIT_FAILURE;
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_SUCCESS;
}
