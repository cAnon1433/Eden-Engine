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

        for (Eden::VoxelVolumeHandle handle : candidates)
        {
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

    void ProcessVoxelCarveInput(GLFWwindow* window, Eden::Renderer& renderer, Eden::Registry& registry,
                                 Eden::VoxelSystemGPU& voxelSystem, const std::vector<Eden::VoxelVolumeHandle>& candidates)
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
                            voxelSystem.Carve(voxelVolume, entry + segmentDir * d, kCarveRadius);
                        }
                    }
                }
                else
                {
                    // Single bite at the FIRST (nearest) segment's entry
                    // point only - the easy-to-test default.
                    voxelSystem.Carve(voxelVolume, segments.front().first, kCarveRadius);
                }

                // One march for the whole click's worth of carving, not
                // one per carve ball or per segment - see Carve()'s own
                // comment on why it's split from MarchDirtyChunks.
                voxelSystem.MarchDirtyChunks(voxelVolume);

                // The actual fix for stale-margin false collisions (see
                // RecomputeExactDistances' own comment for the full
                // mechanism) - run once per completed carve gesture,
                // same batching reasoning as MarchDirtyChunks above, not
                // once per carve ball. Correctness no longer depends on
                // carve radius or coverage after this runs.
                voxelSystem.RecomputeExactDistances(voxelVolume);

                // TEMP DEBUG - testing whether marching cubes' classic
                // face-ambiguity case is actually present at carve
                // sites before committing to a table rewrite over it.
                // Remove once the crack bug is understood.
                voxelSystem.DebugScanAmbiguousCells(voxelVolume);

                // See WakeSleepingBodiesNearVolume's comment - a body
                // asleep on this volume needs to be told its shape just
                // changed, or it never falls into what was just carved.
                WakeSleepingBodiesNearVolume(registry, voxelSystem, voxelVolume);
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
            // - this fluid has no cohesion/surface-tension force, so it
            // doesn't clump into one blob when it settles, it spreads
            // into a puddle with gaps between locally-packed groups.
            // clusterRadius only controls which particles get batched
            // together for reform - it does NOT control whether the
            // resulting geometry visually fuses (see reformBlobRadius
            // below for that).
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

            // Every hardened volume that currently exists - the
            // original test sphere plus anything ReformSystem has
            // created since - so carving and melting can target ANY of
            // them (see FindTargetedVolume), not just the original.
            // Rebuilt each frame since reformedVolumes can grow between
            // frames (H key).
            std::vector<Eden::VoxelVolumeHandle> allVoxelVolumes;
            allVoxelVolumes.reserve(1 + reformedVolumes.size());
            if (voxelSystem.IsValid(voxelVolume)) // same reasoning - the original test sphere is also editor-destroyable
            {
                allVoxelVolumes.push_back(voxelVolume);
            }
            allVoxelVolumes.insert(allVoxelVolumes.end(), reformedVolumes.begin(), reformedVolumes.end());

            ProcessVoxelCarveInput(window, renderer, registry, voxelSystem, allVoxelVolumes);
            ProcessMeltInput(window, renderer, registry, voxelSystem, allVoxelVolumes, particleSystem);
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
                sources.reserve(1 + reformedVolumes.size());
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
