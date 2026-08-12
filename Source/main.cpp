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

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <cstdlib>

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

        // A grid of static cubes - same mesh, different TransformComponent
        // per entity. Crank this back up to 10x10x10 if you want; startup
        // will still be instant since geometry is only uploaded once.
        for (int i = 0; i < 5; ++i)
        {
            for (int j = 0; j < 5; ++j)
            {
                for (int k = 0; k < 5; ++k)
                {
                    Eden::Entity e = registry.CreateEntity();

                    Eden::TransformComponent transform;
                    transform.position = { -4.0f + j * 2.0f, -4.0f + k * 2.0f, i * 2.0f };
                    registry.AddComponent(e, transform);

                    registry.AddComponent(e, Eden::MeshComponent{ cubeMesh });
                    registry.AddComponent(e, Eden::RotationSpeedComponent{ 0.0f });
                    // ColorComponent::color is a glm::vec3, not vec4 - there's
                    // no alpha channel here, the override either applies (this
                    // component present) or doesn't (it's absent). See
                    // ColorComponent.h.
                    registry.AddComponent(e, Eden::ColorComponent{ glm::vec3(1.0f, 0.5f, 0.2f) });
                    registry.AddComponent(e, Eden::LifetimeComponent{ 3.0f }); // Optional: add a lifetime component to destroy the entity after 10 seconds
                }
            }
        }

        // One extra cube with a RotationSpeedComponent - proves SpinSystem
        // is actually doing per-frame work, not just that the grid above
        // holds still. Composition in action: same MeshComponent as every
        // other cube, one extra component, different behavior.
        {
            Eden::Entity e = registry.CreateEntity();

            Eden::TransformComponent transform;
            transform.position = { 0.0f, 3.0f, -2.0f };
            registry.AddComponent(e, transform);

            registry.AddComponent(e, Eden::MeshComponent{ cubeMesh });
            registry.AddComponent(e, Eden::RotationSpeedComponent{ 60.0f });
        }

        // Physics smoke test: a Dynamic cube with gravity enabled, high
        // up with no collider yet (colliders are step 2 of the physics
        // work) - it should just fall straight through the grid below
        // and keep going. This is here to prove PhysicsSystem::Step is
        // actually running at a fixed rate, not that anything stops it
        // yet. Remove once collision response exists and this stops
        // being an interesting test on its own.
        {
            Eden::Entity e = registry.CreateEntity();

            Eden::TransformComponent transform;
            transform.position = { 0.0f, 12.0f, -2.0f };
            registry.AddComponent(e, transform);

            registry.AddComponent(e, Eden::MeshComponent{ cubeMesh });
            registry.AddComponent(e, Eden::ColorComponent{ glm::vec3(0.2f, 0.6f, 1.0f) });

            Eden::RigidBodyComponent body;
            body.type = Eden::BodyType::Dynamic;
            body.inverseMass = 1.0f;
            registry.AddComponent(e, body);

            // Cube mesh is 1.0 units, so half extents of 0.5 match its
            // actual size - see Renderer::CreateCubeMesh(1.0f) above.
            Eden::ColliderComponent collider;
            collider.shape = Eden::ColliderShape::Box;
            collider.halfExtents = glm::vec3(0.5f);
            registry.AddComponent(e, collider);
        }

        // A Static body sitting in the same spot as part of the grid,
        // just to confirm PhysicsSystem::Step correctly leaves Static
        // entities alone (zeroed velocity, transform untouched) even
        // when they carry a RigidBodyComponent. No visual difference
        // expected - this is a "nothing breaks" test, not a "something
        // moves" test.
        {
            Eden::Entity e = registry.CreateEntity();

            Eden::TransformComponent transform;
            transform.position = { 0.0f, -4.0f, -2.0f };
            registry.AddComponent(e, transform);

            registry.AddComponent(e, Eden::MeshComponent{ cubeMesh });
            registry.AddComponent(e, Eden::ColorComponent{ glm::vec3(0.6f, 0.6f, 0.6f) });

            Eden::RigidBodyComponent body;
            body.type = Eden::BodyType::Static;
            body.inverseMass = 0.0f;
            registry.AddComponent(e, body);

            Eden::ColliderComponent collider;
            collider.shape = Eden::ColliderShape::Box;
            collider.halfExtents = glm::vec3(0.5f);
            registry.AddComponent(e, collider);
        }

        Eden::PhysicsSystem physicsSystem;
        Eden::CollisionSystem collisionSystem;
        float physicsAccumulator = 0.0f;

        // --- Particle system (SPH fluid) smoke test -----------------
        // Rendered as round point sprites via Renderer's dedicated
        // point-topology pipeline (see Renderer::GetParticlePointMesh,
        // Shaders/Source/particle_point.vert/.frag) - no per-particle
        // mesh needed, GetParticlePointMesh() is a single shared handle
        // set up once in Renderer::Init().

        Eden::ParticleSystem particleSystem;
        glm::vec3 particleSpawnOrigin{ 0.0f, 6.0f, -2.0f };
        // One box of particles dropped above the static floor/grid so
        // there's something to look at immediately on launch - proves
        // Step() is running and boundary collision against the existing
        // ColliderComponents works, without needing to press Spawn in
        // the UI first. Remove or replace once that's no longer
        // interesting on its own (same reasoning as the physics smoke
        // test cube above it).
        particleSystem.EmitBox(particleSpawnOrigin - glm::vec3(0.5f), particleSpawnOrigin + glm::vec3(0.5f));

        double lastFrameTime = glfwGetTime();

        while (!glfwWindowShouldClose(window))
        {
            double currentFrameTime = glfwGetTime();
            float deltaTime = static_cast<float>(currentFrameTime - lastFrameTime);
            lastFrameTime = currentFrameTime;

            glfwPollEvents();
            ProcessKeyboardInput(window, renderer, deltaTime);

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
                collisionSystem.Step(registry, FIXED_TIMESTEP);
                // Runs after rigid bodies have already moved/resolved
                // for this tick, so particles collide against where
                // solids ended up this step rather than where they
                // started it (see ParticleSystem::Step's comment).
                particleSystem.Step(registry, FIXED_TIMESTEP);
                physicsAccumulator -= FIXED_TIMESTEP;
                ++physicsStepsThisFrame;
            }

            registry.UpdateSystems(deltaTime);
            auto drawList = Eden::RenderSystem::BuildDrawList(registry, renderer);

            // Particles are never ECS entities (see ParticleData.h), so
            // they don't come back from RenderSystem::BuildDrawList -
            // appended here instead. Both lists share the same
            // MAX_INSTANCES_PER_FRAME ceiling; ParticleSystem::
            // BuildDrawList caps itself but doesn't know how much of
            // the budget the ECS draw list already used, so a scene
            // with both a huge entity count AND huge particle count
            // could still exceed the cap in total - not a concern at
            // today's scene sizes, worth revisiting if that changes.
            auto particleDrawList = particleSystem.BuildDrawList(renderer.GetParticlePointMesh());
            drawList.insert(drawList.end(), particleDrawList.begin(), particleDrawList.end());

            renderer.DrawFrame(drawList, [&]()
            {
                Eden::UI::DrawMeshPanel(registry, renderer);
                Eden::UI::DrawPhysicsSettingsPanel(physicsSystem, collisionSystem);
                Eden::UI::DrawParticleSettingsPanel(particleSystem, particleSpawnOrigin, renderer.ParticlePointSize);
            });
        }

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
