#include "EdenUI.h"

#include "../ECS/Components/TransformComponent.h"
#include "../ECS/Components/MeshComponent.h"
#include "../ECS/Components/NameComponent.h"
#include "../ECS/Components/ColorComponent.h"
#include "../ECS/Components/VisibilityComponent.h"
#include "../ECS/Components/LifetimeComponent.h"
#include "../ECS/Components/RotationSpeedComponent.h"
#include "../Physics/RigidBodyComponent.h"
#include "../Physics/ColliderComponent.h"
#include "../Voxel/VoxelSystemGPU.h" // pulls in VoxelTypes.h + VoxelField.h transitively
#include "../Renderer/Raymarch/RaymarchSystem.h"

#include <imgui.h>

#include <cstring>
#include <stdexcept>
#include <exception>
#include <string>

namespace Eden::UI
{
    namespace
    {
        // --- Creation panel state ---
        // ImGui::InputText needs a live, fixed-size char buffer to edit in
        // place - these persist across frames (static) so typed text isn't
        // wiped every frame, same as any other ImGui text field.
        char s_TexturePathBuffer[256] = "";
        char s_ObjPathBuffer[256] = "";
        float s_NewCubeSize = 1.0f;

        glm::vec3 s_NewPosition{ 0.0f, 0.0f, 0.0f };
        glm::vec3 s_NewRotation{ 0.0f, 0.0f, 0.0f };
        glm::vec3 s_NewScale{ 1.0f, 1.0f, 1.0f };

        // Optional components to attach at creation time - unchecked by
        // default so a plain "Create Cube" behaves exactly like it always
        // did (just Transform + Mesh). Values are only used/relevant while
        // their checkbox is on.
        bool s_AddColor = false;
        glm::vec3 s_NewColor{ 1.0f, 1.0f, 1.0f };

        bool s_AddRotationSpeed = false;
        float s_NewRotationSpeed = 30.0f;

        bool s_AddLifetime = false;
        float s_NewLifetimeSeconds = 5.0f;

        bool s_AddName = false;
        char s_NewNameBuffer[128] = "";

        bool s_AddRigidBody = false;
        int s_NewBodyType = static_cast<int>(BodyType::Dynamic);

        bool s_AddCollider = false;
        int s_NewColliderShape = static_cast<int>(ColliderShape::Sphere);

        // --- Raymarch box creation state ---
        // Separate from the MeshComponent cube's s_NewCubeSize (raymarch
        // boxes take a half-extents vector, not a single side length) and
        // from s_AddColor (RaymarchVolumeComponent's tint is unconditional,
        // not an optional override like ColorComponent).
        glm::vec3 s_NewRaymarchHalfExtents{ 0.5f, 0.5f, 0.5f };
        glm::vec3 s_NewRaymarchTint{ 1.0f, 0.5f, 0.2f };
        // Mirrors main.cpp's RaymarchBoxCollision enum (None / Voxel /
        // AnalyticBox) - see that enum's comment for why this isn't a
        // plain bool. Kept as an int + string table here rather than
        // pulling the enum itself in, since it's a Source/main.cpp-local
        // type this module has no header to include. Defaults to 1
        // (Voxel), matching RaymarchBoxCollision::Voxel's own comment on
        // why it's the project-wide default for anything raymarched -
        // real narrow-phase support in both CollisionSystem and
        // ParticleSystemGPU, and collision automatically tracks this
        // box's live shape if it's ever carved/melted/reformed.
        int s_NewRaymarchCollision = 1; // 0=None, 1=Voxel, 2=AnalyticBox

        // --- Inspector state ---
        // Which entity (if any) the "Entities" list has selected for
        // editing below. NullEntity = nothing selected, inspector hidden.
        Entity s_SelectedEntity = NullEntity;

        void ApplyCreationTransform(Registry& registry, Entity entity)
        {
            TransformComponent transform;
            transform.position = s_NewPosition;
            transform.rotationDegrees = s_NewRotation;
            transform.scale = s_NewScale;
            registry.AddComponent(entity, transform);
        }

        void ApplyOptionalCreationComponents(Registry& registry, Entity entity)
        {
            if (s_AddColor)
            {
                registry.AddComponent(entity, ColorComponent{ s_NewColor });
            }
            if (s_AddRotationSpeed)
            {
                registry.AddComponent(entity, RotationSpeedComponent{ s_NewRotationSpeed });
            }
            if (s_AddLifetime)
            {
                registry.AddComponent(entity, LifetimeComponent{ s_NewLifetimeSeconds });
            }
            if (s_AddName && s_NewNameBuffer[0] != '\0')
            {
                registry.AddComponent(entity, NameComponent{ std::string(s_NewNameBuffer) });
            }
            if (s_AddRigidBody)
            {
                RigidBodyComponent body;
                body.type = static_cast<BodyType>(s_NewBodyType);
                body.inverseMass = (body.type == BodyType::Dynamic) ? 1.0f : 0.0f;
                registry.AddComponent(entity, body);
            }
            if (s_AddCollider)
            {
                ColliderComponent collider;
                collider.shape = static_cast<ColliderShape>(s_NewColliderShape);
                registry.AddComponent(entity, collider);
            }
        }

        // Editor-side twin of main.cpp's local SpawnRaymarchBox (see that
        // function's comments for the full rationale on field margin,
        // collision strategy, and the AnalyticBox-vs-Voxel tradeoff) -
        // duplicated rather than shared because main.cpp's version is
        // anonymous-namespace-local to that translation unit. If this
        // drifts out of sync with main.cpp's version, that's the known
        // cost of the duplication; worth factoring into a shared header
        // if a third call site ever needs it.
        //
        // collisionMode: 0=None, 1=Voxel, 2=AnalyticBox - see
        // s_NewRaymarchCollision's comment above for why this is an int
        // rather than main.cpp's RaymarchBoxCollision enum.
        Entity SpawnRaymarchBoxFromEditor(Registry& registry, VoxelSystemGPU& voxelSystem,
                                           const glm::vec3& worldCenter, const glm::vec3& halfExtents,
                                           const glm::vec3& tintColor, int collisionMode)
        {
            constexpr float kFieldMargin = 0.2f;
            glm::vec3 fieldHalfExtents = halfExtents + glm::vec3(kFieldMargin);

            VoxelVolumeDesc desc;
            desc.origin = worldCenter - fieldHalfExtents;
            desc.voxelSize = 0.08f;
            desc.chunkDims = glm::ivec3(2, 2, 2);
            VoxelVolumeHandle handle = voxelSystem.RegisterVolume(desc);

            glm::vec3 fieldExtent = glm::vec3(desc.VoxelDims()) * desc.voxelSize;
            glm::vec3 fieldLocalCenter = fieldExtent * 0.5f;
            voxelSystem.SeedBox(handle, fieldLocalCenter, halfExtents);

            Entity e = registry.CreateEntity();

            TransformComponent transform;
            transform.position = worldCenter;
            registry.AddComponent(e, transform);

            registry.AddComponent(e, VoxelVolumeComponent{ handle });
            registry.AddComponent(e, RaymarchVolumeComponent{ tintColor });

            if (collisionMode != 0) // not None
            {
                RigidBodyComponent body;
                body.type = BodyType::Static;
                body.inverseMass = 0.0f;
                registry.AddComponent(e, body);

                ColliderComponent collider;
                collider.halfExtents = halfExtents;
                if (collisionMode == 1) // Voxel
                {
                    collider.shape = ColliderShape::Voxel;
                    collider.voxelVolume = handle;
                }
                else // AnalyticBox
                {
                    collider.shape = ColliderShape::Box;
                }
                registry.AddComponent(e, collider);
            }

            return e;
        }

        void DrawCreationPanel(Registry& registry, Renderer& renderer, VoxelSystemGPU& voxelSystem)
        {
            ImGui::Text("Create");
            ImGui::Separator();

            ImGui::DragFloat3("Position", &s_NewPosition.x, 0.1f);
            ImGui::DragFloat3("Rotation (deg)", &s_NewRotation.x, 1.0f);
            ImGui::DragFloat3("Scale", &s_NewScale.x, 0.05f, 0.01f, 100.0f);

            if (ImGui::Button("Snap Position to Camera View"))
            {
                Camera& camera = renderer.GetCamera();
                s_NewPosition = camera.Position + camera.Front * 3.0f;
            }

            ImGui::Spacing();
            ImGui::InputFloat("Cube size", &s_NewCubeSize, 0.1f, 1.0f, "%.2f");
            if (s_NewCubeSize < 0.01f)
            {
                s_NewCubeSize = 0.01f; // guard against zero/negative-size geometry
            }

            ImGui::Spacing();
            ImGui::Text("Optional components");
            ImGui::Checkbox("Color override##create", &s_AddColor);
            if (s_AddColor)
            {
                ImGui::SameLine();
                ImGui::ColorEdit3("##createColor", &s_NewColor.x);
            }

            ImGui::Checkbox("Rotation speed##create", &s_AddRotationSpeed);
            if (s_AddRotationSpeed)
            {
                ImGui::SameLine();
                ImGui::DragFloat("deg/sec##createSpin", &s_NewRotationSpeed, 1.0f);
            }

            ImGui::Checkbox("Lifetime##create", &s_AddLifetime);
            if (s_AddLifetime)
            {
                ImGui::SameLine();
                ImGui::DragFloat("seconds##createLifetime", &s_NewLifetimeSeconds, 0.1f, 0.01f, 3600.0f);
            }

            ImGui::Checkbox("Name##create", &s_AddName);
            if (s_AddName)
            {
                ImGui::SameLine();
                ImGui::InputText("##createName", s_NewNameBuffer, sizeof(s_NewNameBuffer));
            }

            ImGui::Checkbox("Rigid Body##create", &s_AddRigidBody);
            if (s_AddRigidBody)
            {
                ImGui::SameLine();
                const char* typeNames[] = { "Static", "Kinematic", "Dynamic" };
                ImGui::Combo("##createBodyType", &s_NewBodyType, typeNames, IM_ARRAYSIZE(typeNames));
            }

            ImGui::Checkbox("Collider##create", &s_AddCollider);
            if (s_AddCollider)
            {
                ImGui::SameLine();
                const char* shapeNames[] = { "Sphere", "Box", "Capsule", "Plane" };
                ImGui::Combo("##createColliderShape", &s_NewColliderShape, shapeNames, IM_ARRAYSIZE(shapeNames));
                ImGui::TextDisabled("Fine-tune shape size/offset in the inspector after creating.");
            }

            ImGui::Spacing();
            if (ImGui::Button("Create Cube"))
            {
                Entity entity = registry.CreateEntity();
                ApplyCreationTransform(registry, entity);

                MeshHandle mesh = renderer.CreateCubeMesh(s_NewCubeSize, InvalidTextureHandle);
                registry.AddComponent(entity, MeshComponent{ mesh });

                ApplyOptionalCreationComponents(registry, entity);
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Text("Raymarch box (SDF, no texture support)");
            ImGui::DragFloat3("Half Extents##raymarch", &s_NewRaymarchHalfExtents.x, 0.02f, 0.05f, 20.0f);
            ImGui::ColorEdit3("Tint##raymarch", &s_NewRaymarchTint.x);
            const char* collisionNames[] = { "None", "Voxel (default - tracks live shape, SPH-safe)", "Analytic Box (static geometry only)" };
            ImGui::Combo("Collision##raymarch", &s_NewRaymarchCollision, collisionNames, IM_ARRAYSIZE(collisionNames));
            ImGui::TextDisabled("Uses Position above, not Scale (Scale only applies to MeshComponent cubes).");
            if (ImGui::Button("Create Raymarch Box"))
            {
                SpawnRaymarchBoxFromEditor(registry, voxelSystem, s_NewPosition, s_NewRaymarchHalfExtents,
                                            s_NewRaymarchTint, s_NewRaymarchCollision);
                // Deliberately does NOT call ApplyOptionalCreationComponents -
                // RaymarchVolumeComponent already carries its own tint (no
                // ColorComponent needed) and SpawnRaymarchBoxFromEditor
                // already adds RigidBody+Collider when collision is
                // requested; running the optional-components block on top
                // would risk a second, conflicting ColliderComponent.
            }

            ImGui::Spacing();
            ImGui::InputText("Texture path", s_TexturePathBuffer, sizeof(s_TexturePathBuffer));
            ImGui::SameLine();
            if (ImGui::Button("Create Textured Cube"))
            {
                if (s_TexturePathBuffer[0] != '\0')
                {
                    // CreateTexture/CreateCubeMesh both throw
                    // std::runtime_error on failure (bad path, decode
                    // failure, etc.) - caught here rather than left to
                    // propagate and crash the whole engine over a typo'd
                    // path in a text field.
                    try
                    {
                        TextureHandle texture = renderer.CreateTexture(s_TexturePathBuffer);

                        Entity entity = registry.CreateEntity();
                        ApplyCreationTransform(registry, entity);

                        MeshHandle mesh = renderer.CreateCubeMesh(s_NewCubeSize, texture);
                        registry.AddComponent(entity, MeshComponent{ mesh });

                        ApplyOptionalCreationComponents(registry, entity);
                    }
                    catch (const std::exception&)
                    {
                        ImGui::OpenPopup("CreateTextureError");
                    }
                }
            }

            if (ImGui::BeginPopup("CreateTextureError"))
            {
                ImGui::Text("Failed to load texture - check the path and the\nconsole output for details.");
                ImGui::EndPopup();
            }

            ImGui::Spacing();
            ImGui::InputText("Model path (.obj/.gltf/.glb)", s_ObjPathBuffer, sizeof(s_ObjPathBuffer));
            ImGui::SameLine();
            if (ImGui::Button("Load Model"))
            {
                if (s_ObjPathBuffer[0] != '\0')
                {
                    try
                    {
                        Entity entity = registry.CreateEntity();
                        ApplyCreationTransform(registry, entity);

                        MeshHandle mesh = renderer.CreateMeshFromFile(s_ObjPathBuffer);
                        registry.AddComponent(entity, MeshComponent{ mesh });

                        ApplyOptionalCreationComponents(registry, entity);
                    }
                    catch (const std::exception&)
                    {
                        ImGui::OpenPopup("LoadObjError");
                    }
                }
            }

            if (ImGui::BeginPopup("LoadObjError"))
            {
                ImGui::Text("Failed to load model - check the path and the\nconsole output for details.");
                ImGui::EndPopup();
            }
        }

        void DrawEntityList(Registry& registry, VoxelSystemGPU& voxelSystem)
        {
            ImGui::Spacing();
            ImGui::Spacing();
            ImGui::Text("Entities");
            ImGui::Separator();

            // Snapshot the entity list before iterating, not while
            // iterating - clicking "Destroy" below calls
            // registry.DestroyEntity() mid-loop, and View() already
            // returns a plain vector<Entity> (not a live iterator into
            // storage), so this is safe by construction rather than by
            // luck. See Registry::View's own comment for why.
            //
            // Two separate View() calls (MeshComponent, then
            // VoxelVolumeComponent) rather than one combined query -
            // MeshComponent and VoxelVolumeComponent entities are
            // mutually exclusive in practice (nothing in this panel
            // creates both on the same entity), and this keeps each list
            // a flat, ordinary View() the same way DrawEntityList always
            // worked, rather than introducing an either/or query type
            // just for this listing.
            auto drawEntityRow = [&](Entity entity, const char* fallbackPrefix)
            {
                ImGui::PushID(static_cast<int>(GetEntityIndex(entity)));

                std::string label = registry.HasComponent<NameComponent>(entity)
                    ? registry.GetComponent<NameComponent>(entity).name
                    : std::string(fallbackPrefix) + std::to_string(GetEntityIndex(entity));

                bool isSelected = (entity == s_SelectedEntity);
                if (ImGui::Selectable(label.c_str(), isSelected))
                {
                    s_SelectedEntity = entity;
                }

                ImGui::SameLine();
                if (ImGui::Button("Destroy"))
                {
                    if (s_SelectedEntity == entity)
                    {
                        s_SelectedEntity = NullEntity;
                    }

                    // Free the GPU/CPU-side volume BEFORE dropping the
                    // ECS side - this used to be the actual live leak
                    // (see UnregisterVolume's header comment): clicking
                    // Destroy here removed the entity but nothing ever
                    // freed the VoxelSystemGPU volume it pointed at.
                    if (registry.HasComponent<VoxelVolumeComponent>(entity))
                    {
                        voxelSystem.UnregisterVolume(registry.GetComponent<VoxelVolumeComponent>(entity).handle);
                    }
                    registry.DestroyEntity(entity);
                }

                ImGui::PopID();
            };

            for (Entity entity : registry.View<MeshComponent>())
            {
                drawEntityRow(entity, "Entity ");
            }
            for (Entity entity : registry.View<VoxelVolumeComponent>())
            {
                drawEntityRow(entity, "Raymarch Box ");
            }
        }

        // One row: "Add X" button when the entity doesn't have the
        // component, or the component's editable field(s) plus a "Remove"
        // button when it does. Every optional component in the inspector
        // follows this exact shape - kept as one small helper instead of
        // repeating the if/else four times below.
        template<typename T, typename DrawFieldsFn, typename MakeDefaultFn>
        void DrawOptionalComponentRow(Registry& registry, Entity entity, const char* label,
                                       DrawFieldsFn drawFields, MakeDefaultFn makeDefault)
        {
            ImGui::PushID(label);

            if (registry.HasComponent<T>(entity))
            {
                ImGui::Text("%s", label);
                ImGui::SameLine();
                if (ImGui::SmallButton("Remove"))
                {
                    registry.RemoveComponent<T>(entity);
                }
                else
                {
                    drawFields(registry.GetComponent<T>(entity));
                }
            }
            else
            {
                if (ImGui::Button((std::string("Add ") + label).c_str()))
                {
                    registry.AddComponent(entity, makeDefault());
                }
            }

            ImGui::PopID();
        }

        void DrawInspector(Registry& registry)
        {
            if (s_SelectedEntity == NullEntity || !registry.IsAlive(s_SelectedEntity))
            {
                s_SelectedEntity = NullEntity;
                return;
            }

            ImGui::Spacing();
            ImGui::Spacing();
            ImGui::Text("Inspector - Entity %u", GetEntityIndex(s_SelectedEntity));
            ImGui::Separator();

            // TransformComponent is always present on anything this panel
            // creates (see ApplyCreationTransform) - not offered as
            // optional add/remove the way the components below are.
            if (registry.HasComponent<TransformComponent>(s_SelectedEntity))
            {
                auto& transform = registry.GetComponent<TransformComponent>(s_SelectedEntity);
                ImGui::DragFloat3("Position##inspector", &transform.position.x, 0.1f);
                ImGui::DragFloat3("Rotation (deg)##inspector", &transform.rotationDegrees.x, 1.0f);
                ImGui::DragFloat3("Scale##inspector", &transform.scale.x, 0.05f, 0.01f, 100.0f);
            }

            // Not routed through DrawOptionalComponentRow like the
            // components below - RaymarchVolumeComponent is mandatory on
            // any entity that has it (RaymarchSystem's own comment: "a
            // RaymarchVolumeComponent entity always has a color", no
            // fallback), so offering a "Remove" button here would let the
            // inspector put a raymarch box into a state RaymarchSystem
            // doesn't expect. Just an unconditional tint edit when present.
            if (registry.HasComponent<RaymarchVolumeComponent>(s_SelectedEntity))
            {
                auto& raymarch = registry.GetComponent<RaymarchVolumeComponent>(s_SelectedEntity);
                ImGui::ColorEdit3("Raymarch Tint##inspector", &raymarch.tintColor.x);
            }

            ImGui::Spacing();

            DrawOptionalComponentRow<ColorComponent>(
                registry, s_SelectedEntity, "Color Override",
                [](ColorComponent& color) { ImGui::ColorEdit3("##color", &color.color.x); },
                []() { return ColorComponent{}; });

            DrawOptionalComponentRow<VisibilityComponent>(
                registry, s_SelectedEntity, "Visibility",
                [](VisibilityComponent& vis) { ImGui::Checkbox("Visible##vis", &vis.visible); },
                []() { return VisibilityComponent{}; });

            DrawOptionalComponentRow<RotationSpeedComponent>(
                registry, s_SelectedEntity, "Rotation Speed",
                [](RotationSpeedComponent& spin) { ImGui::DragFloat("deg/sec##spin", &spin.degreesPerSecond, 1.0f); },
                []() { return RotationSpeedComponent{ 30.0f }; });

            DrawOptionalComponentRow<LifetimeComponent>(
                registry, s_SelectedEntity, "Lifetime",
                [](LifetimeComponent& lifetime) { ImGui::DragFloat("seconds left##lifetime", &lifetime.remainingSeconds, 0.1f, 0.0f, 3600.0f); },
                []() { return LifetimeComponent{ 5.0f }; });

            DrawOptionalComponentRow<RigidBodyComponent>(
                registry, s_SelectedEntity, "Rigid Body",
                [](RigidBodyComponent& body)
                {
                    const char* typeNames[] = { "Static", "Kinematic", "Dynamic" };
                    int typeIndex = static_cast<int>(body.type);
                    if (ImGui::Combo("Body Type##rb", &typeIndex, typeNames, IM_ARRAYSIZE(typeNames)))
                    {
                        body.type = static_cast<BodyType>(typeIndex);
                        // Static/Kinematic take no forces - zero
                        // inverseMass to match what PhysicsSystem::Step
                        // enforces at runtime, so the inspector doesn't
                        // keep showing a stale finite mass next to a body
                        // that's actually being treated as infinite mass.
                        if (body.type != BodyType::Dynamic)
                        {
                            body.inverseMass = 0.0f;
                        }
                        else if (body.inverseMass == 0.0f)
                        {
                            body.inverseMass = 1.0f;
                        }
                    }

                    if (body.type == BodyType::Dynamic)
                    {
                        float mass = body.inverseMass > 0.0f ? 1.0f / body.inverseMass : 0.0f;
                        if (ImGui::DragFloat("Mass##rb", &mass, 0.1f, 0.01f, 1000.0f, "%.2f"))
                        {
                            body.inverseMass = mass > 0.0f ? 1.0f / mass : 0.0f;
                        }
                        ImGui::Checkbox("Use Gravity##rb", &body.useGravity);
                        ImGui::DragFloat("Linear Damping##rb", &body.linearDamping, 0.001f, 0.0f, 1.0f);
                        ImGui::DragFloat("Angular Damping##rb", &body.angularDamping, 0.001f, 0.0f, 1.0f);
                    }

                    ImGui::DragFloat3("Linear Velocity##rb", &body.linearVelocity.x, 0.1f);
                    ImGui::DragFloat3("Angular Velocity (deg/s)##rb", &body.angularVelocity.x, 1.0f);
                },
                []() { return RigidBodyComponent{}; });

            DrawOptionalComponentRow<ColliderComponent>(
                registry, s_SelectedEntity, "Collider",
                [](ColliderComponent& collider)
                {
                    const char* shapeNames[] = { "Sphere", "Box", "Capsule", "Plane" };
                    int shapeIndex = static_cast<int>(collider.shape);
                    if (ImGui::Combo("Shape##col", &shapeIndex, shapeNames, IM_ARRAYSIZE(shapeNames)))
                    {
                        collider.shape = static_cast<ColliderShape>(shapeIndex);
                    }

                    switch (collider.shape)
                    {
                        case ColliderShape::Sphere:
                            ImGui::DragFloat("Radius##col", &collider.radius, 0.05f, 0.01f, 100.0f);
                            break;
                        case ColliderShape::Box:
                            ImGui::DragFloat3("Half Extents##col", &collider.halfExtents.x, 0.05f, 0.01f, 100.0f);
                            break;
                        case ColliderShape::Capsule:
                            ImGui::DragFloat("Radius##col", &collider.radius, 0.05f, 0.01f, 100.0f);
                            ImGui::DragFloat("Half Height##col", &collider.halfHeight, 0.05f, 0.0f, 100.0f);
                            break;
                        case ColliderShape::Plane:
                            ImGui::DragFloat3("Normal##col", &collider.planeNormal.x, 0.01f, -1.0f, 1.0f);
                            ImGui::SameLine();
                            if (ImGui::SmallButton("Normalize##col"))
                            {
                                float length = glm::length(collider.planeNormal);
                                if (length > 0.0001f)
                                {
                                    collider.planeNormal /= length;
                                }
                            }
                            break;
                    }

                    ImGui::DragFloat3("Local Offset##col", &collider.localOffset.x, 0.05f);
                },
                []() { return ColliderComponent{}; });

            ImGui::PushID("NameRow");
            if (registry.HasComponent<NameComponent>(s_SelectedEntity))
            {
                auto& name = registry.GetComponent<NameComponent>(s_SelectedEntity);
                char buffer[128];
                std::strncpy(buffer, name.name.c_str(), sizeof(buffer) - 1);
                buffer[sizeof(buffer) - 1] = '\0';
                if (ImGui::InputText("Name", buffer, sizeof(buffer)))
                {
                    name.name = buffer;
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("Remove##name"))
                {
                    registry.RemoveComponent<NameComponent>(s_SelectedEntity);
                }
            }
            else
            {
                if (ImGui::Button("Add Name"))
                {
                    registry.AddComponent(s_SelectedEntity, NameComponent{ "" });
                }
            }
            ImGui::PopID();
        }
    }

    void DrawMeshPanel(Registry& registry, Renderer& renderer, VoxelSystemGPU& voxelSystem)
    {
        ImGui::Begin("Eden - Mesh Editor");

        DrawCreationPanel(registry, renderer, voxelSystem);
        DrawEntityList(registry, voxelSystem);
        DrawInspector(registry);

        ImGui::End();
    }

    void DrawPhysicsSettingsPanel(PhysicsSystem& physicsSystem, CollisionSystem& collisionSystem)
    {
        ImGui::Begin("Eden - Physics Settings");

        ImGui::Text("Global");
        ImGui::DragFloat3("Gravity##phys", &physicsSystem.gravity.x, 0.1f);

        ImGui::Separator();
        ImGui::Text("Resolution");
        ImGui::DragFloat("Restitution##phys", &collisionSystem.restitution, 0.01f, 0.0f, 1.0f);
        ImGui::DragFloat("Friction##phys", &collisionSystem.friction, 0.01f, 0.0f, 2.0f);
        ImGui::DragFloat("Correction %##phys", &collisionSystem.positionalCorrectionPercent, 0.01f, 0.0f, 1.0f);
        ImGui::DragFloat("Correction Slop##phys", &collisionSystem.positionalCorrectionSlop, 0.001f, 0.0f, 0.5f);
        ImGui::SliderInt("Resolution Iterations##phys", &collisionSystem.resolutionIterations, 1, 12);
        ImGui::TextDisabled("Higher = stacks/chains settle more accurately (costs more per step).");
        ImGui::DragFloat("Max Mover Push Speed##phys", &collisionSystem.maxMoverBorrowedSpeed, 0.1f, 0.1f, 20.0f);
        ImGui::TextDisabled("Caps how hard dragging a Static/Kinematic body can push what's resting on it.");
        ImGui::DragFloat("Max Angular Speed##phys", &collisionSystem.maxAngularSpeed, 0.5f, 1.0f, 50.0f);
        ImGui::TextDisabled("Safety clamp on tumbling/spin speed (rad/s) - prevents solver instability from looking chaotic.");

        ImGui::Separator();
        ImGui::Text("Continuous Collision (tunneling fix)");
        ImGui::Checkbox("Enable Sweep##phys", &collisionSystem.enableContinuousCollisionSweep);
        if (collisionSystem.enableContinuousCollisionSweep)
        {
            ImGui::SliderInt("Max Sweep Iterations##phys", &collisionSystem.maxSweepSubsteps, 1, 48);
        }

        ImGui::Separator();
        ImGui::Text("Broad Phase");
        ImGui::Checkbox("Adaptive Cell Size##phys", &collisionSystem.adaptiveBroadPhaseCellSize);
        if (collisionSystem.adaptiveBroadPhaseCellSize)
        {
            ImGui::BeginDisabled();
            ImGui::DragFloat("Grid Cell Size##phys", &collisionSystem.broadPhaseCellSize, 0.1f, 0.5f, 500.0f);
            ImGui::EndDisabled();
            ImGui::TextDisabled("Auto-computed from current collider sizes.");
        }
        else
        {
            ImGui::DragFloat("Grid Cell Size##phys", &collisionSystem.broadPhaseCellSize, 0.1f, 0.5f, 500.0f);
            ImGui::TextDisabled("Roughly match your typical collider size.");
        }

        ImGui::End();
    }

    void DrawParticleSettingsPanel(ParticleSystem& particleSystem, glm::vec3& spawnOrigin, float& particlePointSize)
    {
        ImGui::Begin("Eden - Particle Settings (SPH)");

        ImGui::Text("Particles: %zu", particleSystem.ParticleCount());
        ImGui::Text("Worker Threads: %u", particleSystem.ThreadCount());

        ImGui::Separator();
        ImGui::Text("Rendering");
        ImGui::DragFloat("Point Size (px)##part", &particlePointSize, 0.1f, 1.0f, 64.0f);
        ImGui::TextDisabled("Screen-space pixel size - does not scale with distance from camera, unlike world-space mesh scale.");

        ImGui::Separator();
        ImGui::Text("Spawn");
        ImGui::DragFloat3("Spawn Origin##part", &spawnOrigin.x, 0.1f);
        if (ImGui::Button("Spawn Box (1x1x1)##part"))
        {
            particleSystem.EmitBox(spawnOrigin - glm::vec3(0.5f), spawnOrigin + glm::vec3(0.5f));
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear All##part"))
        {
            particleSystem.Clear();
        }

        ImGui::Separator();
        ImGui::Text("Solver");
        ImGui::DragFloat("Smoothing Radius##part", &particleSystem.smoothingRadius, 0.01f, 0.05f, 2.0f);
        ImGui::TextDisabled("Changing this after particles exist is fine - takes effect next Step().");
        ImGui::DragFloat("Particle Mass##part", &particleSystem.particleMass, 0.001f, 0.001f, 1.0f);
        ImGui::DragFloat("Rest Density##part", &particleSystem.restDensity, 10.0f, 1.0f, 5000.0f);
        ImGui::DragFloat("Stiffness##part", &particleSystem.stiffness, 10.0f, 0.0f, 20000.0f);
        ImGui::DragFloat("Gamma##part", &particleSystem.gamma, 0.1f, 1.0f, 10.0f);
        ImGui::DragFloat("Viscosity##part", &particleSystem.viscosityCoefficient, 0.01f, 0.0f, 10.0f);
        ImGui::SliderInt("Substeps##part", &particleSystem.substeps, 1, 16);
        ImGui::TextDisabled("WCSPH is stiff - too few substeps at high Stiffness is the usual cause of an explosion.");

        ImGui::Separator();
        ImGui::Text("Boundary Collision (one-way)");
        ImGui::DragFloat3("Gravity##part", &particleSystem.gravity.x, 0.1f);
        ImGui::DragFloat("Boundary Radius##part", &particleSystem.boundaryRadius, 0.01f, 0.01f, 1.0f);
        ImGui::DragFloat("Boundary Restitution##part", &particleSystem.boundaryRestitution, 0.01f, 0.0f, 1.0f);
        ImGui::SliderInt("Max Sweep Iterations##part", &particleSystem.maxSweepIterations, 1, 32);
        ImGui::TextDisabled("Sweep (tunnelling fix) only runs for particles moving faster than Boundary Radius per substep.");
        ImGui::TextDisabled("Particles are deflected by any TransformComponent + ColliderComponent entity; no force is applied back onto it yet.");

        ImGui::End();
    }

    void DrawRendererSettingsPanel(Renderer& renderer)
    {
        ImGui::Begin("Eden - Rendering / Culling");

        ImGui::Checkbox("Occlusion Culling##render", &renderer.EnableOcclusionCulling);
        ImGui::TextDisabled("Frustum culling always runs; this adds a second, coarse CPU pass on top of it.");

        ImGui::SliderInt("Min Occluder Footprint (cells)##render", &renderer.MinOccluderFootprintCells, 1, 64);
        ImGui::TextDisabled("How big (in the occlusion grid) an object's silhouette must be before it's used to hide OTHER objects.");
        ImGui::TextDisabled("Every object is still tested against occluders regardless of its own size - this only controls who contributes depth.");

        ImGui::TextDisabled("Occluders use bounding BOXES, not exact shapes - a rotated or non-box object's box can be");
        ImGui::TextDisabled("bigger than what it actually blocks. If something visible seems to vanish, try toggling this off first.");

        ImGui::End();
    }
}
