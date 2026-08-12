#pragma once

#include "Entity.h"
#include "ComponentStorage.h"
#include "System.h"

#include <unordered_map>
#include <typeindex>
#include <memory>
#include <vector>
#include <utility>
#include <tuple>

namespace Eden
{
    // The ECS world: owns every entity, every component storage, and every
    // registered system. Deliberately simple - no archetypes, no
    // multithreading, no query caching. Good enough for hundreds of
    // thousands of entities at the ECS-bookkeeping level (see the perf
    // work in Systems/); revisit the underlying data structures only if
    // that stops being true.
    class Registry
    {
    public:
        Entity CreateEntity()
        {
            EntityIndex index;
            if (!m_FreeIndices.empty())
            {
                index = m_FreeIndices.back();
                m_FreeIndices.pop_back();
                // Generation for this index was already bumped when it was
                // freed (see DestroyEntity) - m_Generations[index] already
                // holds the correct NEW generation to hand out.
            }
            else
            {
                index = m_NextIndex++;
                m_Generations.push_back(0);
            }

            return MakeEntity(index, m_Generations[index]);
        }

        void DestroyEntity(Entity entity)
        {
            if (!IsAlive(entity))
            {
                // Already destroyed, a stale handle from a previous
                // generation, or NullEntity - nothing to do. This guard is
                // not optional: without it, calling DestroyEntity twice on
                // the same entity (or once on a handle that's already
                // stale) would bump the generation and free the index
                // twice, which could hand the same index out to two
                // "different" live entities at once.
                return;
            }

            for (auto& [type, storage] : m_Storages)
            {
                storage->Remove(entity);
            }

            EntityIndex index = GetEntityIndex(entity);
            m_Generations[index]++;
            m_FreeIndices.push_back(index);
        }

        // True if `entity` refers to a currently-live entity - i.e. its
        // index is in range AND its generation matches what's currently
        // live at that index. False for NullEntity, for a destroyed
        // entity, and for a stale handle whose index has since been
        // recycled for something else. Systems holding onto an Entity
        // across frames (ParentComponent, a spawner's tracking, etc.)
        // should check this before trusting the reference rather than
        // finding out the hard way via a failed component lookup.
        bool IsAlive(Entity entity) const
        {
            if (entity == NullEntity)
            {
                return false;
            }

            EntityIndex index = GetEntityIndex(entity);
            return index < m_Generations.size() && m_Generations[index] == GetEntityGeneration(entity);
        }

        template<typename T>
        void AddComponent(Entity entity, T component)
        {
            GetStorage<T>().Insert(entity, std::move(component));
        }

        template<typename T>
        void RemoveComponent(Entity entity)
        {
            GetStorage<T>().Remove(entity);
        }

        template<typename T>
        bool HasComponent(Entity entity)
        {
            return GetStorage<T>().Has(entity);
        }

        template<typename T>
        T& GetComponent(Entity entity)
        {
            return GetStorage<T>().Get(entity);
        }

        // Returns every entity that has ALL of the given component types.
        // Returns a plain vector rather than a lazy/zero-allocation
        // iterator - simpler to read and reason about, at the cost of one
        // small heap allocation per call. Fine at hundreds of entities; if
        // this ever shows up in a profiler, that's the signal to replace
        // it with a real iterator instead of a vector<Entity> return.
        //
        // Resolves each Rest storage exactly ONCE per View() call (via the
        // tuple below), not once per entity - GetStorage<T>() itself is a
        // std::unordered_map<type_index, ...> lookup, which used to run
        // once per entity per component type here. At small entity counts
        // that's invisible; at tens of thousands it dominates the frame.
        // See GetStorage's comment for the rest of this story.
        template<typename First, typename... Rest>
        std::vector<Entity> View()
        {
            std::vector<Entity> result;
            auto& driverStorage = GetStorage<First>();
            std::tuple<ComponentStorage<Rest>&...> restStorages(GetStorage<Rest>()...);

            for (Entity entity : driverStorage.Entities())
            {
                bool hasAll = std::apply([entity](auto&... storages)
                {
                    return (storages.Has(entity) && ...);
                }, restStorages);

                if (hasAll)
                {
                    result.push_back(entity);
                }
            }
            return result;
        }

        template<typename T>
        void RegisterSystem()
        {
            m_Systems.push_back(std::make_unique<T>());
        }

        void UpdateSystems(float deltaTime)
        {
            for (auto& system : m_Systems)
            {
                system->Update(*this, deltaTime);
            }
        }

        // Public and intended to be called directly by systems that touch
        // many entities per frame (RenderSystem::BuildDrawList, SpinSystem,
        // etc.) - resolve the ComponentStorage<T>& reference ONCE at the
        // top of your loop, then call .Has()/.Get() on it directly instead
        // of Registry::HasComponent<T>()/GetComponent<T>() per entity.
        // Those convenience wrappers still exist and are fine for one-off
        // lookups, but each call re-resolves the storage via a hash-map
        // find - death by a thousand cuts at high entity counts. See
        // Systems/RenderSystem.h for the pattern.
        template<typename T>
        ComponentStorage<T>& GetStorage()
        {
            std::type_index type(typeid(T));
            auto it = m_Storages.find(type);
            if (it == m_Storages.end())
            {
                auto storage = std::make_unique<ComponentStorage<T>>();
                ComponentStorage<T>* raw = storage.get();
                m_Storages.emplace(type, std::move(storage));
                return *raw;
            }
            return static_cast<ComponentStorage<T>&>(*it->second);
        }

    private:
        EntityIndex m_NextIndex = 1; // 0 reserved as NullEntity's index
        std::vector<EntityGeneration> m_Generations{ 0 }; // index -> current generation; slot 0 is an unused placeholder (NullEntity)
        std::vector<EntityIndex> m_FreeIndices;

        std::unordered_map<std::type_index, std::unique_ptr<IComponentStorage>> m_Storages;
        std::vector<std::unique_ptr<System>> m_Systems;
    };
}
