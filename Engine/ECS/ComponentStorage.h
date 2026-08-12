#pragma once

#include "Entity.h"

#include <vector>
#include <cstddef>
#include <cassert>

namespace Eden
{
    // Base class purely so Registry can type-erase "remove this entity from
    // whichever component storages it happens to be in" without knowing
    // every concrete component type up front - see Registry::DestroyEntity.
    class IComponentStorage
    {
    public:
        virtual ~IComponentStorage() = default;
        virtual void Remove(Entity entity) = 0;
        virtual bool Has(Entity entity) const = 0;
    };

    // Sparse set: O(1) add/remove/has/get, and tightly packed dense
    // iteration with zero wasted space for entities that don't have this
    // component. This is the same underlying structure most small-to-mid
    // ECS libraries use (EnTT included) - a "sparse" array indexed
    // directly by entity INDEX, pointing into a "dense" array holding the
    // actual component data with no gaps.
    //
    // Generation-checked: each sparse slot remembers which Entity
    // GENERATION currently occupies it, not just whether it's occupied.
    // This is deliberately enforced HERE, not just in Registry, because
    // several hot-path systems (RenderSystem::BuildDrawList, SpinSystem,
    // etc.) call Has()/Get() directly on a cached storage reference rather
    // than going through Registry::HasComponent/GetComponent - putting the
    // check here means a stale Entity handle gets caught no matter which
    // path touches it, instead of relying on every caller to remember to
    // check Registry::IsAlive() first.
    template<typename T>
    class ComponentStorage : public IComponentStorage
    {
    public:
        void Insert(Entity entity, T component)
        {
            EntityIndex index = GetEntityIndex(entity);

            if (Has(entity))
            {
                m_Dense[m_Sparse[index].denseIndex] = std::move(component);
                return;
            }

            if (index >= m_Sparse.size())
            {
                m_Sparse.resize(index + 1);
            }

            m_Sparse[index] = SparseSlot{ m_Dense.size(), GetEntityGeneration(entity) };
            m_Dense.push_back(std::move(component));
            m_DenseToEntity.push_back(entity);
        }

        void Remove(Entity entity) override
        {
            if (!Has(entity))
            {
                return;
            }

            EntityIndex index = GetEntityIndex(entity);
            size_t denseIndex = m_Sparse[index].denseIndex;
            size_t lastDenseIndex = m_Dense.size() - 1;
            Entity lastEntity = m_DenseToEntity[lastDenseIndex];

            // Swap-and-pop: move the last element into the removed slot so
            // the dense array never develops holes, then shrink by one.
            m_Dense[denseIndex] = std::move(m_Dense[lastDenseIndex]);
            m_DenseToEntity[denseIndex] = lastEntity;
            m_Sparse[GetEntityIndex(lastEntity)].denseIndex = denseIndex;

            m_Dense.pop_back();
            m_DenseToEntity.pop_back();
            m_Sparse[index] = SparseSlot{}; // kInvalidIndex, generation irrelevant once invalid
        }

        bool Has(Entity entity) const override
        {
            EntityIndex index = GetEntityIndex(entity);
            return index < m_Sparse.size()
                && m_Sparse[index].denseIndex != kInvalidIndex
                && m_Sparse[index].generation == GetEntityGeneration(entity);
        }

        T& Get(Entity entity)
        {
            assert(Has(entity) && "Eden: entity does not have this component (or the handle is stale)");
            return m_Dense[m_Sparse[GetEntityIndex(entity)].denseIndex];
        }

        const T& Get(Entity entity) const
        {
            assert(Has(entity) && "Eden: entity does not have this component (or the handle is stale)");
            return m_Dense[m_Sparse[GetEntityIndex(entity)].denseIndex];
        }

        // Both arrays are always the same length and index-aligned:
        // m_DenseToEntity[i] is the entity that owns m_Dense[i]. Used by
        // Registry::View to drive iteration off whichever component type
        // has the fewest entities.
        const std::vector<Entity>& Entities() const { return m_DenseToEntity; }
        size_t Size() const { return m_Dense.size(); }

    private:
        static constexpr size_t kInvalidIndex = static_cast<size_t>(-1);

        struct SparseSlot
        {
            size_t denseIndex = kInvalidIndex;
            EntityGeneration generation = 0;
        };

        std::vector<SparseSlot> m_Sparse;    // entity index -> (dense index, generation)
        std::vector<T> m_Dense;              // packed component data, no gaps
        std::vector<Entity> m_DenseToEntity; // dense index -> entity id
    };
}
