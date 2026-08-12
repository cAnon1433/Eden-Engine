#pragma once

#include <cstdint>

namespace Eden
{
    // A generational handle: the low 32 bits are a slot index, the high 32
    // bits are a generation counter for that slot. Still a plain scalar
    // (uint64_t) rather than a wrapper struct on purpose - it stays usable
    // everywhere the old raw-uint32_t Entity was (unordered_map/
    // unordered_set keys via the default std::hash<uint64_t>, ==
    // comparisons, vector<Entity>) with zero call-site changes anywhere
    // outside Entity.h/ComponentStorage.h/Registry.h.
    //
    // WHY this exists: previously, destroying an entity and later reusing
    // its index for a brand-new entity made the two indistinguishable - a
    // stale Entity value held somewhere (ParentComponent::parent being the
    // obvious case) would silently resolve to whatever new, unrelated
    // entity now occupied that slot. Packing a generation counter in means
    // a stale handle carries the OLD generation, which no longer matches
    // the slot's CURRENT generation once it's been recycled - see
    // Registry::IsAlive and ComponentStorage's generation check in
    // ComponentStorage.h. This doesn't make stale references impossible
    // to hold (nothing stops something from hanging onto an old Entity
    // value), it makes them detectable instead of silently wrong.
    using Entity = uint64_t;
    using EntityIndex = uint32_t;
    using EntityGeneration = uint32_t;

    // index = 0, generation = 0. Registry never hands out index 0 (its
    // index counter starts at 1), so this can never collide with a real
    // entity.
    constexpr Entity NullEntity = 0;

    constexpr Entity MakeEntity(EntityIndex index, EntityGeneration generation)
    {
        return (static_cast<Entity>(generation) << 32) | static_cast<Entity>(index);
    }

    constexpr EntityIndex GetEntityIndex(Entity entity)
    {
        return static_cast<EntityIndex>(entity & 0xFFFFFFFFu);
    }

    constexpr EntityGeneration GetEntityGeneration(Entity entity)
    {
        return static_cast<EntityGeneration>(entity >> 32);
    }
}
