#pragma once

namespace Eden
{
    // Countdown timer in seconds. LifetimeSystem (see
    // Systems/LifetimeSystem.h) ticks this down every frame and calls
    // Registry::DestroyEntity once it hits zero. Useful for anything with
    // a natural expiry - particle-ish effects, temporary markers, entities
    // spawned by SpawnerComponent with a finite lifespan.
    //
    // Anything holding onto this entity's ID elsewhere (ParentComponent,
    // a SpawnerComponent's tracking, etc.) after it's destroyed here now
    // becomes a detectably-stale generational handle rather than a
    // silent wrong-entity bug - see Entity.h and Registry::IsAlive.
    struct LifetimeComponent
    {
        float remainingSeconds = 1.0f;
    };
}
