#pragma once

namespace Eden
{
    // Checked by RenderSystem before an entity is added to the draw list.
    // Lets something be hidden without DestroyEntity-ing it (and losing
    // its Transform/other component state) or ripping its MeshComponent
    // off and back on. An entity with no VisibilityComponent at all is
    // treated as visible - this is opt-in, not required for every
    // renderable entity.
    struct VisibilityComponent
    {
        bool visible = true;
    };
}
