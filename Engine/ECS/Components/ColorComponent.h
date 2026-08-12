#pragma once

#include <glm/glm.hpp>

namespace Eden
{
    // Per-entity color override, read by RenderSystem. Lets one shared
    // MeshHandle (e.g. the stock cube) render as different colors per
    // entity without needing a separate Mesh/vertex buffer per color -
    // the override travels through the existing model-matrix push
    // constant instead of touching mesh data at all.
    //
    // Presence of this component IS the "override active" flag: an
    // entity with a MeshComponent but no ColorComponent just renders
    // its mesh's own vertex colors, same as before this existed.
    struct ColorComponent
    {
        glm::vec3 color{ 1.0f };
    };
}
