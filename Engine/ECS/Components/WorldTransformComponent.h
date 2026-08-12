#pragma once

#include <glm/glm.hpp>

namespace Eden
{
    // Computed output, not something to author by hand. Holds an
    // entity's final world-space model matrix after walking its
    // ParentComponent chain and combining with every ancestor's
    // TransformComponent - see Systems/HierarchySystem.h, which
    // (re)writes this every frame for anything with a TransformComponent.
    //
    // TransformComponent stays purely LOCAL - relative to the parent if
    // there is one, relative to world origin if not. That distinction
    // still needed a place to live, hence a separate component instead
    // of overloading TransformComponent's meaning. RenderSystem prefers
    // this over TransformComponent::GetModelMatrix() when it's present.
    struct WorldTransformComponent
    {
        glm::mat4 matrix{ 1.0f };
    };
}
