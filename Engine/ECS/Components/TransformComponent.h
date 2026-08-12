#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace Eden
{
    struct TransformComponent
    {
        glm::vec3 position{ 0.0f };
        glm::vec3 rotationDegrees{ 0.0f }; // Euler angles, applied Y then X then Z
        glm::vec3 scale{ 1.0f };

        // Value-based dirty check: compares against the last values this
        // was built from, and only redoes the 4 matrix multiplies if
        // something actually changed. Most entities in a typical scene
        // (a static grid, anything without a RotationSpeedComponent /
        // AI driving it) never move once placed - recomputing their model
        // matrix from scratch every single frame is pure waste at scale.
        // At 27,000 entities that waste is no longer negligible.
        //
        // Deliberately NOT an intrusive "who mutated me" dirty flag (which
        // would mean every system/call site that touches position/
        // rotationDegrees/scale directly - SpinSystem, SpawnerSystem,
        // main.cpp, HierarchyUtils - would need to remember to set it,
        // and silently produce stale matrices the moment one of them
        // forgets). Comparing 9 floats is already far cheaper than
        // rebuilding a mat4, so there's no real cost to doing it this way
        // instead.
        glm::mat4 GetModelMatrix() const
        {
            if (m_CacheValid &&
                position == m_CachedPosition &&
                rotationDegrees == m_CachedRotationDegrees &&
                scale == m_CachedScale)
            {
                return m_CachedModel;
            }

            glm::mat4 model = glm::translate(glm::mat4(1.0f), position);
            model = glm::rotate(model, glm::radians(rotationDegrees.y), glm::vec3(0.0f, 1.0f, 0.0f));
            model = glm::rotate(model, glm::radians(rotationDegrees.x), glm::vec3(1.0f, 0.0f, 0.0f));
            model = glm::rotate(model, glm::radians(rotationDegrees.z), glm::vec3(0.0f, 0.0f, 1.0f));
            model = glm::scale(model, scale);

            m_CachedModel = model;
            m_CachedPosition = position;
            m_CachedRotationDegrees = rotationDegrees;
            m_CachedScale = scale;
            m_CacheValid = true;

            return m_CachedModel;
        }

    private:
        // mutable: GetModelMatrix() is logically const (it doesn't change
        // what this transform IS), but needs to write its cache. Same
        // reasoning as any other lazy-computed cache behind a const
        // getter.
        mutable bool m_CacheValid = false;
        mutable glm::mat4 m_CachedModel{ 1.0f };
        mutable glm::vec3 m_CachedPosition{ 0.0f };
        mutable glm::vec3 m_CachedRotationDegrees{ 0.0f };
        mutable glm::vec3 m_CachedScale{ 1.0f };
    };
}
