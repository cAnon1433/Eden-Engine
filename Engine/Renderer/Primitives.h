#pragma once

#include "Vulkan/RendererTypes.h"

#include <vector>
#include <cstdint>

namespace Eden::Primitives
{
    struct IndexedMesh
    {
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
    };

    // Unit cube (extent -0.5..0.5 on each axis before scaling by `size`),
    // centered on the origin. 24 vertices (4 per face) + 36 indices.
    //
    // Not 8 vertices: sharing a single vertex across faces would force it
    // to have one color, but each face here is a different flat color -
    // colors are a per-vertex attribute, so a corner touching 3 differently
    // colored faces needs 3 separate vertex entries at that position, one
    // per face. This is the standard "flat-shaded" cube layout - the same
    // reasoning applies later to face normals, which also can't be shared
    // across faces on a cube.
    inline IndexedMesh MakeCube(float size = 1.0f)
    {
        float h = size * 0.5f;

        glm::vec3 p000{-h,-h,-h}, p001{-h,-h, h}, p010{-h, h,-h}, p011{-h, h, h};
        glm::vec3 p100{ h,-h,-h}, p101{ h,-h, h}, p110{ h, h,-h}, p111{ h, h, h};

        glm::vec3 red{1,0,0}, green{0,1,0}, blue{0,0,1};
        glm::vec3 yellow{1,1,0}, cyan{0,1,1}, magenta{1,0,1};

        IndexedMesh mesh;

        // Each face contributes 4 vertices, in the same corner order as the
        // pre-index version so winding stays identical: (0,1,2) + (0,2,3).
        // UVs are the standard quad layout (a=0,0 b=1,0 c=1,1 d=0,1) -
        // every face gets the full 0..1 texture square, so a tiled/repeat
        // texture shows once per face rather than wrapping oddly. `normal`
        // is the same for all 4 corners of a face (flat shading, matches
        // the same "can't share across faces" reasoning as vertex color
        // above) - passed in explicitly per face rather than computed via
        // cross product, since the face/axis correspondence is already
        // known at each call site below and this avoids any risk of
        // getting the cross-product operand order (and therefore the
        // normal's sign) backwards.
        auto addFace = [&mesh](glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d, glm::vec3 color, glm::vec3 normal)
        {
            uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
            mesh.vertices.push_back({ a, color, {0.0f, 0.0f}, normal });
            mesh.vertices.push_back({ b, color, {1.0f, 0.0f}, normal });
            mesh.vertices.push_back({ c, color, {1.0f, 1.0f}, normal });
            mesh.vertices.push_back({ d, color, {0.0f, 1.0f}, normal });

            mesh.indices.push_back(base + 0);
            mesh.indices.push_back(base + 1);
            mesh.indices.push_back(base + 2);
            mesh.indices.push_back(base + 0);
            mesh.indices.push_back(base + 2);
            mesh.indices.push_back(base + 3);
        };

        addFace(p000, p010, p110, p100, red,     {  0.0f,  0.0f, -1.0f }); // -Z
        addFace(p001, p101, p111, p011, green,   {  0.0f,  0.0f,  1.0f }); // +Z
        addFace(p000, p001, p011, p010, blue,    { -1.0f,  0.0f,  0.0f }); // -X
        addFace(p100, p110, p111, p101, yellow,  {  1.0f,  0.0f,  0.0f }); // +X
        addFace(p000, p100, p101, p001, cyan,    {  0.0f, -1.0f,  0.0f }); // -Y
        addFace(p010, p011, p111, p110, magenta, {  0.0f,  1.0f,  0.0f }); // +Y

        return mesh;
    }
}
