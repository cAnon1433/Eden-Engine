#pragma once

#include "Primitives.h"

#include <string>

namespace Eden::ModelLoader
{
    // Loads a Wavefront OBJ file from disk and returns it in the same
    // IndexedMesh shape Primitives::MakeCube returns, so
    // Renderer::CreateMeshFromFile can feed it straight into
    // CreateIndexedMesh exactly like any procedural mesh - loaded and
    // hand-authored geometry go through the identical GPU-upload path.
    //
    // Vertex color defaults to white (1,1,1) for every loaded vertex - OBJ
    // has no standard per-vertex color channel most exporters actually
    // populate, so visual appearance is expected to come from a texture
    // (see Renderer::CreateTexture) or a ColorComponent override, not
    // vertex color, for loaded models.
    //
    // Deduplicates vertices by (position, normal, texcoord) triple - a
    // face-corner needs an exact match on all three to reuse an existing
    // Eden vertex instead of creating a new one, which is what correctly
    // keeps hard edges/flat-shaded faces sharp instead of averaging their
    // normals together.
    //
    // Throws std::runtime_error (with tinyobjloader's own error text) if
    // the file can't be found or fails to parse. Call during load time,
    // same as Renderer::CreateTexture/CreateMesh - this does real file I/O
    // and CPU-side parsing work, not something to call per-frame.
    Primitives::IndexedMesh LoadObj(const std::string& path);

    // Loads a glTF 2.0 file (.gltf with external buffers, or a single-file
    // .glb) and returns it in the same IndexedMesh shape as LoadObj/
    // MakeCube.
    //
    // Deliberately simplified compared to full glTF support: every mesh
    // and every primitive in the file gets merged into ONE flat
    // IndexedMesh. That means:
    //   - No scene graph / node hierarchy / per-node transforms - if the
    //     file positions multiple parts relative to each other via nodes,
    //     that relative positioning is lost; everything lands in the mesh's
    //     own local space as authored.
    //   - No per-primitive materials - Eden's texture binding is still
    //     per-MESH (see TextureHandle's comment in RendererTypes.h), same
    //     as OBJ loading; a multi-material glTF model will only ever show
    //     whichever single texture you bind the resulting MeshHandle to.
    //   - No skinning/animation/morph targets - static geometry only.
    // This matches the same scope level as LoadObj (flat static geometry,
    // one texture, no scene graph) rather than glTF's full feature set.
    // Revisit if a model actually needs multi-part hierarchy or
    // per-part materials to look right.
    //
    // Throws std::runtime_error if the file can't be found, fails to
    // parse, or contains no usable triangle geometry.
    Primitives::IndexedMesh LoadGltf(const std::string& path);
}
