// tinyobjloader's implementation is generated exactly once, in this
// translation unit - it's a single-header library (see
// ThirdParty/tinyobjloader), and ModelLoader is its only consumer in Eden.
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

// Same reasoning, same file - cgltf.h (see ThirdParty/cgltf) is also a
// single-header library, and having both implementations generated in
// this one translation unit is safe since they share no symbol names
// (tinyobj:: namespace vs cgltf_ prefix).
#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#include "ModelLoader.h"

#include <unordered_map>
#include <stdexcept>
#include <cstdint>

namespace Eden::ModelLoader
{
    namespace
    {
        // A face-corner's full identity for deduplication purposes: which
        // position, which normal, which texcoord. All three now matter -
        // previously (before Eden had a normal field on Vertex) this only
        // tracked position+texcoord, since there was nowhere for a normal
        // to go and no reason to split a vertex over one. Two face-corners
        // with the same position and UV but different normals used to
        // collapse into one Eden vertex; now they correctly stay separate,
        // which matters for anything that isn't perfectly smooth-shaded
        // (hard edges, flat-shaded faces sharing a corner, etc).
        struct VertexKey
        {
            int vertexIndex;
            int normalIndex;
            int texcoordIndex;

            bool operator==(const VertexKey& other) const
            {
                return vertexIndex == other.vertexIndex &&
                       normalIndex == other.normalIndex &&
                       texcoordIndex == other.texcoordIndex;
            }
        };

        struct VertexKeyHash
        {
            size_t operator()(const VertexKey& key) const
            {
                // Good enough for a load-time deduplication map - not
                // security-sensitive, just needs to distribute reasonably.
                size_t h1 = std::hash<int>{}(key.vertexIndex);
                size_t h2 = std::hash<int>{}(key.normalIndex);
                size_t h3 = std::hash<int>{}(key.texcoordIndex);
                return h1 ^ (h2 << 1) ^ (h3 << 2);
            }
        };
    }

    Primitives::IndexedMesh LoadObj(const std::string& path)
    {
        tinyobj::ObjReaderConfig config;
        config.triangulate = true; // Eden's index buffer assumes triangles throughout

        tinyobj::ObjReader reader;
        if (!reader.ParseFromFile(path, config))
        {
            throw std::runtime_error("Eden: failed to load OBJ '" + path + "': " + reader.Error());
        }

        const tinyobj::attrib_t& attrib = reader.GetAttrib();
        const std::vector<tinyobj::shape_t>& shapes = reader.GetShapes();

        Primitives::IndexedMesh mesh;

        // Maps a (position, normal, texcoord) triple already seen to the
        // Eden vertex index it produced, so a face-corner reusing the
        // exact same triple reuses the existing vertex instead of
        // duplicating it - this is what keeps the output an INDEXED mesh
        // instead of one unique vertex per face-corner.
        std::unordered_map<VertexKey, uint32_t, VertexKeyHash> uniqueVertices;

        for (const auto& shape : shapes)
        {
            for (const auto& index : shape.mesh.indices)
            {
                VertexKey key{ index.vertex_index, index.normal_index, index.texcoord_index };

                auto it = uniqueVertices.find(key);
                if (it != uniqueVertices.end())
                {
                    mesh.indices.push_back(it->second);
                    continue;
                }

                Vertex vertex{};

                vertex.position = {
                    attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2],
                };

                // OBJ has no standard per-vertex color most exporters
                // populate - default to white so a bound texture or
                // ColorComponent override shows its true color instead of
                // being tinted by leftover vertex color data.
                vertex.color = { 1.0f, 1.0f, 1.0f };

                if (index.texcoord_index >= 0)
                {
                    vertex.texCoord = {
                        attrib.texcoords[2 * index.texcoord_index + 0],
                        // OBJ's V axis is bottom-up; Vulkan's is top-down -
                        // flip it here once, at load time, instead of
                        // needing every fragment shader invocation to.
                        1.0f - attrib.texcoords[2 * index.texcoord_index + 1],
                    };
                }
                else
                {
                    vertex.texCoord = { 0.0f, 0.0f };
                }

                if (index.normal_index >= 0)
                {
                    vertex.normal = {
                        attrib.normals[3 * index.normal_index + 0],
                        attrib.normals[3 * index.normal_index + 1],
                        attrib.normals[3 * index.normal_index + 2],
                    };
                }
                else
                {
                    // Fallback for OBJ files that don't include normals at
                    // all (rare in practice - most exporters include them
                    // by default). A proper fix would compute a flat
                    // per-face normal from each triangle's geometry; this
                    // just picks "up" so lighting doesn't produce garbage/
                    // uninitialized results, at the cost of visibly wrong
                    // shading on such a model. Known, deliberate
                    // simplification - not silently "handled".
                    vertex.normal = { 0.0f, 1.0f, 0.0f };
                }

                uint32_t newIndex = static_cast<uint32_t>(mesh.vertices.size());
                mesh.vertices.push_back(vertex);
                uniqueVertices[key] = newIndex;
                mesh.indices.push_back(newIndex);
            }
        }

        return mesh;
    }

    Primitives::IndexedMesh LoadGltf(const std::string& path)
    {
        cgltf_options options{};
        cgltf_data* data = nullptr;

        cgltf_result result = cgltf_parse_file(&options, path.c_str(), &data);
        if (result != cgltf_result_success)
        {
            throw std::runtime_error("Eden: failed to parse glTF '" + path + "' (cgltf_result "
                                      + std::to_string(static_cast<int>(result)) + ")");
        }

        // Resolves and loads external .bin buffers referenced by a .gltf
        // file (paths relative to the .gltf file itself), or unpacks the
        // embedded binary chunk of a .glb - either way, after this call
        // every accessor's data is actually readable, not just described.
        result = cgltf_load_buffers(&options, data, path.c_str());
        if (result != cgltf_result_success)
        {
            cgltf_free(data);
            throw std::runtime_error("Eden: failed to load glTF buffers for '" + path + "' (cgltf_result "
                                      + std::to_string(static_cast<int>(result)) + ")");
        }

        Primitives::IndexedMesh mesh;

        for (cgltf_size meshIndex = 0; meshIndex < data->meshes_count; ++meshIndex)
        {
            const cgltf_mesh& gltfMesh = data->meshes[meshIndex];

            for (cgltf_size primIndex = 0; primIndex < gltfMesh.primitives_count; ++primIndex)
            {
                const cgltf_primitive& primitive = gltfMesh.primitives[primIndex];

                if (primitive.type != cgltf_primitive_type_triangles || primitive.indices == nullptr)
                {
                    // Skip anything that isn't an indexed triangle list -
                    // Eden's mesh format assumes triangles throughout (the
                    // same assumption LoadObj makes by forcing
                    // triangulate=true). Non-indexed primitives exist in
                    // the glTF spec but are uncommon enough from real
                    // exporters not to be worth a second code path here.
                    continue;
                }

                const cgltf_accessor* positionAccessor = nullptr;
                const cgltf_accessor* normalAccessor = nullptr;
                const cgltf_accessor* texcoordAccessor = nullptr;

                for (cgltf_size attrIndex = 0; attrIndex < primitive.attributes_count; ++attrIndex)
                {
                    const cgltf_attribute& attribute = primitive.attributes[attrIndex];

                    if (attribute.type == cgltf_attribute_type_position)
                    {
                        positionAccessor = attribute.data;
                    }
                    else if (attribute.type == cgltf_attribute_type_normal)
                    {
                        normalAccessor = attribute.data;
                    }
                    else if (attribute.type == cgltf_attribute_type_texcoord && attribute.index == 0)
                    {
                        // index == 0: glTF allows multiple UV sets
                        // (TEXCOORD_0, TEXCOORD_1, ...) for multi-texture
                        // materials - Eden only has one texture slot per
                        // mesh, so only the first set is meaningful here.
                        texcoordAccessor = attribute.data;
                    }
                }

                if (positionAccessor == nullptr)
                {
                    continue; // no position data - nothing to draw from this primitive
                }

                uint32_t baseVertex = static_cast<uint32_t>(mesh.vertices.size());

                for (cgltf_size v = 0; v < positionAccessor->count; ++v)
                {
                    Vertex vertex{};

                    float position[3] = { 0.0f, 0.0f, 0.0f };
                    cgltf_accessor_read_float(positionAccessor, v, position, 3);
                    vertex.position = { position[0], position[1], position[2] };

                    // Same reasoning as LoadObj: glTF has no standard
                    // per-vertex color most exporters populate, default to
                    // white so a bound texture/ColorComponent shows true.
                    vertex.color = { 1.0f, 1.0f, 1.0f };

                    if (normalAccessor != nullptr && v < normalAccessor->count)
                    {
                        float normal[3] = { 0.0f, 1.0f, 0.0f };
                        cgltf_accessor_read_float(normalAccessor, v, normal, 3);
                        vertex.normal = { normal[0], normal[1], normal[2] };
                    }
                    else
                    {
                        // Same documented fallback as LoadObj's missing-
                        // normal case - picks "up" rather than computing a
                        // real flat per-face normal, so lighting doesn't
                        // read garbage/uninitialized data at the cost of
                        // visibly wrong shading on such a (rare) model.
                        vertex.normal = { 0.0f, 1.0f, 0.0f };
                    }

                    if (texcoordAccessor != nullptr && v < texcoordAccessor->count)
                    {
                        float texcoord[2] = { 0.0f, 0.0f };
                        cgltf_accessor_read_float(texcoordAccessor, v, texcoord, 2);
                        // glTF's UV origin is already top-left, same
                        // convention Vulkan uses - unlike OBJ, no V-flip
                        // needed here.
                        vertex.texCoord = { texcoord[0], texcoord[1] };
                    }
                    else
                    {
                        vertex.texCoord = { 0.0f, 0.0f };
                    }

                    mesh.vertices.push_back(vertex);
                }

                // glTF vertex data is already deduplicated by the exporter
                // (one position+normal+uv per unique vertex, referenced by
                // repeated indices) - unlike OBJ's per-face-corner scheme,
                // there's no dedup work to do here, just offset each
                // primitive's indices by how many vertices came before it
                // in the merged mesh.
                for (cgltf_size i = 0; i < primitive.indices->count; ++i)
                {
                    cgltf_size index = cgltf_accessor_read_index(primitive.indices, i);
                    mesh.indices.push_back(baseVertex + static_cast<uint32_t>(index));
                }
            }
        }

        cgltf_free(data);

        if (mesh.vertices.empty())
        {
            throw std::runtime_error("Eden: glTF '" + path + "' contained no usable triangle geometry");
        }

        return mesh;
    }
}
