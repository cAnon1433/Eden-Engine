#pragma once

#include "ParticleData.h"

#include <glm/glm.hpp>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cstdint>

namespace Eden
{
    // Uniform grid neighbor search for SPH, separate from
    // CollisionSystem's broad-phase spatial hash. Deliberately not
    // shared/generalized between the two: CollisionSystem's grid is sized
    // adaptively around collider extents and exists to cheaply reject
    // non-overlapping AABB pairs; this one is sized to exactly the SPH
    // smoothing radius and exists to answer "which particles are within h
    // of this point" as directly as possible. Forcing one grid to serve
    // both would mean either compromising the cell size for both use
    // cases or adding a parallel sizing mode - not worth the coupling
    // risk to a physics module that's already working, for a data
    // structure this small to write standalone.
    //
    // Cell size is fixed at the smoothing radius `h`: any two particles
    // within h of each other are guaranteed to fall in the same cell or
    // one of the 26 adjacent cells, which is exactly the 3x3x3
    // neighborhood ForEachNeighbor below checks.
    //
    // COUNTING-SORT LAYOUT (revised from an earlier per-cell
    // unordered_map<CellKey, vector<uint32_t>> version): that version
    // rebuilt N separate small vectors from scratch every substep, each
    // one a fresh heap allocation the moment m_Cells.clear() destroyed
    // the previous ones - real, measured cost even at a few hundred
    // particles x 4 substeps, the same class of hot-path allocation
    // churn as the ECS's GetStorage-per-entity bug (see Registry.h).
    // This version keeps exactly two flat, reused vectors
    // (m_CellKeys/m_SortedIndices) that only reallocate when particle
    // count grows past previous capacity - a counting/bucket sort by
    // cell key, not N small containers. The occupied-cell lookup itself
    // is still a hash map (m_CellRanges), but its values are now a
    // cheap {start, count} pair into the flat sorted array rather than
    // an owned vector, so map node churn no longer drags a container
    // allocation along with it.
    class ParticleSpatialHash
    {
    public:
        void Build(const ParticleData& particles, float cellSize)
        {
            m_CellSize = cellSize;
            size_t n = particles.Count();

            // resize(), not clear()+push_back(): reuses existing capacity
            // across calls (particle count is stable frame-to-frame
            // except on Emit/Clear), so this is a no-op allocation-wise
            // in the common case instead of N fresh allocations.
            m_CellKeys.resize(n);
            m_SortedIndices.resize(n);

            for (size_t i = 0; i < n; ++i)
            {
                m_CellKeys[i] = KeyForPosition(particles.positions[i]);
                m_SortedIndices[i] = static_cast<uint32_t>(i);
            }

            // Sorts particle INDICES by cell key, not the particles
            // themselves - ParticleData's arrays never get reordered by
            // this. std::sort over a flat uint32_t array is cache-
            // friendly and allocation-free (in-place); this is the one
            // O(n log n) cost per substep, which is comfortably cheaper
            // than the up-to-27 hash lookups per particle the old
            // version paid on top of its allocation churn.
            std::sort(m_SortedIndices.begin(), m_SortedIndices.end(),
                [this](uint32_t a, uint32_t b) { return m_CellKeys[a] < m_CellKeys[b]; });

            // clear() on an unordered_map destroys elements but
            // (libstdc++/libc++) typically keeps the bucket array
            // allocated - reused capacity across calls, same reasoning
            // as the vectors above. Entries here number one per OCCUPIED
            // cell, not one per particle (typically several particles
            // share a cell at this cellSize), so this was already cheap
            // relative to the old per-particle vector churn even before
            // this rewrite.
            m_CellRanges.clear();

            size_t rangeStart = 0;
            for (size_t i = 0; i < n; ++i)
            {
                bool isLast = (i + 1 == n);
                bool keyChanges = isLast || m_CellKeys[m_SortedIndices[i]] != m_CellKeys[m_SortedIndices[i + 1]];

                if (keyChanges)
                {
                    CellKey key = m_CellKeys[m_SortedIndices[i]];
                    uint32_t count = static_cast<uint32_t>(i + 1 - rangeStart);
                    m_CellRanges[key] = CellRange{ static_cast<uint32_t>(rangeStart), count };
                    rangeStart = i + 1;
                }
            }
        }

        // Invokes callback(neighborIndex) for every particle whose cell is
        // within the 3x3x3 block centered on `position`'s own cell -
        // callers still need their own distance check against `h` since
        // this only narrows to "same or adjacent cell", not "actually
        // within h" (a particle near a cell's far corner can be in an
        // adjacent cell but farther than h away, or in the same cell but
        // that doesn't guarantee within h either - the 3x3x3 block is a
        // superset, not an exact answer).
        template<typename Callback>
        void ForEachNeighbor(const glm::vec3& position, Callback callback) const
        {
            glm::ivec3 center = CellCoordFor(position);

            for (int dz = -1; dz <= 1; ++dz)
            {
                for (int dy = -1; dy <= 1; ++dy)
                {
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        CellKey key = MakeKey(center.x + dx, center.y + dy, center.z + dz);
                        auto it = m_CellRanges.find(key);
                        if (it == m_CellRanges.end())
                        {
                            continue;
                        }

                        // Contiguous slice of one shared flat array,
                        // rather than a separately heap-allocated
                        // per-cell vector - the actual cache-locality
                        // win from the counting-sort layout, on top of
                        // the reduced allocation count above.
                        uint32_t start = it->second.start;
                        uint32_t end = start + it->second.count;
                        for (uint32_t i = start; i < end; ++i)
                        {
                            callback(m_SortedIndices[i]);
                        }
                    }
                }
            }
        }

    private:
        using CellKey = int64_t;

        struct CellRange
        {
            uint32_t start = 0;
            uint32_t count = 0;
        };

        glm::ivec3 CellCoordFor(const glm::vec3& position) const
        {
            return glm::ivec3(
                static_cast<int>(glm::floor(position.x / m_CellSize)),
                static_cast<int>(glm::floor(position.y / m_CellSize)),
                static_cast<int>(glm::floor(position.z / m_CellSize)));
        }

        // Packs three 20-bit signed-ish coordinates into one int64 key.
        // Offsetting by a large constant before packing turns the
        // negative-coordinate case into plain unsigned ranges, avoiding
        // sign-extension collisions between e.g. (-1, 0, 0) and some
        // unrelated positive coordinate. 1u << 20 (~1,048,576 cells in
        // each axis, centered on the origin) is comfortably larger than
        // any scene this project places by hand.
        static constexpr int64_t COORD_OFFSET = 1 << 20;

        static CellKey MakeKey(int x, int y, int z)
        {
            int64_t ux = static_cast<int64_t>(x) + COORD_OFFSET;
            int64_t uy = static_cast<int64_t>(y) + COORD_OFFSET;
            int64_t uz = static_cast<int64_t>(z) + COORD_OFFSET;
            return (ux << 42) | (uy << 21) | uz;
        }

        CellKey KeyForPosition(const glm::vec3& position) const
        {
            glm::ivec3 c = CellCoordFor(position);
            return MakeKey(c.x, c.y, c.z);
        }

        float m_CellSize = 1.0f;
        std::vector<CellKey> m_CellKeys;       // per-particle, same order as ParticleData
        std::vector<uint32_t> m_SortedIndices; // particle indices, sorted by cell key
        std::unordered_map<CellKey, CellRange> m_CellRanges;
    };
}
