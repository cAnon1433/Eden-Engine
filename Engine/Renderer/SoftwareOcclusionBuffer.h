#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <limits>

namespace Eden
{
    // CPU-side software occlusion culling, in the spirit of the classic
    // "rasterize your big occluders into a small depth buffer, test
    // everything else against it before it ever reaches the GPU"
    // technique (the approach behind Vercidium's well-known optimization
    // videos, among others). Separate from and complementary to Frustum
    // culling (see Frustum.h) - frustum culling answers "is this even in
    // view", this answers "is this in view but completely hidden behind
    // something else that's already going to be drawn". Both are coarse,
    // conservative tests: neither needs to be pixel-exact, both only
    // need to never wrongly cull something actually visible.
    //
    // DEPTH CONVENTION: this buffer stores VIEW-SPACE depth (distance
    // along the camera's own forward axis - see ProjectAabb below), not
    // NDC/clip-space z. Deliberately - matching Frustum's near/far
    // planes' own reasoning (see Frustum.h): whether this engine's
    // projection matrices target Vulkan's native [0,1] NDC depth range
    // or GLM's default OpenGL-style [-1,1] one isn't something visibly
    // settled anywhere in this codebase (no GLM_FORCE_DEPTH_ZERO_TO_ONE,
    // no vertex-shader remap), so relying on that convention here would
    // be the same landmine again. View-space depth needs no such
    // assumption - it's just "distance from the camera along where it's
    // looking", always well-defined, and this buffer is used ENTIRELY
    // standalone (written and read only by this class), so it never
    // needs to agree with the GPU's actual depth buffer values, only be
    // internally consistent with itself.
    //
    // RESOLUTION: deliberately low (see Renderer's construction of this
    // class) - this is a coarse conservative test, not a second real
    // depth buffer. A handful of grid cells per occluder's footprint is
    // enough to reliably catch "this object is genuinely, substantially
    // hidden"; it does not need to catch partial/edge-case occlusion,
    // and erring toward "still counts as visible" at the margins is the
    // safe direction to err in (same philosophy as Frustum::ContainsSphere
    // treating anything not PROVABLY outside as visible).
    class SoftwareOcclusionBuffer
    {
    public:
        void Resize(int width, int height)
        {
            m_Width = width;
            m_Height = height;
            m_Depths.assign(static_cast<size_t>(width) * height, std::numeric_limits<float>::max());
        }

        void Clear()
        {
            std::fill(m_Depths.begin(), m_Depths.end(), std::numeric_limits<float>::max());
        }

        // Projects a world-space AABB (given as min/max corners) into
        // this buffer's grid space. Returns false if the box is behind
        // the camera or otherwise couldn't be projected into a valid
        // screen rectangle - callers should treat that as "can't say,
        // assume visible" (frustum culling already handles genuinely
        // off-screen/behind-camera cases; this function isn't trying to
        // duplicate that).
        //
        // `outNearDepth` is the box's own nearest view-space depth
        // (the closest of its 8 corners to the camera along the forward
        // axis) - what RasterizeOccluder writes into covered cells, and
        // what IsVisible compares covered cells against.
        struct ScreenRect
        {
            int minX, minY, maxX, maxY; // inclusive grid-cell bounds
            float nearDepth;
        };

        bool ProjectAabb(const glm::vec3& worldMin, const glm::vec3& worldMax,
                          const glm::mat4& viewProjection, const glm::vec3& cameraPosition, const glm::vec3& cameraForward,
                          ScreenRect& outRect) const
        {
            glm::vec3 corners[8] =
            {
                { worldMin.x, worldMin.y, worldMin.z }, { worldMax.x, worldMin.y, worldMin.z },
                { worldMin.x, worldMax.y, worldMin.z }, { worldMax.x, worldMax.y, worldMin.z },
                { worldMin.x, worldMin.y, worldMax.z }, { worldMax.x, worldMin.y, worldMax.z },
                { worldMin.x, worldMax.y, worldMax.z }, { worldMax.x, worldMax.y, worldMax.z },
            };

            float minNdcX = std::numeric_limits<float>::max(), maxNdcX = -std::numeric_limits<float>::max();
            float minNdcY = std::numeric_limits<float>::max(), maxNdcY = -std::numeric_limits<float>::max();
            float nearestDepth = std::numeric_limits<float>::max();
            bool anyInFront = false;

            for (const glm::vec3& corner : corners)
            {
                float viewSpaceDepth = glm::dot(corner - cameraPosition, cameraForward);
                if (viewSpaceDepth <= 0.0f)
                {
                    // Behind (or exactly at) the camera - this corner's
                    // clip-space x/y isn't meaningful (a perspective
                    // divide by a non-positive w is exactly the kind of
                    // thing that produces garbage screen coordinates).
                    // Skip it for the NDC bounds, but don't invalidate
                    // the whole box - a box straddling the camera plane
                    // can still have some corners meaningfully in front.
                    continue;
                }
                anyInFront = true;
                nearestDepth = glm::min(nearestDepth, viewSpaceDepth);

                glm::vec4 clip = viewProjection * glm::vec4(corner, 1.0f);
                if (clip.w <= 0.0f)
                {
                    continue;
                }
                float ndcX = clip.x / clip.w;
                float ndcY = clip.y / clip.w;
                minNdcX = glm::min(minNdcX, ndcX);
                maxNdcX = glm::max(maxNdcX, ndcX);
                minNdcY = glm::min(minNdcY, ndcY);
                maxNdcY = glm::max(maxNdcY, ndcY);
            }

            if (!anyInFront || minNdcX > maxNdcX)
            {
                return false; // entirely behind the camera, or degenerate
            }

            // NDC x/y in [-1, 1] -> grid cell in [0, width)/[0, height) -
            // this mapping only relies on the LEFT/RIGHT/TOP/BOTTOM
            // clip planes' x/y being in [-1,1] post-divide, which (per
            // Frustum.h's own comment) is true regardless of which
            // depth-range convention the projection targets. Clamped to
            // the buffer's actual bounds - a box that partially extends
            // off-screen still contributes/tests against its on-screen
            // portion.
            auto ndcToGridX = [this](float ndc) { return glm::clamp(static_cast<int>((ndc * 0.5f + 0.5f) * m_Width), 0, m_Width - 1); };
            // NDC y is +1 = up, but grid row 0 is conventionally the
            // top - flip here so RasterizeOccluder/IsVisible's row
            // indexing matches normal image-space expectations without
            // every caller needing to remember the flip themselves.
            auto ndcToGridY = [this](float ndc) { return glm::clamp(static_cast<int>((1.0f - (ndc * 0.5f + 0.5f)) * m_Height), 0, m_Height - 1); };

            outRect.minX = ndcToGridX(minNdcX);
            outRect.maxX = ndcToGridX(maxNdcX);
            outRect.minY = ndcToGridY(maxNdcY); // maxNdcY (top of box) -> smaller grid row
            outRect.maxY = ndcToGridY(minNdcY); // minNdcY (bottom of box) -> larger grid row
            outRect.nearDepth = nearestDepth;
            return true;
        }

        // Writes `rect.nearDepth` into every covered cell, but only
        // where it's actually nearer than what's already stored (an
        // occluder behind another occluder shouldn't overwrite the
        // nearer one's depth) - standard "closer wins" depth-buffer
        // semantics.
        void RasterizeOccluder(const ScreenRect& rect)
        {
            for (int y = rect.minY; y <= rect.maxY; ++y)
            {
                for (int x = rect.minX; x <= rect.maxX; ++x)
                {
                    float& cell = m_Depths[static_cast<size_t>(y) * m_Width + x];
                    cell = glm::min(cell, rect.nearDepth);
                }
            }
        }

        // True if `rect` is NOT provably fully occluded - i.e. at least
        // one covered cell has no occluder nearer than rect.nearDepth,
        // meaning some part of this box could genuinely be visible
        // through/around whatever's been rasterized as an occluder so
        // far. False only when EVERY covered cell is occluded nearer
        // than this box's own nearest point - a real, conservative
        // "definitely hidden" conclusion, not a guess.
        bool IsVisible(const ScreenRect& rect) const
        {
            for (int y = rect.minY; y <= rect.maxY; ++y)
            {
                for (int x = rect.minX; x <= rect.maxX; ++x)
                {
                    float cell = m_Depths[static_cast<size_t>(y) * m_Width + x];
                    if (rect.nearDepth < cell)
                    {
                        return true; // this cell has no nearer occluder - box pokes through here
                    }
                }
            }
            return false; // every covered cell has a nearer occluder - fully hidden
        }

        int Width() const { return m_Width; }
        int Height() const { return m_Height; }

    private:
        int m_Width = 0;
        int m_Height = 0;
        std::vector<float> m_Depths;
    };
}
