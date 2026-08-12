#pragma once

#include <glm/glm.hpp>

namespace Eden::SPH
{
    // Standard WCSPH (weakly-compressible SPH) smoothing kernels, after
    // Muller, Charypar & Gross 2003. Three different kernels are used for
    // three different purposes rather than one kernel for everything -
    // Poly6 is cheap and smooth (good for density, which just needs a
    // plausible falloff), but its gradient goes to zero at the center of
    // the kernel, which makes it useless for pressure force (two
    // coincident particles would compute zero repulsion right when they
    // need it most). Spiky's gradient is linear and never zero for r > 0,
    // which is exactly the "always push apart" behavior pressure needs.
    // Viscosity uses a third kernel because its Laplacian (not gradient)
    // is what the diffusion term needs, and Poly6/Spiky's Laplacians
    // aren't well-behaved for that purpose.
    //
    // All three assume `h` (smoothingRadius) is constant across the
    // simulation - not per-particle variable smoothing lengths, which is
    // a real technique but out of scope for a first working solver.

    // Poly6: used for density estimation.
    // W(r,h) = 315 / (64 * pi * h^9) * (h^2 - r^2)^3, for 0 <= r <= h
    inline float Poly6(float r, float h)
    {
        if (r > h)
        {
            return 0.0f;
        }

        float h2 = h * h;
        float r2 = r * r;
        float diff = h2 - r2;
        constexpr float PI = 3.14159265358979323846f;
        float coefficient = 315.0f / (64.0f * PI * glm::pow(h, 9.0f));
        return coefficient * diff * diff * diff;
    }

    // Spiky gradient: used for pressure force. Returns the full gradient
    // vector (already includes direction), not just the scalar magnitude
    // - callers should NOT re-multiply by a direction vector.
    // grad W(r,h) = -45 / (pi * h^6) * (h - r)^2 * (rVec / r), for 0 < r <= h
    inline glm::vec3 SpikyGradient(const glm::vec3& rVec, float r, float h)
    {
        if (r <= 0.0f || r > h)
        {
            return glm::vec3(0.0f);
        }

        constexpr float PI = 3.14159265358979323846f;
        float coefficient = -45.0f / (PI * glm::pow(h, 6.0f));
        float diff = h - r;
        return coefficient * diff * diff * (rVec / r);
    }

    // Viscosity Laplacian: used for the viscosity (velocity diffusion)
    // force. Unlike Poly6/Spiky above, this one is specifically derived
    // to have a Laplacian that stays well-behaved (doesn't go negative
    // near r = h the way Poly6's does), which is what makes it the
    // conventional choice for viscosity specifically rather than reusing
    // Poly6 here too.
    // lap W(r,h) = 45 / (pi * h^6) * (h - r), for 0 <= r <= h
    inline float ViscosityLaplacian(float r, float h)
    {
        if (r > h)
        {
            return 0.0f;
        }

        constexpr float PI = 3.14159265358979323846f;
        float coefficient = 45.0f / (PI * glm::pow(h, 6.0f));
        return coefficient * (h - r);
    }
}
