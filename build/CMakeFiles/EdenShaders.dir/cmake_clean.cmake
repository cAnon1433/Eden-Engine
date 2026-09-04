file(REMOVE_RECURSE
  "CMakeFiles/EdenShaders"
  "Shaders/Compiled/fluid_blur.frag.spv"
  "Shaders/Compiled/fluid_composite.frag.spv"
  "Shaders/Compiled/fluid_depth.frag.spv"
  "Shaders/Compiled/fluid_depth.vert.spv"
  "Shaders/Compiled/particle_build_grid.comp.spv"
  "Shaders/Compiled/particle_density.comp.spv"
  "Shaders/Compiled/particle_force.comp.spv"
  "Shaders/Compiled/particle_integrate.comp.spv"
  "Shaders/Compiled/particle_point.frag.spv"
  "Shaders/Compiled/particle_point.vert.spv"
  "Shaders/Compiled/particle_point_gpu.vert.spv"
  "Shaders/Compiled/raymarch.frag.spv"
  "Shaders/Compiled/raymarch.vert.spv"
  "Shaders/Compiled/triangle.frag.spv"
  "Shaders/Compiled/triangle.vert.spv"
  "Shaders/Compiled/voxel_march.comp.spv"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/EdenShaders.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
