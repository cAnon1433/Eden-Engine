# Eden - Renderer (Triangle Milestone)

Custom Vulkan renderer for the Eden engine. This checkpoint clears the
screen and draws a hardcoded triangle - confirms the whole Vulkan
bring-up chain (instance -> device -> swapchain -> render pass ->
pipeline -> command submission -> present) actually works end to end.

This was built/compile-tested on Linux (X11, Mesa) before being handed
to you - logic and syntax are verified, but it has NOT been run on
macOS/MoltenVK. If instance or device creation throws on your machine,
that's the first place to check (see MoltenVK note below).

## Dependencies

- Vulkan SDK (LunarG - includes MoltenVK on macOS, validation layers, glslangValidator/glslc)
- GLFW (`brew install glfw`)
- GLM (`brew install glm`)
- CMake (`brew install cmake`)

## Build

```bash
mkdir build && cd build
cmake ..
make
```

CMake will fail loudly if it can't find `glslangValidator`/`glslc` -
make sure `VULKAN_SDK` is set in your shell (the LunarG installer's
`setup-env.sh` does this).

## Run

Compiled shaders (`Shaders/Compiled/*.spv`) are built into `build/Shaders/Compiled`
alongside the executable, so run it from inside `build/`:

```bash
cd build
./Eden
```

## What's implemented

Everything from the triangle milestone, plus:

- `VulkanMemoryAllocator` - wraps VMA (vendored in `ThirdParty/vma/`, actually in use now)
- `VulkanBuffer` - generic VMA-backed buffer, plus a staged-upload path
  (`InitDeviceLocalWithData`) for data that's set once and read every
  frame: host-visible staging buffer -> one-time command buffer copy ->
  fast device-local buffer
- `Vertex` struct (`RendererTypes.h`) with real
  `VkVertexInputBindingDescription`/`VkVertexInputAttributeDescription` -
  position + color, no longer hardcoded in the shader
- The triangle's 3 vertices are now real data bound via
  `vkCmdBindVertexBuffers`, not `gl_VertexIndex` lookups

Runtime-tested (not just compiled) against Mesa's lavapipe software
Vulkan implementation with validation layers enabled - ran multiple
seconds of continuous frame submission with zero VUID errors. Still
untested on actual MoltenVK/macOS hardware.

## What's still a stub

`Resources/Image`, `Resources/Texture`, `Resources/Sampler`,
`Pipeline/ComputePipeline`, `Pipeline/DescriptorSetLayout`,
`Pipeline/DescriptorPool`, `RenderGraph/*`, `Interface/*`. No camera/MVP
transform yet either - that's the next planned milestone (with a depth
buffer, since you can't have one without the other).

## MoltenVK notes (macOS)

- `VulkanInstance` conditionally adds `VK_KHR_portability_enumeration`
  + the `VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR` flag under
  `#ifdef __APPLE__`.
- `VulkanDevice` checks whether the selected physical device exposes
  `VK_KHR_portability_subset` and enables it if so (required by spec
  on MoltenVK, must not be requested on platforms that don't have it).
- Some Vulkan features/extensions common on Windows/Linux discrete
  GPUs simply don't exist under MoltenVK. Not an issue for this
  triangle, but keep it in mind as you build out further.
