# Eden

A custom Vulkan-based 3D engine, built from scratch in C++ around an
Entity-Component-System (ECS) architecture. Eden isn't aimed at being a
general-purpose game engine — it's being built as a self-contained
simulation environment, with a longer-term goal of hosting an AI
"hivemind" agent layer inside it.

This is a **v0.01 demo build**. It is not feature-complete, not
optimized for a release build, and will have bugs — see "What to expect"
below before you dive in.

## AI use disclosure

Eden's code has been developed with substantial AI assistance (Claude,
by Anthropic) working alongside the project owner throughout design,
implementation, and debugging. This isn't a fully hand-written solo
project, and isn't a fully AI-generated one either — it's a
collaborative process the same way a developer might work with a
pair-programmer. Flagging this up front so it's never a surprise later.

## What Eden currently does

- **Vulkan renderer**: full bring-up (instance, device, swapchain,
  render pass, pipeline), VMA-backed memory management, instanced mesh
  rendering (entities sharing a mesh draw in one call, not one draw
  call per object), Blinn-Phong lighting with normals, texture support,
  OBJ and glTF/.glb model loading, frustum culling, and CPU software
  occlusion culling.
- **ECS**: generational entity handles (so a destroyed/reused entity ID
  can't silently resolve to the wrong entity), sparse-set component
  storage, and a working component/system set covering transforms,
  meshes, color overrides, visibility, lifetime, naming, spawners, and
  parent/child hierarchy.
- **Physics**: rigid bodies with semi-implicit Euler integration,
  analytic SDF collision primitives (sphere/box/capsule/plane), impulse
  resolution with friction, sleep/wake islands, and continuous collision
  detection to stop fast-moving objects tunneling through thin
  colliders.
- **GPU particle/fluid simulation**: a full SPH (smoothed-particle
  hydrodynamics) fluid solver running on the GPU via compute shaders —
  32,000+ particles with no CPU-side lag.
- **Deformable SDF/voxel geometry with solid↔liquid state changes**:
  hardened objects are carvable in real time (marching-cubes surface,
  re-triangulated live). Melt a targeted volume into SPH fluid particles,
  let them splash and settle, then reform them back into a solid — the
  reformed shape is whatever the particles actually settled into, not a
  reset to the original. Reformed and original volumes both get real
  collision (sphere/box/capsule against the voxel field), not just
  visuals.
- **In-engine editor (ImGui)**: spawn cubes and textured cubes with
  full transform control, load OBJ/glTF models, select and inspect
  existing entities, and add/remove components live.

## What's NOT in yet (so bug reports on these aren't useful)

- No adjustable/placeable lights — lighting is one hardcoded
  directional light.
- No mesh-swapping in the editor — you can create and destroy entities,
  but not change an existing entity's geometry/texture after creation.
- glTF loading is simplified: no node hierarchy, no per-primitive
  materials, static geometry only.
- Parent/child transform propagation exists in code but isn't wired
  into the main loop yet — moving a parent won't currently move its
  children.
- No joints/constraints in physics.
- No "scenes" concept yet (still an open design question) — nothing
  saves or loads.
- No AI/agent layer at all yet — that's explicitly 1-2 years out and
  waiting on hardware.
- Melting is a manual keypress (`M`), not driven by heat/damage — there's
  no environmental heat system yet.
- Melt/reform transitions snap instantly — no visual blend between the
  solid and liquid states.
- Fluid is single-material — no mixing/reacting between different
  liquids.
- Voxel volumes deleted through the editor now free most of their
  memory, but a fixed-size internal budget (64 volumes total, ever, in
  one run) isn't reclaimed yet — if you melt/reform repeatedly enough
  times in one session, expect an eventual crash on hitting that cap.
  Worth noting in a bug report if you were doing a lot of melt/reform
  cycling right before it happened.

## What to expect from this build

This is a stress-test / bug-hunt build, not a polished demo. Please try
to break it: spawn a lot of entities, load different models, mess with
the fluid sim, resize the window mid-run, alt-tab, disconnect/reconnect
things if you can. If it crashes, hangs, renders garbage, or does
something visually wrong, that's useful information — please note what
you were doing right before it happened.

This build has been developed and tested primarily on **macOS (Apple
Silicon, via MoltenVK)**. The code itself doesn't contain any
Mac-specific logic beyond the required MoltenVK compatibility shims
(which are conditionally compiled and don't affect Windows/Linux), and
the CMake setup targets Vulkan/GLFW/GLM generically rather than any one
platform. That said, **Windows and Linux builds have not actually been
run yet** — if you're testing on either, you may be the first to hit a
platform-specific issue, and that's exactly the kind of thing this
round of testing is for.

## How to run it

### macOS

1. Install dependencies (if you don't already have them):
   ```bash
   brew install cmake glfw glm
   ```
   Also install the [Vulkan SDK](https://vulkan.lunarg.com/) (LunarG) —
   this includes MoltenVK and the shader compiler.
2. Double-click **`Run Eden (Mac).command`**.

### Windows

1. Install the [Vulkan SDK](https://vulkan.lunarg.com/) (LunarG).
2. Install CMake and a C++ compiler (Visual Studio 2019/2022, or the
   Build Tools package).
3. Install GLFW and GLM via [vcpkg](https://github.com/microsoft/vcpkg):
   ```
   git clone https://github.com/microsoft/vcpkg
   .\vcpkg\bootstrap-vcpkg.bat
   .\vcpkg\vcpkg install glfw3 glm
   setx VCPKG_ROOT "C:\path\to\vcpkg"
   ```
   (open a new terminal after `setx` so it picks up the variable)
4. Double-click **`Run Eden (Windows).bat`**.

### Linux

1. Install dependencies, e.g. on Debian/Ubuntu:
   ```bash
   sudo apt install cmake build-essential libglfw3-dev libglm-dev \
                    libvulkan-dev vulkan-tools glslang-tools
   ```
   (or install LunarG's Vulkan SDK directly and make sure `VULKAN_SDK`
   is set)
2. Run **`Run Eden (Linux).sh`** — some file managers won't execute
   `.sh` files on double-click by default; if that happens, either
   enable "allow executing as program" in the file's properties, or run
   it from a terminal:
   ```bash
   ./"Run Eden (Linux).sh"
   ```

All three scripts do the same thing: configure with CMake, build, and
launch — safe to re-run any time, they only rebuild what changed. If a
build fails, the script prints the full log so you can copy/paste it
when reporting a bug.

## Controls

| Input | Action |
|---|---|
| `W` `A` `S` `D` | Move camera forward/left/back/right |
| `Space` | Move camera up |
| `Left Ctrl` | Move camera down |
| Mouse | Look around (while cursor is locked) |
| Left-click | Carve whatever hardened (voxel) surface you're aiming at. Hold `Shift` to drill through multiple layers |
| `M` | Melt the hardened volume you're aiming at into fluid particles |
| `H` | Reform sufficiently-cooled, clustered fluid particles back into a hardened volume |
| `Tab` | Toggle between camera-look and UI-interaction mode |
| `Esc` | Quit |

Camera movement and mouse-look only respond while the cursor is locked
(the default on startup). Press `Tab` to unlock the cursor and interact
with the ImGui panels — the editor lets you spawn entities, load
models, and inspect/edit anything currently in the scene.

## Reporting bugs

For each issue, if you can, note: what platform you're on, what you did
right before it happened, and anything printed to the terminal window
(it stays open — don't close it, that's where errors show up).
