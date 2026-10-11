# Eden

A custom Vulkan-based 3D engine, built from scratch in C++ around an
Entity-Component-System (ECS) architecture. Eden isn't aimed at being a
general-purpose game engine - it's being built as a self-contained
simulation environment: a world whose physical rules (matter, and later
energy, runes and living creatures) are enforced by the engine itself, with a
longer-term goal of hosting AI agents inside it.

This is a **v0.01 demo build**. It is not feature-complete, not tuned for
low-end hardware yet, and will have bugs - see "What to expect" and
"Known issues" below before you dive in.

## AI use disclosure

Eden's code has been developed with substantial AI assistance (Claude,
by Anthropic) working alongside the project owner throughout design,
implementation, and debugging. This isn't a fully hand-written solo
project, and isn't a fully AI-generated one either - it's a
collaborative process the same way a developer might work with a
pair-programmer. Flagging this up front so it's never a surprise later.

## What Eden currently does

- **Vulkan renderer**: instanced mesh rendering, Blinn-Phong lighting,
  textures, OBJ and glTF/.glb loading, frustum culling and CPU software
  occlusion culling, VMA-backed memory management.
- **ECS**: generational entity handles, sparse-set component storage, and
  components/systems for transforms, meshes, colors, visibility, lifetime,
  naming, spawners, and parent/child hierarchy.
- **Physics**: rigid bodies, analytic SDF colliders (sphere/box/capsule/plane),
  impulse resolution with friction, sleep/wake islands, continuous collision
  detection, and collision against voxel volumes.
- **GPU fluid simulation**: SPH (smoothed-particle hydrodynamics) running
  entirely in compute shaders, up to **131,072 particles**, with a hash-grid
  neighbor search and SDF boundary collision. How many you can run smoothly
  depends on your GPU and how much liquid is on screen - see "Performance".
- **Fluid rendering**: liquid is drawn as one continuous, smooth, partially
  transparent surface (screen-space reconstruction: depth splat -> bilateral
  blur -> polish -> temporal smoothing -> lit composite). Opacity comes from
  how much liquid a view ray passes through, so thin edges and lone droplets
  are see-through while deep pools read solid. Isolated droplets get their own
  shading so they look like round drops rather than flat discs. The opaque
  scene (terrain, solids) is rendered first and its depth is used by the fluid
  passes, so liquid behind a solid is hidden and costs almost nothing. An older
  "soft blobs" technique is still available as a checkbox for comparison.
- **Chemicals**: 8 editable liquid types (mass, rest density, viscosity,
  stiffness, cohesion, color, can-harden, hardening threshold), chosen at pour
  time. Different chemicals do **not** mix or react with each other yet.
- **Carvable voxel terrain and objects**: the ground is a field of 20 m x 5 m
  noise-generated terrain tiles (marching-cubes surface, re-triangulated live
  when carved) that stream in and out around the camera. Hardened objects are
  carvable the same way.
- **Solid <-> liquid state changes**: melt a targeted solid into SPH particles,
  let it splash and settle, then reform the cooled, clustered particles back
  into a solid. The reformed shape is whatever the liquid actually settled into,
  not a reset to the original. Melting frees the emptied volume. Reformed and
  original volumes both have real collision.
- **In-engine editor (ImGui)**: spawn cubes, load models, inspect and edit
  entities and their components live, tune fluid and physics settings.
- **Built-in profiler**: a Performance panel with frame-time graph, per-section
  CPU timings, optional per-pass GPU timings, a "slowest recent frame"
  breakdown, and a report you can save and paste into a bug report.

## What's NOT in yet (so bug reports on these aren't useful)

- No adjustable/placeable lights - one hardcoded directional light.
- No mesh-swapping in the editor - you can create and destroy entities, not
  change an existing entity's geometry/texture.
- glTF loading is simplified: no node hierarchy, no per-primitive materials,
  static geometry only.
- Parent/child transform propagation exists in code but isn't wired into the
  main loop yet.
- No joints/constraints in physics.
- **No saving or loading**, and no scene system - the world is regenerated
  each run. (The design is one continuous seed-generated world that persists;
  it isn't built.)
- Melting is a manual keypress (`M`), not driven by heat/damage; there is no
  heat system yet.
- Melt/reform transitions snap instantly - no visual blend.
- A finished solid can't be picked up or used as a tool yet, and voxel volumes
  don't collide with each other (only primitives vs. voxels).
- Chemicals don't react or mix, and there are no gas or energy systems yet.
- No creatures / AI agent layer yet.
- There is a hard limit of **64 voxel volumes alive at once** (terrain tiles
  count toward it). Freed volumes are reused, so melt/reform cycling is fine,
  but spawning very many volumes at once will hit the cap.

## Known issues (real ones - reports on these are welcome but expected)

- **Carving can still hitch** a little, especially when holding the button
  down. It is much better than the first build, but not gone, and it is most
  noticeable on slower CPUs.
- Carve accuracy is rough: very fine detail depends on the voxel resolution.
- Particle liquid physics is not finished - behavior at high particle counts or
  extreme settings can be unstable.
- The most recent round of stability fixes (GPU synchronization) hasn't been
  stress-tested extensively. If you hit a crash while melting, reforming and
  destroying repeatedly, please report exactly what you were doing.
- Lowering "Fluid resolution" (see Performance) makes liquid silhouettes
  slightly stair-stepped; this is expected.
- Liquid is drawn as a single surface (the nearest liquid at each pixel), so
  where a thin or see-through blob sits in front of *other* liquid you will see
  the scene behind it, not the liquid behind it. This is a limitation of the
  technique; fixing it would need a second depth layer.
- Because the liquid's on-screen rectangle is read back from the GPU a couple of
  frames late, liquid that comes into view can take 2-3 frames to appear, and
  turning very fast toward liquid can briefly clip it at a hard edge.
- Getting very close to liquid (so a few particles fill the screen) is the
  most expensive case for the GPU, because every fluid pass touches every
  screen pixel. Use a lower "Fluid resolution" if that's slow on your GPU.

## What to expect from this build

This is a stress-test / bug-hunt build, not a polished demo. Please try to
break it: pour a lot of liquid, carve, melt and reform, load models, resize
the window mid-run, alt-tab. If it crashes, hangs, renders garbage, or does
something visually wrong, that's useful - please note what you were doing
right before it happened.

Eden is developed and tested mainly on **macOS (Apple Silicon, via
MoltenVK)**. **Windows and Linux are supported by the build scripts but have had
much less real-world testing** - on Linux the build has been compiled and run
(including under a software Vulkan driver with validation layers on), but
neither has been tried on a wide range of real GPUs. If you're on either, you
may be the first to hit a platform-specific issue; that's exactly what this
round of testing is for.

## How to run it

All three scripts do the same thing: configure with CMake, build, and
launch. They are safe to re-run - they only rebuild what changed - and if a
build fails they print the full log so you can copy it into a bug report.
**GLFW and GLM are bundled in `ThirdParty/` - you do not need to install them.**

### macOS

1. Install CMake (`brew install cmake`) and the
   [Vulkan SDK](https://vulkan.lunarg.com/) from LunarG (includes MoltenVK
   and the shader compiler).
2. Double-click **`Run Eden (Mac).command`**.

### Windows

1. Install the [Vulkan SDK](https://vulkan.lunarg.com/) (LunarG).
2. Install CMake and Visual Studio 2019/2022 or the Build Tools package
   with the "Desktop development with C++" workload.
3. Double-click **`Run Eden (Windows).bat`**.

### Linux

1. Install dependencies, e.g. on Debian/Ubuntu:
   ```bash
   sudo apt install cmake build-essential libvulkan-dev vulkan-tools \
                    glslang-tools libx11-dev libxrandr-dev libxinerama-dev \
                    libxcursor-dev libxi-dev
   ```
   (or install LunarG's Vulkan SDK and make sure `VULKAN_SDK` is set).
   Wayland desktops work through XWayland.
2. Run **`Run Eden (Linux).sh`**. If your file manager won't execute `.sh` on
   double-click, run it from a terminal: `./"Run Eden (Linux).sh"`

### Build type: Release vs Debug

The scripts build **Release** by default. This matters a lot: a Debug build is
several times slower on the CPU **and** turns on Vulkan validation layers,
which add a large per-frame cost. Use Debug only when you're debugging the
engine itself:

- macOS/Linux: `EDEN_BUILD_TYPE=Debug ./"Run Eden (Mac).command"`
- Windows: `set EDEN_BUILD_TYPE=Debug` in a terminal, then run the `.bat`.

Switching build type triggers a one-time full rebuild.

## Controls

| Input | Action |
|---|---|
| `W` `A` `S` `D` | Move camera forward/left/back/right |
| `Space` | Move camera up |
| `Left Ctrl` | Move camera down |
| Mouse | Look around (while cursor is locked) |
| Left-click (hold) | Carve whatever hardened (voxel) surface you're aiming at. The first carve fires instantly; holding continues at a steady rate. Hold `Shift` to drill through multiple layers. Mold workpieces (`B`) use a smaller brush |
| `M` | Melt the hardened volume you're aiming at into fluid particles |
| `H` | Reform sufficiently-cooled, clustered fluid particles into a hardened volume |
| `P` (hold) | Pour a steady stream of the chemical selected in the ImGui panel at a point in front of the camera |
| `B` | Spawn a fine-grained carvable mold workpiece (a 3 x 0.8 x 3 unit slab) in front of the camera |
| `J` | Print a diagnostic probe of the voxel grid under the crosshair to the terminal (debug aid) |
| `Tab` | Toggle between camera-look and UI-interaction mode |
| `Esc` | Quit |

Camera movement and mouse-look only work while the cursor is locked (the
default on startup). Press `Tab` to unlock the cursor and use the ImGui panels.

## Performance

Open the **"Eden - Performance"** panel (press `Tab` first to use the mouse).

- It shows the frame time, a **GPU-bound / not GPU-bound** verdict, per-section
  CPU timings, and the **slowest recent frame** with what made it slow.
- Tick **"Measure GPU pass times"** to see how long each fluid pass and the
  main scene take on the GPU (approximate on Apple GPUs).
- **Save report** writes `eden_perf.txt` into the working directory (the
  `build/` folder when you launch via the run scripts); **Copy
  report** puts the same text on the clipboard. Please include one when
  reporting a performance problem, ideally taken right after the slowdown.

Things that help on slower machines:

- Use a **Release** build (the default - see above).
- **Fluid resolution** (in the "Fluid surface render" panel) defaults to
  **Auto**: if the game is GPU-bound with liquid on screen for about a second,
  it steps the fluid passes down 100% -> 75% -> 50% -> 35% until the frame rate
  is playable (and never steps back up by itself). You can also pick a fixed
  level. The fluid passes are the main GPU cost when liquid fills the screen,
  and their cost scales with pixel count. Each change rebuilds the fluid render
  targets, causing a brief one-time hitch. Lower levels make small droplets
  look blocky; 35% is a last resort.
- Turn off **Temporal smoothing** if you want to shave off one more pass.
- Liquid that is off-screen or hidden behind terrain/solids is now skipped
  automatically: a small GPU pass works out where the visible liquid is on
  screen, and the fluid passes only run inside that rectangle (or not at all
  if none is visible). Untick "Confine liquid passes..." in the Performance
  panel to compare.
- Raymarched SDF objects (the sphere that starts to the left) are only drawn
  over the part of the screen they can cover, and skipped when off-screen. A
  raymarched object that fills the whole view is still expensive.
- The **Max physics catch-up steps** setting bounds the cost of a slow frame;
  below ~20 fps the simulation runs in slow motion rather than getting even
  slower.

Developer/benchmark environment variables (all optional):
`EDEN_GPU_TIMING=1` (GPU timing on at start), `EDEN_PERF_AUTOREPORT=<seconds>`
(save `eden_perf.txt` after that many seconds and quit), `EDEN_FLUID_SCALE=0.5`,
`EDEN_BUILD_TYPE=Debug` (build scripts).

## Reporting bugs

For each issue, if you can, note: what platform and GPU you're on, what you
did right before it happened, whether it was a Release or Debug build, and
anything printed to the terminal window (it stays open - don't close it,
that's where errors show up). For performance problems, include a saved
report from the Performance panel.

## For the maintainer: GitHub sync and releases

- **Sync (macOS):** `Connect Eden to GitHub (Mac).command` (once, for a freshly
  unzipped folder) and the prompt at the end of `Run Eden (Mac).command`
  push your local state to `main`. They ask for a Personal Access Token each
  time and never store it; the token is written to a temp file outside the
  project for the duration of the push only. Token needs **Contents: read and
  write**, plus the **`workflow` scope** (classic token) or **Workflows: read
  and write** (fine-grained) if the push touches anything under
  `.github/workflows/`.
- Because the project folder is re-created from a zip, it starts without
  `.git/`, `.gitignore` and `.github/`. The scripts keep any file that exists on
  GitHub but not locally (always for dot-files like `.github/workflows/*`;
  for anything else they ask) instead of pushing its deletion, refuse to push
  files over GitHub's 100 MB limit, and print a plain-English hint when a push
  fails. The shared logic lives in `Tools/eden_git_safety.sh`.
- `.gitignore` keeps build output (`build/`, `dist/`), logs and
  `eden_perf.txt` out of commits.
- **Actions:** `.github/workflows/eden-build-check.yml` compiles Eden on Linux,
  Windows and macOS for every push (this is how Windows/Linux build problems
  get found without a Windows/Linux machine; the Windows and macOS jobs are
  new and not yet proven, so read their logs). `.github/workflows/eden-release.yml`
  builds tester zips and publishes a GitHub Release when you push a version tag
  (`git tag v0.01 && git push origin v0.01`).
