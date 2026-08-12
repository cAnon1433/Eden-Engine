# Assets

Put texture and model files here. `Textures/` for images (`.png`, `.jpg`,
etc.), `Models/` for 3D models once mesh loading (OBJ/glTF) lands.

## Why here specifically

CMake copies this whole folder to `build/Assets/` every time you build
(see the `POST_BUILD` step in `CMakeLists.txt`). Eden's working directory
at runtime is `build/` - `Run Eden.command` does `cd build` before
launching the executable - so any path you pass to `Renderer::LoadTexture`
(or, later, a model loader) should be written relative to `build/`, e.g.:

```cpp
renderer.LoadTexture("Assets/Textures/BannerImage.png");
```

not an absolute path to somewhere like `~/Downloads`, and not a path
relative to the project root either - both will fail to `fopen` the file,
since neither matches the actual working directory the process runs with.

Drop a file in `Assets/Textures/`, rebuild (even an unchanged source tree
still re-copies this folder), and reference it by that relative path.
