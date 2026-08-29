#pragma once

#include <filesystem>
#include <string>

namespace Eden
{
    // Resolves resource paths (shaders, assets) against the running
    // executable's own directory, NOT the process's current working
    // directory.
    //
    // Why this exists: every resource load in this engine used to be a
    // literal relative string ("Shaders/Compiled/triangle.vert.spv"),
    // which only resolves correctly if the process's CWD happens to be
    // the build output root - true when launched via the Run Eden
    // (Platform) scripts (which cd there first), false the moment
    // someone unzips a distributed build and double-clicks Eden.exe
    // directly from wherever it landed. CWD is a property of HOW a
    // process was launched, not WHERE its files are - the two are only
    // the same by convention, and that convention breaks for exactly
    // the "hand someone a folder, they double-click it" case this was
    // added for. Resolving against the executable's own on-disk
    // location instead makes resource loading work identically no
    // matter how or from where the exe is launched.
    namespace PathUtils
    {
        // Absolute, symlink-resolved path to the directory containing
        // the currently-running executable. Computed once (via the
        // platform-specific self-path query - GetModuleFileNameW on
        // Windows, _NSGetExecutablePath on macOS, /proc/self/exe on
        // Linux) and cached in a function-local static, since the
        // underlying OS call is a genuine syscall/API call each time
        // and this is asked for on every resource load.
        const std::filesystem::path& GetExecutableDir();

        // Joins `relativePath` onto GetExecutableDir() and returns the
        // result as a plain string ready to hand to std::ifstream etc.
        // An already-absolute `relativePath` is returned unchanged -
        // this is a resolve, not a forced rebase, so a caller that
        // somehow already has a full path isn't double-joined.
        std::string ResolveResourcePath(const std::string& relativePath);
    }
}
