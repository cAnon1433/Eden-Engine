@echo off
setlocal enabledelayedexpansion

REM Double-click this file to build (if needed) and run Eden.
REM Safe to run repeatedly - only rebuilds what actually changed.
REM
REM Requires: Vulkan SDK (LunarG), CMake, and a C++ compiler (Visual
REM Studio 2019/2022 Build Tools or full VS - just the "Desktop
REM development with C++" workload, not the full IDE). GLFW and GLM
REM are vendored directly in ThirdParty/ and build as part of this
REM project - no separate install step (vcpkg or otherwise) needed for
REM them.
REM
REM IMPORTANT: this script explicitly requests the Visual Studio CMake
REM generator (-G "Visual Studio ... " -A x64) instead of letting CMake
REM guess. Left to guess, CMake's default on Windows depends on which
REM shell it's launched from - if this is double-clicked from Explorer
REM (a plain cmd.exe, not a "Developer Command Prompt for VS"), CMake
REM can silently fall back to the NMake Makefiles generator, which
REM needs nmake.exe/cl.exe already on PATH via vcvarsall.bat. Since a
REM plain double-click never runs vcvarsall, that fails with "nmake:
REM no such file or directory" / "CMAKE_CXX_COMPILER not set" even
REM though Visual Studio is installed correctly - confirmed as the
REM actual cause of a real cross-platform test failure. The Visual
REM Studio generator avoids this entirely: it resolves cl.exe's full
REM path itself from the VS install and writes it into the generated
REM .vcxproj files, so configure/build work from any shell, no dev
REM prompt required. It also works with just Build Tools installed,
REM the full IDE isn't required.

cd /d "%~dp0"

echo Eden - build ^& run
echo ==================
echo.

if not exist build mkdir build
cd build

REM Try generators newest-to-oldest. Each attempt writes its own log so
REM a failure shows the real error instead of a generic "config failed".
set GENERATOR_FOUND=0

if not "%GENERATOR_FOUND%"=="1" (
    echo Configuring ^(trying Visual Studio 2022^)...
    cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Debug > eden_cmake.log 2>&1
    if not errorlevel 1 set GENERATOR_FOUND=1
)

if not "%GENERATOR_FOUND%"=="1" (
    echo   ...not found or failed. Trying Visual Studio 2019...
    REM A failed attempt above may have left a partial cache behind -
    REM CMake refuses to reconfigure with a different generator against
    REM an existing cache ("Please delete the cache"), so clear it
    REM first rather than surfacing that as a second, confusing error.
    del /f /q CMakeCache.txt >nul 2>&1
    rmdir /s /q CMakeFiles >nul 2>&1
    cmake .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Debug > eden_cmake.log 2>&1
    if not errorlevel 1 set GENERATOR_FOUND=1
)

if not "%GENERATOR_FOUND%"=="1" (
    echo.
    echo CMake configuration failed. Full log:
    type eden_cmake.log
    echo.
    echo Could not configure with either the Visual Studio 2022 or 2019
    echo CMake generator. Most likely causes:
    echo   1. Visual Studio / VS Build Tools isn't installed, or is
    echo      installed without the "Desktop development with C++"
    echo      workload ^(installer: https://visualstudio.microsoft.com/downloads/^).
    echo   2. The Vulkan SDK isn't installed or VULKAN_SDK isn't set
    echo      ^(the LunarG installer normally sets this for you - if this
    echo      log mentions glslangValidator/glslc, this is the cause^).
    echo.
    pause
    exit /b 1
)

echo.
echo Building...
cmake --build . --config Debug > eden_make.log 2>&1
if errorlevel 1 (
    echo.
    echo Build failed. Full log:
    type eden_make.log
    echo.
    pause
    exit /b 1
)

echo.
echo Build succeeded. Launching Eden...
echo.

if exist Debug\Eden.exe (
    Debug\Eden.exe
) else if exist Eden.exe (
    Eden.exe
) else (
    echo Could not find Eden.exe in build\Debug\ or build\ - build layout
    echo may differ from what this script expects.
    pause
    exit /b 1
)

echo.
echo Eden closed.
pause
