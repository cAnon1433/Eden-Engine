@echo off
setlocal enabledelayedexpansion

REM Builds a Release copy of Eden and packages it into dist\Eden-Windows\
REM plus dist\Eden-Windows.zip - a self-contained folder a tester can
REM unzip anywhere and run by double-clicking Eden.exe, with NO Vulkan
REM SDK, CMake, or Visual Studio required on their end.
REM
REM Builds into a SEPARATE build-release\ folder rather than reusing
REM your normal build\ (which is likely a Debug config, possibly
REM configured with a different generator/cache) - keeps this script
REM safe to run without disturbing your regular dev build.
REM
REM Why this works with no extra runtime installs on the tester's
REM machine:
REM   - CMAKE_MSVC_RUNTIME_LIBRARY is set to the static (/MT) runtime
REM     in CMakeLists.txt, so the VC++ runtime is baked into Eden.exe
REM     itself instead of needing VCRUNTIME140.dll etc. present.
REM   - Vulkan itself doesn't need the SDK on the tester's machine -
REM     only the loader (vulkan-1.dll), which ships with any GPU driver
REM     that supports Vulkan at all. Only building Eden requires the SDK.
REM   - Shaders/Compiled\ and Assets\ are copied next to Eden.exe by
REM     CMakeLists.txt's post-build step, and Engine/Core/PathUtils.cpp
REM     resolves them relative to the exe's own location - not the
REM     working directory - so this works no matter where the folder is
REM     unzipped to or how it's launched.

cd /d "%~dp0"

echo Eden - package for testers
echo ===========================
echo.

if not exist build-release mkdir build-release
cd build-release

set GENERATOR_FOUND=0

if not "%GENERATOR_FOUND%"=="1" (
    echo Configuring Release build ^(trying Visual Studio 2022^)...
    cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release > eden_package_cmake.log 2>&1
    if not errorlevel 1 set GENERATOR_FOUND=1
)

if not "%GENERATOR_FOUND%"=="1" (
    echo   ...not found or failed. Trying Visual Studio 2019...
    del /f /q CMakeCache.txt >nul 2>&1
    rmdir /s /q CMakeFiles >nul 2>&1
    cmake .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release > eden_package_cmake.log 2>&1
    if not errorlevel 1 set GENERATOR_FOUND=1
)

if not "%GENERATOR_FOUND%"=="1" (
    echo.
    echo CMake configuration failed. Full log:
    type eden_package_cmake.log
    echo.
    echo See "Run Eden (Windows).bat"'s own error message for the two
    echo most likely causes ^(missing C++ workload, missing Vulkan SDK^).
    echo.
    pause
    exit /b 1
)

echo.
echo Building Release...
cmake --build . --config Release > eden_package_build.log 2>&1
if errorlevel 1 (
    echo.
    echo Build failed. Full log:
    type eden_package_build.log
    echo.
    pause
    exit /b 1
)

cd ..

if not exist "build-release\Release\Eden.exe" (
    echo.
    echo Build succeeded but build-release\Release\Eden.exe wasn't found -
    echo build layout may differ from what this script expects.
    pause
    exit /b 1
)

echo.
echo Packaging...

if exist "dist\Eden-Windows" rmdir /s /q "dist\Eden-Windows"
mkdir "dist\Eden-Windows"

REM Assets\ and Shaders\Compiled\ are already sitting next to Eden.exe
REM (CMakeLists.txt's post-build copy step put them there) - copying
REM the whole Release\ folder brings them along for free, no need to
REM list them separately here.
xcopy /e /i /y "build-release\Release\*" "dist\Eden-Windows\" >nul

REM .pdb/.ilk are debug-info/incremental-link byproducts, meaningless
REM to a tester and just extra download size - drop them if present.
del /f /q "dist\Eden-Windows\*.pdb" >nul 2>&1
del /f /q "dist\Eden-Windows\*.ilk" >nul 2>&1

echo Zipping...
if exist "dist\Eden-Windows.zip" del /f /q "dist\Eden-Windows.zip"
powershell -NoProfile -Command "Compress-Archive -Path 'dist\Eden-Windows\*' -DestinationPath 'dist\Eden-Windows.zip'"

echo.
echo Done. Ready to send:
echo   dist\Eden-Windows.zip
echo.
echo A tester unzips it anywhere and double-clicks Eden.exe inside -
echo no Vulkan SDK, CMake, or Visual Studio needed on their end.
echo.
pause
