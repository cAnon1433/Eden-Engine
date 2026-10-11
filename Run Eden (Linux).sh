#!/bin/bash
# Run this file to build (if needed) and run Eden.
# Safe to run repeatedly - only rebuilds what actually changed.
#
# If double-clicking does nothing (some Linux file managers won't
# execute .sh files by default), either:
#   - right-click -> Properties -> Permissions -> "Allow executing as
#     program" (varies by desktop environment), or
#   - open a terminal in this folder and run: ./"Run Eden (Linux).sh"

set -e

# Move to this script's own folder, no matter where it's been moved to,
# and no matter if the path has spaces in it.
cd "$(dirname "$0")"

echo "Eden - build & run"
echo "=================="
echo ""

# Self-heal a stale build folder. CMake refuses to reuse a build/ whose
# CMakeCache.txt was created in a different location ("current CMakeCache.txt
# directory is different than ..."): that happens when build/ is copied from
# another folder, the project folder is moved/renamed, or build output was
# ever committed to git and came back with a checkout. build/ is pure output,
# so just start it over.
if [ -f build/CMakeCache.txt ]; then
    CACHED_DIR=$(grep -m1 '^CMAKE_CACHEFILE_DIR:INTERNAL=' build/CMakeCache.txt | cut -d= -f2-)
    if [ "$CACHED_DIR" != "$(pwd -P)/build" ]; then
        echo "The build folder was created for a different location:"
        echo "  $CACHED_DIR"
        echo "Rebuilding it from scratch (one-time, a few minutes)..."
        rm -rf build
    fi
fi

mkdir -p build
cd build

echo "Configuring..."
# Release by default (Debug is much slower and enables Vulkan validation
# layers). Debug build: EDEN_BUILD_TYPE=Debug ./"Run Eden (Linux).sh"
BUILD_TYPE="${EDEN_BUILD_TYPE:-Release}"
echo "Build type: $BUILD_TYPE"
cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE" > /tmp/eden_cmake.log 2>&1 || {
    echo ""
    echo "CMake configuration failed. Full log:"
    cat /tmp/eden_cmake.log
    echo ""
    echo "Common cause on Linux: missing Vulkan SDK, or missing X11"
    echo "development headers that GLFW's own source (vendored in"
    echo "ThirdParty/glfw/, built as part of this project - no separate"
    echo "GLFW/GLM install needed anymore) needs to compile its Linux"
    echo "windowing backend. On Debian/Ubuntu:"
    echo "  sudo apt install cmake build-essential libvulkan-dev \\"
    echo "                   vulkan-tools glslang-tools \\"
    echo "                   libx11-dev libxrandr-dev libxinerama-dev \\"
    echo "                   libxcursor-dev libxi-dev"
    echo "and make sure VULKAN_SDK is set if you installed LunarG's SDK"
    echo "instead of the distro packages."
    echo ""
    read -p "Press Enter to close..."
    exit 1
}

echo "Building..."
make -j"$(nproc)" > /tmp/eden_make.log 2>&1 || {
    echo ""
    echo "Build failed. Full log:"
    cat /tmp/eden_make.log
    echo ""
    read -p "Press Enter to close..."
    exit 1
}

echo ""
echo "Build succeeded. Launching Eden..."
echo ""

./Eden

echo ""
echo "Eden closed."
read -p "Press Enter to close this window..."
