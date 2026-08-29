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

mkdir -p build
cd build

echo "Configuring..."
cmake .. -DCMAKE_BUILD_TYPE=Debug > /tmp/eden_cmake.log 2>&1 || {
    echo ""
    echo "CMake configuration failed. Full log:"
    cat /tmp/eden_cmake.log
    echo ""
    echo "Common cause on Linux: missing Vulkan SDK, or missing X11/Wayland"
    echo "development headers that GLFW's own source (vendored in"
    echo "ThirdParty/glfw/, built as part of this project - no separate"
    echo "GLFW/GLM install needed anymore) needs to compile its Linux"
    echo "windowing backend. On Debian/Ubuntu:"
    echo "  sudo apt install cmake build-essential libvulkan-dev \\"
    echo "                   vulkan-tools glslang-tools \\"
    echo "                   libx11-dev libxrandr-dev libxinerama-dev \\"
    echo "                   libxcursor-dev libxi-dev libwayland-dev \\"
    echo "                   libxkbcommon-dev"
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
