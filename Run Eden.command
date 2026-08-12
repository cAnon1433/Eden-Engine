#!/bin/bash
# Double-click this file to build (if needed) and run Eden.
# Safe to run repeatedly - only rebuilds what actually changed.

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
    read -p "Press Enter to close..."
    exit 1
}

echo "Building..."
make > /tmp/eden_make.log 2>&1 || {
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
