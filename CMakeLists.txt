cmake_minimum_required(VERSION 3.20)
project(Eden VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Statically link the MSVC runtime (/MT, /MTd for Debug) instead of the
# default dynamic one (/MD, /MDd). This ONLY matters for distribution:
# a /MD-built Eden.exe needs VCRUNTIME140.dll (etc.) present on the
# machine it runs on, which every dev box has (VS/Build Tools installs
# it) but a tester's machine generally doesn't - "the app won't even
# start, missing VCRUNTIME140.dll" is the next likely failure once
# build issues are sorted, same category as the NMake generator issue.
# /MT bakes the runtime into Eden.exe itself, so a packaged build runs
# on a bare Windows install with nothing extra installed. No effect on
# Mac/Linux (MSVC_RUNTIME_LIBRARY is a no-op there).
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

find_package(Vulkan REQUIRED)

# GLFW and GLM are vendored directly under ThirdParty/ (same pattern as
# imgui/cgltf/stb/tinyobjloader/vma below - see EDEN_SOURCES' own
# comment on why third-party sources are listed explicitly rather than
# globbed) rather than pulled via find_package(), which used to require
# a separate install step per platform - vcpkg on Windows, a system
# package manager on Mac/Linux - before Eden could even configure.
# Vendoring removes that step entirely: cloning this repo and running
# the launcher scripts is now sufficient, no external package manager
# needed for anything except Vulkan itself (see below for why that one
# genuinely can't be vendored the same way).
#
# GLFW (ThirdParty/glfw/, pinned at release 3.4) is a real compiled
# library with per-platform backends (Win32/Cocoa/X11/Wayland) - built
# as part of THIS project's own build via add_subdirectory, not
# pre-built and dropped in, so it always matches the compiler/runtime
# Eden itself is built with. GLFW_STANDALONE is false when included via
# add_subdirectory (GLFW's own CMakeLists.txt default), which already
# turns off GLFW_BUILD_EXAMPLES/GLFW_BUILD_TESTS - GLFW_BUILD_DOCS and
# GLFW_INSTALL default ON regardless of that, so they're explicitly
# turned off below to keep the build lean and avoid GLFW installing
# itself into the wrong place alongside Eden's own install step (Eden
# doesn't currently have one, but no reason to let GLFW add one).
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory("${CMAKE_SOURCE_DIR}/ThirdParty/glfw")

# GLM (ThirdParty/glm/, pinned at release 1.0.1) is header-only - no
# build step, no compiled artifact, just an include path. INTERFACE
# library so it participates in target_link_libraries below the same
# way GLFW/Vulkan::Vulkan do, without actually compiling anything of
# its own.
add_library(glm INTERFACE)
target_include_directories(glm INTERFACE "${CMAKE_SOURCE_DIR}/ThirdParty/glm")

find_package(Threads REQUIRED) # ParticleThreadPool (Engine/Particles/) - std::thread/std::mutex need explicit pthread linkage on Linux/GCC, usually implicit on macOS/Clang but linked explicitly here too for portability

# --- Shader compilation (GLSL -> SPIR-V) -------------------------------
# Looks in $VULKAN_SDK/bin first (where LunarG's installer puts it on
# macOS/Windows), then falls back to PATH (typical on Linux).
find_program(GLSL_VALIDATOR
    NAMES glslangValidator glslc
    HINTS "$ENV{VULKAN_SDK}/bin"
)

if(NOT GLSL_VALIDATOR)
    message(FATAL_ERROR "Eden: could not find glslangValidator or glslc. Is the Vulkan SDK installed and VULKAN_SDK set?")
endif()

# --target-env syntax differs between the two tools this project
# supports (find_program above accepts either name):
#   glslangValidator wants SPACE-separated: --target-env vulkan1.2
#   glslc wants EQUALS-separated:           --target-env=vulkan1.2
# Passing the wrong form is a hard command-line parse error on
# glslangValidator ("unrecognized command-line option"), not a silent
# fallback - confirmed the hard way on this project's actual dev
# machine, which resolves GLSL_VALIDATOR to glslangValidator. Detect
# which binary this is and set the flag to match, rather than
# guessing one form and hoping it works on both.
get_filename_component(GLSL_VALIDATOR_NAME "${GLSL_VALIDATOR}" NAME_WE)
if(GLSL_VALIDATOR_NAME STREQUAL "glslc")
    set(GLSL_TARGET_ENV_FLAG "--target-env=vulkan1.2")
else()
    set(GLSL_TARGET_ENV_FLAG "--target-env" "vulkan1.2")
endif()

file(GLOB SHADER_SOURCES
    "${CMAKE_SOURCE_DIR}/Shaders/Source/*.vert"
    "${CMAKE_SOURCE_DIR}/Shaders/Source/*.frag"
    "${CMAKE_SOURCE_DIR}/Shaders/Source/*.comp"
)

set(SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/Shaders/Compiled")
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

set(SPIRV_BINARIES "")
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SPIRV "${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv")

    add_custom_command(
        OUTPUT ${SPIRV}
        # GLSL_TARGET_ENV_FLAG is set above based on which binary
        # GLSL_VALIDATOR resolved to (glslangValidator vs glslc take
        # different --target-env syntax - see that comment for why this
        # isn't just hardcoded). Not strictly required by any shader
        # today (an earlier version of raymarch.frag used descriptor
        # indexing / nonuniformEXT, which genuinely needed this - that
        # approach was abandoned after MoltenVK rejected the required
        # layout outright on this project's dev hardware, see
        # RaymarchTypes.h's kRaymarchMaxObjects comment), but left in
        # place since it's harmless and matches what the C++ side
        # already targets. Eden's device/instance setup targets Vulkan
        # 1.2 (see VulkanInstance.cpp's appInfo.apiVersion and
        # VulkanDevice.cpp's
        # VkPhysicalDeviceVulkan12Features usage), so this just makes
        # shader compilation match what the C++ side already assumes.
        COMMAND ${GLSL_VALIDATOR} -V ${GLSL_TARGET_ENV_FLAG} ${SHADER} -o ${SPIRV}
        DEPENDS ${SHADER}
        COMMENT "Compiling shader: ${SHADER_NAME}"
    )
    list(APPEND SPIRV_BINARIES ${SPIRV})
endforeach()

add_custom_target(EdenShaders DEPENDS ${SPIRV_BINARIES})

# --- Engine sources ------------------------------------------------------
file(GLOB_RECURSE EDEN_SOURCES
    "${CMAKE_SOURCE_DIR}/Source/*.cpp"
    "${CMAKE_SOURCE_DIR}/Engine/*.cpp"
    "${CMAKE_SOURCE_DIR}/Engine/*.h"
)

# ImGui lives under ThirdParty/, outside the glob above (deliberately -
# EDEN_SOURCES should only ever pick up Eden's own code automatically;
# vendored third-party sources get listed explicitly, same reasoning as
# stb_image/tinyobjloader being single-.cpp #define-implementation files
# rather than auto-globbed).
set(IMGUI_SOURCES
    "${CMAKE_SOURCE_DIR}/ThirdParty/imgui/imgui.cpp"
    "${CMAKE_SOURCE_DIR}/ThirdParty/imgui/imgui_draw.cpp"
    "${CMAKE_SOURCE_DIR}/ThirdParty/imgui/imgui_tables.cpp"
    "${CMAKE_SOURCE_DIR}/ThirdParty/imgui/imgui_widgets.cpp"
    "${CMAKE_SOURCE_DIR}/ThirdParty/imgui/backends/imgui_impl_glfw.cpp"
    "${CMAKE_SOURCE_DIR}/ThirdParty/imgui/backends/imgui_impl_vulkan.cpp"
)
list(APPEND EDEN_SOURCES ${IMGUI_SOURCES})

add_executable(Eden ${EDEN_SOURCES})
add_dependencies(Eden EdenShaders)

target_include_directories(Eden PRIVATE
    "${CMAKE_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/ThirdParty/vma"
    "${CMAKE_SOURCE_DIR}/ThirdParty/stb"
    "${CMAKE_SOURCE_DIR}/ThirdParty/tinyobjloader"
    "${CMAKE_SOURCE_DIR}/ThirdParty/cgltf"
    "${CMAKE_SOURCE_DIR}/ThirdParty/imgui"
    "${CMAKE_SOURCE_DIR}/ThirdParty/imgui/backends"
)

target_link_libraries(Eden PRIVATE
    Vulkan::Vulkan
    glfw
    glm
    Threads::Threads
)

# Assets/ and compiled shaders are copied to $<TARGET_FILE_DIR:Eden> -
# the directory the actual Eden.exe/Eden binary lands in (e.g.
# build/Debug/ on a Visual Studio multi-config generator, build/ on a
# single-config Makefiles/Ninja generator) - NOT CMAKE_BINARY_DIR.
#
# This used to deliberately be CMAKE_BINARY_DIR instead, because
# Engine/Renderer/Vulkan/Pipeline/VulkanShaderModule.cpp resolved
# "Shaders/Compiled/..." against the process's CURRENT WORKING
# DIRECTORY, and every launcher script cd'd into CMAKE_BINARY_DIR
# before launching so that happened to line up. That made a packaged/
# distributed build fragile: double-clicking Eden.exe directly (a
# tester unzipping a folder, not going through a launcher script that
# cd's first) left the working directory wherever Explorer/Finder
# considers "here", which is NOT necessarily CMAKE_BINARY_DIR, and
# shader loading failed.
#
# Resource loading now resolves against the EXECUTABLE'S OWN location
# instead (see Engine/Core/PathUtils.h/.cpp), which is a property of
# where the .exe file sits, not how the process was launched. So the
# correct place to copy Assets/ and Shaders/Compiled/ is now wherever
# Eden.exe itself actually lands - $<TARGET_FILE_DIR:Eden> - so a
# packaged folder (exe + Shaders/Compiled/ + Assets/, zipped up and
# handed to a tester) is self-contained and correct regardless of
# where it's unzipped to or how it's launched.
add_custom_command(TARGET Eden POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/Assets" "$<TARGET_FILE_DIR:Eden>/Assets"
    COMMENT "Copying Assets/ to build output"
)

# SHADER_OUTPUT_DIR (set above, before the Eden target exists - see its
# own comment) is the shaders' actual compile destination and can't
# itself be $<TARGET_FILE_DIR:Eden> for a multi-config generator's sake
# (that generator-expression target dir isn't resolvable until a
# target exists). So compile there as before, then copy the compiled
# result next to the exe as a separate post-build step, same shape and
# same reasoning as the Assets/ copy just above.
add_custom_command(TARGET Eden POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${SHADER_OUTPUT_DIR}" "$<TARGET_FILE_DIR:Eden>/Shaders/Compiled"
    COMMENT "Copying compiled shaders to build output"
)
