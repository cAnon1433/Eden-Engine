cmake_minimum_required(VERSION 3.20)
project(Eden VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
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

file(GLOB SHADER_SOURCES
    "${CMAKE_SOURCE_DIR}/Shaders/Source/*.vert"
    "${CMAKE_SOURCE_DIR}/Shaders/Source/*.frag"
)

set(SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/Shaders/Compiled")
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

set(SPIRV_BINARIES "")
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SPIRV "${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv")

    add_custom_command(
        OUTPUT ${SPIRV}
        COMMAND ${GLSL_VALIDATOR} -V ${SHADER} -o ${SPIRV}
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
    Threads::Threads
)

# Assets/ (textures, and eventually models) copied to build/Assets/ every
# build - same reasoning as Shaders/Compiled below: Eden's working
# directory at runtime is build/ (Run Eden.command cd's there before
# launching), so relative paths like "Assets/Textures/foo.png" passed to
# Renderer::LoadTexture only resolve correctly if the files actually live
# under build/. copy_directory re-runs every build (cheap - it's a file
# copy, not a compile) so newly added assets show up without needing a
# clean rebuild.
add_custom_command(TARGET Eden POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/Assets" "$<TARGET_FILE_DIR:Eden>/Assets"
    COMMENT "Copying Assets/ to build output"
)

# Compiled shaders land in build/Shaders/Compiled, next to the Eden
# executable, so `./Eden` works when run from build/ (the normal case).
# If you move the binary elsewhere, Shaders/Compiled/ needs to travel
# with it, or adjust the paths in Renderer::Init.
