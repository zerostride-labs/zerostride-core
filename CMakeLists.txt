cmake_minimum_required(VERSION 3.20)
project(zerostride_core VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ZEROSTRIDE_SANITIZE "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

# Strict deterministic C++20 compiler flags (no exceptions, no RTTI, strict conversions)
add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Werror
    -Wconversion
    -Wsign-conversion
    -Wshadow
    -Wnon-virtual-dtor
    -fno-exceptions
    -fno-rtti
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    add_compile_options(-mavx2 -mfma -mf16c)
endif()

if(ZEROSTRIDE_SANITIZE)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

# Optional third_party/ggml integration when populated
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/CMakeLists.txt")
    add_subdirectory(third_party/ggml)
endif()

# Core static library
add_library(zerostride_core STATIC
    src/platform/os_resources.cpp
    src/core/telemetry.cpp
)

target_include_directories(zerostride_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(TARGET ggml)
    target_link_libraries(zerostride_core PUBLIC ggml)
endif()

# Empirical benchmark & telemetry CLI tool
add_executable(zs-bench
    tools/zs-bench/main.cpp
)
target_include_directories(zs-bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(zs-bench PRIVATE zerostride_core)

# Unit tests
enable_testing()
add_executable(test_telemetry
    tests/test_telemetry.cpp
)
target_include_directories(test_telemetry PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_telemetry PRIVATE zerostride_core)
add_test(NAME test_telemetry COMMAND test_telemetry)
