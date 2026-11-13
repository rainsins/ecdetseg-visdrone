cmake_minimum_required(VERSION 3.16)
project(ec_infer LANGUAGES CXX)

# RK3588 board deployment for ECDet-S INT8.
#
# Native build on the board:
#   cmake -S . -B build && cmake --build build -j
#
# Cross build from an x86 host: pass a toolchain file (see the TODO at the
# bottom) and point EC_RKNN_ROOT / OpenCV_DIR at the aarch64 sysroot copies.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -Wall -Wextra on by default; the resampler/postprocess code is warning-clean
# and that is worth keeping, since both are byte-exact ports.
add_compile_options(-Wall -Wextra)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O2)
endif()

# --- Build targets -----------------------------------------------------------
# Two independent targets live here:
#   EC_BUILD_BOARD (default ON)  ec_infer  -- needs librknnrt + OpenCV
#   EC_BUILD_TESTS (default OFF) *_cli     -- host only, needs nothing but C++17
# Keeping them separate matters: the parity harnesses are how the resampler and
# decoder get validated off-board, and they must stay buildable on a laptop with
# no RKNN runtime installed.
option(EC_BUILD_BOARD "Build the board binary (needs librknnrt + OpenCV)" ON)
option(EC_BUILD_TESTS "Build the host-only parity harnesses" OFF)
option(EC_ENABLE_GRID_SAMPLE_CPU "Enable opt-in strict FLOAT32 CPU GridSample callback" OFF)

# --- RKNN runtime -----------------------------------------------------------
# librknnrt ships with rknpu2.  Layout on a stock board image:
#   /usr/lib/librknnrt.so
#   /usr/include/rknn_api.h
# On a cross sysroot it is usually <root>/lib and <root>/include.
set(EC_RKNN_ROOT "" CACHE PATH "Prefix containing lib/librknnrt.so and include/rknn_api.h")
# Explicit overrides win over both the ROOT prefix and the automatic search, so a
# sysroot layout that does not match <root>/include + <root>/lib still works.
set(EC_RKNN_INCLUDE_DIR "" CACHE PATH "Directory containing rknn_api.h (overrides EC_RKNN_ROOT)")
set(EC_RKNN_LIBRARY "" CACHE FILEPATH "Full path to librknnrt.so (overrides EC_RKNN_ROOT)")

if(EC_BUILD_BOARD)
  if(EC_RKNN_ROOT)
    if(NOT EC_RKNN_INCLUDE_DIR)
      set(EC_RKNN_INCLUDE_DIR "${EC_RKNN_ROOT}/include")
    endif()
    if(NOT EC_RKNN_LIBRARY)
      find_library(EC_RKNN_LIBRARY_FOUND NAMES rknnrt
                   PATHS "${EC_RKNN_ROOT}/lib" "${EC_RKNN_ROOT}/lib64" NO_DEFAULT_PATH)
      if(EC_RKNN_LIBRARY_FOUND)
        set(EC_RKNN_LIBRARY "${EC_RKNN_LIBRARY_FOUND}")
      endif()
    endif()
  else()
    if(NOT EC_RKNN_INCLUDE_DIR)
      find_path(EC_RKNN_INCLUDE_DIR rknn_api.h)
    endif()
    if(NOT EC_RKNN_LIBRARY)
      find_library(EC_RKNN_LIBRARY_FOUND NAMES rknnrt)
      if(EC_RKNN_LIBRARY_FOUND)
        set(EC_RKNN_LIBRARY "${EC_RKNN_LIBRARY_FOUND}")
      endif()
    endif()
  endif()

  if(NOT EC_RKNN_INCLUDE_DIR OR NOT EC_RKNN_LIBRARY)
    # Fatal rather than optional: without librknnrt the resulting binary would
    # silently fall back to something else, and the whole point of this target is
    # to exercise the real NPU path.
    message(FATAL_ERROR
      "librknnrt / rknn_api.h not found (EC_BUILD_BOARD is ON).\n"
      "  On the board: install the rknpu2 runtime (expects /usr/lib/librknnrt.so).\n"
      "  Cross build:   -DEC_RKNN_ROOT=/path/to/rknpu2/runtime/Linux/librknn_api\n"
      "  Or set EC_RKNN_INCLUDE_DIR and EC_RKNN_LIBRARY directly.\n"
      "  Host-only parity harnesses: add -DEC_BUILD_BOARD=OFF -DEC_BUILD_TESTS=ON.")
  endif()
  message(STATUS "librknnrt: ${EC_RKNN_LIBRARY}")
  message(STATUS "rknn_api.h: ${EC_RKNN_INCLUDE_DIR}")

  # --- OpenCV ---------------------------------------------------------------
  # Image decode only (cv::imread).  The resize is deliberately NOT OpenCV's --
  # see pil_resize.h -- so a missing imgproc would not break correctness, only
  # the BGR->RGB conversion in main.cpp.
  find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
  message(STATUS "OpenCV: ${OpenCV_VERSION} (${OpenCV_INCLUDE_DIRS})")

  # --- Core library ---------------------------------------------------------
  # Keeping the three stages in a library lets the test executables link them
  # without dragging in main.cpp.
  add_library(ec_core STATIC
    pil_resize.cpp
    postprocess.cpp
    rknn_engine.cpp
  )
  target_include_directories(ec_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${EC_RKNN_INCLUDE_DIR}"
  )
  target_link_libraries(ec_core PUBLIC "${EC_RKNN_LIBRARY}")
  if(EC_ENABLE_GRID_SAMPLE_CPU)
    if(NOT EXISTS "${EC_RKNN_INCLUDE_DIR}/rknn_custom_op.h")
      message(FATAL_ERROR "EC_ENABLE_GRID_SAMPLE_CPU requires rknn_custom_op.h beside rknn_api.h")
    endif()
    target_sources(ec_core PRIVATE grid_sample_cpu.cpp)
    target_compile_definitions(ec_core PUBLIC EC_ENABLE_GRID_SAMPLE_CPU=1)
  endif()

  # --- Board binary ---------------------------------------------------------
  add_executable(ec_infer main.cpp)
  target_link_libraries(ec_infer PRIVATE ec_core ${OpenCV_LIBS})
  target_include_directories(ec_infer PRIVATE ${OpenCV_INCLUDE_DIRS})
else()
  message(STATUS "EC_BUILD_BOARD=OFF: skipping ec_infer (no RKNN/OpenCV required)")
endif()

# --- Host-side parity harnesses --------------------------------------------
# These do not touch the NPU and build on any Linux host with a C++17 compiler,
# which is how the resampler and decoder were validated against Pillow and
# deploy/postprocess.py.  Enable with -DEC_BUILD_TESTS=ON.
if(EC_BUILD_TESTS)
  add_library(ec_host STATIC pil_resize.cpp postprocess.cpp)
  target_include_directories(ec_host PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

  add_executable(pil_resize_cli tests/pil_resize_cli.cpp)
  target_link_libraries(pil_resize_cli PRIVATE ec_host)

  add_executable(postprocess_cli tests/postprocess_cli.cpp)
  target_link_libraries(postprocess_cli PRIVATE ec_host)
endif()

# --- Cross compilation (TODO) ----------------------------------------------
# RK3588 is aarch64.  A working toolchain file needs at minimum:
#
#   set(CMAKE_SYSTEM_NAME Linux)
#   set(CMAKE_SYSTEM_PROCESSOR aarch64)
#   set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
#   set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
#   set(CMAKE_SYSROOT /path/to/sysroot)
#   set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
#   set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
#   set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
#
# then:
#   cmake -S . -B build-arm \
#         -DCMAKE_TOOLCHAIN_FILE=toolchain-aarch64.cmake \
#         -DEC_RKNN_ROOT=$SYSROOT/usr \
#         -DOpenCV_DIR=$SYSROOT/usr/lib/cmake/opencv4
#
# This is not committed as a file because the sysroot path is machine-specific;
# leaving a placeholder toolchain that silently does not work is worse than a
# documented recipe.
