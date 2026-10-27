# Builds mdzplugin.so, a LAMMPS plugin that adds "dump mdz".
#
#   cmake -S lammps -B build-plugin -D LAMMPS_SOURCE_DIR=/path/to/lammps/src \
#         [-D BUILD_MPI=off]   # must match the LAMMPS binary
#   cmake --build build-plugin
#
# Then, in a LAMMPS input script:  plugin load /path/to/mdzplugin.so
#
# The mdz library (C++17, with the C API in mdz/mdz.h) is built from this
# repository and linked statically into the plugin. Its only dependency,
# zstd, is fetched and compiled in. Set MDZ_ZSTD=system to use an installed
# zstd instead.

cmake_minimum_required(VERSION 3.20)
project(mdzplugin LANGUAGES C CXX)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "Use a separate build directory")
endif()

set(LAMMPS_SOURCE_DIR "" CACHE PATH "The src/ directory of the LAMMPS tree the binary was built from")
if(NOT EXISTS "${LAMMPS_SOURCE_DIR}/lammps.h")
  message(FATAL_ERROR "Set LAMMPS_SOURCE_DIR to the LAMMPS src/ directory")
endif()
set(LAMMPS_HEADER_DIR ${LAMMPS_SOURCE_DIR})

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use LAMMPS's plugin support for header paths, MPI (or its serial stubs) and
# integer size settings, so the plugin matches the binary that loads it.
set(CMAKE_MODULE_PATH "${LAMMPS_SOURCE_DIR}/../examples/plugins")
include(CheckIncludeFileCXX)
include(LAMMPSInterfaceCXX)

# --- libmdz ---
# Build only the library, as a static archive of position-independent code
# so it can be linked into the plugin's shared object.
option(MDZ_PARALLEL "Let mdz use worker threads of its own inside LAMMPS" OFF)
set(MDZ_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(MDZ_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MDZ_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(MDZ_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
get_filename_component(MDZ_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
add_subdirectory(${MDZ_ROOT} ${CMAKE_BINARY_DIR}/mdz EXCLUDE_FROM_ALL)

# --- plugin ---
add_library(mdzplugin MODULE mdzplugin.cpp dump_mdz.cpp)
target_include_directories(mdzplugin PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# LAMMPS symbols stay undefined and resolve against the loading binary.
target_link_libraries(mdzplugin PRIVATE lammps mdz::mdz ${CMAKE_DL_LIBS})
if(NOT MDZ_PARALLEL)
  # dump mdz calls mdz::set_num_threads(1) to keep mdz on the calling thread.
  target_compile_definitions(mdzplugin PRIVATE MDZ_PLUGIN_SERIAL)
endif()
set_target_properties(mdzplugin PROPERTIES PREFIX "" SUFFIX ".so")
if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
  set_target_properties(mdzplugin PROPERTIES LINK_FLAGS "-Wl,-undefined,dynamic_lookup")
elseif(NOT CMAKE_SYSTEM_NAME STREQUAL "Windows")
  # keep the symbols of libmdz and zstd out of the global namespace
  set_target_properties(mdzplugin PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")
endif()

