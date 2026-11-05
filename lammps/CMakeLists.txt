# Builds mdzplugin.so, a LAMMPS plugin that adds "dump mdz".
#
#   cmake -S lammps -B build-plugin -D LAMMPS_BUILD_DIR=/path/to/lammps/build
#   cmake --build build-plugin
#
# Then, in a LAMMPS input script:  plugin load /path/to/mdzplugin.so
#
# The plugin must be built with the compiler and C++ standard LAMMPS was
# built with. LAMMPS_BUILD_DIR (the CMake build directory of the LAMMPS
# binary) supplies them, along with LAMMPS's source tree, MPI setting and
# integer sizes, and an explicit setting that differs is an error. Without
# it, set LAMMPS_SOURCE_DIR to the LAMMPS src/ directory and match the rest
# by hand: CMAKE_CXX_COMPILER, CMAKE_CXX_STANDARD, BUILD_MPI, LAMMPS_SIZES.
#
# The mdz library (C++17, with the C API in mdz/mdz.h) is built from this
# repository and linked statically into the plugin. Its only dependency,
# zstd, is fetched and compiled in. Set MDZ_ZSTD=system to use an installed
# zstd instead.
#
# If you build LAMMPS yourself, prefer compiling dump mdz into it as the
# package MDZ (patch_lammps.sh; see README.md).

cmake_minimum_required(VERSION 3.20)

# --- settings of the LAMMPS build, read before project() picks a compiler ---
set(LAMMPS_BUILD_DIR "" CACHE PATH "The CMake build directory of the LAMMPS binary (optional)")
if(LAMMPS_BUILD_DIR)
  if(NOT EXISTS "${LAMMPS_BUILD_DIR}/CMakeCache.txt")
    message(FATAL_ERROR "LAMMPS_BUILD_DIR (${LAMMPS_BUILD_DIR}) is not a CMake build directory")
  endif()
  load_cache("${LAMMPS_BUILD_DIR}" READ_WITH_PREFIX _lmp_
    CMAKE_CXX_COMPILER CMAKE_C_COMPILER lammps_SOURCE_DIR LAMMPS_SIZES BUILD_MPI PKG_PLUGIN)

  # The compiler. Not just the same family: each compiler instantiates the
  # standard library's templates for its own libstdc++ or libc++ version.
  function(_mdz_resolve_compiler name out)
    if(NOT IS_ABSOLUTE "${name}")
      find_program(_found NAMES "${name}" NO_CACHE)
      if(_found)
        set(name "${_found}")
      endif()
    endif()
    get_filename_component(name "${name}" REALPATH)
    set(${out} "${name}" PARENT_SCOPE)
  endfunction()
  if(NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${_lmp_CMAKE_CXX_COMPILER}" CACHE FILEPATH "C++ compiler (from LAMMPS_BUILD_DIR)")
  else()
    _mdz_resolve_compiler("${CMAKE_CXX_COMPILER}" _ours)
    _mdz_resolve_compiler("${_lmp_CMAKE_CXX_COMPILER}" _theirs)
    if(NOT _ours STREQUAL _theirs)
      message(FATAL_ERROR "CMAKE_CXX_COMPILER (${CMAKE_CXX_COMPILER}) is not the compiler LAMMPS was "
        "built with (${_lmp_CMAKE_CXX_COMPILER}, from ${LAMMPS_BUILD_DIR}). The plugin must use the same "
        "compiler: leave CMAKE_CXX_COMPILER unset to take LAMMPS's.")
    endif()
  endif()
  if(NOT CMAKE_C_COMPILER AND _lmp_CMAKE_C_COMPILER)
    # only for zstd, which is C and has no C++ ABI to match
    set(CMAKE_C_COMPILER "${_lmp_CMAKE_C_COMPILER}" CACHE FILEPATH "C compiler (from LAMMPS_BUILD_DIR)")
  endif()

  # The C++ standard. LAMMPS sets it without caching it, so take it from the
  # compile flags of its build (Ninja or Makefiles).
  set(_lmp_std "")
  foreach(_f "${LAMMPS_BUILD_DIR}/build.ninja" "${LAMMPS_BUILD_DIR}/CMakeFiles/lammps.dir/flags.make")
    if(NOT _lmp_std AND EXISTS "${_f}")
      file(STRINGS "${_f}" _lines REGEX "-std=(c|gnu)\\+\\+[0-9a-z]+" LIMIT_COUNT 1)
      if(_lines MATCHES "-std=(c|gnu)\\+\\+([0-9a-z]+)")
        set(_lmp_std "${CMAKE_MATCH_2}")
      endif()
    endif()
  endforeach()
  # the draft names some compilers use
  string(REPLACE "1z" "17" _lmp_std "${_lmp_std}")
  string(REPLACE "2a" "20" _lmp_std "${_lmp_std}")
  string(REPLACE "2b" "23" _lmp_std "${_lmp_std}")
  if(NOT _lmp_std)
    message(WARNING "Could not find the C++ standard of the LAMMPS build in ${LAMMPS_BUILD_DIR}: "
      "set CMAKE_CXX_STANDARD to the one LAMMPS used (its configure summary prints \"C++ Standard\")")
  elseif(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD "${_lmp_std}" CACHE STRING "C++ standard (from LAMMPS_BUILD_DIR)")
  elseif(NOT CMAKE_CXX_STANDARD STREQUAL _lmp_std)
    message(FATAL_ERROR "CMAKE_CXX_STANDARD (${CMAKE_CXX_STANDARD}) is not the C++ standard LAMMPS was "
      "built with (${_lmp_std}, from ${LAMMPS_BUILD_DIR}). LAMMPS's headers differ between them.")
  endif()

  # MPI or its serial stubs, and the integer sizes
  foreach(_v BUILD_MPI LAMMPS_SIZES)
    if(NOT DEFINED _lmp_${_v} OR _lmp_${_v} STREQUAL "")
      continue()
    endif()
    if(NOT DEFINED ${_v})
      set(${_v} "${_lmp_${_v}}" CACHE STRING "${_v} (from LAMMPS_BUILD_DIR)")
    else()
      string(TOLOWER "${${_v}}" _a)
      string(TOLOWER "${_lmp_${_v}}" _b)
      if(_v STREQUAL BUILD_MPI)  # a boolean, spelled any way
        set(_a ${${_v}})
        set(_b ${_lmp_${_v}})
        if(_a)
          set(_a on)
        else()
          set(_a off)
        endif()
        if(_b)
          set(_b on)
        else()
          set(_b off)
        endif()
      endif()
      if(NOT _a STREQUAL _b)
        message(FATAL_ERROR "${_v} (${${_v}}) differs from the LAMMPS build (${_lmp_${_v}}, from "
          "${LAMMPS_BUILD_DIR}): leave it unset to take LAMMPS's.")
      endif()
    endif()
  endforeach()

  if(NOT LAMMPS_SOURCE_DIR AND _lmp_lammps_SOURCE_DIR)
    get_filename_component(_src "${_lmp_lammps_SOURCE_DIR}/../src" ABSOLUTE)
    set(LAMMPS_SOURCE_DIR "${_src}" CACHE PATH "The src/ directory of the LAMMPS tree the binary was built from")
  endif()
  if(DEFINED _lmp_PKG_PLUGIN AND NOT _lmp_PKG_PLUGIN)
    message(WARNING "LAMMPS in ${LAMMPS_BUILD_DIR} was built without the PLUGIN package, so it cannot load plugins")
  endif()
endif()

project(mdzplugin LANGUAGES C CXX)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "Use a separate build directory")
endif()

set(LAMMPS_SOURCE_DIR "" CACHE PATH "The src/ directory of the LAMMPS tree the binary was built from")
if(NOT EXISTS "${LAMMPS_SOURCE_DIR}/lammps.h")
  message(FATAL_ERROR "Set LAMMPS_BUILD_DIR to the LAMMPS build directory, or LAMMPS_SOURCE_DIR to the LAMMPS src/ directory")
endif()
set(LAMMPS_HEADER_DIR ${LAMMPS_SOURCE_DIR})
if(NOT LAMMPS_BUILD_DIR)
  message(WARNING "Without LAMMPS_BUILD_DIR nothing checks that this plugin matches the LAMMPS binary. "
    "Build it with the same compiler (${CMAKE_CXX_COMPILER}) and C++ standard (CMAKE_CXX_STANDARD, "
    "${CMAKE_CXX_STANDARD}) as LAMMPS, and the same BUILD_MPI and LAMMPS_SIZES; or set LAMMPS_BUILD_DIR "
    "to have them taken from the LAMMPS build.")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Use LAMMPS's plugin support for header paths, MPI (or its serial stubs) and
# integer size settings, so the plugin matches the binary that loads it.
set(CMAKE_MODULE_PATH "${LAMMPS_SOURCE_DIR}/../examples/plugins")
include(CheckIncludeFileCXX)
include(LAMMPSInterfaceCXX)

# Everything compiled here, mdz included, is hidden unless marked otherwise
# (only lammpsplugin_init is).
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

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
# LAMMPS's headers differ between C++17 and C++20 (fmt or std::format), so
# the plugin must use the standard LAMMPS was built with; LAMMPS_BUILD_DIR
# sets CMAKE_CXX_STANDARD to match it. C++17 is the minimum.
target_compile_features(mdzplugin PRIVATE cxx_std_17)
# LAMMPS symbols stay undefined and resolve against the loading binary.
target_link_libraries(mdzplugin PRIVATE lammps mdz::mdz ${CMAKE_DL_LIBS})
if(NOT MDZ_PARALLEL)
  # dump mdz calls mdz::set_num_threads(1) to keep mdz on the calling thread.
  target_compile_definitions(mdzplugin PRIVATE MDZ_LAMMPS_SERIAL)
endif()
set_target_properties(mdzplugin PROPERTIES PREFIX "" SUFFIX ".so")

# Export lammpsplugin_init and nothing else. Hidden visibility does not reach
# the standard library's templates: libstdc++ declares namespace std with
# default visibility, so the plugin's own instantiations of, for example,
# std::format (which LAMMPS's fmt:: is in C++20) would be exported as weak
# symbols, and the dynamic linker would bind the plugin's calls to LAMMPS's
# copies of them. Those were compiled by LAMMPS's compiler; a different
# libstdc++ version can lay them out differently (std::format's ABI was not
# stable before GCC 15), and the plugin crashes. The version script makes
# every other symbol defined here local, so the plugin always runs its own
# code. -Bsymbolic-functions would do this only for functions, not for the
# data (static locals, guard variables) such templates also define.
if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
  target_link_options(mdzplugin PRIVATE "LINKER:-undefined,dynamic_lookup"
                      "LINKER:-exported_symbol,_lammpsplugin_init")
elseif(NOT CMAKE_SYSTEM_NAME STREQUAL "Windows")
  target_link_options(mdzplugin PRIVATE "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/mdzplugin.map"
                      "LINKER:--exclude-libs,ALL")
  set_property(TARGET mdzplugin APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mdzplugin.map")
endif()
