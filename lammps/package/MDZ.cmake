# MDZ: dump style mdz, which writes compressed, crash-tolerant .mdz
# trajectories (https://github.com/peterspackman/mdz). Put in place by
# lammps/patch_lammps.sh in the mdz repository.
#
#   MDZ_SOURCE_DIR  the mdz source tree (default: the one that patched this tree)
#   MDZ_PARALLEL    let mdz start worker threads of its own (default off)
#   MDZ_ZSTD        fetch (default: download zstd 1.5.7 and compile it in) or
#                   system (link an installed zstd)
#
# The mdz library is built here, with LAMMPS's compiler, flags and C++
# standard, as a static library of position-independent code (so it also
# goes into a shared liblammps), and linked into LAMMPS. Only the library is
# built: none of mdz's command-line tool, tests, benchmarks or bindings.
set(MDZ_SOURCE_DIR "@MDZ_SOURCE_DIR@" CACHE PATH "mdz source tree")
if(NOT EXISTS "${MDZ_SOURCE_DIR}/CMakeLists.txt" OR NOT EXISTS "${MDZ_SOURCE_DIR}/include/mdz/mdz.h")
  message(FATAL_ERROR "MDZ: MDZ_SOURCE_DIR (${MDZ_SOURCE_DIR}) is not an mdz source tree")
endif()
option(MDZ_PARALLEL "Let mdz use worker threads of its own inside LAMMPS" OFF)

# Included from LAMMPS's own directory, so these would leak into the rest of
# its configuration: mdz's options are all MDZ_*, and BUILD_SHARED_LIBS is
# put back to LAMMPS's setting afterwards.
set(MDZ_BUILD_CLI OFF)
set(MDZ_BUILD_TESTS OFF)
set(MDZ_BUILD_BENCH OFF)
set(MDZ_BUILD_PYTHON OFF)
set(MDZ_BUILD_JS OFF)
set(BUILD_SHARED_LIBS OFF)
add_subdirectory(${MDZ_SOURCE_DIR} ${CMAKE_BINARY_DIR}/mdz EXCLUDE_FROM_ALL)
unset(BUILD_SHARED_LIBS)

target_link_libraries(lammps PRIVATE mdz::mdz)
if(NOT MDZ_PARALLEL)
  # dump mdz then calls mdz::set_num_threads(1), keeping mdz on the thread of
  # the rank that writes the file
  target_compile_definitions(lammps PRIVATE MDZ_LAMMPS_SERIAL)
endif()
