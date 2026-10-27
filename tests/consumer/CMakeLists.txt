# A minimal downstream project that uses mdz in one of the two supported ways:
#   -DMODE=subdir   add_subdirectory(<mdz source>), as the LAMMPS plugin does
#   -DMODE=package  find_package(mdz) against an installed copy
cmake_minimum_required(VERSION 3.18)
project(mdz_consumer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

set(MODE "subdir" CACHE STRING "subdir or package")
if(MODE STREQUAL "subdir")
  add_subdirectory(${MDZ_SOURCE_DIR} ${CMAKE_BINARY_DIR}/mdz EXCLUDE_FROM_ALL)
else()
  find_package(mdz REQUIRED)
endif()

add_executable(consumer consumer.cpp)
target_link_libraries(consumer PRIVATE mdz::mdz)
