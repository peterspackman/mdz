cmake_minimum_required(VERSION 3.18)
project(mdz VERSION 0.4.0 LANGUAGES C CXX)

# When mdz is built on its own it is the top-level project. When another
# project includes it with add_subdirectory, only the library is built.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(MDZ_TOP_LEVEL ON)
else()
  set(MDZ_TOP_LEVEL OFF)
endif()

option(MDZ_BUILD_CLI "Build the mdz command-line tool" ${MDZ_TOP_LEVEL})
option(MDZ_BUILD_TESTS "Build the C++ tests" ${MDZ_TOP_LEVEL})
option(MDZ_BUILD_BENCH "Build the benchmarks" ${MDZ_TOP_LEVEL})
option(MDZ_BUILD_PYTHON "Build the Python extension (nanobind)" OFF)
option(MDZ_UNITY_BUILD "Compile the library as a few large translation units" OFF)
option(MDZ_FUZZ "Build the fuzz targets (libFuzzer with clang)" OFF)
option(MDZ_SANITIZE "Build with address and undefined-behaviour sanitizers" OFF)
set(MDZ_ZSTD "fetch" CACHE STRING "Where zstd comes from: fetch (build 1.5.7 from source) or system")
set_property(CACHE MDZ_ZSTD PROPERTY STRINGS fetch system)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(cmake/zstd.cmake)
include(cmake/deps.cmake)

# ---- library ---------------------------------------------------------------

set(MDZ_SOURCES
  src/bytes.cpp
  src/codec.cpp
  src/column.cpp
  src/columns.cpp
  src/container.cpp
  src/crc32.cpp
  src/error.cpp
  src/file_io.cpp
  src/frame_batch.cpp
  src/kernels.cpp
  src/library.cpp
  src/mapped_file.cpp
  src/metadata.cpp
  src/options.cpp
  src/parallel.cpp
  src/reader.cpp
  src/rigid.cpp
  src/trajectory.cpp
  src/units.cpp
  src/variable.cpp
  src/writer.cpp
  src/xdr3d.cpp
  src/zstd.cpp
  src/streams.cpp
  src/formats/registry.cpp
  src/formats/text.cpp
  src/formats/mdz_format.cpp
  src/formats/dcd.cpp
  src/formats/xtc.cpp
  src/formats/npy.cpp
  src/formats/extxyz.cpp
  src/formats/lammps.cpp
  src/formats/pdb.cpp
  src/formats/xyz.cpp
  src/capi.cpp
)

add_library(mdz ${MDZ_SOURCES})
add_library(mdz::mdz ALIAS mdz)
# C++17 at least; a newer standard set by the compiler or CMAKE_CXX_STANDARD is kept.
target_compile_features(mdz PUBLIC cxx_std_17)
target_include_directories(mdz
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(mdz SYSTEM PRIVATE ${MDZ_HEADER_DEPS})
if(MDZ_ZSTD STREQUAL "fetch")
  # Compile zstd's objects into mdz rather than linking a separate library, so
  # a static libmdz is self-contained and exports no extra target.
  target_sources(mdz PRIVATE $<TARGET_OBJECTS:mdz_zstd>)
  target_include_directories(mdz PRIVATE $<TARGET_PROPERTY:mdz_zstd,INTERFACE_INCLUDE_DIRECTORIES>)
  target_compile_definitions(mdz PRIVATE $<TARGET_PROPERTY:mdz_zstd,INTERFACE_COMPILE_DEFINITIONS>)
else()
  target_link_libraries(mdz PRIVATE ${MDZ_ZSTD_LINK})
  if(MDZ_ZSTD_INCLUDE)
    target_include_directories(mdz SYSTEM PRIVATE ${MDZ_ZSTD_INCLUDE})
  endif()
endif()
find_package(Threads REQUIRED)
target_link_libraries(mdz PUBLIC Threads::Threads)
target_compile_definitions(mdz PRIVATE MDZ_VERSION_STRING="${PROJECT_VERSION}")
if(MDZ_UNITY_BUILD)
  set_target_properties(mdz PROPERTIES UNITY_BUILD ON UNITY_BUILD_BATCH_SIZE 8)
endif()

# Every target that does floating-point work on mdz data must round the same
# way, so fused multiply-adds and reassociation are disabled. The encoder
# predicts from its own reconstruction, and a decoder that rounds differently
# would drift from it.
function(mdz_strict_fp target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -ffp-contract=off -fno-fast-math)
  elseif(MSVC)
    target_compile_options(${target} PRIVATE /fp:precise)
  endif()
endfunction()

# fopen, strerror and getenv are standard C; MSVC deprecates them in favour of
# its _s variants. Applies to this directory tree only.
if(MSVC)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

function(mdz_warnings target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wno-unused-parameter)
  elseif(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8)
  endif()
endfunction()

mdz_strict_fp(mdz)
mdz_warnings(mdz)

if(MDZ_SANITIZE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
  target_compile_options(mdz PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(mdz PUBLIC -fsanitize=address,undefined)
endif()

# ---- tools, tests, bindings ------------------------------------------------

if(MDZ_BUILD_CLI)
  add_subdirectory(cli)
endif()

if(MDZ_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()

if(MDZ_FUZZ)
  add_subdirectory(fuzz)
endif()

if(MDZ_BUILD_BENCH)
  add_subdirectory(bench)
endif()

if(MDZ_BUILD_PYTHON)
  add_subdirectory(bindings/python)
endif()

# ---- install ---------------------------------------------------------------

if(MDZ_TOP_LEVEL AND NOT SKBUILD)
  include(GNUInstallDirs)
  install(TARGETS mdz EXPORT mdzTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(DIRECTORY include/mdz DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(EXPORT mdzTargets NAMESPACE mdz:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mdz)
  include(CMakePackageConfigHelpers)
  configure_package_config_file(cmake/mdzConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/mdzConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mdz)
  write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/mdzConfigVersion.cmake
    COMPATIBILITY SameMinorVersion)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mdzConfig.cmake
                ${CMAKE_CURRENT_BINARY_DIR}/mdzConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mdz)
endif()
