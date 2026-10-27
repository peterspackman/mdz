# zstd for mdz.
#
# MDZ_ZSTD=fetch (the default) compiles zstd 1.5.7 into the mdz library. A
# static libmdz then has no external dependency, and its compressed output is
# byte-identical to other builds of the same mdz version. MDZ_ZSTD=system links
# an installed zstd. Files remain readable either way, but if the zstd version
# differs the compressed bytes may differ from other builds.

if(MDZ_ZSTD STREQUAL "fetch")
  include(FetchContent)
  FetchContent_Declare(mdz_zstd_src
    URL https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz
    URL_HASH SHA256=eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3
    DOWNLOAD_EXTRACT_TIMESTAMP ON)
  FetchContent_GetProperties(mdz_zstd_src)
  if(NOT mdz_zstd_src_POPULATED)
    FetchContent_Populate(mdz_zstd_src)
  endif()
  set(_z ${mdz_zstd_src_SOURCE_DIR}/lib)
  file(GLOB _zstd_sources
    ${_z}/common/*.c
    ${_z}/compress/*.c
    ${_z}/decompress/*.c)
  add_library(mdz_zstd OBJECT ${_zstd_sources})
  target_include_directories(mdz_zstd PUBLIC ${_z})
  # Disable zstd's assembly for portability across toolchains. The C fallback
  # produces the same output.
  target_compile_definitions(mdz_zstd PUBLIC ZSTD_DISABLE_ASM=1 ZSTD_STATIC_LINKING_ONLY=1
                                      PRIVATE XXH_NAMESPACE=MDZ_ZSTD_ ZSTD_LEGACY_SUPPORT=0)
  set_target_properties(mdz_zstd PROPERTIES POSITION_INDEPENDENT_CODE ON C_VISIBILITY_PRESET hidden)
  # Third-party code: build it without warnings, and never fail on them.
  set_target_properties(mdz_zstd PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
  if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mdz_zstd PRIVATE -O3 -w)
  elseif(MSVC)
    target_compile_options(mdz_zstd PRIVATE /w)
  endif()
elseif(MDZ_ZSTD STREQUAL "system")
  # MDZ_ZSTD_LINK is what the mdz target links. It is either an imported
  # target from zstd's own CMake package, which the installed mdzConfig.cmake
  # then finds with find_dependency, or the library paths reported by
  # pkg-config, which the install records directly.
  find_package(zstd CONFIG QUIET)
  if(TARGET zstd::libzstd_static)
    set(MDZ_ZSTD_LINK zstd::libzstd_static)
    set(MDZ_ZSTD_FIND_DEPENDENCY ON)
  elseif(TARGET zstd::libzstd_shared)
    set(MDZ_ZSTD_LINK zstd::libzstd_shared)
    set(MDZ_ZSTD_FIND_DEPENDENCY ON)
  else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(ZSTD REQUIRED libzstd)
    set(MDZ_ZSTD_LINK ${ZSTD_LINK_LIBRARIES})
    set(MDZ_ZSTD_INCLUDE ${ZSTD_INCLUDE_DIRS})
    set(MDZ_ZSTD_FIND_DEPENDENCY OFF)
  endif()
else()
  message(FATAL_ERROR "MDZ_ZSTD must be fetch or system, not '${MDZ_ZSTD}'")
endif()
