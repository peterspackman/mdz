# The Python extension mdz._core (nanobind). Built by scikit-build-core from
# pyproject.toml, or directly with -DMDZ_BUILD_PYTHON=ON.

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

# Use nanobind from the Python environment (it is a build requirement in
# pyproject.toml). Fetch it if it is not installed.
if(NOT nanobind_DIR)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE _mdz_nanobind_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _mdz_nanobind_found
    ERROR_QUIET)
  if(_mdz_nanobind_found EQUAL 0)
    set(nanobind_DIR "${_mdz_nanobind_dir}")
  endif()
endif()
find_package(nanobind CONFIG QUIET)
if(NOT nanobind_FOUND)
  include(FetchContent)
  FetchContent_Declare(nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v2.4.0
    GIT_SHALLOW ON)
  FetchContent_MakeAvailable(nanobind)
endif()

nanobind_add_module(_core
  NB_STATIC
  common.cpp
  module.cpp
  reader.cpp
  writer.cpp)
target_link_libraries(_core PRIVATE mdz::mdz)
mdz_strict_fp(_core)

if(SKBUILD)
  # Install into the mdz package, next to the Python sources (wheel.packages).
  install(TARGETS _core LIBRARY DESTINATION mdz)
endif()
