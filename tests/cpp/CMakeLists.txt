add_executable(mdz_tests
  main.cpp
  test_kernels.cpp
  test_codec.cpp
  test_container.cpp
  test_golden.cpp
  test_writer.cpp
  test_formats_binary.cpp
  test_formats_text.cpp
  test_formats_variable.cpp
  test_variable.cpp
  test_rigid.cpp
)
target_link_libraries(mdz_tests PRIVATE mdz::mdz)
# The tests also use internal headers (kernels, codecs) from src/.
target_include_directories(mdz_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(mdz_tests SYSTEM PRIVATE ${MDZ_HEADER_DEPS})
target_compile_definitions(mdz_tests PRIVATE
  MDZ_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden"
  MDZ_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
  MDZ_EXAMPLES_DIR="${PROJECT_SOURCE_DIR}/examples")
mdz_strict_fp(mdz_tests)
mdz_warnings(mdz_tests)
add_test(NAME mdz_tests COMMAND mdz_tests)
add_subdirectory(capi)
