# The C API test is a C99 program linked against mdz::mdz. It checks that
# mdz.h compiles as C and that it matches the implementation.
add_executable(mdz_capi_test test_capi.c)
set_target_properties(mdz_capi_test PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
                                               LINKER_LANGUAGE CXX)
target_link_libraries(mdz_capi_test PRIVATE mdz::mdz)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(mdz_capi_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_link_libraries(mdz_capi_test PRIVATE m)
elseif(MSVC)
  target_compile_options(mdz_capi_test PRIVATE /W4 /WX)
endif()
add_test(NAME mdz_capi_test
         COMMAND mdz_capi_test ${CMAKE_CURRENT_BINARY_DIR}/capi_test.mdz)
