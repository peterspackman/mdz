add_executable(bench_encode bench_encode.cpp)
target_link_libraries(bench_encode PRIVATE mdz::mdz)
mdz_strict_fp(bench_encode)

add_executable(bench_files bench_files.cpp)
target_link_libraries(bench_files PRIVATE mdz::mdz)
# The raw-floats baseline uses the library's internal zstd wrapper.
target_include_directories(bench_files PRIVATE ${PROJECT_SOURCE_DIR}/src)
mdz_strict_fp(bench_files)
