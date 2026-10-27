# Fuzz targets. With clang, -DMDZ_FUZZ=ON builds them against libFuzzer:
#
#   CC=clang CXX=clang++ cmake -S . -B build-fuzz -DMDZ_FUZZ=ON -DMDZ_SANITIZE=ON
#   build-fuzz/fuzz/gen_corpus corpus
#   build-fuzz/fuzz/fuzz_mdz_file corpus/mdz_file -max_total_time=120
#
# Other compilers get replay binaries that run a target over saved inputs.

add_executable(gen_corpus gen_corpus.cpp)
target_link_libraries(gen_corpus PRIVATE mdz::mdz)
target_include_directories(gen_corpus PRIVATE ${PROJECT_SOURCE_DIR}/src)

foreach(t mdz_file gof_payload metadata formats)
  add_executable(fuzz_${t} fuzz_${t}.cpp)
  target_link_libraries(fuzz_${t} PRIVATE mdz::mdz)
  target_include_directories(fuzz_${t} PRIVATE ${PROJECT_SOURCE_DIR}/src)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(fuzz_${t} PRIVATE -fsanitize=fuzzer)
    target_link_options(fuzz_${t} PRIVATE -fsanitize=fuzzer)
  else()
    target_sources(fuzz_${t} PRIVATE replay_main.cpp)
  endif()
endforeach()
