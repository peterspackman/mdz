# Header-only dependencies used inside the library (never in its public
# headers): fast_float for parsing numbers in text formats, unordered_dense
# for hash maps on hot paths.
include(FetchContent)

FetchContent_Declare(mdz_fast_float
  URL https://github.com/fastfloat/fast_float/archive/refs/tags/v8.0.2.tar.gz
  URL_HASH SHA256=e14a33089712b681d74d94e2a11362643bd7d769ae8f7e7caefe955f57f7eacd
  DOWNLOAD_EXTRACT_TIMESTAMP ON
  SOURCE_SUBDIR none)
FetchContent_Declare(mdz_unordered_dense
  URL https://github.com/martinus/unordered_dense/archive/refs/tags/v4.5.0.tar.gz
  URL_HASH SHA256=2364ce4bc4c23bd02549bbb3a7572d881684cd46057f3737fd53be53669743aa
  DOWNLOAD_EXTRACT_TIMESTAMP ON
  SOURCE_SUBDIR none)
# SOURCE_SUBDIR has no CMakeLists.txt, so MakeAvailable only downloads the
# headers and adds none of the projects' own targets.
FetchContent_MakeAvailable(mdz_fast_float mdz_unordered_dense)
set(MDZ_HEADER_DEPS
  ${mdz_fast_float_SOURCE_DIR}/include
  ${mdz_unordered_dense_SOURCE_DIR}/include)
