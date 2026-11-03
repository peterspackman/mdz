# The WebAssembly core of the JavaScript package: an ES module mdz_core.mjs
# and mdz_core.wasm. Configure with emcmake; scripts/build.sh does it all.

add_executable(mdz_core src/bindings.cpp)
target_link_libraries(mdz_core PRIVATE mdz::mdz)
mdz_strict_fp(mdz_core)
mdz_warnings(mdz_core)
set_target_properties(mdz_core PROPERTIES SUFFIX ".mjs")
target_link_options(mdz_core PRIVATE
  -lembind
  -sMODULARIZE=1
  -sEXPORT_ES6=1
  -sEXPORT_NAME=createMdzCore
  -sENVIRONMENT=web,worker,node
  -sALLOW_MEMORY_GROWTH=1
  -sMAXIMUM_MEMORY=4GB
  # Only the bound functions are needed; nothing reads files or stdin.
  -sFILESYSTEM=0
  -sINCOMING_MODULE_JS_API=locateFile,wasmBinary,instantiateWasm
  # The tests check that failed calls release their memory.
  -sEXPORTED_RUNTIME_METHODS=HEAPU8
  --emit-tsd=mdz_core.d.ts)
