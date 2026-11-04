# Two ES modules from one source: mdz_core (reading .mdz) and mdz_full (also
# writing .mdz and the other formats). The reader is a fraction of the size,
# so a page that only reads never loads the encoder. Configure with emcmake;
# scripts/build.sh does it all.

function(mdz_js_module target export_name)
  add_executable(${target} src/bindings.cpp)
  target_link_libraries(${target} PRIVATE mdz::mdz)
  mdz_strict_fp(${target})
  mdz_warnings(${target})
  set_target_properties(${target} PROPERTIES SUFFIX ".mjs")
  target_link_options(${target} PRIVATE
    -lembind
    -sMODULARIZE=1
    -sEXPORT_ES6=1
    -sEXPORT_NAME=${export_name}
    -sENVIRONMENT=web,worker,node
    -sALLOW_MEMORY_GROWTH=1
    -sMAXIMUM_MEMORY=4GB
    -sFILESYSTEM=0
    -sINCOMING_MODULE_JS_API=locateFile,wasmBinary,instantiateWasm)
endfunction()

mdz_js_module(mdz_core createMdzCore)
target_compile_definitions(mdz_core PRIVATE MDZ_JS_READ_ONLY)
mdz_js_module(mdz_full createMdzFull)
