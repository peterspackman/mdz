#!/usr/bin/env bash
# Build the WebAssembly core with emscripten and copy it into dist/.
#   scripts/build.sh [Release|Debug]
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
root="$(cd "$here/../.." && pwd)"
build="$here/build-wasm"

emcmake cmake -S "$root" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE="${1:-Release}"
cmake --build "$build" --target mdz_core
mkdir -p "$here/dist"
cp "$build/bindings/js/mdz_core.mjs" "$build/bindings/js/mdz_core.wasm" "$here/dist/"
cp "$build/mdz_core.d.ts" "$here/dist/" 2>/dev/null || cp "$build/bindings/js/mdz_core.d.ts" "$here/dist/"
# The package ships the licenses of everything compiled into the wasm.
cp "$root/LICENSE" "$root/THIRD_PARTY_LICENSES" "$here/dist/"
