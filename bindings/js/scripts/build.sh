#!/usr/bin/env bash
# Build the WebAssembly core with emscripten and copy it into dist/.
#   scripts/build.sh [Release|Debug]
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
root="$(cd "$here/../.." && pwd)"
build="$here/build-wasm"

type="${1:-Release}"
case "$type" in
  Release | Debug | RelWithDebInfo | MinSizeRel) ;;
  *) echo "unknown build type '$type' (Release, Debug, RelWithDebInfo or MinSizeRel)" >&2; exit 2 ;;
esac

emcmake cmake -S "$root" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE="$type"
cmake --build "$build" --target mdz_core mdz_full
rm -rf "$here/dist"
mkdir -p "$here/dist"
for m in mdz_core mdz_full; do
  cp "$build/bindings/js/$m.mjs" "$build/bindings/js/$m.wasm" "$here/dist/"
done
# The package ships the licenses of everything compiled into the wasm.
cp "$root/LICENSE" "$root/THIRD_PARTY_LICENSES" "$here/dist/"
