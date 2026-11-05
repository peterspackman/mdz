#!/bin/sh
# Add dump style mdz to a LAMMPS source tree as the package MDZ, then build
# LAMMPS with CMake and -D PKG_MDZ=ON.
#
#   patch_lammps.sh /path/to/lammps          copy the sources in
#   patch_lammps.sh /path/to/lammps --link   symlink them, for working on them
#
# LAMMPS then builds the mdz library from this checkout (recorded in the
# tree; override with -D MDZ_SOURCE_DIR=/path/to/mdz) with its own compiler
# and flags, and links it in. Running the script again updates the tree.
set -eu
usage="usage: patch_lammps.sh /path/to/lammps [--link]"
[ $# -ge 1 ] || { echo "$usage" >&2; exit 1; }
lammps=$(cd "$1" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
mdz=$(dirname "$here")
link=0
case "${2:-}" in
  "") ;;
  --link) link=1 ;;
  *) echo "$usage" >&2; exit 1 ;;
esac
cmake="$lammps/cmake/CMakeLists.txt"
[ -f "$cmake" ] && [ -f "$lammps/src/lammps.h" ] || { echo "not a LAMMPS source tree: $lammps" >&2; exit 1; }
[ -f "$mdz/include/mdz/mdz.h" ] || { echo "not an mdz source tree: $mdz" >&2; exit 1; }

put() {  # put SRC DST: copy, or symlink with --link
  rm -rf "$2"
  if [ $link = 1 ]; then ln -s "$1" "$2"; else cp "$1" "$2"; fi
}

# The package: the dump style alone. mdzplugin.cpp is only for the plugin.
mkdir -p "$lammps/src/MDZ"
put "$here/dump_mdz.h" "$lammps/src/MDZ/dump_mdz.h"
put "$here/dump_mdz.cpp" "$lammps/src/MDZ/dump_mdz.cpp"
put "$here/package/README" "$lammps/src/MDZ/README"
sed "s|@MDZ_SOURCE_DIR@|$mdz|" "$here/package/MDZ.cmake" > "$lammps/cmake/Modules/Packages/MDZ.cmake"

# Register the package: its name in the list of standard packages, and its
# CMake module, included just before LAMMPS registers the styles.
if ! grep -q "^ *MDZ$" "$cmake"; then
  sed -i 's/^set(STANDARD_PACKAGES$/&\n  MDZ/' "$cmake"
fi
if ! grep -q "include(Packages/MDZ)" "$cmake"; then
  sed -i 's/^include(StyleHeaderUtils)$/# package MDZ (dump mdz), added by patch_lammps.sh from mdz\nif(PKG_MDZ)\n  include(Packages\/MDZ)\nendif()\n&/' "$cmake"
fi
grep -q "^ *MDZ$" "$cmake" && grep -q "include(Packages/MDZ)" "$cmake" || {
  echo "could not register MDZ in $cmake" >&2
  exit 1
}
echo "MDZ added to $lammps (mdz: $mdz). Configure LAMMPS with -D PKG_MDZ=ON."
