#!/bin/sh
# usage: run.sh LMP PLUGIN MDZ_CLI   (writes into a temporary directory)
#   PLUGIN is the path to mdzplugin.so, or "builtin" for a LAMMPS with dump
#   mdz compiled in (the package MDZ). MDZ_CLI is the mdz command-line tool
#   (build/cli/mdz). in.rigid needs LAMMPS's MOLECULE and RIGID packages,
#   and is skipped without them.
#   Set MPIRUN to run LAMMPS in parallel, for example MPIRUN="mpirun -np 4".
set -eu
lmp=$1 plugin=$2 cli=$3
here=$(cd "$(dirname "$0")" && pwd)
# the runs are in a temporary directory, so make relative paths absolute
abs() { case $1 in /*) echo "$1" ;; */*) echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")" ;; *) echo "$1" ;; esac; }
lmp=$(abs "$lmp") cli=$(abs "$cli")
help=$("$lmp" -h)
# the dump styles LAMMPS lists, after "* Dump styles:" and up to the next "*"
builtin=$(printf '%s\n' "$help" | sed -n '/^\* Dump styles/,/^\* /p' | grep -cw mdz || true)
if [ "$plugin" = builtin ]; then
  [ "$builtin" -gt 0 ] || { echo "$lmp has no dump mdz compiled in" >&2; exit 1; }
else
  # the inputs load the plugin only without -var builtin yes
  [ "$builtin" -eq 0 ] || {
    echo "$lmp has dump mdz already (compiled in, or a plugin loaded from LAMMPS_PLUGIN_PATH): use PLUGIN=builtin" >&2
    exit 1
  }
  case $plugin in */*) plugin=$(abs "$plugin") ;; *) plugin=$(abs "./$plugin") ;; esac
fi
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
run() {
  input=$here/$1
  if [ "$plugin" = builtin ]; then set -- -var builtin yes; else set -- -var plugin "$plugin"; fi
  ${MPIRUN:-} "$lmp" "$@" -var dir "$here" -in "$input" -log none -screen none
}
run in.smoke
run in.variable
if printf '%s\n' "$help" | grep -q 'rigid/nve' && printf '%s\n' "$help" | grep -qw molecular; then
  run in.rigid
fi
python3 "$here/check.py" "$cli"
