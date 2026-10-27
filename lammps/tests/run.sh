#!/bin/sh
# usage: run.sh LMP PLUGIN MDZ_CLI   (writes into a temporary directory)
#   MDZ_CLI is the mdz command-line tool (build/cli/mdz). in.rigid needs
#   LAMMPS's MOLECULE and RIGID packages, and is skipped without them.
#   Set MPIRUN to run LAMMPS in parallel, for example MPIRUN="mpirun -np 4".
set -eu
lmp=$1 plugin=$2 cli=$3
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
run() { ${MPIRUN:-} "$lmp" -var plugin "$plugin" -var dir "$here" -in "$here/$1" -log none -screen none; }
run in.smoke
run in.variable
if "$lmp" -h | grep -q 'rigid/nve' && "$lmp" -h | grep -qw molecular; then
  run in.rigid
fi
python3 "$here/check.py" "$cli"
