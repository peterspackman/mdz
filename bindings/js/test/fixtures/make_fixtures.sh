#!/usr/bin/env bash
# Regenerate the test fixtures with the native mdz CLI.
#
#   bindings/js/test/fixtures/make_fixtures.sh [path/to/mdz]
#
# The .npy files are the native decoder's output for the same .mdz files, so
# the tests check that the wasm build decodes exactly what the native one does
# (positions in nm).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../../.." && pwd)"
mdz="${1:-$root/build/cli/mdz}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# LAMMPS dump: box, velocities and step, in three groups of frames.
"$mdz" -q compress -f --gof-size 2 "$root/examples/lammps/nacl.lammpstrj" -o "$here/nacl.mdz"
"$mdz" -q decompress -f "$here/nacl.mdz" -o "$here/nacl.positions.npy" 2>/dev/null

# DCD with a unit cell; the first three frames keep the fixture small.
"$mdz" -q compress -f "$root/examples/traj.dcd" -o "$tmp/dcd_full.mdz"
"$mdz" -q extract -f "$tmp/dcd_full.mdz" 0:3 -o "$tmp/dcd3.dcd"
"$mdz" -q compress -f "$tmp/dcd3.dcd" -o "$here/dcd.mdz"
"$mdz" -q decompress -f "$here/dcd.mdz" -o "$here/dcd.positions.npy" 2>/dev/null

# Atoms that come and go, stored exactly.
"$mdz" -q compress -f --lossless "$here/gcmc.lammpstrj" -o "$here/gcmc.mdz"

# A file whose writer never finished: drop the footer and trailer and cut
# into the last group, leaving the first two complete groups readable.
size=$(wc -c < "$here/nacl.mdz")
head -c $((size * 3 / 4)) "$here/nacl.mdz" > "$here/nacl_truncated.mdz"
