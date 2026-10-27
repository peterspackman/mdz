#!/bin/bash
# Example: compress a DCD trajectory with mdz.
#
# mdz converts DCD's Angstrom coordinates to nm on read, so precision is
# always expressed in mdz's own units: --max-error is in Angstroms and
# defaults to 0.005 (XTC parity). Nothing here needs a unit override.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MDZ="${MDZ:-${SCRIPT_DIR}/../build/cli/mdz}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

for INPUT in traj.dcd every-frame.dcd; do
    echo "=================================================================="
    echo " $INPUT"
    echo "=================================================================="
    "$MDZ" info "$SCRIPT_DIR/$INPUT"

    echo ""
    echo "=== Compressing (default 0.005 A max error) ==="
    "$MDZ" compress "$SCRIPT_DIR/$INPUT" -o "$OUT/out.mdz" --force --stats

    echo ""
    echo "=== Round-tripping back to DCD ==="
    "$MDZ" decompress "$OUT/out.mdz" -o "$OUT/recon.dcd" --force

    echo ""
    ls -lh "$SCRIPT_DIR/$INPUT" "$OUT/out.mdz"
    echo ""
done

# traj.dcd saves every 20000 steps, so consecutive frames are nearly
# uncorrelated and mdz falls back to XTC-style intra-frame coding.
# every-frame.dcd saves every step, which is where inter-frame prediction
# pays off, giving a much higher ratio.
