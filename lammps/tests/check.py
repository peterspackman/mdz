"""Check dump mdz output from in.smoke, in.variable and (if it ran) in.rigid
against LAMMPS's own dump custom.

usage: python3 check.py MDZ_CLI   (run in the directory the inputs ran in)

MDZ_CLI is the mdz command-line tool, run as
`MDZ_CLI -q decompress -f IN.mdz -o OUT.lammpstrj`.
"""
import os
import subprocess
import sys


def frames(path):
    lines = open(path).read().split("\n")
    i, out = 0, []
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        fr = {}
        if lines[i] == "ITEM: TIME":
            fr["time"] = float(lines[i + 1])
            i += 2
        assert lines[i] == "ITEM: TIMESTEP", lines[i]
        fr["step"] = int(lines[i + 1])
        n = int(lines[i + 3])
        fr["bounds"] = lines[i + 4]
        fr["box"] = [float(v) for k in range(3) for v in lines[i + 5 + k].split()]
        cols = lines[i + 8].split()[2:]
        fr["atoms"] = {}
        for line in lines[i + 9 : i + 9 + n]:
            v = line.split()
            fr["atoms"][int(v[0])] = dict(zip(cols, map(float, v)))
        out.append(fr)
        i += 9 + n
    return out


def compare(ref, got, rel=None, abs_=None):
    assert len(ref) == len(got), f"{len(ref)} reference frames, {len(got)} in mdz"
    worst = 0.0
    for a, b in zip(ref, got):
        assert a["step"] == b["step"], (a["step"], b["step"])
        assert abs(a["time"] - b["time"]) <= 1e-12 * max(1, abs(a["time"])), (a["time"], b["time"])
        assert a["bounds"] == b["bounds"], (a["bounds"], b["bounds"])
        for x, y in zip(a["box"], b["box"]):
            assert abs(x - y) <= 1e-12 * max(1, abs(x)), (a["box"], b["box"])
        assert a["atoms"].keys() == b["atoms"].keys()
        for i, ra in a["atoms"].items():
            rb = b["atoms"][i]
            for col, v in rb.items():
                if col in ("id", "type", "ix", "iy", "iz"):
                    assert ra[col] == v, (i, col, ra[col], v)
                    continue
                d = abs(ra[col] - v)
                limit = rel * max(abs(ra[col]), 1e-30) if rel else abs_
                if rel and abs(ra[col]) < 1e-20:
                    limit = 1e-30
                assert d <= limit, (a["step"], i, col, ra[col], v)
                if col in "xyz":
                    worst = max(worst, d)
    return worst


def decompress(cli, src, dst):
    subprocess.run([cli, "-q", "decompress", "-f", src, "-o", dst], check=True)


cli = sys.argv[1]
ref1, ref2 = frames("ref.dump"), frames("ref2.dump")

decompress(cli, "exact.mdz", "exact.lammpstrj")
# lossless: each value is rounded to float32 and printed in shortest form
compare(ref1 + ref2, frames("exact.lammpstrj"), rel=1.2e-7)

decompress(cli, "lossy.mdz", "lossy.lammpstrj")
# adaptive precision: error is within 4 quantization steps
worst = compare(ref1, frames("lossy.lammpstrj"), abs_=4 * 0.0005 + 1e-9)
print(f"ok: lossless exact to float32; lossy max error {worst:.5f} A (limit 0.002)")

# atoms deposited, then deleted: each frame has its own atoms, matched by ID
ref = frames("ref_var.dump")
counts = [len(f["atoms"]) for f in ref]
assert len(set(counts)) > 1, counts
decompress(cli, "var.mdz", "var.lammpstrj")
compare(ref, frames("var.lammpstrj"), rel=1.2e-7)
# without variable yes: only frames with the same atoms as the first are kept
decompress(cli, "fixed.mdz", "fixed.lammpstrj")
same = [f for f in ref if f["atoms"].keys() == ref[0]["atoms"].keys()]
compare(same, frames("fixed.lammpstrj"), rel=1.2e-7)
print(f"ok: varying atoms {min(counts)}-{max(counts)} per frame, exact; "
      f"fixed file kept {len(same)} of {len(ref)} frames")

if os.path.exists("ref_rigid.dump"):
    decompress(cli, "rigid.mdz", "rigid.lammpstrj")
    # within 4 steps of the float32 input, plus float32 rounding (~1e-6 for
    # coordinates up to about 16 A)
    worst = compare(frames("ref_rigid.dump"), frames("rigid.lammpstrj"), abs_=4 * 0.0005 + 2e-6)
    print(f"ok: rigid molecules max error {worst:.5f} A (limit 0.002)")
