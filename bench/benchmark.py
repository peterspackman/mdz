"""Benchmark mdz on a set of trajectories and write the results as Markdown.

Runs bench/bench_files on every file given, which measures the codecs in
memory alongside XTC and zstd over the raw floats. It then times the CLI end
to end, including disk I/O, on the largest file, and prints Markdown tables.

    python bench/benchmark.py --build build traj/*.dcd examples/traj.dcd > results.md
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path


def run_bench(exe, files, threads, repeat, tmp):
    cmd = [exe, "--threads", ",".join(map(str, threads)), "--repeat", str(repeat), "--tmp", tmp, *files]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


def mb(n):
    return n / 1e6


def label(path):
    """Split a name like water_100.dcd into ('water', 100). Other names give
    (stem, None)."""
    stem = Path(path).stem
    name, _, interval = stem.rpartition("_")
    if name and interval.isdigit():
        return name, int(interval)
    return stem, None


def sort_key(path):
    name, interval = label(path)
    return (name, interval if interval is not None else -1)


def describe(path, rows):
    r = rows[0]
    name, interval = label(path)
    base = f"{name} ({r['natoms']} atoms, {r['nframes']} frames"
    if interval is not None:
        step_fs = {"water": 2, "argon": 4}.get(name)
        if step_fs:
            ps = interval * step_fs / 1000
            base += f", saved every {ps:g} ps"
        else:
            base += f", every {interval} steps"
    return base + ")"


def size_table(by_file):
    out = [
        "| trajectory | raw MB | XTC | mdz | mdz / XTC size | mdz codec choice | mdz, rigid bodies | lossless mdz | zstd (raw floats) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for path in sorted(by_file, key=sort_key):
        rows = {(r["method"], r["threads"]): r for r in by_file[path]}
        t1 = min(r["threads"] for r in by_file[path])

        def get(m):
            return rows.get((m, t1)) or rows.get((m, 1))

        raw = get("mdz")["raw_bytes"]
        xtc, mdz, ll, z, rig = get("xtc"), get("mdz"), get("mdz-lossless"), get("zstd-raw"), get("mdz-rigid")
        tmp, intra = get("mdz-temporal"), get("mdz-intra")
        choice = "temporal" if mdz["bytes"] == tmp["bytes"] else "intra" if mdz["bytes"] == intra["bytes"] else "mixed"
        out.append(
            f"| {describe(path, by_file[path])} | {mb(raw):.0f} "
            f"| {raw / xtc['bytes']:.1f}x | {raw / mdz['bytes']:.1f}x | {mdz['bytes'] / xtc['bytes']:.2f} "
            f"| {choice} | {raw / rig['bytes']:.1f}x | {raw / ll['bytes']:.2f}x | {raw / z['bytes']:.2f}x |"
        )
    return "\n".join(out)


def precision_table(by_file):
    methods = [("mdz-0.001A", "0.001 Å"), ("mdz", "0.005 Å (default)"), ("mdz-0.05A", "0.05 Å"), ("xtc", "XTC")]
    out = [
        "| trajectory | " + " | ".join(m[1] for m in methods) + " |",
        "|---|" + "---|" * len(methods),
    ]
    for path in sorted(by_file, key=sort_key):
        rows = {r["method"]: r for r in by_file[path] if r["threads"] == min(x["threads"] for x in by_file[path])}
        cells = []
        for m, _ in methods:
            r = rows[m]
            cells.append(f"{r['raw_bytes'] / r['bytes']:.1f}x ({r['max_error_A']:.4f} Å)")
        out.append(f"| {describe(path, by_file[path])} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def speed_table(by_file, threads):
    tmax = max(threads)
    out = [
        f"| trajectory | method | compress MB/s (1 / {tmax} threads) | decompress MB/s (1 / {tmax} threads) "
        "| random frame (ms) | frame-by-frame (ms/frame) |",
        "|---|---|---|---|---|---|",
    ]
    for path in sorted(by_file, key=sort_key):
        rows = defaultdict(dict)
        for r in by_file[path]:
            rows[r["method"]][r["threads"]] = r
        for m in ["mdz", "mdz-temporal", "mdz-intra", "mdz-rigid", "mdz-lossless", "xtc"]:
            if m not in rows:
                continue
            one = rows[m].get(1) or next(iter(rows[m].values()))
            many = rows[m].get(tmax)
            raw = one["raw_bytes"]
            enc = f"{mb(raw) / one['compress_s']:.0f}"
            dec = f"{mb(raw) / one['decompress_s']:.0f}"
            if many and many is not one:
                enc += f" / {mb(raw) / many['compress_s']:.0f}"
                dec += f" / {mb(raw) / many['decompress_s']:.0f}"
            rnd = f"{one['random_us'] / 1000:.2f}" if one["random_us"] >= 0 else "-"
            seq = f"{one['sequential_us'] / 1000:.3f}" if one["sequential_us"] >= 0 else "-"
            out.append(f"| {describe(path, by_file[path])} | {m} | {enc} | {dec} | {rnd} | {seq} |")
    return "\n".join(out)


def cli_table(cli, path, nframes):
    """Time CLI commands end to end, including disk I/O with a warm page cache."""
    rows = []
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        raw = os.path.getsize(path)

        def timed(args):
            best = 1e9
            for _ in range(3):
                t = time.perf_counter()
                subprocess.run([cli, "-q", *map(str, args)], check=True)
                best = min(best, time.perf_counter() - t)
            return best

        t = timed(["compress", path, "-o", d / "t.mdz", "-f"])
        rows.append(("mdz compress", t, raw))
        t = timed(["decompress", d / "t.mdz", "-o", d / "back.dcd", "-f"])
        rows.append(("mdz decompress → .dcd", t, raw))
        t = timed(["convert", path, d / "t.xtc", "-f"])
        rows.append(("mdz convert .dcd → .xtc", t, raw))
        t = timed(["convert", d / "t.xtc", d / "back2.dcd", "-f"])
        rows.append(("mdz convert .xtc → .dcd", t, raw))
        mid = nframes // 2
        t = timed(["extract", d / "t.mdz", f"{mid}:{mid + 10}", "-o", d / "slice.dcd", "-f"])
        rows.append(("mdz extract 10 frames", t, None))
        sizes = {"mdz": os.path.getsize(d / "t.mdz"), "xtc": os.path.getsize(d / "t.xtc")}
    out = [
        f"`{Path(path).name}`, {raw / 1e6:.0f} MB DCD (`.mdz`: {sizes['mdz'] / 1e6:.1f} MB, `.xtc`: {sizes['xtc'] / 1e6:.1f} MB).",
        "",
        "| command | time (s) | MB/s of DCD |",
        "|---|---|---|",
    ]
    for name, t, raw_bytes in rows:
        rate = f"{raw_bytes / 1e6 / t:.0f}" if raw_bytes else "-"
        out.append(f"| `{name}` | {t:.3f} | {rate} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--build", default="build")
    ap.add_argument("--threads", default=f"1,{os.cpu_count()}")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--json", help="also save the raw measurements here")
    ap.add_argument("--tmp", default=tempfile.gettempdir(), help="scratch directory for compressed files")
    a = ap.parse_args()
    threads = [int(t) for t in a.threads.split(",")]
    exe = str(Path(a.build) / "bench" / "bench_files")
    cli = str(Path(a.build) / "cli" / "mdz")

    results = []
    for f in sorted(a.files, key=sort_key):
        print(f"benchmarking {f}", file=sys.stderr, flush=True)
        results += run_bench(exe, [f], threads, a.repeat, a.tmp)
    if a.json:
        Path(a.json).write_text("\n".join(json.dumps(r) for r in results) + "\n")
    by_file = defaultdict(list)
    for r in results:
        by_file[r["file"]].append(r)

    largest = max(a.files, key=os.path.getsize)
    print("## Compression ratio\n")
    print("Ratios are raw float32 positions over compressed size. mdz at its defaults")
    print("(≤ 0.005 Å error, codec chosen per group) against XTC at GROMACS's standard")
    print("precision (0.001 nm, also ≤ 0.005 Å), lossless mdz, and zstd over the raw")
    print("floats.\n")
    print(size_table(by_file))
    print("\n## Size against error budget\n")
    print("Ratio (and the maximum error measured) at three `--max-error` settings.\n")
    print(precision_table(by_file))
    print("\n## Speed\n")
    print("In memory, codec only. MB/s are of raw float32 positions. A random frame is a")
    print("read of one frame from a freshly opened file; frame-by-frame reads every frame")
    print("in order, one call per frame, as an analysis loop would.\n")
    print(speed_table(by_file, threads))
    print("\n## Command line, end to end\n")
    print(cli_table(cli, largest, by_file[largest][0]["nframes"]))


if __name__ == "__main__":
    main()
