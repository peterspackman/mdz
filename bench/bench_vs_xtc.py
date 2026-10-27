"""Benchmark mdz against XTC on real DCD trajectories.

XTC files are read and written with mdtraj, which wraps the C xdrfile library.
"""

import os
import sys
import time
import tempfile

import numpy as np
import mdtraj

import mdz


def benchmark_xtc(traj, precision_nm, label=""):
    """Benchmark XTC compression via mdtraj."""
    natoms = traj.n_atoms
    nframes = traj.n_frames
    raw_bytes = nframes * natoms * 3 * 4

    with tempfile.NamedTemporaryFile(suffix=".xtc", delete=False) as f:
        xtc_path = f.name

    try:
        # mdtraj writes XTC at its default precision of 1000 (0.001 nm), and
        # precision_nm is not passed through. mdtraj works in nm, as mdz does.
        t0 = time.perf_counter()
        traj.save_xtc(xtc_path, force_overwrite=True)
        t_write = time.perf_counter() - t0

        xtc_size = os.path.getsize(xtc_path)

        t0 = time.perf_counter()
        traj_back = mdtraj.load_xtc(xtc_path, top=traj.topology)
        t_read = time.perf_counter() - t0

        err = np.abs(traj.xyz - traj_back.xyz)
        max_err = err.max()
        mean_err = err.mean()

        return {
            "format": "XTC",
            "label": label,
            "natoms": natoms,
            "nframes": nframes,
            "raw_mb": raw_bytes / 1e6,
            "compressed_mb": xtc_size / 1e6,
            "ratio": raw_bytes / xtc_size,
            "write_mb_s": raw_bytes / 1e6 / t_write,
            "read_mb_s": raw_bytes / 1e6 / t_read,
            "max_err_A": max_err * 10,  # nm → Å
            "mean_err_A": mean_err * 10,
        }
    finally:
        os.unlink(xtc_path)


def benchmark_mdz(traj, precision_nm, label="", **opts):
    """Benchmark mdz compression through the Python bindings.

    Measured the same way as :func:`benchmark_xtc`. Both run in-process and
    time compress-and-write and read-and-decompress, including file I/O.
    Running the CLI instead would add process startup time to each
    measurement, which is significant on trajectories of this size.
    """
    natoms = traj.n_atoms
    nframes = traj.n_frames
    raw_bytes = nframes * natoms * 3 * 4

    with tempfile.NamedTemporaryFile(suffix=".mdz", delete=False) as f:
        mdz_path = f.name

    try:
        # XTC stores the box, so mdz is given the box as well.
        boxes = traj.unitcell_vectors
        if boxes is not None:
            boxes = np.ascontiguousarray(boxes, dtype=np.float32)

        t0 = time.perf_counter()
        ct = mdz.compress(traj.xyz, boxes=boxes, precision=precision_nm, **opts)
        mdz.save(mdz_path, ct)
        t_write = time.perf_counter() - t0

        mdz_size = os.path.getsize(mdz_path)

        t0 = time.perf_counter()
        recon, _boxes_back = mdz.decompress(mdz.load(mdz_path))
        t_read = time.perf_counter() - t0

        err = np.abs(traj.xyz - recon)
        return {
            "format": "mdz",
            "label": label,
            "natoms": natoms,
            "nframes": nframes,
            "raw_mb": raw_bytes / 1e6,
            "compressed_mb": mdz_size / 1e6,
            "ratio": raw_bytes / mdz_size if mdz_size > 0 else 0,
            "write_mb_s": raw_bytes / 1e6 / t_write,
            "read_mb_s": raw_bytes / 1e6 / t_read,
            "max_err_A": float(err.max()) * 10,   # nm -> A
            "mean_err_A": float(err.mean()) * 10,
        }
    finally:
        if os.path.exists(mdz_path):
            os.unlink(mdz_path)


def print_results(results):
    header = (
        f"{'Format':<8} {'Label':<25} {'Ratio':>7} {'Size MB':>10} "
        f"{'Write MB/s':>11} {'Read MB/s':>10} {'MaxErr Å':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        if r is None:
            continue
        max_err_str = f"{r['max_err_A']:.4f}" if r['max_err_A'] is not None else "n/a"
        print(
            f"{r['format']:<8} {r['label']:<25} {r['ratio']:>7.1f}x "
            f"{r['compressed_mb']:>9.2f} {r['write_mb_s']:>10.1f} "
            f"{r['read_mb_s']:>9.1f} {max_err_str:>10}"
        )


def main():
    dcd_files = sys.argv[1:] if len(sys.argv) > 1 else ["examples/traj.dcd"]

    for dcd_path in dcd_files:
        if not os.path.exists(dcd_path):
            print(f"File not found: {dcd_path}")
            continue

        file_size_mb = os.path.getsize(dcd_path) / 1e6
        print(f"\n{'='*80}")
        print(f"Trajectory: {dcd_path} ({file_size_mb:.1f} MB)")

        # mdtraj needs the trajectory loaded for the XTC benchmark. Very large
        # files are limited to their first frames.
        max_frames = 500 if file_size_mb > 500 else None

        # mdtraj needs a topology. Use a PDB next to the DCD if there is one.
        pdb_path = dcd_path.replace(".dcd", ".pdb")
        base_dir = os.path.dirname(dcd_path)
        if not os.path.exists(pdb_path):
            # Otherwise use any PDB in the same directory.
            for f in os.listdir(base_dir):
                if f.endswith(".pdb"):
                    pdb_path = os.path.join(base_dir, f)
                    break

        if os.path.exists(pdb_path):
            top = mdtraj.load_pdb(pdb_path).topology
            print(f"Using topology from {pdb_path}")
        else:
            # Otherwise build a placeholder topology, reading the atom count
            # from the DCD header.
            import struct
            with open(dcd_path, "rb") as fh:
                fh.seek(4 + 84 + 4)  # skip first record
                title_sz = struct.unpack("<I", fh.read(4))[0]
                fh.seek(fh.tell() + title_sz + 4)
                fh.seek(fh.tell() + 4)
                natoms_dcd = struct.unpack("<i", fh.read(4))[0]
            top = mdtraj.Topology()
            chain = top.add_chain()
            res = top.add_residue("UNK", chain)
            for i in range(natoms_dcd):
                top.add_atom(f"X{i}", mdtraj.element.virtual, res)
            print(f"Using dummy topology ({natoms_dcd} atoms)")

        print(f"Loading DCD with mdtraj{f' (first {max_frames} frames)' if max_frames else ''}...")
        t0 = time.perf_counter()
        traj = mdtraj.load_dcd(dcd_path, top=top)
        if max_frames and traj.n_frames > max_frames:
            traj = traj[:max_frames]
        load_time = time.perf_counter() - t0
        print(f"  {traj.n_atoms} atoms, {traj.n_frames} frames, loaded in {load_time:.1f}s")
        raw_mb = traj.n_atoms * traj.n_frames * 3 * 4 / 1e6
        print(f"  Raw size: {raw_mb:.1f} MB\n")

        results = []

        print("Benchmarking XTC (default precision 0.001 nm)...")
        results.append(benchmark_xtc(traj, 0.001, "default (0.001 nm)"))

        print("Benchmarking mdz (default)...")
        results.append(benchmark_mdz(traj, 0.001, "default (0.001 nm)"))

        # mdz without adaptive precision or B-frames, which is closer to how
        # XTC quantizes.
        print("Benchmarking mdz (no adaptive, no B-frames)...")
        results.append(benchmark_mdz(
            traj, 0.001, "no-adapt no-B", adaptive=False, anchor_step=1,
        ))

        print()
        print_results(results)


if __name__ == "__main__":
    main()
