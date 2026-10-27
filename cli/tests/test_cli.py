"""End-to-end tests of the mdz command-line tool. They cover exit codes,
format detection, output to stdout, and the guarantees around overwriting,
atomic writes and interruption.

    MDZ_CLI=build/cli/mdz python -m pytest cli/tests
"""

import os
import shutil
import signal
import stat
import subprocess
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
CLI = os.environ.get("MDZ_CLI", str(ROOT / "build" / "cli" / "mdz"))

pytestmark = pytest.mark.skipif(not Path(CLI).exists(), reason=f"mdz CLI not built at {CLI}")


def mdz(*args, stdin=subprocess.DEVNULL, cwd=None):
    return subprocess.run(
        [CLI, "--color", "never", *map(str, args)],
        stdin=stdin,
        capture_output=True,
        cwd=cwd,
    )


def ok(r):
    assert r.returncode == 0, r.stderr.decode()
    return r


def no_temp_files(d: Path):
    left = [p.name for p in d.iterdir() if ".tmp-" in p.name]
    assert not left, f"temporary files left behind: {left}"


@pytest.fixture
def compressed(tmp_path):
    out = tmp_path / "traj.mdz"
    ok(mdz("compress", EXAMPLES / "traj.dcd", "-o", out))
    return out


def test_uppercase_extensions_are_recognised(tmp_path):
    dcd = tmp_path / "TRAJ.DCD"
    shutil.copy(EXAMPLES / "traj.dcd", dcd)
    mdzf = tmp_path / "T.MDZ"
    r = ok(mdz("compress", dcd, "-o", mdzf))
    assert b"does not end in" not in r.stderr
    r = ok(mdz("info", mdzf))
    assert b"MDZ v4" in r.stdout

    xtc = tmp_path / "OUT.XTC"
    ok(mdz("decompress", mdzf, "-o", xtc))
    assert b"XTC" in ok(mdz("info", xtc)).stdout

    lmp = tmp_path / "NACL.LammpsTrj"
    shutil.copy(EXAMPLES / "lammps" / "nacl.lammpstrj", lmp)
    ext = tmp_path / "C.EXTXYZ"
    ok(mdz("convert", lmp, ext))
    assert "Properties=" in ext.read_text()


def test_unrecognised_output_extension_is_an_error(tmp_path, compressed):
    for name in ["out.foo", "noextension"]:
        out = tmp_path / name
        r = mdz("decompress", compressed, "-o", out)
        assert r.returncode == 1
        assert b".extxyz" in r.stderr
        assert not out.exists()

    r = mdz("decompress", compressed, "-o", "-", cwd=tmp_path)
    assert r.returncode == 1
    assert b"--format" in r.stderr
    assert r.stdout == b""
    assert not (tmp_path / "-").exists()

    out = tmp_path / "positions.dat"
    ok(mdz("decompress", compressed, "-o", out, "--format", "raw"))
    assert out.stat().st_size == 25 * 4096 * 3 * 4


def test_stdout_output_with_format(compressed):
    r = ok(mdz("decompress", compressed, "-o", "-", "--format", "xyz"))
    lines = r.stdout.decode().splitlines()
    assert len(lines) == 25 * (4096 + 2)
    assert lines[0] == "4096"
    assert b"warning" in r.stderr, "box loss is warned on stderr"

    r = ok(mdz("extract", compressed, "3:5", "-o", "-", "--format", "lammpstrj"))
    assert r.stdout.decode().count("ITEM: TIMESTEP") == 2

    r = ok(mdz("convert", EXAMPLES / "lammps" / "nacl.lammpstrj", "-", "--format", "extxyz"))
    assert r.stdout.startswith(b"512\n")

    r = mdz("decompress", compressed, "-o", "-", "--format", "dcd")
    assert r.returncode == 1
    assert r.stdout == b""


def test_existing_output_is_refused_without_force(tmp_path, compressed):
    out = tmp_path / "out.xyz"
    out.write_text("ORIGINAL")
    r = mdz("decompress", compressed, "-o", out)
    assert r.returncode == 1
    assert b"--force" in r.stderr
    assert out.read_text() == "ORIGINAL"

    assert mdz("compress", EXAMPLES / "traj.dcd", "-o", compressed).returncode == 1
    assert mdz("convert", EXAMPLES / "traj.dcd", out).returncode == 1
    assert out.read_text() == "ORIGINAL"

    ok(mdz("decompress", compressed, "-o", out, "--force"))
    assert out.read_text().startswith("4096\n")
    no_temp_files(tmp_path)


def write_truncated_dump(path: Path, good: int):
    text = []
    for step in range(good):
        text.append(
            f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n2\n"
            "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
            "ITEM: ATOMS id type x y z\n1 1 1.0 1.0 1.0\n2 1 2.0 2.0 2.0\n"
        )
    text.append(
        "ITEM: TIMESTEP\n999\nITEM: NUMBER OF ATOMS\n2\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
        "ITEM: ATOMS id type x y z\n1 1 garbage 1.0\n"
    )
    path.write_text("".join(text))


def test_failed_write_leaves_destination_untouched(tmp_path):
    src = tmp_path / "bad.lammpstrj"
    write_truncated_dump(src, 250)
    out = tmp_path / "out.xyz"
    out.write_text("ORIGINAL")
    r = mdz("convert", src, out, "--force")
    assert r.returncode == 1
    assert out.read_text() == "ORIGINAL"
    no_temp_files(tmp_path)

    fresh = tmp_path / "fresh.xyz"
    assert mdz("convert", src, fresh).returncode == 1
    assert not fresh.exists()
    no_temp_files(tmp_path)


def test_usage_errors_exit_2_and_runtime_errors_exit_1(tmp_path):
    assert mdz("decompress").returncode == 2
    assert mdz("decompress", "x.mdz", "--format", "nonsense").returncode == 2
    assert mdz("compress", "x.dcd", "--lossless", "--codec", "intra").returncode == 2
    assert mdz("compress", "x.dcd", "--zstd-level", "40").returncode == 2

    r = mdz("decompress", tmp_path / "missing.mdz")
    assert r.returncode == 1
    assert b"error:" in r.stderr
    assert r.stdout == b""

    r = mdz("info", tmp_path / "thing.unknown")
    assert r.returncode == 1
    assert b"error:" in r.stderr
    r = mdz("-q", "info", tmp_path / "thing.unknown")
    assert r.returncode == 1
    assert b"error:" in r.stderr


def test_decompress_reads_stdin(tmp_path, compressed):
    out = tmp_path / "back.dcd"
    with open(compressed, "rb") as f:
        ok(mdz("decompress", "-", "-o", out, stdin=f))
    assert b"Frames:           25" in ok(mdz("info", out)).stdout


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_force_overwrite_keeps_permissions(tmp_path, compressed):
    out = tmp_path / "out.xyz"
    out.write_text("ORIGINAL")
    os.chmod(out, 0o640)
    ok(mdz("decompress", compressed, "-o", out, "--force"))
    assert out.read_text().startswith("4096\n")
    assert stat.S_IMODE(out.stat().st_mode) == 0o640
    no_temp_files(tmp_path)


def test_extract_and_info_report_ranges(tmp_path, compressed):
    r = mdz("extract", compressed, "20:30")
    assert r.returncode == 1 and b"only 25 frames" in r.stderr
    out = tmp_path / "slice.xtc"
    ok(mdz("extract", compressed, "5:9", "-o", out))
    assert b"Atoms:            4096" in ok(mdz("info", out)).stdout
    # The default output name includes the frame range and the source format.
    ok(mdz("extract", compressed, "7"))
    assert (tmp_path / "traj-7-8.dcd").exists()


def test_compress_roundtrip_and_stats(tmp_path):
    out = tmp_path / "t.mdz"
    r = ok(mdz("compress", EXAMPLES / "traj.dcd", "-o", out, "--stats"))
    assert b"Max error:" in r.stderr
    info = ok(mdz("info", out)).stdout.decode()
    assert "Frames:           25" in info and "positions" in info
    lossless = tmp_path / "exact.mdz"
    ok(mdz("compress", EXAMPLES / "traj.dcd", "-o", lossless, "--lossless"))
    assert "lossless" in ok(mdz("info", lossless)).stdout.decode()
    # Several input DCDs are joined into one trajectory.
    joined = tmp_path / "joined.mdz"
    ok(mdz("compress", EXAMPLES / "traj.dcd", EXAMPLES / "traj.dcd", "-o", joined))
    assert "Frames:           50" in ok(mdz("info", joined)).stdout.decode()


def test_completions():
    for shell in ["bash", "zsh", "fish"]:
        r = ok(mdz("completions", shell))
        assert b"compress" in r.stdout
    assert mdz("completions", "tcsh").returncode == 2


@pytest.mark.skipif(os.name != "posix", reason="POSIX signals and FIFOs")
@pytest.mark.parametrize("sig,quiet", [(signal.SIGINT, False), (signal.SIGTERM, True)])
def test_interrupted_write_removes_temp_file(tmp_path, sig, quiet):
    fifo = tmp_path / "in.xyz"
    os.mkfifo(fifo)
    out = tmp_path / "out.xyz"
    out.write_text("ORIGINAL")
    cmd = [CLI, "--color", "never"] + (["-q"] if quiet else []) + ["convert", str(fifo), str(out), "--force"]
    child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    done = threading.Event()

    def feed():
        with open(fifo, "w") as f:
            for step in range(150):
                f.write(f"2\nstep={step}\nH 0.0 0.0 0.0\nH 1.0 1.0 1.0\n")
            f.flush()
            done.wait(30)

    feeder = threading.Thread(target=feed)
    feeder.start()
    try:
        deadline = time.time() + 20
        while not any(".tmp-" in p.name for p in tmp_path.iterdir()):
            assert time.time() < deadline and child.poll() is None, "no temporary file appeared"
            time.sleep(0.01)
        child.send_signal(sig)
        code = child.wait(timeout=20)
    finally:
        done.set()
        feeder.join()
    err = child.stderr.read().decode()
    assert code == 130, err
    assert ("interrupted" in err) == (not quiet), err
    assert out.read_text() == "ORIGINAL"
    no_temp_files(tmp_path)


# --- varying atom counts ---


def write_deposition_dump(path: Path, nframes: int, change_at: int = 0):
    """Write a LAMMPS dump whose atoms change. From frame ``change_at``
    onwards, one atom is deposited every frame and one is removed every third
    frame. Returns the list of atom ids in each frame."""
    alive = {i: (0.5 * i % 9.0, 1.0, 2.0) for i in range(1, 31)}
    nxt = 31
    frames, out = [], []
    for f in range(nframes):
        if f >= max(change_at, 1):
            alive[nxt] = (0.25 * nxt % 9.0, 5.0, 5.0)
            nxt += 1
            if f % 3 == 0:
                del alive[min(alive)]
        ids = sorted(alive)
        frames.append(ids)
        out.append(
            f"ITEM: TIMESTEP\n{f * 10}\nITEM: NUMBER OF ATOMS\n{len(ids)}\n"
            "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\nITEM: ATOMS id type x y z\n"
        )
        for i in ids:
            x, y, z = alive[i]
            out.append(f"{i} {1 + i % 2} {x:.3f} {y + 0.01 * f:.3f} {z:.3f}\n")
    path.write_text("".join(out))
    return frames


def dump_frames(path: Path):
    """Return the list of atom ids in each frame of a LAMMPS dump."""
    lines = path.read_text().splitlines()
    frames, i = [], 0
    while i < len(lines):
        assert lines[i] == "ITEM: TIMESTEP"
        n = int(lines[i + 3])
        cols = lines[i + 8].split()[2:]
        idc = cols.index("id")
        frames.append([int(lines[i + 9 + k].split()[idc]) for k in range(n)])
        i += 9 + n
    return frames


def extxyz_counts(path: Path):
    lines = path.read_text().splitlines()
    counts, i = [], 0
    while i < len(lines):
        n = int(lines[i])
        counts.append(n)
        i += 2 + n
    return counts


def test_varying_atom_count_compress_info_decompress(tmp_path):
    dump = tmp_path / "dep.lammpstrj"
    want = write_deposition_dump(dump, 50)
    out = tmp_path / "dep.mdz"
    r = ok(mdz("compress", dump, "-o", out, "--stats"))
    assert b"(varying)" in r.stderr
    assert b"Max error" in r.stderr

    info = ok(mdz("info", out)).stdout.decode()
    lo, hi = min(map(len, want)), max(map(len, want))
    assert f"varying, {lo}-{hi} per frame" in info

    back = tmp_path / "back.lammpstrj"
    ok(mdz("decompress", out, "-o", back))
    assert dump_frames(back) == want

    # Extract a frame range, and check that formats with one atom count per
    # file are rejected.
    part = tmp_path / "part.lammpstrj"
    ok(mdz("extract", out, "10:20", "-o", part))
    assert dump_frames(part) == want[10:20]
    for ext in ("dcd", "npy", "f32"):
        r = mdz("decompress", out, "-o", tmp_path / f"x.{ext}")
        assert r.returncode == 1
        assert b"one atom count per file" in r.stderr
        assert not (tmp_path / f"x.{ext}").exists()
    no_temp_files(tmp_path)


def test_atoms_changing_after_the_first_chunk_restart_as_varying(tmp_path):
    # Small groups and one thread keep the read chunks small, so the atom
    # count changes long after the first groups were written.
    dump = tmp_path / "late.lammpstrj"
    want = write_deposition_dump(dump, 200, change_at=150)
    out = tmp_path / "late.mdz"
    r = ok(mdz("--threads", "1", "compress", dump, "-o", out, "--gof-size", "2", "--stats"))
    assert b"compressing again" in r.stderr
    assert b"varying" in ok(mdz("info", out)).stdout
    back = tmp_path / "back.lammpstrj"
    ok(mdz("decompress", out, "-o", back))
    assert dump_frames(back) == want
    no_temp_files(tmp_path)


def test_variable_atoms_flag_forces_a_varying_file(tmp_path):
    out = tmp_path / "nacl.mdz"
    ok(mdz("compress", EXAMPLES / "lammps" / "nacl.lammpstrj", "-o", out, "--variable-atoms"))
    assert b"varying" in ok(mdz("info", out)).stdout
    # The atom count never changes, so any output format accepts it.
    ok(mdz("decompress", out, "-o", tmp_path / "nacl.dcd"))
    plain = tmp_path / "plain.mdz"
    ok(mdz("compress", EXAMPLES / "lammps" / "nacl.lammpstrj", "-o", plain))
    assert b"varying" not in ok(mdz("info", plain)).stdout


def test_convert_varying_lammps_to_extxyz_and_back(tmp_path):
    dump = tmp_path / "dep.lammpstrj"
    want = write_deposition_dump(dump, 30)
    ext = tmp_path / "dep.extxyz"
    r = ok(mdz("convert", dump, ext))
    assert b"(varying)" in r.stderr
    assert extxyz_counts(ext) == [len(ids) for ids in want]
    assert ":id:I:1" in ext.read_text()
    again = tmp_path / "again.lammpstrj"
    ok(mdz("convert", ext, again))
    assert dump_frames(again) == want
    r = mdz("convert", dump, tmp_path / "x.dcd")
    assert r.returncode == 1
    assert b"one atom count per file" in r.stderr
    no_temp_files(tmp_path)


def test_convert_to_mdz_compresses(tmp_path):
    dump = tmp_path / "dep.lammpstrj"
    want = write_deposition_dump(dump, 30)
    out = tmp_path / "dep.mdz"
    ok(mdz("convert", dump, out))
    assert b"varying" in ok(mdz("info", out)).stdout
    back = tmp_path / "back.lammpstrj"
    ok(mdz("convert", out, back))
    assert dump_frames(back) == want
    r = mdz("convert", dump, out)
    assert r.returncode == 1  # existing output is refused, as for compress
    no_temp_files(tmp_path)


def test_rigid_compress(tmp_path):
    out = tmp_path / "r.mdz"
    r = ok(mdz("compress", EXAMPLES / "traj.dcd", "-o", out, "--rigid", "--stats"))
    assert b"Max error:" in r.stderr
    assert b"Frames:           25" in ok(mdz("info", out)).stdout
    assert mdz("compress", EXAMPLES / "traj.dcd", "-o", tmp_path / "x.mdz", "--rigid", "--lossless").returncode == 2
