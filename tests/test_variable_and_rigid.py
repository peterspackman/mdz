"""Atom counts that vary from frame to frame, and rigid bodies, from Python."""

import numpy as np
import pytest

import mdz

PRECISION = 0.000125
TOL = 4 * PRECISION + 1e-6


def _gcmc(nframes=23, seed=3):
    """Frames of a run where atoms come and go, as a list of (ids, positions)."""
    rng = np.random.default_rng(seed)
    alive = {i: rng.random(3) * 3.0 for i in range(1, 31)}
    next_id = 31
    frames = []
    for _ in range(nframes):
        for _ in range(2):
            if alive and rng.random() < 0.5:
                del alive[rng.choice(sorted(alive))]
            if rng.random() < 0.5:
                alive[next_id] = rng.random(3) * 3.0
                next_id += 1
        for i in alive:
            alive[i] = alive[i] + rng.normal(0.0, 0.002, 3)
        ids = np.array(sorted(alive), dtype=np.int64)
        frames.append((ids, np.array([alive[i] for i in ids], dtype=np.float32)))
    assert len({len(ids) for ids, _ in frames}) > 1
    return frames


def _write_gcmc(path, frames, **kw):
    with mdz.Writer(path, variable_atoms=True, gof_size=5, **kw) as w:
        for k, (ids, x) in enumerate(frames):
            w.write(x, ids=ids, step=10 * k, boxes=np.eye(3) * 3.0)


def _rings(nmol=40, natoms_per=6, nframes=12, seed=7):
    """Rigid hexagonal rings that tumble and drift. Returns (positions, bodies)."""
    rng = np.random.default_rng(seed)
    ang = 2 * np.pi * np.arange(natoms_per) / natoms_per
    shape = 0.14 * np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], axis=1)
    # Centres sit on a lattice 0.6 nm apart, so no two rings come close
    # enough to be detected as one body.
    grid = np.arange(4) * 0.6
    centres = np.stack(np.meshgrid(grid, grid, grid), axis=-1).reshape(-1, 3)[:nmol].copy()
    out = np.empty((nframes, nmol * natoms_per, 3), dtype=np.float32)
    for f in range(nframes):
        for m in range(nmol):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            w, x, y, z = q
            rot = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                ]
            )
            out[f, m * natoms_per : (m + 1) * natoms_per] = centres[m] + shape @ rot.T
        centres += rng.normal(0.0, 0.01, centres.shape)
    bodies = np.repeat(np.arange(nmol), natoms_per)
    return out, bodies


class TestVariableAtoms:
    def test_writer_and_reader_roundtrip(self, tmp_path):
        frames = _gcmc()
        path = tmp_path / "v.mdz"
        _write_gcmc(path, frames)
        counts = [len(ids) for ids, _ in frames]

        r = mdz.open(path)
        assert r.variable_atoms
        assert r.natoms == 0
        assert len(r) == len(frames)
        assert r.frame_natoms.dtype == np.int64
        assert r.frame_natoms.tolist() == counts
        for i in [0, 4, 5, 13, len(frames) - 1, -1]:
            ids, x = frames[i]
            got = r.read_frame(i)
            assert got.shape == (len(ids), 3)
            np.testing.assert_allclose(got, x, atol=TOL)
            np.testing.assert_array_equal(r[i], got)
            np.testing.assert_array_equal(r.atom_ids(i), ids)
        np.testing.assert_allclose(r.cell(3), np.eye(3) * 3.0, atol=1e-6)

    def test_ranges_are_lists_of_frames(self, tmp_path):
        frames = _gcmc()
        path = tmp_path / "v.mdz"
        _write_gcmc(path, frames)
        r = mdz.open(path)

        def same(got, want):
            assert isinstance(got, list)
            assert len(got) == len(want)
            for g, (_, x) in zip(got, want):
                np.testing.assert_allclose(g, x, atol=TOL)

        same(r.read_all(), frames)
        same(r.read_frames(3, 17), frames[3:17])
        same(r.read_frames(20, 100), frames[20:])
        same(r[2:9], frames[2:9])
        same(r[1:20:3], frames[1:20:3])
        same(r[::-4], frames[::-4])
        same(list(r), frames)

    def test_columns_are_flat(self, tmp_path):
        frames = _gcmc()
        path = tmp_path / "v.mdz"
        _write_gcmc(path, frames)
        r = mdz.open(path)
        total = sum(len(ids) for ids, _ in frames)
        assert "mdz.atoms" not in r.column_names()
        pos = r.read_column("positions")
        assert pos.shape == (total, 3)
        np.testing.assert_allclose(pos, np.concatenate([x for _, x in frames]), atol=TOL)
        per_frame = np.split(pos, np.cumsum(r.frame_natoms)[:-1])
        for got, (_, want) in zip(per_frame, frames):
            np.testing.assert_allclose(got, want, atol=TOL)
        assert r.read_column("step").tolist() == [10 * k for k in range(len(frames))]
        with pytest.raises(KeyError):
            r.read_column("mdz.atoms")

    def test_load_info_and_decompress(self, tmp_path):
        frames = _gcmc()
        path = tmp_path / "v.mdz"
        _write_gcmc(path, frames)
        info = mdz.info(path)
        assert info["variable_atoms"] is True
        assert info["natoms"] == 0
        assert info["rigid_bodies"] == 0
        assert "mdz.atoms" not in info["columns"]
        assert info["n_columns"] == len(info["columns"])

        ct = mdz.load(path)
        assert ct.variable_atoms and ct.natoms == 0
        assert ct.frame_natoms.tolist() == [len(ids) for ids, _ in frames]
        assert "mdz.atoms" not in ct.column_names
        pos, boxes = mdz.decompress(ct)
        assert isinstance(pos, list) and len(pos) == len(frames)
        for got, (_, x) in zip(pos, frames):
            np.testing.assert_allclose(got, x, atol=TOL)
        assert boxes.shape == (len(frames), 3, 3)
        assert ct.read_column("positions").shape == (sum(len(i) for i, _ in frames), 3)

    def test_without_ids_atoms_are_matched_by_position(self, tmp_path):
        path = tmp_path / "n.mdz"
        rng = np.random.default_rng(0)
        xs = [rng.random((n, 3)).astype(np.float32) for n in (5, 7, 6, 6)]
        with mdz.Writer(path, variable_atoms=True) as w:
            for x in xs:
                w.write(x)
        r = mdz.open(path)
        assert r.frame_natoms.tolist() == [5, 7, 6, 6]
        np.testing.assert_array_equal(r.atom_ids(1), np.arange(7))
        for got, x in zip(r, xs):
            np.testing.assert_allclose(got, x, atol=TOL)

    def test_several_frames_per_call_share_a_count(self, tmp_path):
        path = tmp_path / "m.mdz"
        rng = np.random.default_rng(1)
        a = rng.random((3, 4, 3)).astype(np.float32)
        b = rng.random((2, 6, 3)).astype(np.float32)
        with mdz.Writer(path, variable_atoms=True, gof_size=4) as w:
            w.write(a, ids=[10, 11, 12, 13], step=[0, 1, 2])
            w.write(b, ids=[11, 12, 14, 15, 16, 17], step=[3, 4])
        r = mdz.open(path)
        assert r.frame_natoms.tolist() == [4, 4, 4, 6, 6]
        np.testing.assert_array_equal(r.atom_ids(4), [11, 12, 14, 15, 16, 17])
        np.testing.assert_allclose(r[4], b[1], atol=TOL)
        with mdz.Writer(tmp_path / "bad.mdz", variable_atoms=True) as w:
            with pytest.raises(ValueError, match="ids"):
                w.write(a, ids=[1, 2, 3])
            with pytest.raises(ValueError, match="32 bits"):
                w.write(a, ids=[1, 2, 3, 1 << 40])
            w.write(a)

    def test_fixed_writer_explains_the_option(self, tmp_path):
        with mdz.Writer(tmp_path / "f.mdz") as w:
            w.write(np.zeros((5, 3)))
            with pytest.raises(ValueError, match="variable_atoms=True"):
                w.write(np.zeros((6, 3)))
            with pytest.raises(ValueError, match="variable_atoms=True"):
                w.write(np.zeros((5, 3)), ids=np.arange(5))
            w.write(np.zeros((5, 3)))
        r = mdz.open(tmp_path / "f.mdz")
        assert not r.variable_atoms
        assert r.frame_natoms.tolist() == [5, 5]
        np.testing.assert_array_equal(r.atom_ids(0), np.arange(5))
        assert mdz.info(tmp_path / "f.mdz")["variable_atoms"] is False

    def test_bad_combinations(self, tmp_path):
        with pytest.raises(ValueError, match="ids"):
            mdz.Writer(tmp_path / "a.mdz", variable_atoms=True, atom_names=["X"])
        with pytest.raises(ValueError, match="fixed atom count"):
            mdz.Writer(tmp_path / "b.mdz", variable_atoms=True, rigid=True)

    def test_append_continues_a_varying_file(self, tmp_path):
        frames = _gcmc(nframes=17)
        path = tmp_path / "a.mdz"
        _write_gcmc(path, frames[:9])
        # variable_atoms is taken from the file and need not be repeated.
        with mdz.Writer(path, append=True, gof_size=5) as w:
            for k, (ids, x) in enumerate(frames[9:], start=9):
                w.write(x, ids=ids, step=10 * k, boxes=np.eye(3) * 3.0)
        r = mdz.open(path)
        assert r.frame_natoms.tolist() == [len(i) for i, _ in frames]
        for i, (ids, x) in enumerate(frames):
            np.testing.assert_allclose(r[i], x, atol=TOL)
            np.testing.assert_array_equal(r.atom_ids(i), ids)
        # A fixed-count file cannot be appended to as a variable-atom file.
        fixed = tmp_path / "fixed.mdz"
        with mdz.Writer(fixed) as w:
            w.write(np.zeros((3, 3)))
        with pytest.raises(ValueError, match="fixed atom count"):
            mdz.Writer(fixed, append=True, variable_atoms=True)


class TestRigid:
    def test_compress_with_given_and_detected_bodies(self):
        pos, bodies = _rings()
        plain = mdz.compress(pos)
        given = mdz.compress(pos, bodies=bodies)
        detected = mdz.compress(pos, rigid=True)
        for ct in (given, detected):
            back, _ = mdz.decompress(ct)
            assert np.abs(back - pos).max() <= TOL
            assert ct.rigid_bodies == 40
            assert ct.compressed_size < plain.compressed_size
            assert len(ct.metadata()["bodies"]) == pos.shape[1]
        assert given.metadata()["bodies"] == bodies.tolist()
        assert plain.rigid_bodies == 0 and "bodies" not in plain.metadata()
        # Rigid coding does not change the set of columns.
        assert given.column_names == plain.column_names

    def test_compress_batch_and_boxes(self):
        pos, bodies = _rings(nframes=6)
        boxes = np.broadcast_to(np.eye(3) * 5.0, (6, 3, 3))
        ct = mdz.compress_batch(pos, boxes=boxes, step=np.arange(6), rigid=True, bodies=bodies)
        assert ct.rigid_bodies == 40
        np.testing.assert_allclose(ct.read_column("positions"), pos, atol=TOL)
        _, b = mdz.decompress(mdz.compress(pos, boxes=boxes, rigid=True))
        np.testing.assert_allclose(b, boxes)

    def test_bad_bodies(self):
        pos, bodies = _rings(nframes=3)
        with pytest.raises(ValueError, match="bodies"):
            mdz.compress(pos, bodies=bodies[:-1])
        with pytest.raises(ValueError, match="bodies"):
            mdz.compress(pos, bodies=bodies - 2)
        with pytest.raises(ValueError, match="bodies"):
            mdz.compress_batch(pos, bodies=bodies[:-1])
        with pytest.raises(ValueError, match="bodies"):
            mdz.compress(pos, bodies=bodies.reshape(-1, 2))

    def test_lossless_ignores_rigid(self):
        pos, bodies = _rings(nframes=3)
        back, _ = mdz.decompress(mdz.compress(pos, bodies=bodies, lossless=True))
        np.testing.assert_array_equal(back, pos)

    def test_writer_streams_and_appends(self, tmp_path):
        pos, bodies = _rings(nframes=14)
        path = tmp_path / "r.mdz"
        with mdz.Writer(path, rigid=True, gof_size=4) as w:
            for x in pos[:10]:
                w.write(x)
        with mdz.Writer(path, append=True, rigid=True, gof_size=4) as w:
            w.write(pos[10:])
        r = mdz.open(path)
        assert r.rigid_bodies == 40
        assert mdz.info(path)["rigid_bodies"] == 40
        np.testing.assert_allclose(r.read_all(), pos, atol=TOL)
        assert r.metadata()["bodies"] is not None

        given = tmp_path / "g.mdz"
        with mdz.Writer(given, bodies=bodies, gof_size=4) as w:
            w.write(pos)
            with pytest.raises(ValueError, match="atoms"):
                w.write(pos[0, :10])
        assert mdz.open(given).metadata()["bodies"] == bodies.tolist()
        np.testing.assert_allclose(mdz.open(given).read_all(), pos, atol=TOL)
