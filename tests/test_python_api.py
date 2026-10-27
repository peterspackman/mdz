"""Input handling, validation and reader behaviour of the Python bindings."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os

import numpy as np
import pytest

import mdz

PRECISION = 0.001  # nm; the adaptive maximum error is 4 * PRECISION
TOL = 4 * PRECISION + 1e-6


def _pos(nframes=20, natoms=10, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.random((nframes, natoms, 3)) * 3.0).astype(np.float32)


def _roundtrip(arr, **kw):
    recon, _ = mdz.decompress(mdz.compress(arr, precision=PRECISION, **kw))
    return recon


class TestInputLayouts:
    def test_float64_is_accepted(self):
        p = _pos().astype(np.float64)
        recon = _roundtrip(p)
        assert recon.dtype == np.float32
        assert np.abs(recon - p).max() <= TOL

    def test_nested_list_is_accepted(self):
        p = _pos(nframes=3, natoms=4)
        recon = _roundtrip(p.tolist())
        assert np.abs(recon - p).max() <= TOL

    def test_fortran_order_is_not_scrambled(self):
        p = _pos()
        recon = _roundtrip(np.asfortranarray(p))
        assert np.abs(recon - p).max() <= TOL

    @pytest.mark.parametrize(
        "view",
        [
            lambda p: p[:, ::2, :],
            lambda p: p[::3],
            lambda p: p[:, :, ::-1],
            lambda p: np.ascontiguousarray(p.transpose(1, 0, 2)).transpose(1, 0, 2),
        ],
    )
    def test_strided_views(self, view):
        v = view(_pos())
        assert not v.flags.c_contiguous
        recon = _roundtrip(v)
        assert recon.shape == v.shape
        assert np.abs(recon - v).max() <= TOL

    def test_fortran_boxes(self):
        p = _pos()
        boxes = np.asfortranarray(
            np.stack([np.array([[3, 0, 0], [0.5, 3, 0], [0.2, 0.1, 3]]) + i for i in range(20)])
        )
        _, rb = mdz.decompress(mdz.compress(p, boxes=boxes, precision=PRECISION))
        np.testing.assert_allclose(rb, boxes, atol=1e-6)

    def test_compress_batch_columns_any_layout(self):
        p = _pos()
        vel = np.asfortranarray(_pos(seed=2).astype(np.float64))
        charge = _pos(seed=3)[:, :, 0].T.copy().T  # F-ordered (nframes, natoms)
        ct = mdz.compress_batch(
            p.astype(np.float64), velocities=vel, charge=charge, precision=PRECISION
        )
        assert np.abs(ct.read_column("velocities") - vel).max() <= TOL
        assert np.abs(ct.read_column("charge") - charge).max() <= TOL


class TestShapeValidation:
    @pytest.mark.parametrize(
        "shape", [(20, 10, 4), (20, 10, 2), (20, 10), (20, 10, 3, 1), (0, 10, 3), (20, 0, 3)]
    )
    def test_bad_positions_shape(self, shape):
        with pytest.raises(ValueError):
            mdz.compress(np.zeros(shape, dtype=np.float32))
        with pytest.raises(ValueError):
            mdz.compress_batch(np.zeros(shape, dtype=np.float32))

    @pytest.mark.parametrize("shape", [(20, 3, 4), (19, 3, 3), (20, 9), (20, 3)])
    def test_bad_boxes_shape(self, shape):
        with pytest.raises(ValueError, match="boxes"):
            mdz.compress(_pos(), boxes=np.ones(shape, dtype=np.float32))
        with pytest.raises(ValueError, match="boxes"):
            mdz.compress_batch(_pos(), boxes=np.ones(shape, dtype=np.float32))

    @pytest.mark.parametrize(
        "kw",
        [
            {"velocities": np.zeros((20, 10, 2))},
            {"velocities": np.zeros((20, 9, 3))},
            {"forces": np.zeros((19, 10, 3))},
            {"charge": np.zeros((20, 11))},
            {"charge": np.zeros((20, 10, 1))},
            {"cell": np.zeros((20, 5))},
            {"cell": np.zeros(120)},
            {"time": np.zeros(19)},
            {"energy": np.zeros((20, 1))},
            {"step": np.zeros(21, dtype=np.int64)},
            {"atom_names": ["X"] * 9},
            {"atom_types": [1] * 11},
            {"atom_ids": [1] * 3},
        ],
    )
    def test_bad_column_shapes(self, kw):
        with pytest.raises(ValueError):
            mdz.compress_batch(_pos(), **kw)

    def test_non_finite_box_rejected(self):
        boxes = np.ones((20, 3, 3), dtype=np.float32)
        boxes[3, 1, 1] = np.nan
        with pytest.raises(ValueError, match="finite"):
            mdz.compress(_pos(), boxes=boxes)


class TestParameterValidation:
    @pytest.mark.parametrize(
        "kw",
        [
            {"predictor_order": 5},
            {"predictor_order": 9},
            {"predictor_order": -1},
            {"zstd_level": 0},
            {"zstd_level": 23},
            {"zstd_level": 100},
            {"gof_size": 0},
            {"gof_size": -1},
            {"gof_size": 65536},
            {"anchor_step": 0},
            {"anchor_step": 256},
            {"precision": 0.0},
            {"precision": -1.0},
            {"precision": float("nan")},
            {"codec": "bogus"},
        ],
    )
    def test_out_of_range_rejected(self, kw):
        with pytest.raises(ValueError):
            mdz.compress(_pos(), **kw)
        kw.pop("codec", None)
        if kw:
            with pytest.raises(ValueError):
                mdz.compress_batch(_pos(), **kw)

    @pytest.mark.parametrize(
        "kw",
        [{"predictor_order": 4}, {"zstd_level": 1}, {"zstd_level": 22}, {"gof_size": 1}],
    )
    def test_range_limits_accepted(self, kw):
        mdz.compress(_pos(), precision=PRECISION, **kw)

    def test_defaults_match_cli(self):
        ct = mdz.compress(_pos(nframes=45))
        assert ct.gof_size == 20
        assert mdz.compress_batch(_pos(nframes=45)).gof_size == 20

    def test_step_beyond_2_53_exact(self):
        # Values beyond 2**53 would be rounded if passed through float64.
        step = np.arange(20, dtype=np.int64) + 2**53 + 1
        ct = mdz.compress_batch(_pos(), step=step, precision=PRECISION)
        np.testing.assert_array_equal(ct.read_column("step"), step)

    def test_step_near_int64_max_exact(self):
        step = np.arange(20, dtype=np.int64) + (2**63 - 100)
        ct = mdz.compress_batch(_pos(), step=step, precision=PRECISION)
        np.testing.assert_array_equal(ct.read_column("step"), step)


class TestCellColumns:
    def test_box_and_cell_column_shapes(self):
        n = 20
        boxes = np.tile(np.eye(3, dtype=np.float32) * 3.0, (n, 1, 1))
        cell = np.tile([3.0, 90.0, 3.0, 90.0, 90.0, 3.0], (n, 1))
        ct = mdz.compress_batch(
            _pos(), boxes=boxes, cell=cell, time=np.arange(n) * 0.5, precision=PRECISION
        )
        box_col = ct.read_column("box")
        assert box_col.shape == (n, 3, 3)
        np.testing.assert_array_equal(box_col, boxes)
        assert ct.read_column("cell").shape == (n, 6)
        assert ct.read_column("time").shape == (n,)

    def test_decompress_returns_cells_for_cell_only_file(self, tmp_path):
        n = 20
        cell = np.tile([3.0, 90.0, 4.0, 90.0, 90.0, 5.0], (n, 1))
        ct = mdz.compress_batch(_pos(), cell=cell, precision=PRECISION)
        assert "box" not in ct.column_names
        _, boxes = mdz.decompress(ct)
        assert boxes is not None and boxes.shape == (n, 3, 3)
        np.testing.assert_allclose(boxes[0], np.diag([3.0, 4.0, 5.0]), atol=1e-5)
        path = str(tmp_path / "cell.mdz")
        ct.save(path)
        np.testing.assert_allclose(mdz.open(path).cells, boxes, atol=0)
        np.testing.assert_allclose(mdz.decode_boxes(path), boxes, atol=0)


@pytest.fixture
def reader_file(tmp_path):
    pos = _pos(nframes=47, natoms=30)
    boxes = np.stack([np.eye(3, dtype=np.float32) * (3.0 + 0.01 * i) for i in range(47)])
    path = tmp_path / "r.mdz"
    mdz.save(path, mdz.compress(pos, boxes=boxes, precision=PRECISION, gof_size=10))
    return path, pos, boxes


class TestReader:
    def test_iteration_matches_read_all(self, reader_file):
        path, _, _ = reader_file
        r = mdz.open(path)
        frames = list(r)
        assert len(frames) == len(r) == 47
        np.testing.assert_array_equal(np.stack(frames), r.read_all())

    def test_iterators_are_independent(self, reader_file):
        r = mdz.open(reader_file[0])
        a, b = iter(r), iter(r)
        next(a)
        np.testing.assert_array_equal(next(b), r[0])
        np.testing.assert_array_equal(next(a), r[1])

    @pytest.mark.parametrize(
        "key",
        [slice(None, None, 3), slice(None, None, -1), slice(40, 5, -7), slice(-5, None, 2)],
    )
    def test_strided_slices(self, reader_file, key):
        r = mdz.open(reader_file[0])
        np.testing.assert_array_equal(r[key], r.read_all()[key])

    def test_negative_frame_index(self, reader_file):
        r = mdz.open(reader_file[0])
        np.testing.assert_array_equal(r[-1], r.read_all()[-1])
        np.testing.assert_array_equal(r.read_frame(-2), r.read_all()[-2])
        with pytest.raises(IndexError):
            r[47]
        with pytest.raises(IndexError):
            r[-48]

    def test_cell_indexing(self, reader_file):
        path, _, boxes = reader_file
        r = mdz.open(path)
        np.testing.assert_allclose(r.cell(0), boxes[0], atol=0)
        np.testing.assert_allclose(r.cell(-1), boxes[-1], atol=0)
        np.testing.assert_allclose(r.cell(-47), boxes[0], atol=0)
        for bad in (47, 999, -48):
            with pytest.raises(IndexError):
                r.cell(bad)

    def test_cell_without_cell_is_none_but_still_bounds_checked(self, tmp_path):
        path = tmp_path / "nocell.mdz"
        mdz.save(path, mdz.compress(_pos(), precision=PRECISION))
        r = mdz.open(path)
        assert r.cell(0) is None and r.cell(-1) is None and r.cells is None
        with pytest.raises(IndexError):
            r.cell(20)

    def test_pathlike_accepted(self, reader_file):
        path, _, _ = reader_file
        assert isinstance(path, os.PathLike)
        assert mdz.info(path)["nframes"] == 47
        assert mdz.load(path).nframes == 47


class TestPackaging:
    def test_version_single_sourced(self):
        assert mdz.__version__ == importlib.metadata.version("mdz")

    def test_typing_files_ship(self):
        pkg = os.path.dirname(mdz.__file__)
        assert os.path.exists(os.path.join(pkg, "py.typed"))
        assert os.path.exists(os.path.join(pkg, "_core.pyi"))

    def test_stub_lists_every_public_name(self):
        pkg = os.path.dirname(mdz.__file__)
        stub = open(os.path.join(pkg, "_core.pyi")).read()
        for name in mdz.__all__:
            if name != "__version__":
                assert f"def {name}(" in stub or f"class {name}" in stub, name


@pytest.mark.skipif(importlib.util.find_spec("ovito") is None, reason="needs OVITO")
def test_ovito_reader_builds_cell_from_box(tmp_path):  # pragma: no cover
    from ovito.io import import_file

    from mdz.ovito_reader import MdzFileReader

    boxes = np.tile(np.array([[3, 0, 0], [1, 3, 0], [0, 0, 3]], np.float32), (20, 1, 1))
    path = str(tmp_path / "box.mdz")
    mdz.save(path, mdz.compress(_pos(), boxes=boxes))
    data = import_file(path, input_format=MdzFileReader(scale=1.0)).compute(0)
    np.testing.assert_allclose(np.asarray(data.cell)[:, :3], boxes[0].T, atol=1e-5)


def _steps_boxes(nframes, first_step=0):
    steps = np.arange(nframes, dtype=np.int64) * 1000 + first_step
    times = np.arange(nframes, dtype=np.float64) * 0.002
    boxes = np.stack(
        [np.array([[3, 0, 0], [0.5, 3, 0], [0.2, 0.1, 3]]) + 0.01 * i for i in range(nframes)]
    ).astype(np.float32)
    return steps, times, boxes


def _write_in_pieces(w, pos, steps, times, boxes, sizes):
    """Write frames in chunks of ``sizes``. A chunk of size 1 is passed as a
    single (natoms, 3) frame."""
    i = 0
    for n in sizes:
        if n == 1:
            w.write(pos[i], step=int(steps[i]), time=float(times[i]), boxes=boxes[i])
        else:
            sl = slice(i, i + n)
            w.write(pos[sl], step=steps[sl], time=times[sl], boxes=boxes[sl])
        i += n
    assert i == len(pos)


class TestWriter:
    SIZES = [1, 7, 13, 1, 1, 27]  # 50 frames, not aligned to groups

    def test_pieces_match_one_shot_compress(self, tmp_path):
        pos = _pos(nframes=50, natoms=12)
        steps, times, boxes = _steps_boxes(50)
        path = tmp_path / "w.mdz"
        with mdz.Writer(path, precision=PRECISION, gof_size=10) as w:
            _write_in_pieces(w, pos, steps, times, boxes, self.SIZES)
            assert w.frames_written == 50
        assert w.closed and w.frames_written == 50 and w.frames_committed == 50

        ref_pos, ref_boxes = mdz.decompress(
            mdz.compress(pos, boxes=boxes, precision=PRECISION, gof_size=10)
        )
        ct = mdz.load(path)
        got_pos, got_boxes = mdz.decompress(ct)
        np.testing.assert_array_equal(got_pos, ref_pos)
        np.testing.assert_array_equal(got_boxes, ref_boxes)
        np.testing.assert_array_equal(ct.read_column("step"), steps)
        np.testing.assert_array_equal(ct.read_column("time"), times)
        np.testing.assert_array_equal(mdz.open(path).read_all(), ref_pos)

    def test_matches_compress_batch_with_columns_and_metadata(self, tmp_path):
        pos, vel = _pos(nframes=30, natoms=8), _pos(nframes=30, natoms=8, seed=5)
        names = [f"A{i}" for i in range(8)]
        types = list(range(1, 9))
        path = tmp_path / "cols.mdz"
        with mdz.Writer(
            path, atom_names=names, atom_types=types, precision=PRECISION, gof_size=10
        ) as w:
            for a, b in [(0, 4), (4, 5), (5, 30)]:
                w.write(pos[a:b], velocities=vel[a:b])
        ref = mdz.compress_batch(
            pos, velocities=vel, atom_names=names, atom_types=types,
            precision=PRECISION, gof_size=10,
        )
        ct = mdz.load(path)
        for col in ("positions", "velocities"):
            np.testing.assert_array_equal(ct.read_column(col), ref.read_column(col))
        assert ct.metadata() == ref.metadata()
        r = mdz.open(path)
        assert r.atom_names == names and r.atom_types == types

    def test_lossless_is_exact(self, tmp_path):
        pos = _pos(nframes=13, natoms=5)
        path = tmp_path / "ll.mdz"
        with mdz.Writer(path, lossless=True, gof_size=4) as w:
            for x in pos:
                w.write(x)
        np.testing.assert_array_equal(mdz.open(path).read_all(), pos)

    def test_append_continues_file(self, tmp_path):
        pos = _pos(nframes=50, natoms=12)
        steps, times, boxes = _steps_boxes(50)
        names = [f"C{i}" for i in range(12)]
        path = tmp_path / "app.mdz"
        kw = dict(precision=PRECISION, gof_size=10)
        with mdz.Writer(path, atom_names=names, **kw) as w:
            w.write(pos[:23], step=steps[:23], time=times[:23], boxes=boxes[:23])
        # Static metadata is taken from the file and need not be repeated.
        with mdz.Writer(path, append=True, **kw) as w:
            assert w.frames_written == 23
            w.write(pos[23:], step=steps[23:], time=times[23:], boxes=boxes[23:])
            assert w.frames_written == 50

        r = mdz.open(path)
        assert len(r) == 50 and r.atom_names == names
        ref_pos, _ = mdz.decompress(mdz.compress(pos, precision=PRECISION, gof_size=10))
        # The first writer closed frames 20-22 as a short group, so only the
        # first 20 frames match a one-shot compression exactly.
        np.testing.assert_allclose(r.read_all(), pos, atol=TOL, rtol=0)
        np.testing.assert_array_equal(r.read_all()[:20], ref_pos[:20])
        np.testing.assert_array_equal(r.read_column("step"), steps)
        np.testing.assert_array_equal(r.read_column("time"), times)
        np.testing.assert_allclose(r.cells, boxes, atol=0)

    def test_append_with_other_settings_raises(self, tmp_path):
        path = tmp_path / "app.mdz"
        with mdz.Writer(path, precision=PRECISION, gof_size=5) as w:
            w.write(_pos(nframes=5))
        w = mdz.Writer(path, append=True, lossless=True, gof_size=5)
        with pytest.raises(ValueError, match="settings"):
            w.write(_pos(nframes=5))
        w.close()
        assert mdz.open(path).nframes == 5

    def test_append_requires_existing_file(self, tmp_path):
        with pytest.raises(OSError):
            mdz.Writer(tmp_path / "missing.mdz", append=True)

    def test_flushed_but_never_closed_file_loads(self, tmp_path):
        import shutil

        pos = _pos(nframes=40, natoms=6)
        path = tmp_path / "live.mdz"
        w = mdz.Writer(path, precision=PRECISION, gof_size=10)
        w.write(pos[:25], step=np.arange(25))
        assert (w.frames_written, w.frames_committed) == (25, 20)
        # A copy taken at this point is what a crash would leave: only the
        # complete groups.
        shutil.copy(path, tmp_path / "before_flush.mdz")
        w.flush()
        assert w.frames_committed == 25
        shutil.copy(path, tmp_path / "after_flush.mdz")
        w.write(pos[25:], step=np.arange(25, 40))
        w.close()

        r = mdz.open(tmp_path / "before_flush.mdz")
        assert len(r) == 20
        np.testing.assert_allclose(r.read_all(), pos[:20], atol=TOL, rtol=0)
        r = mdz.open(tmp_path / "after_flush.mdz")
        assert len(r) == 25
        np.testing.assert_allclose(r.read_all(), pos[:25], atol=TOL, rtol=0)
        np.testing.assert_array_equal(r.read_column("step"), np.arange(25))
        assert mdz.load(tmp_path / "after_flush.mdz").nframes == 25
        assert mdz.open(path).nframes == 40

        # A file that was never closed can be appended to.
        with mdz.Writer(tmp_path / "after_flush.mdz", append=True,
                        precision=PRECISION, gof_size=10) as w2:
            w2.write(pos[25:], step=np.arange(25, 40))
        r = mdz.open(tmp_path / "after_flush.mdz")
        assert len(r) == 40
        np.testing.assert_array_equal(r.read_column("step"), np.arange(40))

    def test_huge_steps_are_exact(self, tmp_path):
        steps = np.array([2**53 + 1, 2**62 + 3, 2**63 - 1, -(2**60) - 7], dtype=np.int64)
        path = tmp_path / "steps.mdz"
        with mdz.Writer(path, gof_size=3) as w:
            w.write(_pos(nframes=1, natoms=4)[0], step=int(steps[0]))  # Python int
            w.write(_pos(nframes=3, natoms=4), step=steps[1:])
        got = mdz.open(path).read_column("step")
        assert got.dtype == np.int64
        assert got.tolist() == steps.tolist()

    def test_write_after_close_raises(self, tmp_path):
        path = tmp_path / "c.mdz"
        w = mdz.Writer(path, gof_size=5)
        w.write(_pos(nframes=3))
        w.close()
        w.close()  # idempotent
        assert w.closed and w.frames_written == 3
        with pytest.raises(ValueError, match="closed"):
            w.write(_pos(nframes=1))
        with pytest.raises(ValueError, match="closed"):
            w.flush()
        assert mdz.open(path).nframes == 3

    def test_mismatched_atom_count_raises(self, tmp_path):
        path = tmp_path / "m.mdz"
        with mdz.Writer(path, gof_size=4) as w:
            w.write(_pos(nframes=2, natoms=10))  # still buffered
            with pytest.raises(ValueError, match="atom"):
                w.write(_pos(nframes=1, natoms=11))
            w.write(_pos(nframes=2, natoms=10))  # completes the first group
            with pytest.raises(ValueError, match="atoms"):
                w.write(_pos(nframes=1, natoms=9))
            assert w.frames_written == 4
        assert mdz.open(path).nframes == 4

    def test_bad_shapes_and_metadata_raise(self, tmp_path):
        with mdz.Writer(tmp_path / "s.mdz", atom_names=["X"] * 3) as w:
            with pytest.raises(ValueError, match="atom_names"):
                w.write(_pos(nframes=2, natoms=4))
            with pytest.raises(ValueError, match="positions"):
                w.write(np.zeros((4, 2)))
            with pytest.raises(ValueError, match="step"):
                w.write(_pos(nframes=2, natoms=3), step=[1, 2, 3])
            w.write(_pos(nframes=2, natoms=3))

    def test_exception_in_block_is_not_masked(self, tmp_path):
        path = tmp_path / "e.mdz"
        with pytest.raises(RuntimeError, match="boom"):
            with mdz.Writer(path) as w:
                raise RuntimeError("boom")  # close also fails, as there are no frames
        with pytest.raises(RuntimeError, match="boom"):
            with mdz.Writer(path, gof_size=4) as w:
                w.write(_pos(nframes=6))
                raise RuntimeError("boom")
        assert mdz.open(path).nframes == 6  # the file was still closed properly

    def test_empty_writer_close_raises(self, tmp_path):
        w = mdz.Writer(tmp_path / "empty.mdz")
        with pytest.raises(ValueError, match="no frames"):
            w.close()
        assert w.closed

    def test_bad_parameters_raise(self, tmp_path):
        with pytest.raises(ValueError, match="gof_size"):
            mdz.Writer(tmp_path / "p.mdz", gof_size=0)
        with pytest.raises(ValueError):
            mdz.Writer(tmp_path / "p.mdz", codec="nope")


@pytest.mark.skipif(importlib.util.find_spec("ovito") is None, reason="needs OVITO")
def test_ovito_reader_varying_atoms(tmp_path):  # pragma: no cover
    from ovito.io import import_file

    from mdz.ovito_reader import MdzFileReader

    path = str(tmp_path / "vary.mdz")
    rng = np.random.default_rng(0)
    frames = []
    with mdz.Writer(path, variable_atoms=True, gof_size=4) as w:
        for k in range(10):
            ids = np.arange(k, 20 + 2 * k)
            x = rng.uniform(0, 3, (len(ids), 3)).astype(np.float32)
            frames.append((ids, x))
            w.write(x, ids=ids, step=100 * k, boxes=np.eye(3) * 3.0)
    assert MdzFileReader.detect(path)
    pipeline = import_file(path, input_format=MdzFileReader(scale=1.0))
    assert pipeline.num_frames == 10
    for k in (0, 3, 9):
        data = pipeline.compute(k)
        ids, x = frames[k]
        assert data.particles.count == len(ids)
        np.testing.assert_array_equal(data.particles["Particle Identifier"], ids)
        np.testing.assert_allclose(data.particles.positions, x, atol=1e-3)
        assert data.attributes["step"] == 100 * k
        np.testing.assert_allclose(np.asarray(data.cell)[:, :3], np.eye(3) * 3.0, atol=1e-6)
