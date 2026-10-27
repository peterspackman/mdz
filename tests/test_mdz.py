"""Round-trip and public-API tests for the mdz codec.

These exercise the compiled extension (``mdz._core``) through the ``mdz``
package: compress/decompress, save/load, the random-access reader,
multi-column batches, and the precision contract.
"""

from __future__ import annotations

import numpy as np
import pytest

import mdz

PRECISION = 0.001  # nm


def _traj(natoms=200, nframes=40, seed=42, dt=0.002, sigma=0.05):
    """A damped random walk around fixed sites, in nm. 30% of the atoms move
    ten times less than the rest."""
    rng = np.random.default_rng(seed)
    box_size = (natoms / 30.0) ** (1.0 / 3.0)
    eq = rng.uniform(0, box_size, (natoms, 3)).astype(np.float32)
    mobility = np.ones((natoms, 1), dtype=np.float32)
    mobility[: int(natoms * 0.3)] = 0.1

    pos = np.empty((nframes, natoms, 3), dtype=np.float32)
    pos[0] = eq + rng.normal(0, sigma, (natoms, 3)).astype(np.float32) * mobility
    vel = rng.normal(0, sigma / dt, (natoms, 3)).astype(np.float32) * 0.1 * mobility
    friction = 0.9
    for i in range(1, nframes):
        noise = rng.normal(0, sigma * np.sqrt(1 - friction**2), (natoms, 3)).astype(np.float32) * mobility
        force = -0.5 * (pos[i - 1] - eq)
        vel = friction * vel + force * dt + noise
        pos[i] = pos[i - 1] + vel * dt
    return pos


class TestRoundtrip:
    def test_shape_and_dtype_preserved(self):
        pos = _traj()
        ct = mdz.compress(pos, precision=PRECISION)
        recon, boxes = mdz.decompress(ct)
        assert recon.shape == pos.shape
        assert recon.dtype == np.float32
        assert boxes is None
        assert ct.natoms == pos.shape[1]
        assert ct.nframes == pos.shape[0]

    def test_adaptive_precision_contract(self):
        # Documented contract: with adaptive precision the maximum error is at
        # most 4 * precision.
        pos = _traj()
        ct = mdz.compress(pos, precision=PRECISION, adaptive=True)
        recon, _ = mdz.decompress(ct)
        assert np.abs(recon - pos).max() <= PRECISION * 4 + 1e-6

    def test_nonadaptive_precision_contract(self):
        pos = _traj()
        ct = mdz.compress(pos, precision=PRECISION, adaptive=False)
        recon, _ = mdz.decompress(ct)
        assert np.abs(recon - pos).max() <= PRECISION + 1e-6

    @pytest.mark.parametrize("codec", ["auto", "temporal", "intra"])
    def test_codec_modes_roundtrip(self, codec):
        pos = _traj()
        ct = mdz.compress(pos, precision=PRECISION, codec=codec)
        recon, _ = mdz.decompress(ct)
        assert recon.shape == pos.shape
        # A loose bound that catches gross corruption with any codec.
        assert np.abs(recon - pos).max() <= PRECISION * 8

    def test_single_frame(self):
        pos = _traj(natoms=64, nframes=1)
        ct = mdz.compress(pos, precision=PRECISION)
        recon, _ = mdz.decompress(ct)
        assert recon.shape == pos.shape
        assert np.abs(recon - pos).max() <= PRECISION * 4 + 1e-6

    def test_single_atom(self):
        pos = _traj(natoms=1, nframes=20)
        ct = mdz.compress(pos, precision=PRECISION)
        recon, _ = mdz.decompress(ct)
        assert recon.shape == pos.shape
        assert np.abs(recon - pos).max() <= PRECISION * 4 + 1e-6


class TestLossless:
    """``lossless=True`` guarantees bit-exact output, so these tests compare
    bit patterns rather than using a tolerance."""

    def test_bit_exact_round_trip(self):
        pos = _traj(natoms=120, nframes=25)
        ct = mdz.compress(pos, lossless=True)
        recon, _ = mdz.decompress(ct)
        assert np.array_equal(recon.view(np.uint32), pos.view(np.uint32))

    def test_awkward_values_survive(self):
        # Signed zeros, denormals and the float32 extremes are the values most
        # likely to expose a codec that is only exact for typical input. The
        # bit comparison makes -0.0 decoding as 0.0 a failure.
        awkward = np.array(
            [0.0, -0.0, 1.0, -1.0, 1e-40, -1e-40,
             np.finfo(np.float32).max, np.finfo(np.float32).min,
             np.finfo(np.float32).tiny, 1 / 3, -2.5e-7, 12345.678],
            dtype=np.float32,
        )
        pos = np.tile(awkward, (4, 1)).reshape(2, 8, 3).astype(np.float32)
        ct = mdz.compress(pos, lossless=True)
        recon, _ = mdz.decompress(ct)
        assert np.array_equal(recon.view(np.uint32), pos.view(np.uint32))

    def test_overrides_the_rounding_options(self):
        # lossless=True overrides the precision and codec options.
        pos = _traj(natoms=60, nframes=20)
        ct = mdz.compress(pos, lossless=True, precision=0.1, codec="intra")
        recon, _ = mdz.decompress(ct)
        assert np.array_equal(recon.view(np.uint32), pos.view(np.uint32))

    def test_costs_more_than_lossy(self):
        # The docs state that lossless output is larger than lossy output.
        pos = _traj(natoms=200, nframes=40)
        lossy = mdz.compress(pos, precision=PRECISION).compressed_size
        exact = mdz.compress(pos, lossless=True).compressed_size
        assert exact > lossy


class TestInputValidation:
    def test_nan_rejected(self):
        pos = _traj(natoms=16, nframes=5)
        pos[2, 3, 1] = np.nan
        with pytest.raises(ValueError):
            mdz.compress(pos, precision=PRECISION)

    def test_inf_rejected(self):
        pos = _traj(natoms=16, nframes=5)
        pos[0, 0, 0] = np.inf
        with pytest.raises(ValueError):
            mdz.compress(pos, precision=PRECISION)

    def test_nonpositive_precision_rejected(self):
        pos = _traj(natoms=16, nframes=5)
        with pytest.raises(ValueError):
            mdz.compress(pos, precision=0.0)

    def test_zero_gof_size_rejected(self):
        pos = _traj(natoms=16, nframes=5)
        with pytest.raises(ValueError):
            mdz.compress(pos, precision=PRECISION, gof_size=0)


class TestBoxes:
    def _boxes(self, nframes, edge=3.0):
        # A diagonal cell that grows slightly each frame.
        boxes = np.zeros((nframes, 3, 3), dtype=np.float32)
        for i in range(nframes):
            d = edge + 0.001 * i
            boxes[i] = np.diag([d, d, d])
        return boxes

    def test_boxes_roundtrip(self):
        pos = _traj(natoms=50, nframes=10)
        boxes = self._boxes(10)
        ct = mdz.compress(pos, boxes=boxes, precision=PRECISION)
        recon, rboxes = mdz.decompress(ct)
        assert rboxes is not None
        assert rboxes.shape == boxes.shape
        assert np.abs(rboxes - boxes).max() <= PRECISION + 1e-5

    def test_decode_boxes_matches_decompress(self, tmp_path):
        pos = _traj(natoms=50, nframes=12)
        boxes = self._boxes(12)
        ct = mdz.compress(pos, boxes=boxes, precision=PRECISION)
        path = str(tmp_path / "boxes.mdz")
        mdz.save(path, ct)
        only_boxes = mdz.decode_boxes(path)
        _, rboxes = mdz.decompress(mdz.load(path))
        np.testing.assert_allclose(only_boxes, rboxes, atol=0)


class TestSaveLoad:
    def test_save_load_roundtrip(self, tmp_path):
        pos = _traj()
        ct = mdz.compress(pos, precision=PRECISION)
        path = str(tmp_path / "traj.mdz")
        nbytes = mdz.save(path, ct)
        assert nbytes > 0
        ct2 = mdz.load(path)
        assert ct2.natoms == ct.natoms
        assert ct2.nframes == ct.nframes
        recon1, _ = mdz.decompress(ct)
        recon2, _ = mdz.decompress(ct2)
        np.testing.assert_array_equal(recon1, recon2)

    def test_info_reports_dimensions(self, tmp_path):
        pos = _traj(natoms=128, nframes=30)
        ct = mdz.compress(pos, precision=PRECISION)
        path = str(tmp_path / "traj.mdz")
        mdz.save(path, ct)
        meta = mdz.info(path)
        assert meta["natoms"] == 128
        assert meta["nframes"] == 30


class TestReader:
    def test_reader_random_access_matches_decompress(self, tmp_path):
        pos = _traj(natoms=80, nframes=50)
        ct = mdz.compress(pos, precision=PRECISION)
        path = str(tmp_path / "traj.mdz")
        mdz.save(path, ct)
        recon, _ = mdz.decompress(mdz.load(path))

        reader = mdz.open(path)
        assert reader.natoms == 80
        assert reader.nframes == 50

        np.testing.assert_array_equal(reader.read_frame(0), recon[0])
        np.testing.assert_array_equal(reader.read_frame(49), recon[49])
        np.testing.assert_array_equal(reader.read_frames(10, 20), recon[10:20])
        np.testing.assert_array_equal(reader.read_all(), recon)


class TestMultiColumn:
    def test_compress_batch_multi_column_roundtrip(self, tmp_path):
        pos = _traj(natoms=64, nframes=20)
        vel = _traj(natoms=64, nframes=20, seed=7) * 0.1
        nframes = pos.shape[0]
        cell = np.tile([3.0, 90.0, 3.0, 90.0, 90.0, 3.0], (nframes, 1)).astype(np.float64)

        ct = mdz.compress_batch(
            pos, velocities=vel, cell=cell, precision=PRECISION
        )
        names = set(ct.column_names)
        assert {"positions", "velocities"} <= names

        path = str(tmp_path / "traj_multi.mdz")
        ct.save(path)
        ct2 = mdz.load(path)

        rpos = ct2.read_column("positions")
        assert rpos.shape == pos.shape
        assert np.abs(rpos - pos).max() <= PRECISION * 4 + 1e-6

    def test_step_beyond_int32_roundtrips(self, tmp_path):
        # Step counters beyond the int32 range must not saturate. They are
        # stored as int64.
        pos = _traj(natoms=32, nframes=10)
        nframes = pos.shape[0]
        base = 5_000_000_000  # > 2**31 - 1
        step = np.array([base + i * 1_000_000 for i in range(nframes)], dtype=np.int64)
        assert step.max() > np.iinfo(np.int32).max

        ct = mdz.compress_batch(pos, step=step, precision=PRECISION)
        path = str(tmp_path / "step.mdz")
        ct.save(path)
        ct2 = mdz.load(path)

        rstep = ct2.read_column("step")
        assert rstep.dtype == np.int64
        np.testing.assert_array_equal(rstep, step)


class TestConcurrency:
    """The bindings release the GIL while compressing and decompressing.
    These tests run real Python threads to check that results stay correct
    when calls overlap."""

    def test_concurrent_compress_decompress_matches_serial(self):
        import threading

        trajs = [_traj(natoms=64, nframes=30, seed=s) for s in range(8)]
        expected = [mdz.decompress(mdz.compress(p, precision=PRECISION))[0] for p in trajs]

        results: list = [None] * len(trajs)

        def worker(i):
            ct = mdz.compress(trajs[i], precision=PRECISION)
            results[i] = mdz.decompress(ct)[0]

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(trajs))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for got, want in zip(results, expected):
            np.testing.assert_array_equal(got, want)

    def test_concurrent_reads_on_shared_reader(self, tmp_path):
        import threading

        pos = _traj(natoms=80, nframes=60)
        path = str(tmp_path / "traj.mdz")
        mdz.save(path, mdz.compress(pos, precision=PRECISION))
        reader = mdz.open(path)
        expected = reader.read_all()

        results: dict = {}

        def worker(i):
            results[i] = reader.read_frames(0, 60)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i in range(8):
            np.testing.assert_array_equal(results[i], expected)


class TestReaderProtocol:
    """The random-access reader behaves like a sequence of frames and exposes
    every column, the per-frame cell and the atom metadata."""

    def _make_multi(self, tmp_path, natoms=24, nframes=12):
        pos = _traj(natoms=natoms, nframes=nframes)
        vel = _traj(natoms=natoms, nframes=nframes, seed=3) * 0.1
        cell = np.tile([3.0, 90.0, 3.0, 90.0, 90.0, 3.0], (nframes, 1)).astype(np.float64)
        step = np.arange(nframes, dtype=np.int64) * 5
        names = ["C" if i % 2 == 0 else "H" for i in range(natoms)]
        ct = mdz.compress_batch(
            pos, velocities=vel, cell=cell, step=step, atom_names=names, precision=PRECISION
        )
        path = str(tmp_path / "reader.mdz")
        ct.save(path)
        return path, pos, vel, step, names

    def test_len_index_slice_iter(self, tmp_path):
        path, pos, _, _, _ = self._make_multi(tmp_path)
        r = mdz.open(path)
        n = pos.shape[0]
        tol = PRECISION * 4 + 1e-6

        assert len(r) == n
        np.testing.assert_allclose(r[0], pos[0], atol=tol)
        np.testing.assert_allclose(r[-1], pos[-1], atol=tol)  # negative index

        sl = r[2:6]
        assert sl.shape == (4, pos.shape[1], 3)
        np.testing.assert_allclose(sl, pos[2:6], atol=tol)

        frames = list(r)  # iteration
        assert len(frames) == n
        np.testing.assert_allclose(frames[5], pos[5], atol=tol)

        with pytest.raises(IndexError):
            _ = r[n]

    def test_cell_columns_metadata(self, tmp_path):
        path, pos, vel, step, names = self._make_multi(tmp_path)
        r = mdz.open(path)

        # The per-frame cell is a 3x3 matrix, here cubic with ~3 nm edges.
        c0 = r.cell(0)
        assert c0.shape == (3, 3)
        for d in range(3):
            assert abs(c0[d, d] - 3.0) < 1e-3
        assert r.cells.shape == (pos.shape[0], 3, 3)

        assert {"positions", "velocities", "cell", "step"} <= set(r.column_names())
        np.testing.assert_allclose(r.read_column("velocities"), vel, atol=0.01)
        np.testing.assert_array_equal(r.read_column("step"), step)

        assert list(r.atom_names) == names

        with mdz.open(path) as r2:
            assert len(r2) == pos.shape[0]

    def test_set_num_threads_noop(self):
        # 0 selects the default thread count and must never raise.
        assert mdz.set_num_threads(0) is None
