"""Tests for reading and writing the supported trajectory formats."""

import os
import tempfile
import warnings
import numpy as np
import pytest


class TestExtxyz:
    """Extended XYZ format roundtrip tests."""

    def _write_extxyz(self, path, natoms, nframes, positions, atom_names=None, cell=None):
        """Write a simple extxyz file for testing."""
        if atom_names is None:
            atom_names = ["X"] * natoms
        with open(path, "w") as f:
            for frame in range(nframes):
                f.write(f"{natoms}\n")
                comment_parts = ["Properties=species:S:1:pos:R:3"]
                if cell is not None:
                    a, b, c = cell
                    comment_parts.append(
                        f'Lattice="{a:.6f} 0.0 0.0 0.0 {b:.6f} 0.0 0.0 0.0 {c:.6f}"'
                    )
                comment_parts.append('pbc="T T T"')
                f.write(" ".join(comment_parts) + "\n")
                for atom in range(natoms):
                    x = positions[frame, atom, 0]
                    y = positions[frame, atom, 1]
                    z = positions[frame, atom, 2]
                    f.write(f"{atom_names[atom]} {x:.6f} {y:.6f} {z:.6f}\n")

    def _read_extxyz(self, path):
        """Read an extxyz file back."""
        frames = []
        atom_names = None
        with open(path) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                natoms = int(line.strip())
                comment = f.readline()  # comment line
                positions = []
                names = []
                for _ in range(natoms):
                    parts = f.readline().split()
                    names.append(parts[0])
                    positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                frames.append(positions)
                if atom_names is None:
                    atom_names = names
        return np.array(frames, dtype=np.float32), atom_names

    def test_extxyz_write_read(self):
        """Write and read back an extxyz file."""
        natoms = 5
        nframes = 3
        # Positions in Angstroms for the file
        positions = np.random.default_rng(42).random((nframes, natoms, 3)).astype(np.float32) * 10.0
        names = ["C", "O", "H", "H", "N"]

        with tempfile.NamedTemporaryFile(suffix=".extxyz", delete=False) as f:
            path = f.name

        try:
            self._write_extxyz(path, natoms, nframes, positions, names, cell=[30.0, 30.0, 30.0])
            read_pos, read_names = self._read_extxyz(path)

            assert read_pos.shape == positions.shape
            np.testing.assert_allclose(read_pos, positions, atol=1e-4)
            assert read_names == names
        finally:
            os.unlink(path)


class TestXyz:
    """Plain XYZ format tests."""

    def test_xyz_roundtrip(self):
        natoms = 3
        nframes = 2
        # Positions in Angstroms
        positions_ang = np.array([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]],
        ], dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False, mode="w") as f:
            path = f.name
            for frame in range(nframes):
                f.write(f"{natoms}\n")
                f.write("comment\n")
                for atom in range(natoms):
                    f.write(f"X {positions_ang[frame, atom, 0]:.6f} "
                            f"{positions_ang[frame, atom, 1]:.6f} "
                            f"{positions_ang[frame, atom, 2]:.6f}\n")

        try:
            frames = []
            with open(path) as fh:
                while True:
                    line = fh.readline()
                    if not line:
                        break
                    n = int(line.strip())
                    fh.readline()  # comment
                    frame = []
                    for _ in range(n):
                        parts = fh.readline().split()
                        frame.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    frames.append(frame)

            read_pos = np.array(frames, dtype=np.float32)
            np.testing.assert_allclose(read_pos, positions_ang, atol=1e-4)
        finally:
            os.unlink(path)


class TestLammps:
    """LAMMPS dump format tests."""

    def _write_lammps(self, path, natoms, nframes, positions_ang, box_size=30.0):
        with open(path, "w") as f:
            for frame in range(nframes):
                f.write("ITEM: TIMESTEP\n")
                f.write(f"{frame}\n")
                f.write("ITEM: NUMBER OF ATOMS\n")
                f.write(f"{natoms}\n")
                f.write("ITEM: BOX BOUNDS pp pp pp\n")
                f.write(f"0.0 {box_size}\n")
                f.write(f"0.0 {box_size}\n")
                f.write(f"0.0 {box_size}\n")
                f.write("ITEM: ATOMS id type x y z\n")
                for atom in range(natoms):
                    f.write(f"{atom+1} 1 {positions_ang[frame, atom, 0]:.6f} "
                            f"{positions_ang[frame, atom, 1]:.6f} "
                            f"{positions_ang[frame, atom, 2]:.6f}\n")

    def _read_lammps(self, path):
        frames = []
        with open(path) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if "ITEM: TIMESTEP" in line:
                    f.readline()  # step
                    f.readline()  # ITEM: NUMBER
                    natoms = int(f.readline().strip())
                    f.readline()  # ITEM: BOX
                    for _ in range(3):
                        f.readline()
                    f.readline()  # ITEM: ATOMS
                    frame = []
                    for _ in range(natoms):
                        parts = f.readline().split()
                        frame.append([float(parts[2]), float(parts[3]), float(parts[4])])
                    frames.append(frame)
        return np.array(frames, dtype=np.float32)

    def test_lammps_roundtrip(self):
        natoms = 4
        nframes = 3
        positions = np.random.default_rng(42).random((nframes, natoms, 3)).astype(np.float32) * 10.0

        with tempfile.NamedTemporaryFile(suffix=".lammpstrj", delete=False) as f:
            path = f.name

        try:
            self._write_lammps(path, natoms, nframes, positions)
            read_pos = self._read_lammps(path)
            assert read_pos.shape == positions.shape
            np.testing.assert_allclose(read_pos, positions, atol=1e-4)
        finally:
            os.unlink(path)


class TestConvert:
    """mdz.convert between trajectory formats and .mdz."""

    NAMES = ["C", "O", "H", "N"]

    def _extxyz(self, tmp_path, natoms=4, nframes=3):
        pos = np.random.default_rng(7).random((nframes, natoms, 3)).astype(np.float32) * 10.0
        path = str(tmp_path / "in.extxyz")
        TestExtxyz()._write_extxyz(path, natoms, nframes, pos, self.NAMES, cell=[30.0, 30.0, 30.0])
        return path, pos

    def test_extxyz_to_mdz_keeps_positions_names_and_cell(self, tmp_path):
        import mdz

        src, pos = self._extxyz(tmp_path)
        out = str(tmp_path / "out.mdz")
        mdz.convert(src, out)
        r = mdz.open(out)
        # extxyz is in Angstrom, .mdz in nm.
        np.testing.assert_allclose(r.read_all(), pos / 10.0, atol=5e-4)
        assert r.atom_names == self.NAMES
        np.testing.assert_allclose(r.cell(0), np.eye(3) * 3.0, atol=1e-5)

    def test_format_is_detected_from_extension(self, tmp_path):
        import mdz

        src, pos = self._extxyz(tmp_path)
        mdz_path = str(tmp_path / "out.mdz")
        mdz.convert(src, mdz_path)
        for ext in ["xyz", "extxyz", "pdb", "lammpstrj", "dcd", "xtc"]:
            out = str(tmp_path / f"back.{ext}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # lossy targets warn; tested separately
                mdz.convert(mdz_path, out)
            # Converting back to .mdz reads the file with that format's reader.
            again = str(tmp_path / f"again_{ext}.mdz")
            mdz.convert(out, again)
            np.testing.assert_allclose(
                mdz.open(again).read_all(), pos / 10.0, atol=2e-3, err_msg=ext
            )

    def test_npy_output(self, tmp_path):
        import mdz

        src, pos = self._extxyz(tmp_path)
        out = str(tmp_path / "out.npy")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdz.convert(src, out)
        np.testing.assert_allclose(np.load(out), pos / 10.0, atol=1e-5)

    def test_unknown_output_format_raises_before_writing(self, tmp_path):
        import mdz

        src, _ = self._extxyz(tmp_path)
        out = tmp_path / "out.foo"
        with pytest.raises(ValueError, match="unsupported output format"):
            mdz.convert(src, str(out))
        assert not out.exists()

    def test_warns_about_dropped_data(self, tmp_path):
        import mdz

        src, _ = self._extxyz(tmp_path)
        mdz_path = str(tmp_path / "out.mdz")
        mdz.convert(src, mdz_path)
        with pytest.warns(UserWarning, match="cell"):
            mdz.convert(mdz_path, str(tmp_path / "out.xyz"))

    def test_warns_about_dropped_box(self, tmp_path):
        import mdz

        pos = np.random.default_rng(3).random((4, 5, 3)).astype(np.float32)
        boxes = np.tile(np.eye(3, dtype=np.float32) * 2.0, (4, 1, 1))
        mdz_path = str(tmp_path / "box.mdz")
        mdz.save(mdz_path, mdz.compress(pos, boxes=boxes))
        with pytest.warns(UserWarning, match="box"):
            mdz.convert(mdz_path, str(tmp_path / "out.xyz"))

    def test_refuses_to_overwrite(self, tmp_path):
        import mdz

        src, pos = self._extxyz(tmp_path)
        out = tmp_path / "out.mdz"
        out.write_bytes(b"precious")
        with pytest.raises(FileExistsError):
            mdz.convert(src, str(out))
        assert out.read_bytes() == b"precious"
        mdz.convert(src, out, overwrite=True)  # os.PathLike is accepted too
        assert mdz.open(str(out)).nframes == pos.shape[0]

    def test_refuses_to_convert_onto_itself(self, tmp_path):
        import mdz

        src, _ = self._extxyz(tmp_path)
        before = open(src, "rb").read()
        with pytest.raises(ValueError, match="same file"):
            mdz.convert(src, src, overwrite=True)
        assert open(src, "rb").read() == before
