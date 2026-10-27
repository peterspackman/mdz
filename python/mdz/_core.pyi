"""Type stubs for the mdz C++ extension (``mdz._core``).

Keep in sync with ``bindings/python/*.cpp``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from types import TracebackType
from typing import Any, Literal, final, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Path = str | os.PathLike[str]

__all__ = [
    "Trajectory",
    "MdzReader",
    "Writer",
    "compress",
    "decompress",
    "save",
    "load",
    "open",
    "info",
    "decode_boxes",
    "convert",
    "compress_batch",
    "set_num_threads",
]

@final
class Trajectory:
    """A compressed trajectory, holding the columns, their descriptors and the
    metadata.

    Returned by :func:`compress`, :func:`compress_batch` and :func:`load`.
    Write it to a file with :meth:`save`.
    """

    @property
    def natoms(self) -> int:
        """Atoms per frame, or 0 if the count varies (see ``variable_atoms``)."""
    @property
    def nframes(self) -> int: ...
    @property
    def variable_atoms(self) -> bool:
        """Whether the atom count varies between frames."""
    @property
    def frame_natoms(self) -> NDArray[np.int64]:
        """Number of atoms in each frame, as an (nframes,) array."""
    @property
    def rigid_bodies(self) -> int:
        """Number of rigid bodies recorded in the metadata (0 if none)."""
    @property
    def gof_size(self) -> int: ...
    @property
    def compressed_size(self) -> int: ...
    @property
    def precision(self) -> float:
        """Quantization step of the positions column, in nm, or 0.0 if lossless."""
    @property
    def ratio(self) -> float:
        """Ratio of uncompressed to compressed size, over all stored columns."""
    @property
    def column_names(self) -> list[str]:
        """Names of all readable columns. ``cell`` is derived from ``box``."""
    def columns(self) -> list[dict[str, Any]]:
        """Column descriptors (name, dtype, width, per_frame, lossy, precision, unit)."""
    def metadata(self) -> dict[str, Any]:
        """Metadata dict (atom_names, atom_types, atom_ids, residues, bodies, ...)."""
    def read_column(self, name: str) -> NDArray[Any]:
        """Decode a column by name.

        Shapes:

        - per-atom, width 3: (nframes, natoms, 3)
        - per-atom scalar: (nframes, natoms)
        - ``box``: (nframes, 3, 3)
        - ``cell``: (nframes, 6), derived from ``box``
        - per-frame scalar: (nframes,)

        When the atom count varies, per-atom columns are flat, with shape
        (total_atoms, 3) or (total_atoms,). Split them by frame with
        ``np.split(a, np.cumsum(frame_natoms)[:-1])``. Raises KeyError for an
        unknown name.
        """
    def save(self, path: _Path) -> int:
        """Save to a .mdz file. Returns the number of bytes written."""

@final
class MdzReader:
    """Random-access reader for .mdz files, behaving as a read-only sequence
    of frames.

    Only the groups that contain the requested frames are decompressed.

    When the atom count varies (``variable_atoms``), ``natoms`` is 0 and
    frame i has shape ``(frame_natoms[i], 3)``. A range of frames
    (``read_frames``, a slice, ``read_all``) is returned as a list of such
    arrays, one per frame. ``atom_ids(i)`` gives the atom ids of frame i.
    """

    @property
    def natoms(self) -> int:
        """Atoms per frame, or 0 if the count varies (see ``variable_atoms``)."""
    @property
    def nframes(self) -> int: ...
    @property
    def variable_atoms(self) -> bool:
        """Whether the atom count varies between frames."""
    @property
    def frame_natoms(self) -> NDArray[np.int64]:
        """Number of atoms in each frame, as an (nframes,) array."""
    @property
    def rigid_bodies(self) -> int:
        """Number of rigid bodies recorded in the metadata (0 if none)."""
    def atom_ids(self, frame: int) -> NDArray[np.int32]:
        """Ids of the atoms in one frame, in the order the positions are returned.

        For a fixed atom count these are the ``atom_ids`` metadata, or
        0..natoms-1 if there is none. Negative indices count from the end.
        """
    @property
    def precision(self) -> float: ...
    @property
    def gof_size(self) -> int: ...
    @property
    def n_gofs(self) -> int: ...
    @property
    def cells(self) -> NDArray[np.float32] | None:
        """All per-frame unit cells as (nframes, 3, 3) in nm, or None."""
    @property
    def atom_names(self) -> list[str] | None: ...
    @property
    def atom_types(self) -> list[int] | None: ...
    def read_frames(self, start: int, stop: int) -> NDArray[np.float32] | list[NDArray[np.float32]]:
        """Frames [start, stop) as a (k, natoms, 3) array. The range is clipped.

        Returns a list of k (natoms_i, 3) arrays if the atom count varies.
        """
    def read_frame(self, frame_idx: int) -> NDArray[np.float32]:
        """One frame as (natoms, 3). Negative indices count from the end."""
    def read_all(self) -> NDArray[np.float32] | list[NDArray[np.float32]]:
        """All frames as (nframes, natoms, 3), or a list of (natoms_i, 3) arrays if the count varies."""
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: int, /) -> NDArray[np.float32]: ...
    @overload
    def __getitem__(self, key: slice, /) -> NDArray[np.float32] | list[NDArray[np.float32]]: ...
    def __iter__(self) -> Iterator[NDArray[np.float32]]: ...
    def __enter__(self) -> MdzReader: ...
    def __exit__(
        self,
        _exc_type: type[BaseException] | None = None,
        _exc_value: BaseException | None = None,
        _traceback: TracebackType | None = None,
    ) -> bool: ...
    def cell(self, frame: int) -> NDArray[np.float32] | None:
        """Unit cell of one frame as a (3, 3) array in nm, or None if the file has none.

        Negative indices count from the end. An index out of range raises
        IndexError.
        """
    def column_names(self) -> list[str]: ...
    def read_column(self, name: str) -> NDArray[Any]:
        """Decode a whole column by name (see :meth:`Trajectory.read_column`).

        When the atom count varies, per-atom columns, including ``positions``,
        are flat with shape (total_atoms, 3) or (total_atoms,).
        """
    def metadata(self) -> dict[str, Any]: ...

@final
class Writer:
    """Write a .mdz file incrementally, as frames arrive.

    Frames are buffered until a complete group of ``gof_size`` frames is
    ready. The group is then encoded and written to the OS. If the process
    dies, every complete group can still be read from the file.
    :meth:`close`, or leaving the ``with`` block, encodes the remaining frames
    and writes the footer.

    The compression keywords are those of :func:`compress` and
    :func:`compress_batch`. ``atom_names``, ``atom_types`` and ``atom_ids``
    are static, each of length natoms. With ``append=True``, an existing .mdz
    file is continued after its last complete group, whether or not it was
    closed cleanly. Pass the settings it was created with. Static metadata
    that is not given is taken from the file, and a file with a varying atom
    count keeps that setting.

    ``variable_atoms=True`` allows the atom count to change between frames.
    Each call to :meth:`write` then passes the ``ids`` of its frames, and
    atoms are matched across frames by id. The static ``atom_names``,
    ``atom_types`` and ``atom_ids`` cannot be used in this mode. Without it, a
    frame with a different atom count raises ValueError. ``rigid`` and
    ``bodies`` model rigid bodies as in :func:`compress`. Bodies detected in
    the first group are used for the whole file. They cannot be used with
    ``variable_atoms``.

    Raises ValueError on invalid arguments, or on frames or settings that do
    not match earlier frames or the file being appended to. Raises OSError on
    I/O failures.
    """

    def __init__(
        self,
        path: _Path,
        atom_names: list[str] | None = None,
        atom_types: list[int] | None = None,
        atom_ids: list[int] | None = None,
        precision: float = 0.000125,
        column_precision: dict[str, float] | None = None,
        gof_size: int = 20,
        predictor_order: int = 0,
        zstd_level: int = 3,
        adaptive: bool = True,
        anchor_step: int = 1,
        codec: Literal["auto", "temporal", "intra", "lossless"] = "auto",
        lossless: bool = False,
        append: bool = False,
        variable_atoms: bool = False,
        rigid: bool = False,
        bodies: ArrayLike | None = None,
    ) -> None: ...
    def write(
        self,
        positions: ArrayLike,
        velocities: ArrayLike | None = None,
        forces: ArrayLike | None = None,
        charge: ArrayLike | None = None,
        cell: ArrayLike | None = None,
        time: ArrayLike | float | None = None,
        energy: ArrayLike | float | None = None,
        step: ArrayLike | int | None = None,
        boxes: ArrayLike | None = None,
        ids: ArrayLike | None = None,
    ) -> None:
        """Add one frame, with positions ``(natoms, 3)``, or several, with
        positions ``(nframes, natoms, 3)``. Positions are in nm.

        The other arrays match the positions. For one frame, velocities and
        forces are (natoms, 3), charge is (natoms,), cell is (6,), boxes is
        (3, 3), and time, energy and step are scalars. For several frames,
        each has a leading nframes axis, as for :func:`compress_batch`. Every
        call must pass the same arrays and atom count. ``step`` is stored
        exactly as int64. Raises ValueError if the writer is closed.

        With ``variable_atoms=True``, the atom count may change between
        calls. All frames in one call share it, so write one frame per call
        when the count changes every frame. ``ids`` is a ``(natoms,)`` array
        of 32-bit integers that identifies the atoms. Without ``ids``, atoms
        are matched by their position in the frame. Passing ``ids`` to a
        fixed-count writer is an error.
        """
    def flush(self) -> None:
        """Encode the buffered frames as a group, which may be short, and write them to the OS."""
    def close(self) -> None:
        """Encode the buffered frames and write the footer. Closing again does nothing."""
    @property
    def frames_written(self) -> int:
        """Frames written so far, including buffered frames and, when appending,
        the frames already in the file."""
    @property
    def frames_committed(self) -> int:
        """Frames already encoded into the file, which a crash would not lose."""
    @property
    def closed(self) -> bool: ...
    def __enter__(self) -> Writer: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        _exc_value: BaseException | None = None,
        _traceback: TracebackType | None = None,
    ) -> bool: ...

def compress(
    positions: ArrayLike,
    boxes: ArrayLike | None = None,
    precision: float = 0.000125,
    gof_size: int = 20,
    predictor_order: int = 0,
    zstd_level: int = 3,
    adaptive: bool = True,
    anchor_step: int = 1,
    codec: Literal["auto", "temporal", "intra", "lossless"] = "auto",
    lossless: bool = False,
    rigid: bool = False,
    bodies: ArrayLike | None = None,
) -> Trajectory:
    """Compress positions (nframes, natoms, 3) in nm, with optional
    (nframes, 3, 3) boxes.

    Any float dtype and memory layout is accepted. Valid ranges are gof_size
    1-65535, predictor_order 0-4, zstd_level 1-22 and anchor_step 1-255.
    Raises ValueError on wrong shapes, out-of-range parameters, or values
    that cannot be stored.

    ``rigid=True`` also encodes the positions as rigid bodies and keeps the
    smaller encoding for each group. The error bound is unchanged. It is
    ignored when ``lossless`` is True. ``bodies`` is an (natoms,) integer
    array giving each atom's body, or -1 for none. It implies ``rigid``.
    Without it, bodies are detected. The bodies are stored in the metadata
    (``metadata()["bodies"]``).
    """

def compress_batch(
    positions: ArrayLike,
    velocities: ArrayLike | None = None,
    forces: ArrayLike | None = None,
    charge: ArrayLike | None = None,
    cell: ArrayLike | None = None,
    time: ArrayLike | None = None,
    energy: ArrayLike | None = None,
    step: ArrayLike | None = None,
    atom_names: list[str] | None = None,
    atom_types: list[int] | None = None,
    atom_ids: list[int] | None = None,
    precision: float = 0.000125,
    column_precision: dict[str, float] | None = None,
    gof_size: int = 20,
    predictor_order: int = 0,
    zstd_level: int = 3,
    adaptive: bool = True,
    anchor_step: int = 1,
    lossless: bool = False,
    boxes: ArrayLike | None = None,
    rigid: bool = False,
    bodies: ArrayLike | None = None,
) -> Trajectory:
    """Compress a multi-column trajectory.

    Shapes:

    - positions, velocities, forces: (nframes, natoms, 3)
    - charge: (nframes, natoms)
    - cell: (nframes, 6)
    - boxes: (nframes, 3, 3)
    - time, energy, step: (nframes,)

    ``step`` is stored exactly as int64. ``cell`` holds A, gamma, B, beta,
    alpha, C in nm and degrees, and is stored as the equivalent box matrix.
    If both ``cell`` and ``boxes`` are given, ``boxes`` takes precedence.
    ``rigid`` and ``bodies`` behave as in :func:`compress`.
    """

def decompress(
    ct: Trajectory,
) -> tuple[NDArray[np.float32], NDArray[np.float32] | None]:
    """Return ``(positions, boxes)``, with shapes (nframes, natoms, 3) and
    (nframes, 3, 3). ``boxes`` is None if the trajectory has no box.

    If the atom count varies, ``positions`` is a list of (natoms_i, 3) arrays.
    """

def save(path: _Path, ct: Trajectory) -> int:
    """Save a compressed trajectory to a .mdz file. Returns the number of bytes written."""

def load(path: _Path) -> Trajectory:
    """Load a compressed trajectory from a .mdz file."""

def open(path: _Path) -> MdzReader:
    """Open a .mdz file for random-access reading."""

def info(path: _Path) -> dict[str, Any]:
    """Summary of a .mdz file.

    Keys: version, natoms (0 if the count varies), nframes, gof_size,
    n_columns, columns, compressed_size, variable_atoms (bool) and
    rigid_bodies (number of rigid bodies recorded in the metadata).
    """

def decode_boxes(path: _Path) -> NDArray[np.float32] | None:
    """Per-frame (nframes, 3, 3) box matrices in nm, or None if the file has no box.

    Only the box column is decoded.
    """

def convert(input: _Path, output: _Path, overwrite: bool = False) -> None:
    """Convert between trajectory formats, inferred from the file extensions.

    Raises FileExistsError if ``output`` exists and ``overwrite`` is False,
    and ValueError if input and output are the same file. Issues a
    UserWarning for data the output format cannot store.
    """

def set_num_threads(n: int) -> None:
    """Set the number of worker threads.

    0 restores the default, which is ``MDZ_NUM_THREADS`` if set and otherwise
    one thread per logical core.
    """
