"""mdz - MD trajectory compression with fast random access.

This package re-exports the C++ extension module ``mdz._core``, built with
nanobind. The codec and the ``.mdz`` file format are implemented only in the
C++ library.

Quick start
-----------
>>> import numpy as np, mdz
>>> positions = np.random.randn(100, 500, 3).astype(np.float32) * 0.1
>>> ct = mdz.compress(positions)              # -> compressed trajectory
>>> recon, boxes = mdz.decompress(ct)         # -> (positions, boxes|None)
>>> mdz.save("traj.mdz", ct)
>>> ct = mdz.load("traj.mdz")

The random-access reader behaves like a read-only sequence of frames. It also
gives access to the unit cell and every stored column, which is useful for
tools such as MDAnalysis:

>>> reader = mdz.open("traj.mdz")
>>> len(reader)                       # number of frames
>>> reader[42]                        # frame 42, shape (natoms, 3)
>>> reader[-1]                        # last frame
>>> reader[10:20]                     # frames 10 to 19, shape (10, natoms, 3)
>>> for frame in reader: ...          # iterate frames
>>> reader.cell(42)                   # (3, 3) box matrix (nm), or None
>>> reader.read_column("velocities")  # whole column as numpy
>>> reader.atom_names                 # list[str] or None
>>> with mdz.open("traj.mdz") as r: ...

``mdz.Writer`` writes a file incrementally, frame by frame. Use it for
trajectories that do not fit in memory or that come from a running
simulation:

>>> with mdz.Writer("traj.mdz") as w:
...     for x in frames:                  # each (natoms, 3) in nm
...         w.write(x)

When atoms are added or removed (deposition, evaporation, grand-canonical
runs), open the writer with ``variable_atoms=True`` and pass the atom ids of
each frame. The reader then returns each frame with its own atom count:

>>> with mdz.Writer("gcmc.mdz", variable_atoms=True) as w:
...     for x, ids in frames:             # (n, 3) and (n,), n changing
...         w.write(x, ids=ids)
>>> r = mdz.open("gcmc.mdz")
>>> r.frame_natoms                        # atoms per frame
>>> r[7], r.atom_ids(7)                   # (n7, 3) and (n7,)
>>> r[0:10]                               # a list of 10 arrays

Rigid molecules compress better when ``compress``, ``compress_batch`` or
``Writer`` is given ``rigid=True`` (bodies are detected) or ``bodies=`` (a
body index per atom).

Call ``mdz.set_num_threads(n)`` to limit the number of worker threads.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    # scikit-build-core takes the package version from CMakeLists.txt.
    __version__ = _version("mdz")
except PackageNotFoundError:  # pragma: no cover - running from a bare checkout
    __version__ = "0+unknown"

try:
    from mdz._core import (
        MdzReader,
        Writer,
        Trajectory,
        compress,
        compress_batch,
        decompress,
        save,
        load,
        open,
        info,
        convert,
        decode_boxes,
        set_num_threads,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mdz requires its compiled C++ extension (mdz._core), which was not "
        "found. Build it from the repository root with `pip install .` (or "
        "`pip install -e . --no-build-isolation`), or install a prebuilt wheel."
    ) from exc

__all__ = [
    # Core types
    "Trajectory",
    "MdzReader",
    "Writer",
    # Compress / decompress in memory
    "compress",
    "compress_batch",
    "decompress",
    # Files
    "save",
    "load",
    "open",
    "info",
    "convert",
    "decode_boxes",
    # Utilities
    "set_num_threads",
    "__version__",
]
