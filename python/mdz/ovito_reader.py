"""OVITO Pro file reader for ``.mdz`` molecular dynamics trajectories.

Install it into the Python interpreter bundled with OVITO Pro::

    /Applications/OvitoPro.app/Contents/MacOS/ovitos -m pip install --user mdz

The ``OVITO.FileReader`` entry point in ``pyproject.toml`` registers this
reader with OVITO Pro's format auto-detection. ``.mdz`` files can then be
opened from the GUI's *File > Load* dialog, or with
:func:`ovito.io.import_file` without an explicit ``input_format``.

Frames are decoded on demand, so opening a large trajectory is cheap. The
reader maps positions, velocities, forces and charge to particle properties,
atom ids to ``Particle Identifier``, and atom names or types to ``Particle
Type``. Files whose atom count varies between frames are supported. The
per-frame box becomes the simulation cell, and other per-frame values
(``time``, ``step``, ``energy``, ...) become global attributes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import traits.api
from ovito.data import DataCollection
from ovito.io import FileReaderInterface

# mdz stores lengths in nm, as GROMACS does. OVITO's defaults (element radii,
# cell rendering, chemistry-aware modifiers) assume Angstrom, so lengths are
# converted by default. Pass input_format=MdzFileReader(scale=1.0) to keep nm.
NM_TO_ANGSTROM = 10.0

MDZ_MAGIC = b"MDZ\x04"

# Per-atom columns mapped to OVITO properties, and whether they are lengths
# (scaled with `scale`).
_PARTICLE_COLUMNS = {
    "velocities": ("Velocity", True),
    "forces": ("Force", False),
    "charge": ("Charge", False),
}


class MdzFileReader(FileReaderInterface):
    """FileReaderInterface subclass for ``.mdz`` trajectories.

    Parameters exposed to :func:`ovito.io.import_file`:

    :param scale: Factor applied to positions, velocities and cell vectors.
        Defaults to ``10.0``, which converts nm to Angstrom. Pass
        ``scale=1.0`` to keep nm.
    """

    scale = traits.api.Float(NM_TO_ANGSTROM)

    def __init__(self, **traits_kwargs):
        super().__init__(**traits_kwargs)
        self._path: str | None = None
        self._reader = None
        self._per_frame: list[str] = []
        self._per_atom: list[str] = []
        self._columns: dict[str, np.ndarray] = {}  # decoded on first use
        self._offsets: np.ndarray | None = None  # frame starts, varying atoms only

    @staticmethod
    def detect(filename: str) -> bool:
        try:
            with open(filename, "rb") as f:
                return f.read(4) == MDZ_MAGIC
        except OSError:
            return False

    def _ensure_open(self, filename: str) -> None:
        if self._path == filename and self._reader is not None:
            return
        try:
            import mdz
        except ImportError:
            raise RuntimeError("the mdz reader needs mdz's compiled extension (mdz._core)")
        self._reader = mdz.open(filename)
        self._path = filename
        self._columns = {}
        descs = mdz.load(filename).columns()
        self._per_frame = [d["name"] for d in descs if d["per_frame"] and d["width"] == 1]
        self._per_atom = [d["name"] for d in descs if not d["per_frame"]]
        self._offsets = None
        if self._reader.variable_atoms:
            self._offsets = np.concatenate(([0], np.cumsum(self._reader.frame_natoms)))

    def _column(self, name: str) -> np.ndarray:
        if name not in self._columns:
            self._columns[name] = np.asarray(self._reader.read_column(name))
        return self._columns[name]

    def _atom_values(self, name: str, frame: int) -> np.ndarray:
        """One frame of a per-atom column."""
        col = self._column(name)
        if self._offsets is None:
            return col[frame]
        return col[self._offsets[frame] : self._offsets[frame + 1]]

    # --- OVITO interface ---

    def scan(self, filename: str, register_frame: Callable[..., None]) -> None:
        self._ensure_open(filename)
        for i in range(self._reader.nframes):
            register_frame(frame_info=i, label=i)

    def parse(
        self,
        data: DataCollection,
        *,
        filename: str,
        frame_index: int,
        frame_info: Any = None,
        is_new_file: bool = False,
        **kwargs: Any,
    ) -> None:
        self._ensure_open(filename)
        frame = frame_info if isinstance(frame_info, int) else frame_index
        r = self._reader
        s = float(self.scale)

        pos = np.asarray(r.read_frame(frame), dtype=np.float32) * s
        n = pos.shape[0]
        particles = data.create_particles(count=n)
        particles.create_property("Position", data=np.ascontiguousarray(pos))

        for col, (prop, is_length) in _PARTICLE_COLUMNS.items():
            if col in self._per_atom:
                v = self._atom_values(col, frame).astype(np.float32)
                particles.create_property(prop, data=np.ascontiguousarray(v * s if is_length else v))

        # OVITO tracks atoms across frames by Particle Identifier.
        if r.variable_atoms or "atom_ids" in r.metadata():
            particles.create_property("Particle Identifier", data=np.asarray(r.atom_ids(frame), dtype=np.int64))

        names = r.atom_names
        types = r.atom_types
        if names and len(names) == n:
            ptype = particles.create_property("Particle Type")
            with ptype as t:
                for i, sym in enumerate(names):
                    t[i] = ptype.add_type_name(sym, particles).id
        elif types and len(types) == n:
            ptype = particles.create_property("Particle Type", data=np.asarray(types, dtype=np.int32))
            # Register each type id so that the GUI lists the types.
            for tid in sorted(set(int(v) for v in types)):
                ptype.add_type_id(tid, particles)

        box = r.cell(frame)
        if box is not None:
            # mdz stores cell vectors as rows; OVITO expects columns.
            cell = data.create_cell(matrix=np.asarray(box, dtype=np.float64).T * s, pbc=(True, True, True))
            cell[:, 3] = 0.0

        for name in self._per_frame:
            col = self._column(name)
            if frame < len(col):
                data.attributes[name] = col[frame].item()
        data.attributes["Frame"] = frame
