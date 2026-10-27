"""Generate a multi-column .mdz sample for OVITO integration testing.

Run with ovitos or any Python that has the current mdz wheel installed:

    /Applications/OvitoPro.app/Contents/MacOS/ovitos examples/make_multicolumn.py

Produces examples/traj_multicolumn.mdz, a small water-box-like trajectory
with positions, velocities, forces, charge, per-frame cell and time, and
atom names + types in metadata.
"""

from __future__ import annotations

import os
import numpy as np

from mdz._core import compress_batch


def main():
    rng = np.random.default_rng(0)
    nframes = 100
    # 20 water molecules: O, H, H triples
    nmol = 20
    natoms = nmol * 3

    # Positions: a slowly drifting cloud in a 2 nm box
    base = rng.uniform(0, 2, (natoms, 3)).astype(np.float32)
    drift = rng.normal(0, 0.002, (nframes, natoms, 3)).astype(np.float32).cumsum(axis=0)
    positions = (base[None, :, :] + drift).astype(np.float32)

    # Velocities in nm/ps, thermal scale ~0.3
    velocities = rng.normal(0, 0.3, (nframes, natoms, 3)).astype(np.float32)

    # Forces in kJ/mol/nm
    forces = rng.normal(0, 200.0, (nframes, natoms, 3)).astype(np.float32)

    # Partial charges (static per atom, but stored per-frame for simplicity)
    charge_per_atom = np.tile([-0.834, 0.417, 0.417], nmol).astype(np.float32)
    charge = np.broadcast_to(charge_per_atom, (nframes, natoms)).copy()

    # Cubic box varying slightly (A, gamma, B, beta, alpha, C)
    cell = np.tile([2.0, 90.0, 2.0, 90.0, 90.0, 2.0], (nframes, 1)).astype(np.float64)
    cell[:, 0] += np.linspace(0, 0.01, nframes)

    # Per-frame time and energy
    time = np.arange(nframes, dtype=np.float64) * 0.002
    energy = -1234.5 - np.arange(nframes, dtype=np.float64) * 0.1

    atom_names = []
    atom_types = []
    for _ in range(nmol):
        atom_names.extend(["O", "H", "H"])
        atom_types.extend([8, 1, 1])

    ct = compress_batch(
        positions,
        velocities=velocities,
        forces=forces,
        charge=charge,
        cell=cell,
        time=time,
        energy=energy,
        atom_names=atom_names,
        atom_types=atom_types,
        precision=0.001,
        column_precision={"forces": 1.0, "velocities": 0.01, "charge": 0.0001},
        gof_size=10,
        anchor_step=5,
        zstd_level=9,
    )

    out = os.path.join(os.path.dirname(__file__), "traj_multicolumn.mdz")
    ct.save(out)
    raw = positions.nbytes + velocities.nbytes + forces.nbytes + charge.nbytes
    print(f"{out}")
    print(f"  natoms={ct.natoms} nframes={ct.nframes} columns={ct.column_names}")
    print(f"  compressed={ct.compressed_size} bytes, raw≈{raw}, ratio={raw / ct.compressed_size:.1f}x")


if __name__ == "__main__":
    main()
