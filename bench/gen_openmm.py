"""Generate benchmark trajectories with OpenMM.

Each of two systems is equilibrated once and then run from the same state at
several save intervals. The save interval matters because it determines how
well mdz's temporal prediction works.

  water   TIP3P water, PME, 2 fs steps, HBond constraints, 300 K
  argon   liquid argon (Lennard-Jones, 0.34 nm / 0.996 kJ/mol), 4 fs, 90 K

Output files are <out>/<system>_<interval>.dcd, holding positions and box as
written by OpenMM's DCDReporter.

    python bench/gen_openmm.py OUT [--water-box 7.0] [--argon-atoms 50000] [--quick]

The CUDA platform is used if available, then OpenCL, then CPU.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u

# (save interval in steps, frames) for each system. Longer intervals use fewer
# frames to keep the run time reasonable on a CPU.
INTERVALS = [(1, 500), (10, 500), (100, 500), (1000, 250), (5000, 100)]


def water_system(box_nm: float):
    ff = app.ForceField("tip3p.xml")
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(ff, boxSize=mm.Vec3(box_nm, box_nm, box_nm) * u.nanometer, model="tip3p")
    system = ff.createSystem(
        modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * u.nanometer, constraints=app.HBonds
    )
    integ = mm.LangevinMiddleIntegrator(300 * u.kelvin, 1.0 / u.picosecond, 2.0 * u.femtosecond)
    return modeller.topology, modeller.positions, system, integ, 300


def argon_system(natoms: int):
    # Liquid argon near its triple point, 0.0213 atoms/A^3.
    density = 21.3  # atoms / nm^3
    box = (natoms / density) ** (1 / 3)
    n = int(np.ceil(natoms ** (1 / 3)))
    grid = (np.indices((n, n, n)).reshape(3, -1).T[:natoms] + 0.5) * (box / n)
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(box, 0, 0), mm.Vec3(0, box, 0), mm.Vec3(0, 0, box))
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(1.0)
    nb.setUseDispersionCorrection(True)
    for _ in range(natoms):
        system.addParticle(39.948)
        nb.addParticle(0.0, 0.34, 0.996)
    system.addForce(nb)
    top = app.Topology()
    chain = top.addChain()
    for _ in range(natoms):
        res = top.addResidue("AR", chain)
        top.addAtom("Ar", app.Element.getBySymbol("Ar"), res)
    top.setPeriodicBoxVectors(system.getDefaultPeriodicBoxVectors())
    integ = mm.LangevinMiddleIntegrator(90 * u.kelvin, 1.0 / u.picosecond, 4.0 * u.femtosecond)
    return top, grid * u.nanometer, system, integ, 90


def platform():
    names = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
    for name, props in [("CUDA", {"Precision": "mixed"}), ("OpenCL", {"Precision": "mixed"}), ("CPU", {})]:
        if name in names:
            return mm.Platform.getPlatformByName(name), props


def run(name, build, out: Path, equil_steps: int, intervals):
    top, pos, system, integ, temp = build()
    plat, props = platform()
    sim = app.Simulation(top, system, integ, plat, props)
    sim.context.setPositions(pos)
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(temp * u.kelvin)
    t = time.time()
    sim.step(equil_steps)
    rate = equil_steps / (time.time() - t)
    print(f"{name}: {system.getNumParticles()} atoms, {rate:.0f} steps/s on {plat.getName()}", flush=True)
    state = sim.context.getState(getPositions=True, getVelocities=True)
    for interval, frames in intervals:
        path = out / f"{name}_{interval}.dcd"
        sim.context.setState(state)
        sim.reporters.clear()
        sim.reporters.append(app.DCDReporter(str(path), interval, enforcePeriodicBox=False))
        t = time.time()
        sim.step(interval * frames)
        sim.reporters.clear()
        print(f"  {path.name}: {frames} frames every {interval} steps ({time.time() - t:.0f} s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--water-box", type=float, default=7.0, help="cube edge, nm")
    ap.add_argument("--argon-atoms", type=int, default=50000)
    ap.add_argument("--quick", action="store_true", help="tiny runs, to check the script")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    intervals = [(i, 5) for i, _ in INTERVALS[:2]] if a.quick else INTERVALS
    equil = 2000 if a.quick else 20000
    run("water", lambda: water_system(a.water_box), out, equil, intervals)
    run("argon", lambda: argon_system(a.argon_atoms), out, equil, intervals)


if __name__ == "__main__":
    main()
