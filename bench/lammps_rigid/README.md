# Rigid clusters in a Lennard-Jones solvent

Inputs for the rigid-body benchmark in docs/BENCHMARKS.md. They require a
LAMMPS build with the MOLECULE and RIGID packages. The OPENMP package is
optional but makes the runs much faster.

```bash
lmp -in in.rigid                                        # equilibrate, writes eq.restart
for e in 1 10 100 1000; do lmp -sf omp -pk omp 16 -var every $e -in in.dump; done
bench_files --lammps-units lj --bodies-from mol clusters_*.lammpstrj
```
