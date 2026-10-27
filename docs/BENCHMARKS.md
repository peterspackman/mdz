# Benchmarks

This page compares mdz with XTC, and with zstd applied to the raw floats, on
trajectories generated with OpenMM and LAMMPS. It also reports compression,
decompression and random-access speeds.

## Setup

- **Trajectories** (`bench/gen_openmm.py`, OpenMM 8.6 on an NVIDIA GeForce RTX 4080 SUPER):
  - **water**: TIP3P water, 33840 atoms, PME, 2 fs steps, rigid water
    (constrained), 300 K.
  - **argon**: liquid argon, 50000 atoms (Lennard-Jones), 4 fs steps, 90 K.

  Each system is equilibrated and then run from the same state, saving every
  1, 10, 100, 1000 and 5000 steps. This spans frames that are nearly identical
  (every step) to frames 10-20 ps apart that are effectively uncorrelated,
  which is the range that determines how much temporal prediction can gain.
  The two sample DCDs in `examples/` are also included.
- **Machine**: AMD Ryzen 7 7800X3D 8-Core Processor, 16 threads, 15 GB, Linux (WSL2), GCC 13, Release build.
- **Measurements** (`bench/bench_files`, run by `bench/benchmark.py`).
  Frames are loaded once. Every codec runs in memory, and compressed files are
  read back through mmap from the page cache, so the speeds measure the codecs
  rather than disk I/O. Each figure is the best of three runs. Ratios are the
  size of the raw float32 positions divided by the compressed size, so box and
  frame metadata count only on the compressed side.

To reproduce the results:

```bash
python bench/gen_openmm.py traj          # ~10 min on a GPU
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
python bench/benchmark.py --build build traj/*.dcd examples/*.dcd > results.md
```

## Summary

- **Size.** At the same error, mdz files are never larger than XTC, and they
  are much smaller when frames are saved often. At XTC's error (0.005 Å),
  mdz with default settings matches XTC once frames are a few picoseconds
  apart, where neither format can predict one frame from the previous one.
  When frames are femtoseconds apart, mdz files are 2.5-3.4x smaller (argon
  saved every 4 fs: 8.5x against 2.5x).
- **Rigid bodies** help most for large bodies. For 13-atom rigid clusters in
  a LAMMPS run, `--rigid` makes files 23-35% smaller (a 29-53% higher
  ratio), with larger gains when frames are further apart. Compression is up
  to 1.8x slower (see the table below). For three-atom water, files are
  11-15% smaller at 2-200 fs intervals but only 3-6% smaller at 2-10 ps, and
  compression is about 3x slower. Argon has no rigid bodies, so `--rigid`
  changes nothing.
- **Lossless** compression gives 1.3-3x, against 1.1x for zstd on the same
  floats.
- **Speed.** Compression runs at 200-350 MB/s on one thread and
  0.6-1.4 GB/s on 16. Decompression of temporally coded groups runs at
  2 GB/s on one thread and 11-12 GB/s on 16. Intra-coded groups decompress
  at about 400-600 MB/s on one thread and 2-4 GB/s on 16. mdz's own XTC
  implementation runs at 320-560 MB/s in either direction on one thread.
- **Random access.** Reading one frame from a freshly opened file takes
  1.5-7 ms for these 34-50k-atom systems. This decodes a group of up to 20
  frames, or only the requested frame for intra-coded groups. Reading frame
  by frame takes 0.3-1.6 ms per frame.
- **Command line.** The CLI compresses a 300 MB DCD in 0.42 s (713 MB/s,
  including disk I/O and DCD parsing) and decompresses it in 0.38 s.

## Rigid bodies in a LAMMPS run of rigid clusters

The inputs are in `bench/lammps_rigid/`. The system is 717 rigid 13-atom
icosahedral clusters (`fix rigid/small`) in 10000 Lennard-Jones solvent atoms,
19321 atoms in total. It was run with LAMMPS (OpenMP) and dumped with
`dump custom` (unwrapped coordinates and molecule ids) every 1-1000 steps, 300
frames per dump. The dumps are read in LJ units (σ taken as 1 nm), with the
bodies taken from the molecule ids (`bench_files --lammps-units lj
--bodies-from mol`). `bench/lammps_rigid/README.md` gives the commands to
reproduce these runs.

| saved every | XTC | mdz | mdz `--rigid` | rigid / plain size | compress MB/s, plain / rigid (1 thread) | random frame, plain / rigid (ms) |
|---|---|---|---|---|---|---|
| 1 step | 1.9x | 9.1x | 11.7x | 0.77 | 287 / 161 | 1.6 / 1.6 |
| 10 steps | 1.9x | 3.2x | 4.2x | 0.75 | 213 / 144 | 2.3 / 2.1 |
| 100 steps | 1.9x | 2.2x | 3.3x | 0.67 | 244 / 138 | 2.4 / 1.7 |
| 1000 steps | 1.8x | 2.1x | 3.3x | 0.65 | 257 / 142 | 2.2 / 1.8 |

All results are at an error of at most 0.005 Å (0.0005 σ). Automatic body
detection finds no bodies in this system. Detection looks for atoms within
0.16 nm of each other, a threshold chosen for atomistic data, and the bonds
here are 1 σ long.

## Compression ratio

Ratios are the size of the raw float32 positions divided by the compressed
size. The table compares mdz with default settings (error at most 0.005 Å,
codec chosen per group) against XTC at the standard GROMACS precision
(0.001 nm, also at most 0.005 Å error), lossless mdz, and zstd on the raw
floats.

| trajectory | raw MB | XTC | mdz | mdz / XTC size | mdz codec choice | mdz, rigid bodies | lossless mdz | zstd (raw floats) |
|---|---|---|---|---|---|---|---|---|
| argon (50000 atoms, 500 frames, saved every 0.004 ps) | 300 | 2.5x | 8.5x | 0.29 | temporal | 8.5x | 3.01x | 1.12x |
| argon (50000 atoms, 500 frames, saved every 0.04 ps) | 300 | 2.5x | 6.7x | 0.36 | temporal | 6.7x | 2.05x | 1.12x |
| argon (50000 atoms, 500 frames, saved every 0.4 ps) | 300 | 2.3x | 2.5x | 0.93 | mixed | 2.5x | 1.59x | 1.12x |
| argon (50000 atoms, 250 frames, saved every 4 ps) | 150 | 2.2x | 2.2x | 1.00 | intra | 2.2x | 1.50x | 1.12x |
| argon (50000 atoms, 100 frames, saved every 20 ps) | 60 | 2.1x | 2.1x | 1.00 | intra | 2.1x | 1.44x | 1.11x |
| every-frame (769 atoms, 120 frames) | 1 | 3.2x | 9.3x | 0.35 | temporal | 9.3x | 2.20x | 1.13x |
| traj (4096 atoms, 25 frames) | 1 | 3.4x | 3.4x | 1.00 | intra | 3.4x | 1.35x | 1.12x |
| water (33840 atoms, 500 frames, saved every 0.002 ps) | 203 | 3.3x | 8.2x | 0.40 | temporal | 9.6x | 2.29x | 1.08x |
| water (33840 atoms, 500 frames, saved every 0.02 ps) | 203 | 3.3x | 3.9x | 0.84 | temporal | 4.4x | 1.55x | 1.08x |
| water (33840 atoms, 500 frames, saved every 0.2 ps) | 203 | 3.3x | 3.3x | 1.00 | intra | 3.8x | 1.46x | 1.08x |
| water (33840 atoms, 250 frames, saved every 2 ps) | 102 | 3.2x | 3.2x | 1.00 | intra | 3.4x | 1.37x | 1.08x |
| water (33840 atoms, 100 frames, saved every 10 ps) | 41 | 3.2x | 3.2x | 1.00 | intra | 3.3x | 1.32x | 1.08x |

## Size against error budget

Compression ratio, and the maximum error measured, at three `--max-error`
settings.

| trajectory | 0.001 Å | 0.005 Å (default) | 0.05 Å | XTC |
|---|---|---|---|---|
| argon (50000 atoms, 500 frames, saved every 0.004 ps) | 7.8x (0.0010 Å) | 8.5x (0.0050 Å) | 17.4x (0.0500 Å) | 2.5x (0.0050 Å) |
| argon (50000 atoms, 500 frames, saved every 0.04 ps) | 4.4x (0.0010 Å) | 6.7x (0.0050 Å) | 8.7x (0.0500 Å) | 2.5x (0.0050 Å) |
| argon (50000 atoms, 500 frames, saved every 0.4 ps) | 2.1x (0.0010 Å) | 2.5x (0.0050 Å) | 6.3x (0.0500 Å) | 2.3x (0.0050 Å) |
| argon (50000 atoms, 250 frames, saved every 4 ps) | 1.9x (0.0010 Å) | 2.2x (0.0050 Å) | 4.9x (0.0500 Å) | 2.2x (0.0050 Å) |
| argon (50000 atoms, 100 frames, saved every 20 ps) | 1.9x (0.0010 Å) | 2.1x (0.0050 Å) | 3.1x (0.0500 Å) | 2.1x (0.0050 Å) |
| every-frame (769 atoms, 120 frames) | 7.7x (0.0010 Å) | 9.3x (0.0050 Å) | 18.1x (0.0500 Å) | 3.2x (0.0050 Å) |
| traj (4096 atoms, 25 frames) | 2.7x (0.0010 Å) | 3.4x (0.0050 Å) | 5.2x (0.0500 Å) | 3.4x (0.0050 Å) |
| water (33840 atoms, 500 frames, saved every 0.002 ps) | 7.4x (0.0010 Å) | 8.2x (0.0050 Å) | 11.5x (0.0500 Å) | 3.3x (0.0050 Å) |
| water (33840 atoms, 500 frames, saved every 0.02 ps) | 2.6x (0.0010 Å) | 3.9x (0.0050 Å) | 7.9x (0.0500 Å) | 3.3x (0.0050 Å) |
| water (33840 atoms, 500 frames, saved every 0.2 ps) | 2.6x (0.0010 Å) | 3.3x (0.0050 Å) | 6.4x (0.0500 Å) | 3.3x (0.0050 Å) |
| water (33840 atoms, 250 frames, saved every 2 ps) | 2.6x (0.0010 Å) | 3.2x (0.0050 Å) | 4.6x (0.0500 Å) | 3.2x (0.0050 Å) |
| water (33840 atoms, 100 frames, saved every 10 ps) | 2.6x (0.0010 Å) | 3.2x (0.0050 Å) | 4.4x (0.0500 Å) | 3.2x (0.0050 Å) |

## Speed

Speeds are measured in memory and cover the codec only. MB/s are of raw
float32 positions. "Random frame" is the time to read one frame from a freshly
opened file. "Frame-by-frame" reads every frame in order, one call per frame,
as an analysis loop would.

| trajectory | method | compress MB/s (1 / 16 threads) | decompress MB/s (1 / 16 threads) | random frame (ms) | frame-by-frame (ms/frame) |
|---|---|---|---|---|---|
| argon (50000 atoms, 500 frames, saved every 0.004 ps) | mdz | 283 / 1227 | 1954 / 11149 | 4.45 | 0.467 |
| argon (50000 atoms, 500 frames, saved every 0.004 ps) | mdz-temporal | 394 | 1949 | 4.44 | 0.449 |
| argon (50000 atoms, 500 frames, saved every 0.004 ps) | mdz-intra | 275 | 445 | 2.53 | 1.439 |
| argon (50000 atoms, 500 frames, saved every 0.004 ps) | mdz-rigid | 280 | 1964 | 4.48 | 0.460 |
| argon (50000 atoms, 500 frames, saved every 0.004 ps) | mdz-lossless | 530 / 1105 | 2478 / 8350 | 4.02 | 0.433 |
| argon (50000 atoms, 500 frames, saved every 0.004 ps) | xtc | 331 | 421 | - | - |
| argon (50000 atoms, 500 frames, saved every 0.04 ps) | mdz | 259 / 1140 | 1993 / 10841 | 4.35 | 0.451 |
| argon (50000 atoms, 500 frames, saved every 0.04 ps) | mdz-temporal | 351 | 1990 | 4.39 | 0.452 |
| argon (50000 atoms, 500 frames, saved every 0.04 ps) | mdz-intra | 263 | 432 | 2.58 | 1.481 |
| argon (50000 atoms, 500 frames, saved every 0.04 ps) | mdz-rigid | 262 | 2025 | 4.42 | 0.451 |
| argon (50000 atoms, 500 frames, saved every 0.04 ps) | mdz-lossless | 600 / 901 | 2778 / 7659 | 3.57 | 0.371 |
| argon (50000 atoms, 500 frames, saved every 0.04 ps) | xtc | 324 | 405 | - | - |
| argon (50000 atoms, 500 frames, saved every 0.4 ps) | mdz | 248 / 885 | 916 / 6724 | 5.63 | 0.829 |
| argon (50000 atoms, 500 frames, saved every 0.4 ps) | mdz-temporal | 386 | 1346 | 6.26 | 0.638 |
| argon (50000 atoms, 500 frames, saved every 0.4 ps) | mdz-intra | 230 | 392 | 2.83 | 1.616 |
| argon (50000 atoms, 500 frames, saved every 0.4 ps) | mdz-rigid | 247 | 917 | 5.66 | 0.832 |
| argon (50000 atoms, 500 frames, saved every 0.4 ps) | mdz-lossless | 384 / 727 | 2972 / 7661 | 3.37 | 0.352 |
| argon (50000 atoms, 500 frames, saved every 0.4 ps) | xtc | 323 | 375 | - | - |
| argon (50000 atoms, 250 frames, saved every 4 ps) | mdz | 200 / 746 | 410 / 3034 | 3.73 | 1.589 |
| argon (50000 atoms, 250 frames, saved every 4 ps) | mdz-temporal | 383 | 1502 | 5.52 | 0.584 |
| argon (50000 atoms, 250 frames, saved every 4 ps) | mdz-intra | 256 | 410 | 3.73 | 1.572 |
| argon (50000 atoms, 250 frames, saved every 4 ps) | mdz-rigid | 201 | 413 | 3.70 | 1.539 |
| argon (50000 atoms, 250 frames, saved every 4 ps) | mdz-lossless | 395 / 701 | 2971 / 7754 | 3.22 | 0.347 |
| argon (50000 atoms, 250 frames, saved every 4 ps) | xtc | 318 | 384 | - | - |
| argon (50000 atoms, 100 frames, saved every 20 ps) | mdz | 199 / 593 | 389 / 1892 | 6.98 | 1.648 |
| argon (50000 atoms, 100 frames, saved every 20 ps) | mdz-temporal | 348 | 1509 | 5.25 | 0.586 |
| argon (50000 atoms, 100 frames, saved every 20 ps) | mdz-intra | 249 | 387 | 6.89 | 1.644 |
| argon (50000 atoms, 100 frames, saved every 20 ps) | mdz-rigid | 194 | 390 | 6.85 | 1.636 |
| argon (50000 atoms, 100 frames, saved every 20 ps) | mdz-lossless | 466 / 703 | 3216 / 8450 | 2.72 | 0.318 |
| argon (50000 atoms, 100 frames, saved every 20 ps) | xtc | 343 | 374 | - | - |
| every-frame (769 atoms, 120 frames) | mdz | 292 / 788 | 2066 / 3691 | 0.06 | 0.007 |
| every-frame (769 atoms, 120 frames) | mdz-temporal | 422 | 2006 | 0.06 | 0.007 |
| every-frame (769 atoms, 120 frames) | mdz-intra | 294 | 554 | 0.06 | 0.018 |
| every-frame (769 atoms, 120 frames) | mdz-rigid | 274 | 2013 | 0.07 | 0.007 |
| every-frame (769 atoms, 120 frames) | mdz-lossless | 518 / 817 | 1963 / 3805 | 0.07 | 0.008 |
| every-frame (769 atoms, 120 frames) | xtc | 433 | 518 | - | - |
| traj (4096 atoms, 25 frames) | mdz | 236 / 326 | 584 / 739 | 0.28 | 0.091 |
| traj (4096 atoms, 25 frames) | mdz-temporal | 420 | 1553 | 0.17 | 0.047 |
| traj (4096 atoms, 25 frames) | mdz-intra | 354 | 590 | 0.26 | 0.091 |
| traj (4096 atoms, 25 frames) | mdz-rigid | 201 | 612 | 0.26 | 0.091 |
| traj (4096 atoms, 25 frames) | mdz-lossless | 396 / 446 | 3348 / 3951 | 0.11 | 0.027 |
| traj (4096 atoms, 25 frames) | xtc | 426 | 570 | - | - |
| water (33840 atoms, 500 frames, saved every 0.002 ps) | mdz | 294 / 1367 | 1942 / 11399 | 3.06 | 0.321 |
| water (33840 atoms, 500 frames, saved every 0.002 ps) | mdz-temporal | 403 | 1954 | 3.03 | 0.313 |
| water (33840 atoms, 500 frames, saved every 0.002 ps) | mdz-intra | 342 | 528 | 1.43 | 0.818 |
| water (33840 atoms, 500 frames, saved every 0.002 ps) | mdz-rigid | 118 | 1054 | 3.84 | 0.494 |
| water (33840 atoms, 500 frames, saved every 0.002 ps) | mdz-lossless | 447 / 939 | 2780 / 8359 | 2.35 | 0.248 |
| water (33840 atoms, 500 frames, saved every 0.002 ps) | xtc | 492 | 519 | - | - |
| water (33840 atoms, 500 frames, saved every 0.02 ps) | mdz | 345 / 1222 | 1802 / 11932 | 2.95 | 0.306 |
| water (33840 atoms, 500 frames, saved every 0.02 ps) | mdz-temporal | 474 | 1816 | 2.98 | 0.307 |
| water (33840 atoms, 500 frames, saved every 0.02 ps) | mdz-intra | 359 | 581 | 1.30 | 0.742 |
| water (33840 atoms, 500 frames, saved every 0.02 ps) | mdz-rigid | 108 | 685 | 3.73 | 0.681 |
| water (33840 atoms, 500 frames, saved every 0.02 ps) | mdz-lossless | 369 / 723 | 2842 / 8367 | 2.32 | 0.248 |
| water (33840 atoms, 500 frames, saved every 0.02 ps) | xtc | 456 | 556 | - | - |
| water (33840 atoms, 500 frames, saved every 0.2 ps) | mdz | 255 / 1012 | 528 / 4155 | 1.46 | 0.824 |
| water (33840 atoms, 500 frames, saved every 0.2 ps) | mdz-temporal | 380 | 1282 | 4.32 | 0.436 |
| water (33840 atoms, 500 frames, saved every 0.2 ps) | mdz-intra | 332 | 529 | 1.43 | 0.823 |
| water (33840 atoms, 500 frames, saved every 0.2 ps) | mdz-rigid | 95 | 646 | 4.18 | 0.739 |
| water (33840 atoms, 500 frames, saved every 0.2 ps) | mdz-lossless | 393 / 699 | 3108 / 8937 | 2.10 | 0.217 |
| water (33840 atoms, 500 frames, saved every 0.2 ps) | xtc | 463 | 505 | - | - |
| water (33840 atoms, 250 frames, saved every 2 ps) | mdz | 256 / 773 | 535 / 4278 | 1.94 | 0.812 |
| water (33840 atoms, 250 frames, saved every 2 ps) | mdz-temporal | 392 | 1479 | 3.71 | 0.394 |
| water (33840 atoms, 250 frames, saved every 2 ps) | mdz-intra | 332 | 526 | 1.93 | 0.820 |
| water (33840 atoms, 250 frames, saved every 2 ps) | mdz-rigid | 93 | 628 | 4.68 | 0.778 |
| water (33840 atoms, 250 frames, saved every 2 ps) | mdz-lossless | 508 / 741 | 2943 / 8528 | 2.26 | 0.239 |
| water (33840 atoms, 250 frames, saved every 2 ps) | xtc | 465 | 509 | - | - |
| water (33840 atoms, 100 frames, saved every 10 ps) | mdz | 256 / 727 | 539 / 2547 | 3.37 | 0.808 |
| water (33840 atoms, 100 frames, saved every 10 ps) | mdz-temporal | 378 | 1522 | 3.51 | 0.387 |
| water (33840 atoms, 100 frames, saved every 10 ps) | mdz-intra | 343 | 537 | 3.39 | 0.806 |
| water (33840 atoms, 100 frames, saved every 10 ps) | mdz-rigid | 93 | 636 | 5.43 | 0.783 |
| water (33840 atoms, 100 frames, saved every 10 ps) | mdz-lossless | 465 / 660 | 2730 / 8335 | 2.26 | 0.262 |
| water (33840 atoms, 100 frames, saved every 10 ps) | xtc | 456 | 505 | - | - |

## Command line, end to end

The input is `argon_1.dcd`, a 300 MB DCD file. It compresses to 35.4 MB as
`.mdz` and 119.9 MB as `.xtc`.

| command | time (s) | MB/s of DCD |
|---|---|---|
| `mdz compress` | 0.421 | 713 |
| `mdz decompress → .dcd` | 0.384 | 782 |
| `mdz convert .dcd → .xtc` | 1.082 | 277 |
| `mdz convert .xtc → .dcd` | 1.335 | 225 |
| `mdz extract 10 frames` | 0.052 | - |
