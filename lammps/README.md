# dump mdz for LAMMPS

A LAMMPS dump style that writes `.mdz` trajectories:

- compressed: much smaller than `dump custom` text, with a position precision
  you choose (or lossless)
- any fields: it takes the same fields as `dump custom`, including computes,
  fixes and per-atom variables
- crash-tolerant: frames are compressed a group at a time and each finished
  group goes to the file at once. A run killed at its wall-time limit leaves a
  readable file holding every complete group.
- appendable: after a restart, `dump_modify ID append yes` continues the same
  file
- faithful: values are stored in the run's unit style, and the file records
  that style, the box (triclinic tilt and origin), the boundary flags, the
  timestep and the simulation time

There are two ways to add it to LAMMPS:

- **Compiled in, as the package MDZ** (recommended if you build LAMMPS):
  LAMMPS's own CMake build compiles dump mdz and the mdz library with its
  compiler and flags, so nothing can mismatch.
- **As a plugin**, for a LAMMPS binary you cannot rebuild. It needs the
  `PLUGIN` package in that binary, and must be built with the same
  compiler and C++ standard as it.

## Compiling it into LAMMPS

Put the package into your LAMMPS source tree, then configure LAMMPS with
`-D PKG_MDZ=ON` as well as your usual settings:

```sh
lammps/patch_lammps.sh /path/to/lammps        # --link to symlink, not copy
cmake -S /path/to/lammps/cmake -B build-lammps -D PKG_MDZ=ON ...
cmake --build build-lammps
```

The script copies `dump_mdz.cpp` and `dump_mdz.h` into `src/MDZ`, adds
`cmake/Modules/Packages/MDZ.cmake`, and registers the package in
`cmake/CMakeLists.txt`. Running it again updates the tree. LAMMPS then
builds the mdz library (only the library, as static, position-independent
code) from this checkout, whose path the script records; set
`-D MDZ_SOURCE_DIR=/path/to/mdz` to use another. zstd is downloaded and
compiled in at configure time; `-D MDZ_ZSTD=system` uses an installed zstd
instead. Only the CMake build of LAMMPS is supported.

It works with C++17 and C++20 builds, with MPI or serial, and with the
KOKKOS package: dump mdz runs on the host, and LAMMPS copies the atom data
there before it writes any dump.

## Building the plugin

You need CMake (3.20 or later), a C++17 compiler, and the CMake build
directory of the LAMMPS binary (or at least the source tree it was built
from):

```sh
cmake -S lammps -B build-plugin -D LAMMPS_BUILD_DIR=/path/to/lammps/build
cmake --build build-plugin
```

`LAMMPS_BUILD_DIR` supplies what the plugin must share with LAMMPS: the C++
compiler, the C++ standard (read from LAMMPS's compile flags), `BUILD_MPI`,
`LAMMPS_SIZES` and the LAMMPS source tree. Setting any of these to
something else is an error.

Without a LAMMPS build directory, give the source tree and match the rest by
hand:

```sh
cmake -S lammps -B build-plugin \
      -D LAMMPS_SOURCE_DIR=/path/to/lammps/src \
      -D CMAKE_CXX_COMPILER=...  # the compiler LAMMPS was built with
      -D CMAKE_CXX_STANDARD=20   # the standard LAMMPS was built with
      -D BUILD_MPI=on            # on with MPI, off for a serial LAMMPS
```

The LAMMPS configure summary prints the compiler and standard ("C++
Compiler", "C++ Standard"). LAMMPS builds with the KOKKOS package use C++20.

**Why the compiler matters, not just the standard.** Built as C++20 (with a
standard library that has it), LAMMPS's `fmt::` is `std::format`, whose code
is instantiated from templates in each binary that uses it, the plugin
included. Its layout was not ABI-stable before GCC 15, so the copies in a
LAMMPS built with one compiler or libstdc++ version and in a plugin built
with another can disagree. A plugin that called LAMMPS's copies crashed formatting a
message of its own (seen with LAMMPS built by hipcc on GCC 15's libstdc++
and the plugin by GCC 14). A plugin built as C++17 against such
a LAMMPS fails to load instead, with an `undefined symbol` error naming
`LAMMPS_NS::Error`.

The plugin guards against this as far as it can: it is linked with a
version script that exports only `lammpsplugin_init`, so it always runs its
own instantiations rather than binding to LAMMPS's. Messages it hands to
LAMMPS (errors, warnings) still cross between the two, so build it with
LAMMPS's compiler, which `LAMMPS_BUILD_DIR` does for you.

This builds the mdz library from this repository (C++17 or later; its one
dependency, zstd, is downloaded and compiled in) and links it statically
into `build-plugin/mdzplugin.so`. To use an installed zstd instead, add
`-D MDZ_ZSTD=system`.

Then load it in the input script, before the first `dump mdz`:

```
plugin load /path/to/mdzplugin.so
```

## Threads and checking the build

By default dump mdz restricts mdz to a single thread, the thread of the
rank that writes the file. `-D MDZ_PARALLEL=on` (for the package or the
plugin) lets mdz start its own worker threads (one per core, or
`MDZ_NUM_THREADS`). mdz parallelizes across groups of frames, and dump mdz
compresses one group at a time, so this option currently has no effect on
speed.

To check the build, run the smoke test against your LAMMPS binary, with the
plugin or with `builtin` for the package. It needs the mdz command-line
tool, built from the repository root:

```sh
cmake -S . -B build && cmake --build build --target mdz_cli
lammps/tests/run.sh /path/to/lmp build-plugin/mdzplugin.so build/cli/mdz
lammps/tests/run.sh /path/to/lmp builtin build/cli/mdz
```

It also checks a run whose atoms change (`in.variable`, with `fix deposit`
and `delete_atoms`) and, if LAMMPS has the MOLECULE and RIGID packages,
rigid molecules (`in.rigid`). Set `MPIRUN`, for example `MPIRUN="mpirun -np
2"`, to run LAMMPS in parallel.

## Usage

```
compute pe all pe/atom
dump traj all mdz 100 traj.mdz id type x y z ix iy iz vx vy vz c_pe
dump_modify traj precision 0.001
```

The fields are those of `dump custom`. With none, they are `id type x y z ix
iy iz`. Positions are required, as `x y z` or as `xu yu zu` for unwrapped
coordinates, and `id` is added if it is not listed.

- `x y z`, `vx vy vz` and `fx fy fz` are stored to a precision (see
  `precision` below).
- `ix iy iz` and `type` are stored exactly.
- Every other field (`q`, `mol`, `mass`, `c_ID`, `c_ID[2]`, `f_ID`,
  `v_name`, `i_name`, `d_name`, ...) is stored under its own name, exactly
  as a float or, for integer fields, as an int. `precision/column` stores a
  float field to a precision instead.

Text fields (`element`, type labels) and scaled coordinates are not
supported.

The dump always sorts atoms by ID. By default every frame holds the same
atoms with the same types, and atom IDs and types are stored once per file;
with `variable yes` the atoms may change from frame to frame (see below).

| `dump_modify` keyword | value | default |
|---|---|---|
| `precision` | position step, in length units; the maximum error is up to 4 steps | 0 (default: a maximum error of 0.005 Å) |
| `velocity/precision`, `force/precision` | step for velocities and forces | 0 (default) |
| `precision/column` | field and step: store a general float field to that step, in its own units, so the error is at most half a step | exact |
| `lossless` | `yes`: store every value exactly as a float | `no` |
| `gof` | frames per compressed group. This is the granularity of random access and the most a crash can lose. | 20 |
| `variable` | `yes`: let the group's atoms change from frame to frame (atoms lost, deleted, deposited, inserted) | `no` |
| `rigid` | `yes`: model rigid bodies, detected from the first frames; `mol`: each molecule (molecule ID > 0) is a body | `no` |
| `append` | `yes`: continue an existing file (standard keyword) | `no` |

The standard `every`, `first`, `delay`, `skip` and `pbc` keywords work as
usual, as do `dump custom`'s `region` and `thresh`. `flush` is not supported, because the file is written a group at a
time.

Set the content keywords before the first `run`; after the file is open they
are fixed.

## Atoms that come and go

With `dump_modify ID variable yes` each frame stores the atoms the group has
at that step, with their IDs and types, so `fix deposit`, `fix evaporate`,
`fix gcmc`, `delete_atoms` between runs, or atoms lost through a
non-periodic boundary (`thermo_modify lost ignore`) are all recorded as
they happen. Readers return each frame with its own atom count. For a fixed
set of atoms, this mode makes the file less than 2% larger. It is not the
default.

Without it, a frame whose atoms differ from the first frame's (a different
count, or different IDs) cannot be stored. Such frames are skipped with a
warning and the run continues, so that a long run that loses an atom is not
stopped by its dump. The warning suggests `variable yes`. Later frames that
match the first frame again are written as usual.

```
dump traj all mdz 100 traj.mdz
dump_modify traj variable yes
```

`variable` must be set before the first run, and an appended file must
have been written with the same setting.

## Rigid bodies

`dump_modify ID rigid yes` also encodes positions as rigid bodies, with a
centre and an orientation per body and frame plus small residuals. For each
group of frames, the smaller of the two encodings is kept. The error bound is
unchanged. Bodies are detected in the first group as atoms within 1.6 Å of
each other whose distances stay within 0.02 Å.

`rigid mol` instead treats each molecule as a body. It needs the `mol` field
in the dump command, and atoms with molecule ID 0 belong to no body. This suits
`fix rigid` and SHAKE-constrained molecules.

Rigid-body encoding helps most for bodies with many atoms (rings, rigid
fragments) and roughly breaks even for water. It is ignored with `lossless`
and cannot be combined with `variable yes`.

## Restarting after a crash

If a run dies, the file keeps every group that was complete. With the default
`gof 20` and a dump every 100 steps, that is everything up to the last
multiple of 2000 steps. Restart from a restart file written at or before that
point and append:

```
read_restart run.restart
...
dump traj all mdz 100 traj.mdz
dump_modify traj append yes     # plus the same settings as before
```

The settings must match the ones the file was written with. To have no gap or
overlap, restart from the step where the file ends; `mdz info traj.mdz`
reports the frame count.

## Reading the files

- **Command line:** `mdz info traj.mdz`, or `mdz decompress traj.mdz -o
  traj.lammpstrj` to get a dump in the original units. Other formats are
  converted, for example to `.xtc`.
- **Python:** `mdz.open("traj.mdz")`.
- **C:** `include/mdz/mdz.h` (C++: `include/mdz/mdz.hpp`).

## Limitations

- **Single precision:** per-atom values are stored as floats. `lossless`
  means exact to float precision (about 7 significant digits). The box,
  step and time keep full double precision.
- **Fixed atom set by default:** without `variable yes`, the group's atom
  count, IDs and types must stay the same for the life of the file; frames
  that differ are skipped (with a warning).
- **Restricted triclinic box:** the box is stored in LAMMPS's restricted
  triclinic form, as LAMMPS holds it internally. A general triclinic box is
  stored in that restricted frame.
- **Plugin: not in `plugin list`:** LAMMPS's plugin loader does not handle dump
  styles, so the plugin adds `dump mdz` to the style table itself. It works
  with current and older LAMMPS versions, but it does not appear in `plugin
  list` and cannot be unloaded.
