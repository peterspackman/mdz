# Using mdz

## Precision

`--max-error` is the largest coordinate error you will accept, in Å. The
default is 0.005 Å, which is what XTC gives at its usual precision.

Internally mdz works with a quantization step. With adaptive precision (the
default) every atom gets a step of 1, 2, 4 or 8 times the base step depending
on how much it moves, so the worst-case error is 4x the base step. The CLI
does this conversion for you: 0.005 Å max error means a 0.00125 Å base step.

```bash
mdz compress traj.dcd -o out.mdz --max-error 0.001          # tighter
mdz compress traj.dcd -o out.mdz --max-error 0.05           # looser
mdz compress traj.dcd -o out.mdz --precision 0.01 --no-adaptive   # XTC's exact grid
mdz compress traj.extxyz -o out.mdz --precision-col velocities=0.01
```

`--precision` sets the base step directly, in Å. `--precision-col` sets the
step for one column, in that column's own unit (nm/ps for velocities,
kJ/mol/nm for forces).

In Python and C++, `precision` is the base step in nm. The default, 0.000125
nm, is the same setting as the CLI default.

## Units

mdz stores values in nm and ps. DCD, PDB and XYZ files are in Å and are
converted on the way in and out; XTC is already in nm.

LAMMPS dumps are kept in their own unit style rather than converted, so a dump
compressed and decompressed again comes back with the same numbers. A dump
doesn't say which style it uses, so give it with `--lammps-units` (default
`real`). Converting to any other format converts the units properly.

## Choosing a codec

By default each group of frames is coded both ways and the smaller result is
kept. You can force one:

```bash
mdz compress traj.dcd -o out.mdz --codec temporal   # frames saved often
mdz compress traj.dcd -o out.mdz --codec intra      # frames far apart
```

## Lossless

```bash
mdz compress traj.dcd -o out.mdz --lossless
```

Every float is stored bit for bit. This uses the same prediction as the
temporal codec, but takes the residual between the integer bit patterns of the
actual and predicted values, which is exactly reversible.

Expect 1.3-3x, against 2-10x for the lossy codecs and about 1.1x for zstd on
the raw floats. The low mantissa bits of float32 coordinates are essentially
noise, so there is a limit to what any lossless method can do. Use it when you
need the original bits back, not as a default.

## Frames with different atoms

GCMC, `fix deposit`, evaporation and dumps of a region all give frames whose
atoms change. `mdz compress` notices this and writes a file that allows it
(`--variable-atoms` forces it). Atoms are matched by id where the format has
one, such as the LAMMPS `id` column, an extxyz `id` property or PDB serial
numbers, and by position otherwise.

```bash
mdz compress gcmc.lammpstrj -o gcmc.mdz
mdz info gcmc.mdz          # Atoms: varying, 412-540 per frame
```

```python
r = mdz.open("gcmc.mdz")
r.frame_natoms             # atoms in each frame
r[42]                      # (natoms in frame 42, 3)
r.atom_ids(42)

with mdz.Writer("out.mdz", variable_atoms=True) as w:
    for x, ids in frames:
        w.write(x, ids=ids)
```

Each group is stored over every atom that appears in any of its frames, with
gaps filled from the nearest frame that has the atom, so an atom that comes
and goes costs very little. The overhead on a trajectory whose atoms don't
change is under 2%. DCD, `.npy` and raw float32 can't hold a varying atom
count.

In LAMMPS use `dump_modify ID variable yes`; in C, the `variable_atoms` writer
option.

## Rigid bodies

```bash
mdz compress clusters.lammpstrj -o clusters.mdz --rigid
```

With `--rigid`, a rigid body is stored as a centre and an orientation per
frame, plus small residuals for its atoms so the error bound still holds. It
is worth it for large rigid bodies: on a LAMMPS run of rigid 13-atom clusters
files were 23-35% smaller, at the cost of compression up to 1.8x slower. For
three-atom water it saves 3-6% at typical save intervals, which is rarely
worth the slower compression.

Pass the bodies if you have them: `bodies=` in Python (a body index per atom,
-1 for none), or `dump_modify ID rigid mol` in LAMMPS. Otherwise mdz looks for
groups of atoms within 0.16 nm of each other whose distances stay fixed to
0.002 nm. Those thresholds assume atomistic data in nm, so detection won't
find anything in LJ units.

## Writing as you go

`mdz.Writer` (and `mdz::Writer` in C++) writes each group to disk as soon as
it is full. `flush()` writes whatever is buffered as a short group. If the
process dies, everything up to the last complete group is readable, and
`append=True` continues from there. When appending, use the same precision and
codec settings the file was created with, and the same columns. On Windows,
close any readers of the file first: appending shortens the file, and Windows
won't do that while another program has it open.

```python
with mdz.Writer("traj.mdz", append=True) as w:
    w.write(x, step=step, boxes=box)   # the same columns as before
```

## Threads

Compression and decompression run in parallel over groups, using every core
by default. `--threads N`, `mdz.set_num_threads(n)` or
`mdz::set_num_threads(n)` changes that; inside MPI or OpenMP programs set it
to 1.

## C++

Use mdz from another CMake project with `add_subdirectory(mdz)` or
`find_package(mdz)`, then link `mdz::mdz`. zstd is downloaded and built in;
pass `-DMDZ_ZSTD=system` to use an installed one.

```cpp
#include "mdz/mdz.hpp"

auto w = mdz::Writer::create("traj.mdz", mdz::CompressOptions{});
mdz::FrameBatch f(natoms, 1);
f.columns["positions"] = positions_nm;     // natoms * 3 floats
f.int_frame_props["step"] = {step};
w.write(f);
w.close();

auto r = mdz::Reader::open("traj.mdz");
std::vector<float> frame = r.read_frame(42);
```

Errors are thrown as `mdz::Error`. The C API is in
[`include/mdz/mdz.h`](../include/mdz/mdz.h).
