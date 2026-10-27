# mdz

mdz compresses molecular dynamics trajectories to a fixed maximum error
(0.005 Å by default, the same as XTC). Any frame can be read without
decompressing the rest of the file.

Frames are predicted from the ones before them, so files are 2.5-3.5x
smaller than XTC when frames are saved every few femtoseconds, and never
bigger than XTC. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

mdz is a C++17 library with a command-line tool, Python bindings, a C API and
a [LAMMPS dump style](lammps/README.md). It reads and writes DCD, XTC, XYZ,
extended XYZ, PDB, LAMMPS dumps and `.npy`.

## Install

```bash
pip install mdz
```

Prebuilt `mdz` binaries are on the
[releases page](https://github.com/peterspackman/mdz/releases). To build
from source you need CMake 3.18+ and a C++17 compiler:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

## Usage

```bash
mdz compress traj.dcd -o traj.mdz
mdz decompress traj.mdz -o traj.xtc
mdz extract traj.mdz 100:200 -o slice.pdb
mdz info traj.mdz
```

```python
import mdz

r = mdz.open("traj.mdz")
r[42]           # one frame, (natoms, 3), in nm
r[1000:2000]    # a range of frames

with mdz.Writer("out.mdz") as w:
    for step, x in simulation():
        w.write(x, step=step)
```

[docs/USAGE.md](docs/USAGE.md) covers precision, units, lossless mode,
varying atom counts, rigid bodies and the C++ API.
[docs/FORMAT.md](docs/FORMAT.md) describes the file format.

## Development

```bash
ctest --test-dir build
MDZ_CLI=$PWD/build/cli/mdz python -m pytest cli/tests
pip install . && python -m pytest tests
```

## License

MIT. To cite mdz, see [CITATION.cff](CITATION.cff).
