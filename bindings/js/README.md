# @peterspackman/mdz

Read and write `.mdz` trajectories in browsers and Node, and parse or convert
xyz, extxyz, LAMMPS dumps, PDB, DCD, XTC and npy.

```js
import { open, createWriter, compress, parse, convert } from '@peterspackman/mdz';

const t = await open(file, { lengthUnit: 'angstrom' });
const xyz = await t.readFrame(42);     // Float32Array, natoms * 3
await t.readFrame(43, xyz);            // fill an existing buffer
t.peek(44);                            // sync; null unless 44's group is decoded
await t.prefetch(44, 20, { direction: -1, wrap: true });
await t.cell(42);                      // Float32Array(9), rows a, b, c, or null
await t.frame(42);                     // { positions, cell, scalars, columns }
await t.readColumn('energy');          // { dtype, shape, data }
await t.atomUniverse();                // { ids, elements, names, types }

const w = await createWriter({ elements, lengthUnit: 'angstrom' });
w.write({ positions, cell, energy });
const bytes = w.close();

await compress(positions, { natoms, cells });
await parse(file);                     // format from file.name, or { format }
await convert(file, { to: 'mdz' });
```

`open` reads only the header and index; groups of frames are read on demand
and cached (`cacheBytes`, default 256 MiB). Writing and the other formats load
a second, larger module the first time they are used.

## Build

```sh
npm run build   # needs emcc, cmake, ninja
npm test
```
