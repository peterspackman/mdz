# @peterspackman/mdz

Read `.mdz` trajectories in browsers and Node.

```js
import { open } from '@peterspackman/mdz';

const t = await open(file, { lengthUnit: 'angstrom' });
t.natoms, t.nframes, t.metadata.elements;

const xyz = await t.readFrame(42);     // Float32Array, natoms * 3
await t.readFrame(43, xyz);            // fill an existing buffer
t.peek(44);                            // sync; null unless 44's group is decoded
await t.prefetch(44, 20, { direction: -1, wrap: true });

await t.cell(42);                      // Float32Array(9), rows a, b, c, or null
await t.frame(42);                     // { positions, cell, scalars, columns }
await t.readColumn('energy');          // { dtype, shape, data }
await t.atomUniverse();                // { ids, elements, names, types }
t.close();
```

`open` takes a `File`/`Blob`, `ArrayBuffer`, `Uint8Array` or any
`{ size, read(start, end) }`. Only the header and index are read up front;
groups of frames are read and decoded on demand and cached (`cacheBytes`,
default 256 MiB).

## Build

```sh
npm run build   # needs emcc, cmake, ninja
npm test
```
