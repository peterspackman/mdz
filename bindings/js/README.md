# @peterspackman/mdz

Read `.mdz` molecular dynamics trajectories in browsers and Node, through
WebAssembly. The API follows the Python package (`mdz.open`), in camelCase.

```js
import { open } from '@peterspackman/mdz';

const t = await open(file, { lengthUnit: 'angstrom' }); // File, Blob, ArrayBuffer or Uint8Array
t.natoms, t.nframes, t.metadata.elements;
const xyz = await t.readFrame(42);          // Float32Array, natoms * 3
await t.readFrame(43, xyz);                 // or fill your own buffer
const cell = await t.cell(42);              // Float32Array(9), rows a, b, c, or null
const fr = await t.frame(42);               // {positions, cell, scalars: {time, energy, step}, columns}
const vel = await t.readColumn('velocities', 0, 10); // {dtype, shape, data}
t.close();
```

Positions and cells are in nm unless `lengthUnit: 'angstrom'` is given. Other
columns are in their stored units (see `t.columns`). Every returned array
belongs to the caller. Errors are `MdzError`, whose `kind` is `InvalidInput`,
`Corrupt`, `OutOfRange`, ...

## Building

Needs emscripten (`emcc`), CMake and Ninja.

```sh
npm run build   # scripts/build.sh: emcmake build into build-wasm/, copied to dist/
npm test        # node --test
```

Test fixtures are regenerated with `test/fixtures/make_fixtures.sh`, which uses
the native `mdz` CLI.
