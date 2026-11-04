# mdz for JavaScript / WebAssembly

Goal: an npm package (`@peterspackman/mdz`) that reads (and later writes)
`.mdz` files in browsers and Node, with an API that follows the Python one.
The first user is CrystalExplorer web (cebuns).

## What we know

**The library already builds under emscripten.** `emcmake cmake -DMDZ_BUILD_CLI=OFF
-DMDZ_BUILD_TESTS=OFF -DMDZ_BUILD_BENCH=OFF` compiles every source with no
changes, including the zstd we fetch.

**Size.** A probe that binds `Reader::from_bytes`, `natoms`, `nframes` and
`read_positions` with embind links to 235 KB of wasm (94 KB gzipped) at -O3.

**It runs.** In Node, the probe reads `nacl.mdz` (converted with the native
CLI) and returns correct positions.

**Exceptions.** Emscripten turns off C++ exception catching by default
(`-fignore-exceptions`), and every mdz error is a throw. We build the library
*and* the binding with `-fwasm-exceptions`, which all current browsers and
Node 20+ support. A corrupt file then gives a JS exception, and
`getExceptionMessage` returns `corrupt .mdz file: not an .mdz file`.

**Threads.** The JS build is single-threaded: no pthreads and no
SharedArrayBuffer. `parallel_for` already runs inline when `num_threads()==1`.
We force it to 1 at module init so that `std::thread` is never reached.

**In-memory I/O is already supported.** We have `Reader::from_bytes`,
`load_bytes`, `to_bytes`, `memory_input` / `memory_output`, and
`Writer::create(std::unique_ptr<OutputStream>)`. Nothing in the binding needs
a filesystem, so there is no MEMFS, no `.data` file and no `FS`.

**The format suits range reads.** It has a 24-byte header, then the schema,
then self-contained group records (`GOF\x01`), then the footer index (group
offsets and frame ranges), then a 16-byte trailer. To open a file you read the
head, the tail and the footer. To read a frame you read one record. The
current `Reader` needs the whole file in memory, though. Its constructor also
decodes every box up front.

### What cebuns needs (from the cebuns session)

- **Sizes.** Typical runs are 100–5k atoms × 100–10k frames. The largest real
  case is 78k atoms × 643 frames, from a 1.8 GB LAMMPS dump.
- **Access.**
  - Random access for scrubbing, which matters most.
  - Sequential playback with prefetch.
  - Load-all for small files.
  - Memory must stay bounded: cebuns' `LazyTrajectory` uses an LRU cache with
    a byte budget (256 MB).
- **Byte source.** Bytes come from a `File`/`Blob` via
  `ByteSource { size; read(start, end): Promise<Uint8Array> }`. cebuns asked
  for "async range reads matter more than load-all".
- **Data.**
  - Essential: positions and the per-frame cell (3x3, row-major).
  - Once per file: elements, names and types.
  - Shown and plotted: per-frame time and energy.
  - Colour-by: extra per-atom columns.
  - Nice to have: velocities and forces.
  - Also: variable atom counts (GCMC).
- **Return types.**
  - A flat `Float32Array` of n*3 per frame.
  - Ideally filling a caller-provided buffer.
  - A cell as a `Float32Array(9)`.
  - TypeScript `.d.ts` files.
- **Writing.** Read-only is fine at first. Export to `.mdz` comes later.
- **Runtime.**
  - Vite, ESM and TypeScript.
  - occjs is loaded from `public/wasm` with pthreads, but a non-threaded
    decoder is preferred.
  - Running in a worker is welcome but not required.

### occ's setup: what to copy and what to avoid

- **Copy:**
  - embind.
  - `MODULARIZE` / `EXPORT_ES6`.
  - `ALLOW_MEMORY_GROWTH`.
  - The `packages/occjs` layout with vitest.
- **Avoid:**
  - Building it into the main CMake with global `CMAKE_EXE_LINKER_FLAGS`.
  - `.data` preload files.
  - pthreads.
  - The hand-written wasm path resolution in `module-loader.js`. Emscripten's
    ES6 output already uses `new URL('mdz.wasm', import.meta.url)`, which Vite
    and Node resolve without help.
  - Element-wise setters such as `Mat3N.set(i, j, v)`. Arrays should cross the
    boundary as one typed-array copy.

## Design

There are two layers.

1. **A small synchronous embind core (C++).** It decodes bytes it is given and
   never does I/O. Range reads are async in JS, while wasm is sync. This split
   avoids Asyncify and JSPI entirely: JS fetches the bytes a call needs, then
   calls into wasm.
2. **A TypeScript wrapper.** It handles byte sources, the group LRU cache,
   prefetch, units and the public API.

### C++: a lazy, record-at-a-time reader (new, also useful natively)

This goes in a new public header, `include/mdz/lazy_reader.hpp`; the name is
open. It is built and tested natively like the rest of the library, so the
logic is not wasm-only. It is useful for HTTP range reads and object storage
too.

```cpp
// Step 1: header + schema. Returns the bytes needed if `head` is too short.
static LazyReader open(ByteSpan head);           // throws Error(Corrupt)
std::size_t schema_end() const;
// Step 2: index. From the trailer + footer (complete file), or by scanning
// record headers (incomplete file; JS supplies 8-byte record headers).
std::optional<uint64_t> footer_offset(ByteSpan last16) const;
void set_index(ByteSpan footer);                 // validates CRC, fills groups
// Step 3: decode one group from its record bytes.
FrameBatch decode_group(std::size_t gi, ByteSpan record) const;
void decode_positions(std::size_t gi, ByteSpan record, float* out) const;
```

It reuses `container.cpp`'s record, schema and footer parsing, which is split
out of `load_container`. It also reuses `decompress()` on a one-group
`CompressedTrajectory`. Metadata (atom names, elements, types, ids, bodies,
residues) and the rigid model live in the schema, so they are available after
step 1.

### JS public API (follows Python, in camelCase)

```ts
import { open, info, fromBlob } from '@peterspackman/mdz';

interface ByteSource { size: number; read(start: number, end: number): Promise<Uint8Array>; }
type Input = Blob | ArrayBuffer | Uint8Array | ByteSource;

const t = await open(input, { lengthUnit: 'nm' | 'angstrom', cacheBytes: 256 << 20 });
t.natoms; t.nframes; t.variableAtoms; t.complete; t.precision; t.gofSize; t.nGroups;
t.columnNames; t.columns;          // [{name, dtype, width, perFrame, lossy, precision, unit}]
t.metadata;                        // {atomNames?, elements?, atomTypes?, atomIds?, residues?, bodies?, ...}

await t.readFrame(i, out?);              // Float32Array(natoms*3); fills `out` if given
await t.readFrames(start, stop, out?);   // packed Float32Array, [start, stop) clipped
await t.frame(i);                        // {positions, cell?, time?, energy?, step?, columns: {...}}
await t.cell(i);                         // Float32Array(9) | null
await t.readColumn(name, start?, stop?); // like Python read_column; shape in result.shape
await t.prefetch(start, stop);           // warm the cache for playback
t.frameNatoms(i); await t.atomIds(i);    // variable atom counts
t.close();
```

How the Python API maps onto it:

| Python | JS |
|---|---|
| `open` | `open` |
| `MdzReader.read_frame` / `read_frames` / `read_all` | `readFrame` / `readFrames` / `readFrames(0, nframes)` |
| `cell(i)` | `cell(i)` |
| `cells` | `await t.cells()` |
| `read_column` | `readColumn` |
| `metadata()` | `metadata` |
| `info` | `info` |

Rules for every function:

- Results are JS-owned copies. They are never views of `HEAPF32`, because a
  view detaches when wasm memory grows.
- Decoded groups are cached on the JS heap, in an LRU with a byte budget.
  `readFrame` copies one frame out of a group (it returns a `subarray`
  only when the caller asks for views).
- Lengths are in nm, as in Python. `lengthUnit: 'angstrom'` scales positions,
  boxes and cells; cebuns works in Å.
- A `Uint8Array`/`ArrayBuffer` that is under a threshold, or `{ eager: true }`,
  is loaded with `Reader::from_bytes` (the existing path). Anything else
  takes the lazy path.

**Requirements from the cebuns review of the draft API:**

- **Sync cache hit.** `t.peek(i, out?)` returns the frame without awaiting if
  its group is decoded, and `null` otherwise. Playback renders synchronously,
  and a peek saves cebuns from keeping a second cache. `peekFrame(i)` returns
  the full `frame()` object the same way.
- **Shared in-flight work.** Concurrent calls for frames in the same group
  that is not yet decoded share one read and one decode. Pending work is kept
  in a per-group `Map<gi, Promise>`.
- **Ownership.** Arrays returned without `out` are JS-owned copies that stay
  valid after later calls.
- **Cell layout.** A cell is a `Float32Array(9)` with the lattice vectors as
  rows: `[ax,ay,az, bx,by,bz, cx,cy,cz]`. That is already mdz's `Box32`
  layout. Cells are in Å when `lengthUnit: 'angstrom'`.
- **Prefetch.** It is called with wrap-around and backwards playback. The
  signature is `prefetch(i, count, { direction: 1 | -1, wrap: true })`, and it
  works in whole groups.
- **Variable atom counts.** cebuns uses a fixed superset of atoms with a
  `present` mask. In mdz, ids are arbitrary int32 values, not dense. Each
  group stores the union of its atoms, and the header has no list for the
  whole file.
  - `await t.atomUniverse()` returns `{ids: Int32Array, elements?: string[],
    names?: string[], types?: Int32Array}` for every atom in the file, sorted
    by id. Each array runs parallel to `ids`.
    - It reads only each group's `mdz.atoms` blob (the membership and the
      string table) and its `mdz.element`, `mdz.name` and `mdz.type` blobs.
      It skips through the blob lengths and never reads positions. This is
      option (b) below, applied to a few columns.
    - An atom takes the attributes it has in the first group it appears in.
    - cebuns builds the topology of the whole superset at open, so this has to
      be ready before the first frame.
  - `frame(i)` also returns `ids` and `present` relative to its group's union.
  - The adapter on cebuns' side maps id to slot.
- **Whole-file scalars.** `scalarSeries(name)` (energy, time, step) over all
  frames uses the same blob-skipping reads as `atomUniverse()`. It is
  welcome but not blocking.

**Open question: per-frame cells, time and energy for the whole file.** They
are stored in each group record, so getting all of them means touching every
record. We have three options:

- (a) `cells()` reads every record. This is fine for small files.
- (b) Read only each record's blob-length prefixes and the box/time/energy
  blobs, skipping positions. This needs about n_columns small reads per
  group.
- (c) Later, a format revision could put a per-frame summary in the footer.

v1 does (a), with lazy per-frame access. We do (b) if cebuns needs instant
energy plots on very large files.

### Packaging

- Code lives in `bindings/js/`, next to `bindings/python/`:
  - `CMakeLists.txt`: the `mdzjs` target, built only under `EMSCRIPTEN`, with
    flags per target and not global.
  - `src/bindings.cpp`, `ts/*.ts`, `package.json`, `vitest` tests.
- Output:
  - ESM `mdz.js` + `mdz.wasm`, built with `-sMODULARIZE -sEXPORT_ES6
    -sENVIRONMENT=web,worker,node -sALLOW_MEMORY_GROWTH -sMAXIMUM_MEMORY=4GB
    -fwasm-exceptions`.
  - Types are generated with `--emit-tsd` for the core; the wrapper's types
    are written by hand.
- `SINGLE_FILE` (base64 inlined) stays an option if bundlers give trouble. It
  costs about 33% in size and loses streaming compilation.
- CMake gets a `MDZ_BUILD_JS` option (default ON under emscripten). It turns off
  the CLI, tests and bench, which do not apply there.

## Stage 1: Emscripten build and minimal reader
**Goal**: `bindings/js` builds `mdz.js`/`mdz.wasm`, which can be loaded in
Node, with an eager `Reader` (from bytes) and `readFrame`, `cell`, `metadata`
and `columns`.
**Success Criteria**: `npm test` in `bindings/js` passes in Node. A corrupt
file throws a JS `Error` that carries the mdz message.
**Tests**:
- Fixture `.mdz` files are made by the native CLI from `examples/`: DCD (box),
  LAMMPS (extra columns) and a variable-atom file.
- Positions match the native decoder bit for bit. Native writes `.npy`
  references with `mdz decompress`.
- Truncated and garbage inputs throw.
- Reading an incomplete file returns its complete groups.
**Status**: Complete
- `bindings/js`: `scripts/build.sh` (emcmake) builds `dist/mdz_core.{mjs,wasm}`.
  `lib/index.js` + `lib/index.d.ts` hold the async public API.
- `npm test` runs `node --test` with no dependencies, and all 17 tests pass.
  Positions are bit-identical to the native decoder.
- C++ errors become `MdzError` with a `kind`. A probe confirmed that C++
  destructors run when a JS error unwinds through wasm under
  `-fwasm-exceptions`, so the error path does not leak.
- The npm package is 128 kB packed; the wasm is 274 kB.

## Stage 2: LazyReader in C++
**Goal**: the `LazyReader` above, with `load_container` refactored to share
the record, schema and footer parsing.
**Success Criteria**: the existing tests pass. The new lazy reader decodes
every frame of every fixture identically to `Reader`.
**Tests**:
- Complete and incomplete files.
- A `head` too short returns the bytes it needs.
- A bad footer CRC.
- A record from the wrong offset raises Corrupt.
- Variable atom counts and rigid bodies.
- A fuzz target for `open` and `decode_group`.
**Status**: Complete

## Stage 3: TypeScript API with async range reads
**Goal**: `open(Blob | ByteSource | bytes)`, a group LRU with a byte budget,
`prefetch`, `lengthUnit`, and `.d.ts` files. It runs in a browser worker and
on the main thread.
**Success Criteria**: random access to a large file reads only the head, the
tail, the footer and the records it needs, which tests check by counting
`ByteSource` reads. Memory stays under the budget.
**Tests**:
- A counting `ByteSource` mock.
- An LRU eviction test.
- `peek` returns null before a decode and the frame after it.
- Concurrent reads of one group make a single read.
- `prefetch` wraps around and runs backwards.
- `readFrame(i, out)` fills the caller's buffer.
- The Å scaling.
- A browser smoke test, a page in `examples/` driven by vitest browser
  mode or Playwright.
**Status**: Complete

## Stage 4: CI, release, cebuns integration
**Goal**: a CI job that runs a pinned emsdk build and the Node tests, and npm
publishing from `release.yml`. cebuns gets an `MdzTrajectory` that implements
its `LazyTrajectory` surface.
**Success Criteria**: CI is green. cebuns opens a `.mdz` by drag-drop and
scrubs a 78k-atom trajectory.
**Status**: In Progress. CI and release are done. The cebuns integration is not started.

## Stage 5 (later): writing and extras
- Write `.mdz`:
  - A `Writer` over `memory_output`, with `write({positions, cell, time,
    energy, ...})` and `close() → Uint8Array`.
  - A streaming sink callback for large exports.
  - `compress(positions, opts)` to match Python.
- Option (b) above, for per-frame scalars without decoding positions.
- Optional format converters (xyz, lammps, dcd → mdz in the browser). They are
  not needed by cebuns, which has its own parsers, and they cost size.
- `-msimd128`, if profiling shows decode time matters.
**Status**: Complete
- `createWriter` / `compress` cover writing `.mdz`, `parse` / `convert` cover the other formats, and `formats` lists them.
- Writing and the formats live in a second module (`mdz_full`, 404 kB gzipped) that is loaded on first use. The reader stays at 101 kB gzipped.
- Not done: the cheaper whole-file scalars (option (b)) and `-msimd128`.
