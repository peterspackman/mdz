// mdz for JavaScript: read .mdz trajectories in browsers and Node.
//
//   import { open } from '@peterspackman/mdz';
//   const t = await open(file, { lengthUnit: 'angstrom' });
//   const xyz = await t.readFrame(42);   // Float32Array, natoms * 3
//   const box = await t.cell(42);        // Float32Array(9), rows a, b, c, or null
//
// The API follows the Python package (mdz.open / MdzReader), in camelCase.
// Methods that decode are async, so that later versions can read large
// files in byte ranges without changing the API.

import createMdzCore from '../dist/mdz_core.mjs';

/** Error raised by mdz. `kind` is InvalidInput, Corrupt, Io, OutOfRange,
 * OutOfMemory or Internal. */
export class MdzError extends Error {
  constructor(message, kind = 'Internal') {
    super(message);
    this.name = 'MdzError';
    this.kind = kind;
  }
}

let corePromise = null;

/** Load the WebAssembly module. Called by open(); call it yourself to load
 * early or to pass emscripten options such as `locateFile` or `wasmBinary`.
 * Options only take effect on the first call. */
export function init(options = {}) {
  if (!corePromise) {
    corePromise = createMdzCore({ ...options, MdzError }).catch((e) => {
      corePromise = null;
      throw e;
    });
  }
  return corePromise;
}

/** The version of the mdz library. */
export async function version() {
  return (await init()).version();
}

const LENGTH_SCALE = { nm: 1, angstrom: 10 };

async function toBytes(input) {
  if (input instanceof Uint8Array) return input;
  if (input instanceof ArrayBuffer) return new Uint8Array(input);
  if (ArrayBuffer.isView(input)) return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  if (typeof Blob !== 'undefined' && input instanceof Blob) return new Uint8Array(await input.arrayBuffer());
  throw new MdzError('open() takes a Uint8Array, ArrayBuffer or Blob', 'InvalidInput');
}

/**
 * Open a .mdz file held in memory or in a Blob (such as a File from a file
 * picker).
 * @param {Uint8Array | ArrayBuffer | ArrayBufferView | Blob} input
 * @param {{ lengthUnit?: 'nm' | 'angstrom' }} [options]
 */
export async function open(input, options = {}) {
  const lengthUnit = options.lengthUnit ?? 'nm';
  const scale = LENGTH_SCALE[lengthUnit];
  if (scale === undefined) throw new MdzError(`unknown lengthUnit '${lengthUnit}' (use 'nm' or 'angstrom')`, 'InvalidInput');
  const [core, bytes] = await Promise.all([init(), toBytes(input)]);
  return new Trajectory(core.Reader.fromBytes(bytes), lengthUnit, scale);
}

function scaleInPlace(a, s) {
  if (s !== 1) for (let i = 0; i < a.length; ++i) a[i] *= s;
  return a;
}

export class Trajectory {
  #r;
  #scale;
  #columns;
  #metadata;
  #frameNatoms;

  /** @internal Use open(). */
  constructor(reader, lengthUnit, scale) {
    this.#r = reader;
    this.#scale = scale;
    this.lengthUnit = lengthUnit;
  }

  #reader() {
    if (!this.#r) throw new MdzError('the trajectory is closed', 'InvalidInput');
    return this.#r;
  }

  // Resolve a frame index, counting from the end if negative.
  #index(i) {
    const n = this.nframes;
    if (!Number.isInteger(i)) throw new MdzError(`frame index must be an integer, got ${i}`, 'InvalidInput');
    const f = i < 0 ? i + n : i;
    if (f < 0 || f >= n) throw new MdzError(`frame ${i} out of range (nframes=${n})`, 'OutOfRange');
    return f;
  }

  /** Atoms per frame, or 0 if the count varies (see variableAtoms). */
  get natoms() { return this.#reader().natoms; }
  get nframes() { return this.#reader().nframes; }
  /** Whether the atom count varies between frames. */
  get variableAtoms() { return this.#reader().variableAtoms; }
  /** Whether the writer closed the file. An unclosed file holds its complete groups. */
  get complete() { return this.#reader().complete; }
  /** Quantization step of positions in lengthUnit, or 0 if they are exact. */
  get precision() { return this.#reader().precision * this.#scale; }
  get gofSize() { return this.#reader().gofSize; }
  get nGroups() { return this.#reader().nGroups; }

  /** Names of every readable column, with 'positions' first. 'cell' is derived from 'box'. */
  get columnNames() { return this.#reader().columnNames(); }
  /** Column descriptors: name, dtype, width, perFrame, lossy, precision, unit. */
  get columns() { return (this.#columns ??= this.#reader().columns()); }
  /** Static metadata: atomNames, elements, atomTypes, atomIds, residues, bodies, attributes, ... */
  get metadata() { return (this.#metadata ??= this.#reader().metadata()); }
  /** Atoms in each frame, as a Float64Array of length nframes. */
  get frameNatoms() { return (this.#frameNatoms ??= this.#reader().frameNatoms()); }

  /** Atom ids of one frame, in the order its positions are returned. */
  async atomIds(i) { return this.#reader().atomIds(this.#index(i)); }

  /**
   * Positions of one frame, natoms * 3 values in lengthUnit. Fills and returns
   * `out` if given, which must hold exactly that many values.
   */
  async readFrame(i, out) {
    const f = this.#index(i);
    return this.#deliver(this.#reader().readPositions(f, 1), out);
  }

  /**
   * Positions of frames [start, stop), packed frame after frame. The range is
   * clipped to the file. Fills and returns `out` if given.
   */
  async readFrames(start = 0, stop = this.nframes, out) {
    const n = this.nframes;
    const lo = Math.max(0, Math.min(start, n));
    const hi = Math.max(lo, Math.min(stop, n));
    if (hi === lo) return this.#deliver(new Float32Array(0), out);
    return this.#deliver(this.#reader().readPositions(lo, hi - lo), out);
  }

  #deliver(positions, out) {
    scaleInPlace(positions, this.#scale);
    if (out === undefined) return positions;
    if (!(out instanceof Float32Array) || out.length !== positions.length)
      throw new MdzError(`out must be a Float32Array of length ${positions.length}`, 'InvalidInput');
    out.set(positions);
    return out;
  }

  /** The cell of one frame as a Float32Array(9), rows a, b, c, in lengthUnit; null if the file has none. */
  async cell(i) {
    const c = this.#reader().cell(this.#index(i));
    return c && scaleInPlace(c, this.#scale);
  }

  /**
   * One frame with every column:
   * { natoms, positions, cell, scalars: {time, energy, step, ...}, columns: {velocities, ...} }.
   * positions and cell are in lengthUnit; other columns in their stored units
   * (see `columns`). Files whose atom count varies also give ids, elements,
   * names and types for the frame.
   */
  async frame(i) {
    const fr = this.#reader().frame(this.#index(i));
    scaleInPlace(fr.positions, this.#scale);
    if (fr.cell) scaleInPlace(fr.cell, this.#scale);
    return fr;
  }

  /**
   * Decode a column for frames [start, stop) as { name, dtype, shape, data }.
   * Shapes follow the Python package. 'positions' is in lengthUnit; other
   * columns are in their stored units. int64 columns (step) come back as
   * Float64Array.
   */
  async readColumn(name, start = 0, stop = this.nframes) {
    const n = this.nframes;
    const lo = Math.max(0, Math.min(start, n));
    const hi = Math.max(lo, Math.min(stop, n));
    if (hi === lo && n > 0) throw new MdzError(`empty frame range [${start}, ${stop})`, 'OutOfRange');
    const col = this.#reader().readColumn(name, lo, hi - lo);
    if (name === 'positions') scaleInPlace(col.data, this.#scale);
    return col;
  }

  /** Release the WebAssembly memory held by this trajectory. Closing again does nothing. */
  close() {
    this.#r?.delete();
    this.#r = null;
  }

  [Symbol.dispose]() { this.close(); }
}
