// Read and write .mdz trajectories, and parse and convert other formats, in
// browsers and Node.
//
// open() reads the header and the group index; each group of frames is read
// and decoded when a frame in it is first asked for, and kept in an LRU cache.
// Writing and the other formats use a larger module (mdz_full), loaded the
// first time one of them is used.

import createMdzCore from '../dist/mdz_core.mjs';

export class MdzError extends Error {
  constructor(message, kind = 'Internal') {
    super(message);
    this.name = 'MdzError';
    this.kind = kind;
  }
}

let corePromise = null;

export function init(options = {}) {
  if (!corePromise) {
    corePromise = createMdzCore({ ...options, MdzError }).catch((e) => {
      corePromise = null;
      throw e;
    });
  }
  return corePromise;
}

let fullPromise = null;

export function initFull(options = {}) {
  if (!fullPromise) {
    fullPromise = import('../dist/mdz_full.mjs')
      .then((m) => m.default({ ...options, MdzError }))
      .catch((e) => {
        fullPromise = null;
        throw e;
      });
  }
  return fullPromise;
}

export async function version() {
  return (await init()).version();
}

const LENGTH_SCALE = { nm: 1, angstrom: 10 };
const HEAD_GUESS = 64 * 1024;
const DEFAULT_CACHE_BYTES = 256 * 1024 * 1024;

export function blobSource(blob) {
  return {
    size: blob.size,
    read: async (start, end) => new Uint8Array(await blob.slice(start, end).arrayBuffer()),
  };
}

function memorySource(bytes) {
  return { size: bytes.length, read: (start, end) => bytes.subarray(start, end) };
}

function toSource(input) {
  if (input instanceof Uint8Array) return memorySource(input);
  if (input instanceof ArrayBuffer) return memorySource(new Uint8Array(input));
  if (ArrayBuffer.isView(input)) return memorySource(new Uint8Array(input.buffer, input.byteOffset, input.byteLength));
  if (typeof Blob !== 'undefined' && input instanceof Blob) return blobSource(input);
  if (input && typeof input.size === 'number' && typeof input.read === 'function') return input;
  throw new MdzError('open() takes a Uint8Array, ArrayBuffer, Blob or byte source', 'InvalidInput');
}

async function readRange(src, start, end) {
  const bytes = await src.read(start, end);
  if (!(bytes instanceof Uint8Array) || bytes.length !== end - start)
    throw new MdzError(`byte source returned ${bytes?.length} bytes for [${start}, ${end})`, 'Io');
  return bytes;
}

async function scanRecords(core, lazy, src) {
  for (;;) {
    const at = lazy.scanOffset();
    if (at + core.recordHeadSize > src.size) return;
    const n = lazy.recordSize(await readRange(src, at, at + core.recordHeadSize));
    if (!n || !lazy.addRecord(await readRange(src, at, at + n))) return;
  }
}

export async function open(input, options = {}) {
  const lengthUnit = options.lengthUnit ?? 'nm';
  const scale = LENGTH_SCALE[lengthUnit];
  if (scale === undefined) throw new MdzError(`unknown lengthUnit '${lengthUnit}' (use 'nm' or 'angstrom')`, 'InvalidInput');
  const src = toSource(input);
  const core = await init();
  const size = src.size;
  if (size < core.headerSize) throw new MdzError('not an .mdz file: too short', 'Corrupt');
  let head = await readRange(src, 0, Math.min(size, HEAD_GUESS));
  const need = core.LazyReader.headSize(head);
  if (need > head.length && need <= size) head = await readRange(src, 0, need);
  const lazy = core.LazyReader.open(head, size);
  try {
    let closed = false;
    if (size >= core.trailerSize) {
      const range = lazy.footerRange(await readRange(src, size - core.trailerSize, size));
      if (range) closed = lazy.readFooter(range[0], await readRange(src, range[0], range[1]));
    }
    if (!closed) await scanRecords(core, lazy, src);
  } catch (e) {
    lazy.delete();
    throw e;
  }
  return new Trajectory(lazy, src, lengthUnit, scale, options.cacheBytes ?? DEFAULT_CACHE_BYTES);
}

function iota(n) {
  const a = new Int32Array(n);
  for (let i = 0; i < n; ++i) a[i] = i;
  return a;
}

function concat(chunks, Type) {
  const out = new Type(chunks.reduce((n, c) => n + c.length, 0));
  let at = 0;
  for (const c of chunks) {
    out.set(c, at);
    at += c.length;
  }
  return out;
}

// (a, gamma, b, beta, alpha, c) of a row-major box, angles in degrees.
function cellParams(m) {
  const len = (i) => Math.hypot(m[i], m[i + 1], m[i + 2]);
  const angle = (i, j) => {
    const d = (m[i] * m[j] + m[i + 1] * m[j + 1] + m[i + 2] * m[j + 2]) / (len(i) * len(j));
    return (Math.acos(Math.max(-1, Math.min(1, d))) * 180) / Math.PI;
  };
  return [len(0), angle(0, 3), len(3), angle(0, 6), angle(3, 6), len(6)];
}

function groupBytes(g) {
  let n = 0;
  const add = (a) => {
    if (a) n += a.byteLength ?? a.length * 16;
  };
  [g.natoms, g.offsets, g.positions, g.cells, g.origins, g.ids, g.types, g.elements, g.names].forEach(add);
  Object.values(g.scalars).forEach(add);
  Object.values(g.columns).forEach(add);
  return n;
}

export class Trajectory {
  #lazy;
  #src;
  #scale;
  #budget;
  #groups;
  #cache = new Map();
  #cached = 0;
  #pending = new Map();
  #columns;
  #metadata;

  constructor(lazy, src, lengthUnit, scale, cacheBytes) {
    this.#lazy = lazy;
    this.#src = src;
    this.#scale = scale;
    this.#budget = cacheBytes;
    this.#groups = lazy.groups();
    this.lengthUnit = lengthUnit;
  }

  #reader() {
    if (!this.#lazy) throw new MdzError('the trajectory is closed', 'InvalidInput');
    return this.#lazy;
  }

  #index(i) {
    const n = this.nframes;
    if (!Number.isInteger(i)) throw new MdzError(`frame index must be an integer, got ${i}`, 'InvalidInput');
    const f = i < 0 ? i + n : i;
    if (f < 0 || f >= n) throw new MdzError(`frame ${i} out of range (nframes=${n})`, 'OutOfRange');
    return f;
  }

  #clip(start, stop) {
    const n = this.nframes;
    const lo = Math.max(0, Math.min(start, n));
    return [lo, Math.max(lo, Math.min(stop, n))];
  }

  #first(gi) {
    return this.#groups[4 * gi + 2];
  }

  #groupOf(f) {
    let lo = 0;
    let hi = this.nGroups - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (this.#first(mid) <= f) lo = mid;
      else hi = mid - 1;
    }
    return lo;
  }

  async #decode(gi, columns) {
    const offset = this.#groups[4 * gi];
    const bytes = await readRange(this.#src, offset, offset + this.#groups[4 * gi + 1]);
    const g = this.#reader().decodeGroup(gi, bytes, columns, this.#scale);
    g.first = this.#first(gi);
    g.offsets = new Float64Array(g.nframes + 1);
    for (let k = 0; k < g.nframes; ++k) g.offsets[k + 1] = g.offsets[k] + g.natoms[k];
    return g;
  }

  // A decoded group with every column, from the cache or the source.
  #load(gi) {
    const hit = this.#peekGroup(gi);
    if (hit) return Promise.resolve(hit);
    let p = this.#pending.get(gi);
    if (!p) {
      p = this.#decode(gi, [])
        .then((g) => {
          if (this.#lazy) this.#insert(gi, g);
          return g;
        })
        .finally(() => this.#pending.delete(gi));
      this.#pending.set(gi, p);
    }
    return p;
  }

  // Only some columns, from the cache if the group is there, without caching.
  #loadColumns(gi, columns) {
    const hit = this.#peekGroup(gi);
    return hit ? Promise.resolve(hit) : this.#decode(gi, columns);
  }

  #peekGroup(gi) {
    const g = this.#cache.get(gi);
    if (g) {
      this.#cache.delete(gi);
      this.#cache.set(gi, g);
    }
    return g;
  }

  #insert(gi, g) {
    g.bytes = groupBytes(g);
    this.#cache.set(gi, g);
    this.#cached += g.bytes;
    for (const [k, old] of this.#cache) {
      if (this.#cached <= this.#budget || this.#cache.size <= 1) break;
      this.#cache.delete(k);
      this.#cached -= old.bytes;
    }
  }

  #frameOf(g, f) {
    const k = f - g.first;
    const a0 = g.offsets[k];
    const a1 = g.offsets[k + 1];
    const atoms = (arr) => {
      const w = arr.length / g.offsets[g.nframes];
      return arr.slice(a0 * w, a1 * w);
    };
    const fr = {
      natoms: a1 - a0,
      positions: g.positions.slice(a0 * 3, a1 * 3),
      cell: g.cells ? g.cells.slice(9 * k, 9 * k + 9) : null,
      scalars: {},
      columns: {},
    };
    if (g.origins) fr.origin = g.origins.slice(3 * k, 3 * k + 3);
    for (const [name, v] of Object.entries(g.scalars)) fr.scalars[name] = v[k];
    for (const [name, v] of Object.entries(g.columns)) fr.columns[name] = atoms(v);
    for (const name of ['ids', 'types', 'elements', 'names']) if (g[name]) fr[name] = atoms(g[name]);
    return fr;
  }

  #positionsOf(g, f, out) {
    const k = f - g.first;
    const p = g.positions.subarray(g.offsets[k] * 3, g.offsets[k + 1] * 3);
    if (out === undefined) return p.slice();
    if (!(out instanceof Float32Array) || out.length !== p.length)
      throw new MdzError(`out must be a Float32Array of length ${p.length}`, 'InvalidInput');
    out.set(p);
    return out;
  }

  get natoms() { return this.#reader().natoms; }
  get nframes() { return this.#reader().nframes; }
  get variableAtoms() { return this.#reader().variableAtoms; }
  get complete() { return this.#reader().complete; }
  get precision() { return this.#reader().precision * this.#scale; }
  get gofSize() { return this.#reader().gofSize; }
  get nGroups() { return this.#groups.length / 4; }
  get columns() { return (this.#columns ??= this.#reader().columns()); }
  get metadata() { return (this.#metadata ??= this.#reader().metadata()); }
  get cachedBytes() { return this.#cached; }

  get columnNames() {
    const names = this.columns.map((c) => c.name);
    if (names.includes('box')) names.push('cell');
    return names;
  }

  async readFrame(i, out) {
    const f = this.#index(i);
    return this.#positionsOf(await this.#load(this.#groupOf(f)), f, out);
  }

  // Positions of frame i if its group is decoded, else null. Never waits.
  peek(i, out) {
    const f = this.#index(i);
    const g = this.#peekGroup(this.#groupOf(f));
    return g ? this.#positionsOf(g, f, out) : null;
  }

  async frame(i) {
    const f = this.#index(i);
    return this.#frameOf(await this.#load(this.#groupOf(f)), f);
  }

  peekFrame(i) {
    const f = this.#index(i);
    const g = this.#peekGroup(this.#groupOf(f));
    return g ? this.#frameOf(g, f) : null;
  }

  async cell(i) {
    const f = this.#index(i);
    const g = await this.#load(this.#groupOf(f));
    const k = f - g.first;
    return g.cells ? g.cells.slice(9 * k, 9 * k + 9) : null;
  }

  async atomIds(i) {
    const f = this.#index(i);
    if (!this.variableAtoms) return this.metadata.atomIds?.slice() ?? iota(this.natoms);
    return this.#frameOf(await this.#load(this.#groupOf(f)), f).ids;
  }

  async readFrames(start = 0, stop = this.nframes, out) {
    const [lo, hi] = this.#clip(start, stop);
    const chunks = [];
    for (let f = lo; f < hi; ) {
      const g = await this.#load(this.#groupOf(f));
      const end = Math.min(hi, g.first + g.nframes);
      chunks.push(g.positions.slice(g.offsets[f - g.first] * 3, g.offsets[end - g.first] * 3));
      f = end;
    }
    const all = concat(chunks, Float32Array);
    if (out === undefined) return all;
    if (!(out instanceof Float32Array) || out.length !== all.length)
      throw new MdzError(`out must be a Float32Array of length ${all.length}`, 'InvalidInput');
    out.set(all);
    return out;
  }

  // Decode the groups that hold count frames from frame i, stepping by
  // direction and wrapping around the ends if wrap is set.
  async prefetch(i, count = this.gofSize, { direction = 1, wrap = false } = {}) {
    const n = this.nframes;
    const groups = new Set();
    let f = this.#index(i);
    for (let k = 0; k < count; ++k) {
      groups.add(this.#groupOf(f));
      f += direction;
      if (f < 0 || f >= n) {
        if (!wrap) break;
        f = (f + n) % n;
      }
    }
    await Promise.all([...groups].map((gi) => this.#load(gi)));
  }

  async readColumn(name, start = 0, stop = this.nframes) {
    const [lo, hi] = this.#clip(start, stop);
    const n = hi - lo;
    const natoms = this.natoms;
    const varying = this.variableAtoms;
    if (name === 'positions') {
      const data = await this.readFrames(lo, hi);
      return { name, dtype: 'f32', shape: varying ? [data.length / 3, 3] : [n, natoms, 3], data };
    }
    const stored = name === 'cell' ? 'box' : name;
    const desc = this.columns.find((c) => c.name === stored);
    if (!desc) throw new MdzError(`no column named '${name}'`, 'InvalidInput');
    const chunks = [];
    for (let f = lo; f < hi; ) {
      const gi = this.#groupOf(f);
      const g = await this.#loadColumns(gi, [stored]);
      const k0 = f - g.first;
      const k1 = Math.min(hi, g.first + g.nframes) - g.first;
      if (stored === 'box') chunks.push(g.cells.slice(9 * k0, 9 * k1));
      else if (stored === 'origin') chunks.push(g.origins.slice(3 * k0, 3 * k1));
      else if (desc.perFrame) chunks.push(g.scalars[stored].slice(k0, k1));
      else {
        const w = desc.width;
        chunks.push(g.columns[stored].slice(g.offsets[k0] * w, g.offsets[k1] * w));
      }
      f = g.first + k1;
    }
    if (name === 'cell') {
      const boxes = concat(chunks, Float32Array);
      const data = new Float64Array(6 * n);
      for (let k = 0; k < n; ++k) data.set(cellParams(boxes.subarray(9 * k, 9 * k + 9)), 6 * k);
      return { name, dtype: 'f64', shape: [n, 6], data };
    }
    if (stored === 'box') return { name, dtype: 'f32', shape: [n, 3, 3], data: concat(chunks, Float32Array) };
    if (stored === 'origin') return { name, dtype: 'f32', shape: [n, 3], data: concat(chunks, Float32Array) };
    if (desc.perFrame) {
      const data = concat(chunks, Float64Array);
      return { name, dtype: desc.dtype === 'i64' ? 'i64' : 'f64', shape: [n], data };
    }
    const Type = chunks[0] instanceof Int32Array ? Int32Array : Float32Array;
    const data = concat(chunks, Type);
    const w = desc.width;
    const rows = data.length / w;
    const shape = varying ? (w === 1 ? [rows] : [rows, w]) : w === 1 ? [n, natoms] : [n, natoms, w];
    return { name, dtype: Type === Int32Array ? 'i32' : 'f32', shape, data };
  }

  // Atoms in each frame.
  async frameNatoms() {
    if (!this.variableAtoms) return new Float64Array(this.nframes).fill(this.natoms);
    const chunks = [];
    for (let gi = 0; gi < this.nGroups; ++gi) chunks.push((await this.#loadColumns(gi, ['mdz.atoms'])).natoms);
    return concat(chunks, Float64Array);
  }

  // Every atom in the file, sorted by id, with its element, name and type
  // where stored. In a file whose atom count varies, an atom takes the
  // values of the first group it appears in.
  async atomUniverse() {
    if (!this.variableAtoms) {
      const m = this.metadata;
      const out = { ids: m.atomIds?.slice() ?? iota(this.natoms) };
      if (m.elements) out.elements = m.elements.slice();
      if (m.atomNames) out.names = m.atomNames.slice();
      if (m.atomTypes) out.types = m.atomTypes.slice();
      return out;
    }
    const seen = new Map();
    for (let gi = 0; gi < this.nGroups; ++gi) {
      const g = await this.#loadColumns(gi, ['mdz.atoms', 'mdz.element', 'mdz.name', 'mdz.type']);
      for (let a = 0; a < g.ids.length; ++a) {
        const id = g.ids[a];
        if (!seen.has(id)) seen.set(id, { element: g.elements?.[a], name: g.names?.[a], type: g.types?.[a] });
      }
    }
    const ids = Int32Array.from(seen.keys()).sort();
    const out = { ids };
    const first = seen.get(ids[0]) ?? {};
    if (first.element !== undefined) out.elements = Array.from(ids, (id) => seen.get(id).element);
    if (first.name !== undefined) out.names = Array.from(ids, (id) => seen.get(id).name);
    if (first.type !== undefined) out.types = Int32Array.from(ids, (id) => seen.get(id).type);
    return out;
  }

  close() {
    this.#lazy?.delete();
    this.#lazy = null;
    this.#cache.clear();
    this.#cached = 0;
  }

  [Symbol.dispose]() {
    this.close();
  }
}

// --- other formats ---

async function wholeBytes(input) {
  if (typeof input === 'string') return new TextEncoder().encode(input);
  if (input instanceof Uint8Array) return input;
  if (input instanceof ArrayBuffer) return new Uint8Array(input);
  if (ArrayBuffer.isView(input)) return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  if (typeof Blob !== 'undefined' && input instanceof Blob) return new Uint8Array(await input.arrayBuffer());
  if (input && typeof input.size === 'number' && typeof input.read === 'function') return readRange(input, 0, input.size);
  throw new MdzError('expected a string, Uint8Array, ArrayBuffer, Blob or byte source', 'InvalidInput');
}

async function formatOf(core, given, input, what) {
  if (given) return given;
  const name = typeof input?.name === 'string' ? input.name : '';
  const detected = name && core.detectFormat(name);
  if (!detected) throw new MdzError(`cannot tell the ${what} format: pass { ${what === 'input' ? 'from' : 'to'} }`, 'InvalidInput');
  return detected;
}

function lengthScale(options) {
  const unit = options.lengthUnit ?? 'nm';
  const scale = LENGTH_SCALE[unit];
  if (scale === undefined) throw new MdzError(`unknown lengthUnit '${unit}' (use 'nm' or 'angstrom')`, 'InvalidInput');
  return scale;
}

// Options for the C++ side: precision in nm, everything else as given.
function coreOptions(options, scale) {
  const out = { ...options, lengthScale: scale };
  if (options.precision !== undefined) out.precision = options.precision / scale;
  return out;
}

/** The trajectory formats, with their file extensions. */
export async function formats() {
  return (await initFull()).formats();
}

/**
 * Parse a whole trajectory file in any supported format. `format` defaults to
 * the one matching `input.name` (a File). Lengths are in lengthUnit; other
 * values are in mdz's internal units (nm/ps, kJ/mol/nm, ...).
 */
export async function parse(input, options = {}) {
  const scale = lengthScale(options);
  const core = await initFull();
  const format = await formatOf(core, options.format, input, 'input');
  const f = core.parse(await wholeBytes(input), format, { ...options, lengthScale: scale });
  f.frameNatoms = f.natoms;
  f.variableAtoms = f.ids !== undefined;
  f.natoms = f.variableAtoms ? 0 : (f.frameNatoms[0] ?? 0);
  f.cells ??= null;
  return f;
}

/**
 * Convert a trajectory between formats, returning the bytes of the output.
 * `from` defaults to the format matching `input.name`; `to` is a format name
 * such as 'mdz', 'xyz' or 'lammps'. Compression options apply to 'mdz'.
 */
export async function convert(input, options = {}) {
  const scale = lengthScale(options);
  const core = await initFull();
  const from = await formatOf(core, options.from, input, 'input');
  if (!options.to) throw new MdzError('convert() needs { to }, the output format', 'InvalidInput');
  return core.convert(await wholeBytes(input), from, options.to, coreOptions(options, scale));
}

// --- writing .mdz ---

/**
 * Write a .mdz file frame by frame. Bytes are handed to `onData` as groups
 * are completed, or returned by close() if there is no onData.
 */
export async function createWriter(options = {}) {
  const scale = lengthScale(options);
  const core = await initFull();
  return new Writer(new core.Writer(coreOptions(options, scale)), options.onData);
}

export class Writer {
  #w;
  #onData;
  #chunks = [];

  constructor(writer, onData) {
    this.#w = writer;
    this.#onData = onData;
  }

  #writer() {
    if (!this.#w) throw new MdzError('the writer is closed', 'InvalidInput');
    return this.#w;
  }

  #drain() {
    const bytes = this.#w.take();
    if (bytes.length === 0) return;
    if (this.#onData) this.#onData(bytes);
    else this.#chunks.push(bytes);
  }

  /**
   * Add a frame: { positions, cell?, velocities?, forces?, charge?, time?,
   * energy?, step?, scalars?, columns?, ids?, elements?, names?, types? }.
   * positions and cell are in lengthUnit.
   */
  write(frame) {
    const w = this.#writer();
    if (!frame?.positions?.length) throw new MdzError('write() needs a frame with positions', 'InvalidInput');
    w.write(frame);
    this.#drain();
  }

  /** Encode buffered frames as a group, which may be short. */
  flush() {
    this.#writer().flush();
    this.#drain();
  }

  /** Finish the file. Returns its bytes, unless onData was given. */
  close() {
    const w = this.#writer();
    try {
      w.close();
      this.#drain();
    } finally {
      w.delete();
      this.#w = null;
    }
    if (this.#onData) return undefined;
    const out = concat(this.#chunks, Uint8Array);
    this.#chunks = [];
    return out;
  }

  get framesWritten() { return this.#writer().framesWritten; }
  get framesCommitted() { return this.#writer().framesCommitted; }
}

/**
 * Compress positions, nframes * natoms * 3 values in lengthUnit, with optional
 * cells (9 per frame), to the bytes of a .mdz file.
 */
export async function compress(positions, options = {}) {
  const { natoms } = options;
  if (!ArrayBuffer.isView(positions)) positions = Float32Array.from(positions);
  const cells = options.cells && (ArrayBuffer.isView(options.cells) ? options.cells : Float64Array.from(options.cells));
  if (!Number.isInteger(natoms) || natoms <= 0 || positions.length % (natoms * 3) !== 0)
    throw new MdzError('compress() needs natoms, with positions holding natoms * 3 values per frame', 'InvalidInput');
  const nframes = positions.length / (natoms * 3);
  if (cells && cells.length !== nframes * 9)
    throw new MdzError(`cells must hold 9 values per frame (${nframes * 9})`, 'InvalidInput');
  const w = await createWriter({ ...options, natoms: undefined, cells: undefined });
  for (let f = 0; f < nframes; ++f) {
    const frame = { positions: positions.subarray(f * natoms * 3, (f + 1) * natoms * 3) };
    if (cells) frame.cell = cells.subarray(9 * f, 9 * f + 9);
    w.write(frame);
  }
  return w.close();
}
