// Types for @peterspackman/mdz. Keep in sync with lib/index.js.

export type ErrorKind = 'InvalidInput' | 'Corrupt' | 'Io' | 'OutOfRange' | 'OutOfMemory' | 'Internal';

export class MdzError extends Error {
  constructor(message: string, kind?: ErrorKind);
  readonly name: 'MdzError';
  readonly kind: ErrorKind;
}

export type LengthUnit = 'nm' | 'angstrom';

export interface OpenOptions {
  /** Unit of positions, cells and precision. Default 'nm'. */
  lengthUnit?: LengthUnit;
  /** Budget for decoded groups kept in memory. Default 256 MiB. */
  cacheBytes?: number;
}

/** Random access to the bytes of a file. */
export interface ByteSource {
  size: number;
  read(start: number, end: number): Uint8Array | Promise<Uint8Array>;
}

export function blobSource(blob: Blob): ByteSource;

/** Emscripten module options, such as `locateFile` or `wasmBinary`. */
export interface InitOptions {
  locateFile?: (path: string, prefix: string) => string;
  wasmBinary?: ArrayBuffer | Uint8Array;
  instantiateWasm?: (imports: WebAssembly.Imports, done: (instance: WebAssembly.Instance) => void) => unknown;
}

export function init(options?: InitOptions): Promise<unknown>;
export function version(): Promise<string>;
export function open(
  input: Uint8Array | ArrayBuffer | ArrayBufferView | Blob | ByteSource,
  options?: OpenOptions,
): Promise<Trajectory>;

export type Dtype = 'f32' | 'f64' | 'i32' | 'u32' | 'u8' | 'i64';

export interface ColumnInfo {
  name: string;
  dtype: Dtype;
  /** Values per atom, or per frame for per-frame columns. */
  width: number;
  perFrame: boolean;
  lossy: boolean;
  /** Quantization step in the column's unit; 0 if stored exactly. */
  precision: number;
  unit: string;
}

export interface Metadata {
  atomNames?: string[];
  elements?: string[];
  atomTypes?: Int32Array;
  atomIds?: Int32Array;
  residues?: string[];
  residueIds?: Int32Array;
  chainIds?: string[];
  hetatm?: Uint8Array;
  /** Rigid body index of each atom, or -1 for none. */
  bodies?: Int32Array;
  sourceFormat?: string;
  attributes: Record<string, string>;
}

export interface Frame {
  natoms: number;
  /** natoms * 3 values in lengthUnit. */
  positions: Float32Array;
  /** 3x3 row-major, rows a, b, c, in lengthUnit; null if the file has none. */
  cell: Float32Array | null;
  /** Lower corner of the box, in lengthUnit, if stored. */
  origin?: Float32Array;
  /** Per-frame values such as time, energy and step. */
  scalars: Record<string, number>;
  /** Per-atom columns other than positions, in their stored units. */
  columns: Record<string, Float32Array | Int32Array>;
  /** Only when the atom count varies: the frame's atom ids and attributes. */
  ids?: Int32Array;
  elements?: string[];
  names?: string[];
  types?: Int32Array;
}

export interface Column {
  name: string;
  dtype: 'f32' | 'f64' | 'i32' | 'i64';
  shape: number[];
  /** Flat, row-major. int64 columns are returned as Float64Array. */
  data: Float32Array | Float64Array | Int32Array;
}

export interface AtomUniverse {
  /** Sorted atom ids. The other arrays are parallel to it. */
  ids: Int32Array;
  elements?: string[];
  names?: string[];
  types?: Int32Array;
}

export interface PrefetchOptions {
  direction?: 1 | -1;
  wrap?: boolean;
}

export class Trajectory implements Disposable {
  readonly lengthUnit: LengthUnit;
  /** Atoms per frame, or 0 if the count varies. */
  readonly natoms: number;
  readonly nframes: number;
  readonly variableAtoms: boolean;
  /** Whether the writer closed the file. */
  readonly complete: boolean;
  /** Quantization step of positions in lengthUnit, or 0 if exact. */
  readonly precision: number;
  readonly gofSize: number;
  readonly nGroups: number;
  readonly columnNames: string[];
  readonly columns: ColumnInfo[];
  readonly metadata: Metadata;
  /** Bytes of decoded groups held in the cache. */
  readonly cachedBytes: number;

  /** Negative frame indices count from the end. */
  readFrame(frame: number, out?: Float32Array): Promise<Float32Array>;
  /** Positions of a frame whose group is already decoded, else null. */
  peek(frame: number, out?: Float32Array): Float32Array | null;
  frame(frame: number): Promise<Frame>;
  peekFrame(frame: number): Frame | null;
  readFrames(start?: number, stop?: number, out?: Float32Array): Promise<Float32Array>;
  cell(frame: number): Promise<Float32Array | null>;
  atomIds(frame: number): Promise<Int32Array>;
  prefetch(frame: number, count?: number, options?: PrefetchOptions): Promise<void>;
  /** Lengths (positions, box, cell, origin) are in lengthUnit; others in stored units. */
  readColumn(name: string, start?: number, stop?: number): Promise<Column>;
  frameNatoms(): Promise<Float64Array>;
  atomUniverse(): Promise<AtomUniverse>;
  close(): void;
  [Symbol.dispose](): void;
}
