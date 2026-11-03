// Tests for the JavaScript reader. Fixtures are made by
// test/fixtures/make_fixtures.sh with the native CLI; the .npy files hold the
// native decoder's positions for the same .mdz files.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { MdzError, init, open, version } from '../lib/index.js';

const fixture = (name) => readFileSync(new URL(`./fixtures/${name}`, import.meta.url));

// A minimal .npy reader for little-endian float32 arrays.
function readNpy(name) {
  const buf = fixture(name);
  const headerLen = buf.readUInt16LE(8);
  const header = buf.subarray(10, 10 + headerLen).toString('latin1');
  assert.match(header, /'descr': '<f4'/);
  const shape = header.match(/'shape': \(([^)]*)\)/)[1].split(',').filter((s) => s.trim()).map(Number);
  const data = buf.subarray(10 + headerLen);
  const copy = new Uint8Array(data.length);
  copy.set(data);
  return { shape, data: new Float32Array(copy.buffer) };
}

// The wasm build must decode exactly what the native build does: the same
// float32 bits, in nm.
function assertMatchesNative(actual, expected) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; ++i)
    if (actual[i] !== expected[i]) assert.fail(`value ${i}: wasm ${actual[i]}, native ${expected[i]}`);
}

test('version is the library version', async () => {
  assert.match(await version(), /^\d+\.\d+\.\d+/);
});

test('header of a LAMMPS-derived file', async () => {
  const t = await open(fixture('nacl.mdz'));
  assert.equal(t.natoms, 512);
  assert.equal(t.nframes, 6);
  assert.equal(t.nGroups, 3);
  assert.equal(t.gofSize, 2);
  assert.equal(t.complete, true);
  assert.equal(t.variableAtoms, false);
  assert.deepEqual(t.columnNames, ['positions', 'velocities', 'box', 'origin', 'step', 'cell']);
  const vel = t.columns.find((c) => c.name === 'velocities');
  assert.deepEqual({ ...vel, precision: undefined }, {
    name: 'velocities', dtype: 'f32', width: 3, perFrame: false, lossy: true, precision: undefined, unit: 'Å/fs',
  });
  assert.ok(t.precision > 0);
  t.close();
});

test('every frame matches the native decoder', async () => {
  for (const name of ['nacl', 'dcd']) {
    const t = await open(fixture(`${name}.mdz`));
    const ref = readNpy(`${name}.positions.npy`);
    assert.deepEqual(ref.shape, [t.nframes, t.natoms, 3]);
    assertMatchesNative(await t.readFrames(), ref.data);
    const per = t.natoms * 3;
    for (let f = 0; f < t.nframes; ++f)
      assertMatchesNative(await t.readFrame(f), ref.data.subarray(f * per, (f + 1) * per));
    t.close();
  }
});

test('readFrame fills a caller buffer and returns it', async () => {
  const t = await open(fixture('nacl.mdz'));
  const out = new Float32Array(t.natoms * 3);
  const got = await t.readFrame(3, out);
  assert.equal(got, out);
  assert.deepEqual(out, await t.readFrame(3));
  await assert.rejects(t.readFrame(3, new Float32Array(5)), { name: 'MdzError', kind: 'InvalidInput' });
});

test('returned arrays are not invalidated by later reads', async () => {
  const t = await open(fixture('dcd.mdz'));
  const first = await t.readFrame(0);
  const kept = first.slice();
  for (let f = 0; f < t.nframes; ++f) await t.frame(f);
  await t.readColumn('positions');
  assert.deepEqual(first, kept);
});

test('negative indices count from the end, and out of range throws', async () => {
  const t = await open(fixture('nacl.mdz'));
  assert.deepEqual(await t.readFrame(-1), await t.readFrame(t.nframes - 1));
  await assert.rejects(t.readFrame(t.nframes), { kind: 'OutOfRange' });
  await assert.rejects(t.readFrame(-t.nframes - 1), { kind: 'OutOfRange' });
  await assert.rejects(t.readFrame(1.5), { kind: 'InvalidInput' });
});

test('readFrames clips the range', async () => {
  const t = await open(fixture('nacl.mdz'));
  assert.equal((await t.readFrames(4, 100)).length, 2 * t.natoms * 3);
  assert.equal((await t.readFrames(10, 20)).length, 0);
});

test('lengthUnit angstrom scales positions, cells and precision', async () => {
  const nm = await open(fixture('nacl.mdz'));
  const a = await open(fixture('nacl.mdz'), { lengthUnit: 'angstrom' });
  const pNm = await nm.readFrame(2);
  const pA = await a.readFrame(2);
  for (let i = 0; i < pNm.length; ++i) assert.ok(Math.abs(pA[i] - pNm[i] * 10) < 1e-5);
  const cNm = await nm.cell(2);
  const cA = await a.cell(2);
  for (let i = 0; i < 9; ++i) assert.ok(Math.abs(cA[i] - cNm[i] * 10) < 1e-5);
  assert.ok(Math.abs(a.precision - nm.precision * 10) < 1e-9);
  await assert.rejects(open(fixture('nacl.mdz'), { lengthUnit: 'bohr' }), { kind: 'InvalidInput' });
});

test('cell is a row-major 3x3 with rows a, b, c', async () => {
  // nacl.lammpstrj: an orthogonal box from -1.41005 to 21.150751 Å.
  const t = await open(fixture('nacl.mdz'), { lengthUnit: 'angstrom' });
  const c = await t.cell(0);
  assert.equal(c.length, 9);
  const L = 21.150751 + 1.41005;
  const expected = [L, 0, 0, 0, L, 0, 0, 0, L];
  for (let i = 0; i < 9; ++i) assert.ok(Math.abs(c[i] - expected[i]) < 1e-4, `cell[${i}] = ${c[i]}`);
});

test('frame() gives positions, cell, scalars and columns', async () => {
  const t = await open(fixture('nacl.mdz'));
  const fr = await t.frame(1);
  assert.equal(fr.natoms, 512);
  assert.deepEqual(fr.positions, await t.readFrame(1));
  assert.deepEqual(fr.cell, await t.cell(1));
  assert.equal(typeof fr.scalars.step, 'number');
  assert.ok(fr.columns.velocities instanceof Float32Array);
  assert.equal(fr.columns.velocities.length, 512 * 3);
  assert.equal(fr.ids, undefined);
});

test('readColumn follows the Python shapes', async () => {
  const t = await open(fixture('nacl.mdz'));
  const box = await t.readColumn('box');
  assert.deepEqual(box.shape, [6, 3, 3]);
  const cell = await t.readColumn('cell');
  assert.deepEqual(cell.shape, [6, 6]);
  assert.ok(Math.abs(cell.data[1] - 90) < 1e-9, 'gamma is 90 degrees');
  const step = await t.readColumn('step');
  assert.deepEqual(step.shape, [6]);
  assert.ok(step.data instanceof Float64Array);
  const vel = await t.readColumn('velocities', 2, 4);
  assert.deepEqual(vel.shape, [2, 512, 3]);
  const pos = await t.readColumn('positions', 0, 2);
  assert.deepEqual(pos.data, await t.readFrames(0, 2));
  await assert.rejects(t.readColumn('nope'), { kind: 'InvalidInput' });
  await assert.rejects(t.readColumn('mdz.atoms'), { kind: 'InvalidInput' });
});

test('atoms that come and go', async () => {
  const t = await open(fixture('gcmc.mdz'), { lengthUnit: 'angstrom' });
  assert.equal(t.variableAtoms, true);
  assert.equal(t.natoms, 0);
  assert.deepEqual(Array.from(t.frameNatoms), [3, 4, 2]);
  assert.deepEqual(Array.from(await t.atomIds(1)), [1, 2, 5, 9]);
  assert.deepEqual(Array.from(await t.atomIds(2)), [1, 9]);
  // Stored exactly: the values come back as written, in Å.
  assert.deepEqual(Array.from(await t.readFrame(2)), [1.2, 1, 1, 5.1, 5, 5].map(Math.fround));
  const fr = await t.frame(1);
  assert.deepEqual(Array.from(fr.ids), [1, 2, 5, 9]);
  assert.deepEqual(fr.elements, ['O', 'H', 'H', 'Na']);
  assert.deepEqual(Array.from(fr.types), [1, 2, 2, 3]);
  assert.equal(fr.scalars.step, 10);
  assert.equal(t.columnNames.includes('mdz.atoms'), false);
});

test('an unclosed file opens with its complete groups', async () => {
  const t = await open(fixture('nacl_truncated.mdz'));
  assert.equal(t.complete, false);
  assert.equal(t.nframes, 4);
  const full = await open(fixture('nacl.mdz'));
  assert.deepEqual(await t.readFrames(), await full.readFrames(0, 4));
});

test('inputs that are not .mdz files throw MdzError', async () => {
  await assert.rejects(open(new Uint8Array(100)), (e) => e instanceof MdzError && e.kind === 'Corrupt');
  await assert.rejects(open(new Uint8Array(0)), { name: 'MdzError' });
  const bytes = fixture('nacl.mdz');
  await assert.rejects(open(bytes.subarray(0, 20)), { name: 'MdzError' });
  await assert.rejects(open('nacl.mdz'), { kind: 'InvalidInput' });
});

test('failed opens release their memory', async () => {
  // Each failed open copies 16 MB into wasm memory before rejecting it. If
  // the copy were not freed, 40 failures would grow the heap by 640 MB.
  const core = await init();
  const garbage = new Uint8Array(16 << 20);
  await assert.rejects(open(garbage));
  const before = core.HEAPU8.length;
  for (let i = 0; i < 40; ++i) await assert.rejects(open(garbage), { kind: 'Corrupt' });
  assert.ok(core.HEAPU8.length - before <= 32 << 20, `heap grew by ${core.HEAPU8.length - before} bytes`);
});

test('Blob, ArrayBuffer and typed-array views are accepted', async () => {
  const bytes = fixture('nacl.mdz');
  const ab = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  for (const input of [new Blob([bytes]), ab, new DataView(ab)]) {
    const t = await open(input);
    assert.equal(t.nframes, 6);
    t.close();
  }
});

test('a closed trajectory refuses to read', async () => {
  const t = await open(fixture('nacl.mdz'));
  t.close();
  t.close();
  assert.throws(() => t.nframes, { kind: 'InvalidInput' });
});
