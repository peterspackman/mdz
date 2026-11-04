// Writing .mdz, and parsing and converting other formats.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { compress, convert, createWriter, formats, open, parse } from '../lib/index.js';

const fixture = (name) => readFileSync(new URL(`./fixtures/${name}`, import.meta.url));
const example = (name) => readFileSync(new URL(`../../../examples/${name}`, import.meta.url));

function frames(natoms, nframes) {
  const out = [];
  for (let f = 0; f < nframes; ++f) {
    const p = new Float32Array(natoms * 3);
    for (let i = 0; i < p.length; ++i) p[i] = (i % 17) * 0.11 + 0.003 * f * ((i % 3) + 1);
    out.push(p);
  }
  return out;
}

function maxDiff(a, b) {
  assert.equal(a.length, b.length);
  let m = 0;
  for (let i = 0; i < a.length; ++i) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

const CELL = [2, 0, 0, 0.3, 2.5, 0, 0, 0, 3];

test('written frames read back within the error bound', async () => {
  const input = frames(50, 25);
  const w = await createWriter({ gofSize: 10, elements: Array.from({ length: 50 }, (_, i) => (i % 2 ? 'H' : 'O')) });
  input.forEach((positions, f) => w.write({ positions, cell: CELL, time: f * 0.5, energy: -f, step: f * 100 }));
  assert.equal(w.framesWritten, 25);
  const t = await open(w.close());
  assert.equal(t.nframes, 25);
  assert.equal(t.nGroups, 3);
  assert.equal(t.metadata.elements[1], 'H');
  for (let f = 0; f < 25; ++f) assert.ok(maxDiff(await t.readFrame(f), input[f]) <= 4 * 0.000125 + 1e-7);
  const fr = await t.frame(7);
  assert.deepEqual(fr.scalars, { time: 3.5, energy: -7, step: 700 });
  assert.deepEqual(Array.from(fr.cell), CELL.map(Math.fround));
});

test('lossless writing is exact, and lengthUnit applies to input', async () => {
  const input = frames(20, 6);
  const w = await createWriter({ lossless: true, lengthUnit: 'angstrom' });
  for (const positions of input) w.write({ positions, cell: CELL });
  const t = await open(w.close(), { lengthUnit: 'angstrom' });
  assert.equal(t.precision, 0);
  for (let f = 0; f < 6; ++f) assert.ok(maxDiff(await t.readFrame(f), input[f]) < 1e-5);
  // In nm, lossless round-trips every bit.
  const nm = await open(await compress(input[0], { natoms: 20, lossless: true }));
  assert.deepEqual(await nm.readFrame(0), input[0]);
});

test('onData receives the file as it is written', async () => {
  const chunks = [];
  const w = await createWriter({ gofSize: 2, onData: (b) => chunks.push(b) });
  const input = frames(10, 5);
  input.forEach((positions) => w.write({ positions }));
  assert.ok(chunks.length >= 2, 'complete groups are handed over before close');
  assert.equal(w.framesCommitted, 4);
  assert.equal(w.close(), undefined);
  const bytes = new Uint8Array(chunks.reduce((n, c) => n + c.length, 0));
  let at = 0;
  for (const c of chunks) {
    bytes.set(c, at);
    at += c.length;
  }
  assert.equal((await open(bytes)).nframes, 5);
});

test('per-atom columns and extra scalars', async () => {
  const w = await createWriter({ lossless: true });
  const charge = Float32Array.from([0.5, -0.5, 1]);
  const image = Int32Array.from([0, 1, -1]);
  w.write({ positions: new Float32Array(9), charge, columns: { image }, scalars: { pressure: 1.5 } });
  const fr = await (await open(w.close())).frame(0);
  assert.deepEqual(fr.columns.charge, charge);
  assert.deepEqual(fr.columns.image, image);
  assert.equal(fr.scalars.pressure, 1.5);
});

test('atoms that come and go', async () => {
  const w = await createWriter({ variableAtoms: true, lossless: true, gofSize: 2 });
  w.write({ positions: [0, 0, 0, 1, 1, 1], ids: [3, 7], elements: ['O', 'H'] });
  w.write({ positions: [0, 0, 0.1, 2, 2, 2, 1, 1, 1.1], ids: [3, 9, 7], elements: ['O', 'Na', 'H'] });
  w.write({ positions: [2, 2, 2.2], ids: [9], elements: ['Na'] });
  const t = await open(w.close());
  assert.equal(t.variableAtoms, true);
  assert.deepEqual(Array.from(await t.frameNatoms()), [2, 3, 1]);
  const u = await t.atomUniverse();
  assert.deepEqual(Array.from(u.ids), [3, 7, 9]);
  assert.deepEqual(u.elements, ['O', 'H', 'Na']);
  assert.deepEqual(Array.from(await t.readFrame(2)), [2, 2, 2.2].map(Math.fround));
});

test('writer errors', async () => {
  const w = await createWriter();
  w.write({ positions: new Float32Array(6) });
  assert.throws(() => w.write({ positions: new Float32Array(9) }), { kind: 'InvalidInput' });
  assert.throws(() => w.write({ positions: new Float32Array(6), ids: [1, 2] }), { kind: 'InvalidInput' });
  assert.throws(() => w.write({}), { kind: 'InvalidInput' });
  w.close();
  assert.throws(() => w.write({ positions: new Float32Array(6) }), { kind: 'InvalidInput' });
  await assert.rejects(createWriter({ gofSize: 0 }), { kind: 'InvalidInput' });
  await assert.rejects(createWriter({ variableAtoms: true, elements: ['H'] }), { kind: 'InvalidInput' });
  const empty = await createWriter();
  assert.throws(() => empty.close(), { kind: 'InvalidInput' });
});

test('compress takes packed frames and cells', async () => {
  const input = frames(8, 4);
  const packed = new Float32Array(8 * 3 * 4);
  input.forEach((p, f) => packed.set(p, f * 24));
  const cells = new Float32Array(36);
  for (let f = 0; f < 4; ++f) cells.set(CELL, 9 * f);
  const t = await open(await compress(packed, { natoms: 8, cells, gofSize: 3 }));
  assert.equal(t.nframes, 4);
  assert.equal(t.nGroups, 2);
  assert.ok(maxDiff(await t.readFrames(), packed) < 1e-3);
  await assert.rejects(compress(packed, { natoms: 7 }), { kind: 'InvalidInput' });
});

test('formats lists what can be read and written', async () => {
  const list = await formats();
  const byName = Object.fromEntries(list.map((f) => [f.name, f]));
  for (const name of ['mdz', 'xyz', 'extxyz', 'lammps', 'dcd', 'xtc', 'pdb', 'npy']) {
    assert.ok(byName[name], name);
    assert.equal(byName[name].writable, true);
  }
  assert.ok(byName.lammps.extensions.includes('lammpstrj'));
});

test('parse a LAMMPS dump', async () => {
  const f = await parse(fixture('gcmc.lammpstrj'), { format: 'lammps', lengthUnit: 'angstrom' });
  assert.equal(f.nframes, 3);
  assert.equal(f.variableAtoms, true);
  assert.deepEqual(Array.from(f.frameNatoms), [3, 4, 2]);
  assert.deepEqual(Array.from(f.ids), [1, 2, 5, 1, 2, 5, 9, 1, 9]);
  assert.deepEqual(f.elements.slice(3, 7), ['O', 'H', 'H', 'Na']);
  assert.ok(maxDiff(f.positions.subarray(0, 3), [1, 1, 1]) < 1e-5);
  assert.ok(maxDiff(f.cells.subarray(0, 9), [10, 0, 0, 0, 10, 0, 0, 0, 10]) < 1e-5);
  assert.deepEqual(Array.from(f.scalars.step), [0, 10, 20]);
});

test('parse takes text and detects the format from a file name', async () => {
  const xyz = '2\nwater bits\nO 0 0 0\nH 0.96 0 0\n2\n\nO 0 0 0.1\nH 0.96 0 0.1\n';
  const f = await parse(xyz, { format: 'xyz', lengthUnit: 'angstrom' });
  assert.equal(f.nframes, 2);
  assert.equal(f.natoms, 2);
  assert.deepEqual(f.metadata.atomNames, ['O', 'H']);
  assert.ok(Math.abs(f.positions[8] - 0.1) < 1e-6); // frame 1, atom 0, z
  const file = new File([xyz], 'water.xyz');
  assert.equal((await parse(file)).nframes, 2);
  await assert.rejects(parse(xyz), { kind: 'InvalidInput' });
  await assert.rejects(parse(xyz, { format: 'nope' }), { kind: 'InvalidInput' });
  await assert.rejects(parse('garbage', { format: 'xyz' }), { name: 'MdzError' });
});

test('convert a LAMMPS dump to .mdz and back to xyz', async () => {
  const dump = example('lammps/nacl.lammpstrj');
  const mdzBytes = await convert(new File([dump], 'nacl.lammpstrj'), { to: 'mdz', gofSize: 2 });
  const t = await open(mdzBytes, { lengthUnit: 'angstrom' });
  assert.equal(t.nframes, 6);
  assert.equal(t.nGroups, 3);
  const parsed = await parse(dump, { format: 'lammps', lengthUnit: 'angstrom' });
  assert.ok(maxDiff(await t.readFrames(), parsed.positions) < 0.005 + 1e-4);
  const xyz = new TextDecoder().decode(await convert(mdzBytes, { from: 'mdz', to: 'xyz' }));
  assert.equal(xyz.split('\n')[0].trim(), '512');
  assert.equal((await parse(xyz, { format: 'xyz' })).nframes, 6);
});

test('convert to .mdz restarts when the atoms change', async () => {
  const bytes = await convert(fixture('gcmc.lammpstrj'), { from: 'lammps', to: 'mdz', lossless: true });
  const t = await open(bytes);
  assert.equal(t.variableAtoms, true);
  assert.deepEqual(Array.from(await t.frameNatoms()), [3, 4, 2]);
});

test('convert errors', async () => {
  await assert.rejects(convert('x', { from: 'xyz' }), { kind: 'InvalidInput' });
  await assert.rejects(convert(fixture('gcmc.lammpstrj'), { from: 'lammps', to: 'dcd' }), { name: 'MdzError' });
});
