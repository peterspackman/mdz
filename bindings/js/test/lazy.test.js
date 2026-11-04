// Range reads, caching and the frame-access helpers.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { blobSource, open } from '../lib/index.js';

const fixture = (name) => readFileSync(new URL(`./fixtures/${name}`, import.meta.url));

// A byte source that records every read.
function counting(bytes) {
  const src = {
    size: bytes.length,
    reads: [],
    async read(start, end) {
      src.reads.push([start, end]);
      return bytes.subarray(start, end);
    },
  };
  src.bytesRead = () => src.reads.reduce((n, [a, b]) => n + b - a, 0);
  return src;
}

test('opening reads the head, trailer and footer only', async () => {
  const bytes = fixture('nacl.mdz');
  const src = counting(bytes);
  const t = await open(src);
  assert.equal(t.nframes, 6);
  assert.equal(src.reads.length, 3);
  assert.ok(src.reads[0][1] <= bytes.length);
  // The head guess covers the whole of this small file, so only the
  // group reads that follow tell us anything.
  const before = src.reads.length;
  await t.readFrame(3);
  assert.equal(src.reads.length, before + 1);
  const [a, b] = src.reads.at(-1);
  assert.ok(b - a < bytes.length / 2, 'one group, not the file');
});

test('a Blob is read in ranges', async () => {
  const bytes = fixture('dcd.mdz');
  const slices = [];
  const blob = new Blob([bytes]);
  const orig = blob.slice.bind(blob);
  blob.slice = (a, b) => {
    slices.push([a, b]);
    return orig(a, b);
  };
  const t = await open(blob);
  assert.deepEqual(await t.readFrame(1), await (await open(bytes)).readFrame(1));
  assert.ok(slices.every(([a, b]) => b - a < bytes.length || a === 0));
  assert.equal(typeof blobSource(blob).read, 'function');
});

test('peek is null until the group is decoded', async () => {
  const t = await open(fixture('nacl.mdz'));
  assert.equal(t.peek(4), null);
  assert.equal(t.peekFrame(4), null);
  const p = await t.readFrame(4);
  assert.deepEqual(t.peek(4), p);
  assert.deepEqual(t.peek(5), await t.readFrame(5), 'same group');
  assert.equal(t.peek(0), null, 'another group');
  const out = new Float32Array(p.length);
  assert.equal(t.peek(4, out), out);
  assert.deepEqual(t.peekFrame(4), await t.frame(4));
});

test('concurrent reads of one group share a single read', async () => {
  const src = counting(fixture('nacl.mdz'));
  const t = await open(src);
  const before = src.reads.length;
  await Promise.all([t.readFrame(2), t.readFrame(3), t.frame(2), t.cell(3)]);
  assert.equal(src.reads.length, before + 1);
});

test('the cache keeps to its budget and evicts the oldest group', async () => {
  const src = counting(fixture('dcd.mdz'));
  // dcd.mdz: 4096 atoms; a decoded group of 3 frames is about 150 kB.
  const t = await open(src, { cacheBytes: 1 });
  await t.readFrame(0);
  const one = t.cachedBytes;
  assert.ok(one > 0, 'the last group is always kept');
  const src2 = counting(fixture('nacl.mdz'));
  const u = await open(src2, { cacheBytes: 0 });
  await u.readFrame(0);
  await u.readFrame(2); // another group evicts the first
  assert.equal(u.peek(0), null);
  assert.notEqual(u.peek(2), null);
  const reads = src2.reads.length;
  await u.readFrame(0);
  assert.equal(src2.reads.length, reads + 1, 'evicted group is read again');
});

test('prefetch runs backwards and wraps around', async () => {
  const t = await open(fixture('nacl.mdz')); // groups of 2 frames: [0,1] [2,3] [4,5]
  await t.prefetch(1, 3, { direction: -1, wrap: true }); // frames 1, 0, 5
  assert.notEqual(t.peek(0), null);
  assert.notEqual(t.peek(5), null);
  assert.equal(t.peek(2), null);
  const u = await open(fixture('nacl.mdz'));
  await u.prefetch(5, 3); // no wrap: frame 5 only
  assert.notEqual(u.peek(5), null);
  assert.equal(u.peek(0), null);
});

test('results are copies that later reads and eviction do not change', async () => {
  const t = await open(fixture('nacl.mdz'), { cacheBytes: 0 });
  const a = await t.readFrame(0);
  const kept = a.slice();
  a[0] = 1234;
  await t.readFrame(4);
  assert.deepEqual(await t.readFrame(0), kept);
});

test('atomUniverse of a file whose atoms come and go', async () => {
  const t = await open(fixture('gcmc.mdz'));
  const u = await t.atomUniverse();
  assert.deepEqual(Array.from(u.ids), [1, 2, 5, 9]);
  assert.deepEqual(u.elements, ['O', 'H', 'H', 'Na']);
  assert.deepEqual(Array.from(u.types), [1, 2, 2, 3]);
});

test('atomUniverse of a fixed file comes from the metadata', async () => {
  const t = await open(fixture('nacl.mdz'));
  const u = await t.atomUniverse();
  assert.equal(u.ids.length, 512);
  assert.deepEqual(u.ids, t.metadata.atomIds);
  assert.deepEqual(u.ids, await t.atomIds(0));
});

test('per-frame scalars over the whole file', async () => {
  const t = await open(fixture('nacl.mdz'));
  const step = await t.readColumn('step');
  assert.equal(step.data.length, 6);
  for (let f = 0; f < 6; ++f) assert.equal(step.data[f], (await t.frame(f)).scalars.step);
});

test('a source that returns the wrong number of bytes is an Io error', async () => {
  const bytes = fixture('nacl.mdz');
  const src = { size: bytes.length, read: (a, b) => bytes.subarray(a, Math.max(a, b - 1)) };
  await assert.rejects(open(src), { kind: 'Io' });
});

test('a damaged group fails when read, not when opened', async () => {
  const bytes = new Uint8Array(fixture('nacl.mdz'));
  const t0 = await open(bytes);
  const n = bytes.length;
  bytes[Math.floor(n / 2)] ^= 0xff;
  const t = await open(bytes);
  assert.equal(t.nframes, t0.nframes);
  const results = await Promise.allSettled([0, 2, 4].map((f) => t.readFrame(f)));
  const failed = results.filter((r) => r.status === 'rejected');
  assert.equal(failed.length, 1);
  assert.equal(failed[0].reason.kind, 'Corrupt');
});
