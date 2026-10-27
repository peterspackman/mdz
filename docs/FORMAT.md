# The .mdz file format

This document specifies version 4 of the `.mdz` container. It covers the
layout, how writers append, how readers recover from a crash, the column and
metadata conventions, and the units.

Within a group, each column's data is an opaque *blob*. The blob encodings
(temporal prediction, the XTC-style intra codec, the lossless codec, rigid
bodies and the simple column codecs) are defined by the reference
implementation in the following files:

- `src/codec.cpp`: the group codecs
- `src/kernels.cpp`: quantization, zigzag and varints
- `src/predict.hpp`: the polynomial predictor
- `src/xdr3d.cpp`: the XTC coordinate codec
- `src/rigid.cpp`: rigid bodies
- `src/variable.cpp`: membership for varying atom counts
- `src/columns.cpp`: the simple column codecs

A coordinate group's blob is zstd(`3 | order u8 | codec u8 | payload`). The
codec byte is 0 for temporal, 1 for intra (XTC-style), 2 for lossless and 3
for rigid bodies. Decoding a rigid-body group requires the metadata key
`bodies` (see Metadata). Its payload holds the bodies' reference geometry,
their centres and orientations, the free atoms, and the residual of every body
atom. Each of these parts is itself a coordinate payload.

All integers are little-endian unless stated otherwise. CRCs are CRC-32 (the
IEEE polynomial, as used by zlib and gzip).

## Layout

```text
header   magic "MDZ\x04" | version u16 = 4 | flags u16 | natoms u64
         | schema_len u32 | schema_crc u32                         (24 bytes)
schema   schema_len bytes, CRC-32 schema_crc
record*  "GOF\x01" | body_len u32 | body | crc32(body) u32
footer   "IDX\x01" | body_len u64 | body | crc32(body) u32         (on close)
trailer  footer_offset u64 | "MDZ4END\0"                           (16 bytes)
```

The file is written front to back as frames arrive. No earlier part of the
file is ever rewritten, so any prefix of the file can be read.

### Header

| field | type | meaning |
|---|---|---|
| magic | 4 bytes | `4D 44 5A 04` (`"MDZ\x04"`) |
| version | u16 | 4 |
| flags | u16 | bit 0: the atom count varies (see Varying atom counts). Readers refuse a file with any flag they do not recognise. |
| natoms | u64 | atoms in every frame; 0 if the count varies |
| schema_len | u32 | length of the schema in bytes |
| schema_crc | u32 | CRC-32 of the schema bytes |

### Schema

The schema holds everything that is fixed when the file is created:

```text
gof_size u32                 nominal frames per group (informational)
n_columns u16
n_columns × descriptor
meta_len u32 | meta          metadata block (see Metadata), raw or as a zstd frame; meta_len 0 = none
```

A column descriptor:

| field | type | meaning |
|---|---|---|
| name_len, name | u8, UTF-8 | column name, e.g. `positions` |
| dtype | u8 | 0 F32, 1 F64, 2 I32, 3 U32, 4 U8, 5 I64 |
| width | u8 | values per atom, or per frame for per-frame columns |
| per_frame | u8 | 1 = one row of `width` values per frame; 0 = per atom per frame |
| lossy | u8 | 1 if values were quantized |
| precision | f32 | quantization step, in the column's unit (0 = lossless) |
| pred_order | u8 | predictor order (0 = auto); informational |
| codec_flags | u8 | bit 0 adaptive precision, bit 1 B-frames, bit 2 spatial delta |
| anchor_step | u8 | B-frame anchor spacing (1 = none) |
| unit_len, unit | u8, UTF-8 | unit label, e.g. `nm`, `Å`, `kcal/mol/Å` |

With adaptive precision (bit 0), the maximum error of a lossy column is up
to 4 × `precision`. Without it, the maximum error is `precision` / 2.

### Group records

Each record holds one group of consecutive frames, with one blob per schema
column, in schema order.

```text
"GOF\x01" | body_len u32 | body | crc32(body) u32
body = first_frame u64 | nframes u32 | n_columns × (blob_len u32 | blob)
```

- `first_frame` must equal the number of frames in all earlier records, so
  groups are contiguous.
- `nframes` is at least 1. Groups need not be `gof_size` frames long, since
  a writer may flush a short group at any time.
- Every record can be decoded on its own, including its per-frame columns.
  This is what makes random access and crash recovery possible.
- `body_len` covers the body only. The record occupies `body_len + 12` bytes.

### Footer and trailer

When a writer closes the file, it appends an index of the records and a
fixed-size trailer.

```text
"IDX\x01" | body_len u64 | body | crc32(body) u32
body = n_groups u64 | n_groups × (offset u64 | first_frame u64 | nframes u32)
trailer = footer_offset u64 | "MDZ4END\0"
```

`offset` is the byte offset of each record's `GOF\x01` magic, and
`footer_offset` is the offset of `IDX\x01`.

## Reading

1. Check the header, the version and the flags. Read the schema and check
   its CRC.
2. Look for a valid trailer and footer. The footer is valid if all of the
   following hold. The last 16 bytes end in `MDZ4END\0`. `footer_offset`
   points to `IDX\x01` after the schema. The footer CRC matches. The footer
   ends exactly where the trailer starts. The body length is
   `8 + 20 × n_groups`.
3. **Complete file** (valid footer). Parse records from the end of the schema
   up to `footer_offset`. Every record must be valid, and the records must
   match the footer index exactly. Anything else is corruption.
4. **Incomplete file** (no valid footer). The writer crashed or is still
   running. Parse records from the end of the schema, and stop at the first
   one that is truncated, fails its CRC, or does not continue the frame
   count. The groups before that point make up the trajectory. Any bytes
   after it are a partial final write.

## Appending

To continue a file, read it as above and truncate it just past the last
valid record, which removes any footer or partial final write. Then write
further records with `first_frame` continuing the count, and a new footer on
close. The schema
(columns, codec settings, metadata) must stay the same. A writer must refuse
frames whose encoding would need a different schema.

## Columns

Columns are identified by name. Readers ignore columns they do not recognise.

| name | dtype × width | kind | meaning |
|---|---|---|---|
| `positions` | F32 × 3 | per atom | coordinates; required |
| `velocities` | F32 × 3 | per atom | |
| `forces` | F32 × 3 | per atom | |
| `image` | I32 × 3 | per atom | periodic image flags |
| `box` | F64 × 9 | per frame | cell vectors a, b, c as the rows of a row-major matrix |
| `origin` | F64 × 3 | per frame | the box's lower corner |
| `time` | F64 × 1 | per frame | simulation time, always in ps |
| `step` | I64 × 1 | per frame | timestep number, exact |
| `energy` | F64 × 1 | per frame | |

Any other per-atom column, float or integer, or per-frame property may be
present under its own name.

## Varying atom counts

With header flag bit 0 set, frames may hold different atoms (grand-canonical
runs, deposition, evaporation, a dump of a region). Atoms are identified by
an integer id. Each group of frames is stored over the *union* of the atoms
present in any of its frames, in ascending id order, exactly as a
fixed-count group of that many atoms: every per-atom blob holds values for
every atom of the union in every frame of the group. An atom absent from a
frame is given values of the encoder's choosing, which readers ignore. mdz
repeats the values from the nearest frame that has the atom, so they cost
almost nothing.

The column `mdz.atoms` (I64 × 1, per atom) records, for each group, which
atoms are present in which frames. Its blob is zstd-compressed and contains
the following.

```text
0x25 | count u32 | count ids, ascending, as zigzag LEB128 deltas (the first from 0)
     | nframes u32 | nframes rows of ceil(count / 8) bytes:
                     bit a (LSB first) of row f set if atom a is in frame f
     | nstrings u32 | nstrings × (len u16 | UTF-8)
```

`nframes` must equal the group's frame count. The strings form the table
that the group's per-atom string columns index into. Static per-atom metadata
cannot describe atoms that come and go, so in these files it is stored as
per-atom I32 columns instead.

| column | instead of |
|---|---|
| `mdz.type` | `atom_types` |
| `mdz.resid` | `residue_ids` |
| `mdz.hetatm` | `hetatm` |
| `mdz.name` | `atom_names` (index into the group's strings) |
| `mdz.element` | `elements` (index) |
| `mdz.resname` | `residues` (index) |
| `mdz.chain` | `chain_ids` (index) |

Atom ids are not stored as a separate column, because the membership already
holds them.

## Units

By default, values are in mdz's internal units:

| quantity | unit |
|---|---|
| length (positions, box, origin) | nm |
| velocity | nm/ps |
| force | kJ/mol/nm |
| time | ps |

A file may instead keep values in a LAMMPS unit style, so that they
round-trip exactly. In that case the metadata attribute `mdz.units` is
`lammps:<style>`, where `<style>` is one of `real`, `metal`, `si`, `cgs`,
`electron`, `micro`, `nano`, `lj`. Lengths, velocities and forces are then
in that style's units, and so are the `precision` fields of their columns.
The column `unit` labels give the same information in readable form. Time
is always in ps, whatever the style.

A reader converts to the units it wants. Files without `mdz.units` are in
internal units.

## Metadata

The schema's metadata block is a versioned list of keyed fields.

```text
version u8 = 1 | n_fields u8 | n_fields × (key_len u8 | key | tag u8 | payload)
```

Writers may store the block zstd-compressed. A block that begins with the
zstd frame magic `28 B5 2F FD` is a zstd frame containing the block above. A
raw block begins with its version byte. mdz compresses any block of 64 bytes
or more.

| tag | payload |
|---|---|
| 0 string array | count u32, then count × (len u16, UTF-8 bytes) |
| 1 i32 array | count u32, then count × i32 |
| 2 string | len u16, UTF-8 bytes |

Readers skip keys they do not recognise. An unknown tag is an error, because
the length of its payload cannot be determined.

Known keys:

| key | tag | meaning |
|---|---|---|
| `atom_names` | 0 | per atom |
| `elements` | 0 | per atom |
| `atom_types` | 1 | per atom |
| `atom_ids` | 1 | per atom |
| `residues` | 0 | per atom |
| `residue_ids` | 1 | per atom |
| `chain_ids` | 0 | per atom |
| `hetatm` | 1 | per atom (0/1) |
| `bodies` | 1 | per atom: the rigid body it belongs to (-1: none), for rigid-body groups |
| `source_format` | 2 | the format the trajectory was read from |
| `attr.<name>` | 2 | free-form attribute `<name>` |

Attributes in use:

| attribute | meaning |
|---|---|
| `mdz.units` | `lammps:<style>`, the unit style the values are kept in (see Units) |
| `lammps.units` | the LAMMPS unit style of the source |
| `lammps.boundary` | boundary flags, e.g. `pp pp ff` |
| `lammps.coords` | `wrapped`, `unwrapped`, `scaled` or `scaled-unwrapped` |
| `lammps.box` | `orthogonal`, `triclinic` or `general` |
| `lammps.columns` | the source dump's `ITEM: ATOMS` column names |
