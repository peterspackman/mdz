/*
 * C API for writing and reading .mdz molecular dynamics trajectories.
 *
 * Writing. Each group of frames is compressed and appended to the file as
 * soon as it is complete. A program that dies, such as a simulation killed at
 * its wall-time limit, leaves a file holding every complete group. After a
 * restart, the program can append to that file.
 *
 * Units. Values go in and come out in one LAMMPS unit style ("real",
 * "metal", "si", "cgs", "electron", "micro", "nano" or "lj"), which is named
 * when the writer is opened and recorded in the file. Lengths (positions,
 * box, origin and precision) are in the style's length unit, and velocities
 * and forces in its velocity and force units. The file stores values in the
 * style as given, so reading them back in the same style returns what was
 * written, up to the requested precision. Per-atom values are stored in
 * single precision, so with `lossless` each value reads back rounded to the
 * nearest float (a relative error below 6e-8). The box, origin, step and
 * time are kept in full double precision.
 *
 * Errors. Functions return MDZ_OK (0) on success and a negative MDZ_ERR_*
 * code on failure. mdz_last_error() describes the most recent failure on
 * the calling thread. No function throws or aborts the process; internal
 * errors are reported as MDZ_ERR_INTERNAL.
 *
 * Threads. Handles are not thread-safe, so use each from one thread at a
 * time. Each call compresses or decompresses at most one group of frames, on
 * the calling thread. The C API therefore starts no threads, whatever worker
 * count is set with mdz::set_num_threads or MDZ_NUM_THREADS.
 *
 * Linking. The implementation is C++ (libmdz, the mdz::mdz CMake target).
 * When linking a C program against the static library, link with a C++
 * compiler or add the C++ standard library and -lpthread.
 *
 * ABI. The option and frame structs start with a `struct_size` member, which
 * the *_init functions set. It lets later versions add fields at the end of
 * these structs and still accept callers built against this header. A struct
 * smaller than the one declared here is refused with MDZ_ERR_INVALID.
 *
 * Varying atom counts. A writer opened with `variable_atoms` accepts frames
 * whose atom count changes, for example as atoms are lost, deleted or
 * inserted. Each frame gives its atom count and the atoms' IDs, which are
 * used to match atoms across frames. Read such a file one frame at a time,
 * sizing buffers with mdz_reader_frame_natoms.
 *
 * Columns. Besides the fields mdz_frame names, a writer can store any number
 * of general per-atom columns, such as per-atom energies or stresses,
 * declared in mdz_writer_options.columns. Each is stored exactly unless it is
 * given a precision. Their values are in whatever unit the caller uses, and
 * are never converted between unit styles.
 *
 * Rigid bodies. A writer opened with `rigid` also codes positions as rigid
 * bodies (a centre and orientation per body and frame, plus residuals) and
 * keeps whichever coding is smaller for each group. The error bound is the
 * same either way. Bodies are given with mdz_writer_set_bodies or detected
 * in the first group.
 */
#ifndef MDZ_H
#define MDZ_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MDZ_OK 0
/* An argument was invalid: for example a NULL where a value is required, an
 * unknown unit style, or a frame whose fields differ from the first frame's. */
#define MDZ_ERR_INVALID (-1)
/* Reading or writing the file failed, or it is not a valid .mdz file. */
#define MDZ_ERR_IO (-2)
/* A frame index past the end of the trajectory. */
#define MDZ_ERR_RANGE (-3)
/* An internal error or out-of-memory condition. Other than out of memory,
 * this indicates a bug in mdz. */
#define MDZ_ERR_INTERNAL (-4)

/* The library version, for example "0.4.0". */
const char *mdz_version(void);

/* A description of the last error on this thread, or "" if the last call
 * succeeded. The string is valid until the next mdz call on the same
 * thread. */
const char *mdz_last_error(void);

/* --- Writing --- */

typedef struct mdz_writer mdz_writer;

/* Value types of a general column. */
#define MDZ_FLOAT 0 /* passed as double, stored as float */
#define MDZ_INT 1   /* int32_t */

/* A general per-atom column. */
typedef struct mdz_column {
    /* A unique name, other than positions, velocities, forces and image,
     * and not starting with "mdz.". */
    const char *name;
    /* MDZ_FLOAT or MDZ_INT. */
    int type;
    /* Values per atom, from 1 to 255. */
    uint32_t width;
    /* For MDZ_FLOAT, the quantization step in the column's own unit, so the
     * error is at most half of it, or 0 to store each value exactly as a
     * float. Must be 0 for MDZ_INT. Ignored with `lossless`. */
    double precision;
} mdz_column;

typedef struct mdz_writer_options {
    /* sizeof(mdz_writer_options), set by mdz_writer_options_init. */
    uint32_t struct_size;
    /* LAMMPS unit style of every value passed in. NULL means "real". */
    const char *units;
    /* Quantization step for positions, in the style's length unit. 0 selects
     * the default, which matches XTC: a maximum error of 0.005 Angstrom. With
     * adaptive precision (the default) the maximum error can reach 4 times
     * this step. Ignored when `lossless` is set. */
    double precision;
    /* Quantization steps for velocities and forces, in the style's units.
     * 0 selects the defaults, 0.000125 nm/ps and 0.000125 kJ/mol/nm. */
    double velocity_precision;
    double force_precision;
    /* Nonzero to store every value exactly as a float, giving larger files. */
    int lossless;
    /* Frames per independently compressed group. This sets the granularity
     * of random access and how many frames a crash can lose. 0 selects 20. */
    uint32_t gof_size;
    /* Nonzero to open an existing file and continue after its last complete
     * group. A missing file is created. The settings above must match those
     * the file was created with. */
    int append;
    /* LAMMPS boundary flags, for example "pp pp ff". NULL means "pp pp pp". */
    const char *boundary;
    /* Nonzero if positions are unwrapped (x + image * box). */
    int unwrapped;

    /* Nonzero to allow the atom count to change between frames. Every frame
     * must then set mdz_frame.natoms and mdz_frame.ids, the natoms passed to
     * mdz_writer_open is ignored, and mdz_writer_set_atoms cannot be used.
     * When appending, the file must have been written the same way. */
    int variable_atoms;
    /* Nonzero to model rigid bodies (see the top of this file). Bodies come
     * from mdz_writer_set_bodies, or are detected in the first group and
     * kept for the whole file. Ignored with `lossless`. Cannot be combined
     * with variable_atoms. */
    int rigid;
    /* ncolumns general columns that every frame carries, in the order of
     * mdz_frame.columns. The array is copied by mdz_writer_open. When
     * appending, give the file's columns again. */
    const mdz_column *columns;
    uint32_t ncolumns;
} mdz_writer_options;

/* Fill `opts` with defaults. Call this before setting any field. */
void mdz_writer_options_init(mdz_writer_options *opts);

/* One frame to write. A NULL pointer means the field is absent. The first
 * frame written fixes which fields are present, and every later frame must
 * match it. */
typedef struct mdz_frame {
    /* sizeof(mdz_frame), set by mdz_frame_init. */
    uint32_t struct_size;
    /* Timestep number. */
    int64_t step;
    /* Simulation time in the style's time unit. Used only if has_time is
     * nonzero. */
    double time;
    int has_time;
    /* The box: its origin (lower corner, 3 values) and the cell vectors a,
     * b and c as the rows of a row-major 3x3 matrix (9 values). The LAMMPS
     * restricted triclinic box is a = (xprd, 0, 0), b = (xy, yprd, 0) and
     * c = (xz, yz, zprd). Leave both NULL for no box. origin requires box;
     * a box without an origin has its origin at 0. */
    const double *origin;
    const double *box;
    /* natoms x 3 values each, atom by atom, in a fixed atom order (for
     * LAMMPS, sorted by atom ID). positions is required unless the frame has
     * no atoms. */
    const double *positions;
    const double *velocities;
    const double *forces;
    /* natoms x 3 image flags. */
    const int32_t *image;

    /* The fields below are for a writer opened with variable_atoms;
     * otherwise leave them 0 and NULL (natoms may also equal the writer's
     * atom count). The arrays above then hold this frame's natoms atoms. */
    int64_t natoms;
    /* natoms atom IDs, required unless natoms is 0. Each must fit in 32 bits
     * and none may repeat. A frame with no atoms may leave every per-atom
     * array NULL. */
    const int64_t *ids;
    /* natoms atom types, or NULL. The first frame fixes whether types are
     * present. */
    const int32_t *types;

    /* One pointer per column in mdz_writer_options.columns, in that order,
     * each to natoms x width values: double for MDZ_FLOAT, int32_t for
     * MDZ_INT. Required if the writer has columns, except in a frame with no
     * atoms. */
    const void *const *columns;
} mdz_frame;

/* Reset `frame` so every field is absent. Call this before setting any
 * field. */
void mdz_frame_init(mdz_frame *frame);

/* Open `path` for writing frames of `natoms` atoms (ignored with
 * opts->variable_atoms). `opts` may be NULL for the defaults. On success *out
 * holds the writer, which must be freed with mdz_writer_close; on failure
 * *out is NULL. */
int mdz_writer_open(const char *path, int64_t natoms, const mdz_writer_options *opts,
                    mdz_writer **out);

/* Set static per-atom data, stored once per file: atom IDs and types, each
 * natoms values long and in the order frames list atoms. Either may be NULL.
 * Call this before the first frame. IDs must fit in 32 bits. Returns
 * MDZ_ERR_INVALID for a variable_atoms writer, whose frames carry their own
 * IDs and types. */
int mdz_writer_set_atoms(mdz_writer *w, const int64_t *ids, const int32_t *types);

/* Set the rigid body of each atom: natoms values, in the order frames list
 * atoms, each a body index >= 0 or -1 for an atom in no body. Only for a
 * writer opened with `rigid`, and only before the first frame. If this is
 * not called, the writer detects bodies itself. When appending to a file
 * that has bodies, the file's bodies are used, and any given here must match
 * them. */
int mdz_writer_set_bodies(mdz_writer *w, const int32_t *bodies);

/* Add one frame. Frames are buffered until a group is complete. A frame
 * that does not match the first frame is rejected with MDZ_ERR_INVALID, and
 * the frames before it are kept. */
int mdz_writer_write_frame(mdz_writer *w, const mdz_frame *frame);

/* Encode any buffered frames as a short group and hand everything written
 * to the operating system, so that a crash of the program afterwards loses
 * none of these frames. This is not an fsync. Short groups compress slightly
 * worse, so flush at checkpoints rather than after every frame. */
int mdz_writer_flush(mdz_writer *w);

/* Write buffered frames and the footer that marks the file complete, then
 * free the writer. The writer is freed even on error. If no frame was ever
 * written to a new file, the file is removed instead. NULL is a no-op. */
int mdz_writer_close(mdz_writer *w);

/* Frames written so far, including buffered frames and, when appending,
 * those already in the file. Returns -1 for a NULL writer. */
int64_t mdz_writer_frames(const mdz_writer *w);

/* --- Reading --- */

typedef struct mdz_reader mdz_reader;

/* Open `path`, which may be any .mdz file, for random access. A file whose
 * writer never finished opens with every complete group; see
 * mdz_reader_complete. On failure *out is NULL. */
int mdz_reader_open(const char *path, mdz_reader **out);

/* Free the reader. NULL is a no-op. */
void mdz_reader_close(mdz_reader *r);

/* Atoms per frame, or 0 if the count varies (see mdz_reader_frame_natoms).
 * Returns -1 for a NULL reader. */
int64_t mdz_reader_natoms(const mdz_reader *r);
/* Frames in the file, or -1 for a NULL reader. */
int64_t mdz_reader_nframes(const mdz_reader *r);

/* 1 if the atom count varies from frame to frame, else 0. */
int mdz_reader_variable_atoms(const mdz_reader *r);

/* Atoms in frame `index`. Returns -1 for a NULL reader or an index out of
 * range, with the reason in mdz_last_error. */
int64_t mdz_reader_frame_natoms(const mdz_reader *r, int64_t index);

/* The number of rigid bodies the file records, or 0 if none. If `bodies`
 * is non-NULL and the file has bodies, each atom's body index (or -1 for
 * none) is copied into it, natoms values. Returns -1 for a NULL reader. */
int64_t mdz_reader_bodies(const mdz_reader *r, int32_t *bodies);

/* 1 if the file was closed cleanly by its writer, else 0. */
int mdz_reader_complete(const mdz_reader *r);

/* The LAMMPS unit style frames are returned in. This defaults to the style
 * the file was written in, or to "metal" (Angstrom, ps) for files stored in
 * mdz's internal units, such as those converted from other formats. The
 * string is valid until the next mdz_reader_set_units or until the reader
 * is closed. Returns NULL for a NULL reader. */
const char *mdz_reader_units(const mdz_reader *r);

/* Return frames in another LAMMPS unit style from now on, converting as
 * needed. */
int mdz_reader_set_units(mdz_reader *r, const char *units);

/* 1 if the file stores the named field (for example "velocities",
 * "forces", "image" or "box"), else 0. */
int mdz_reader_has(const mdz_reader *r, const char *column);

/* The file's general per-atom columns: every stored per-atom column other
 * than positions, velocities, forces and image. Returns -1 for a NULL
 * reader. */
int64_t mdz_reader_ncolumns(const mdz_reader *r);

/* Describe general column `i`, from 0 to mdz_reader_ncolumns - 1. The name
 * stays valid until the reader is closed. precision is 0 for a column stored
 * exactly. */
int mdz_reader_column(const mdz_reader *r, int64_t i, mdz_column *column);

/* Copy static per-atom IDs or types into a caller buffer of natoms values.
 * Returns MDZ_ERR_INVALID if the file has none. A file whose atom count
 * varies never has static IDs or types; read each frame's through
 * mdz_frame_out.ids and mdz_frame_out.types instead. */
int mdz_reader_atom_ids(const mdz_reader *r, int64_t *ids);
int mdz_reader_atom_types(const mdz_reader *r, int32_t *types);

/* One frame read back. The caller sets the buffer pointers for the fields
 * it wants, each sized natoms x 3 for the frame's atom count (see
 * mdz_reader_frame_natoms); NULL buffers are skipped. Requesting a field the
 * file does not have fails with MDZ_ERR_INVALID. The step, time, box and
 * origin are always filled in, with the has_* flags saying which are
 * present. */
typedef struct mdz_frame_out {
    uint32_t struct_size;
    int64_t step;
    int has_step;
    double time;
    int has_time;
    double origin[3];
    double box[9];
    int has_box;
    double *positions;
    double *velocities;
    double *forces;
    int32_t *image;

    /* Set on return to the number of atoms in this frame. */
    int64_t natoms;
    /* If non-NULL, natoms values each: the frame's atom IDs and types. For a
     * fixed atom count, the IDs are the file's static IDs, or 0..natoms-1 if
     * it has none. Requesting types from a file without them fails with
     * MDZ_ERR_INVALID. */
    int64_t *ids;
    int32_t *types;
    /* If non-NULL, mdz_reader_ncolumns pointers, one per general column.
     * Each is NULL to skip the column, or a buffer for natoms x width values
     * of its type (see mdz_reader_column). The values are not converted by
     * mdz_reader_set_units. */
    void *const *columns;
} mdz_frame_out;

/* Reset `frame` so every buffer pointer is NULL. Call this before setting
 * any field. */
void mdz_frame_out_init(mdz_frame_out *frame);

/* Read frame `index` (0-based). The reader caches the last group decoded, so
 * reading frames in order decodes each group once. */
int mdz_reader_read_frame(mdz_reader *r, int64_t index, mdz_frame_out *frame);

#ifdef __cplusplus
}
#endif

#endif /* MDZ_H */
