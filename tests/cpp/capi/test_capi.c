/* Exercises mdz.h from C. The test writes a trajectory, reads it back,
 * appends to it, recovers from a simulated crash, and checks that misuse is
 * reported as an error code rather than crashing. It is compiled as C99 and
 * registered with CTest by the CMakeLists.txt in this directory.
 *
 * usage: test_capi [scratch.mdz] */
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mdz/mdz.h"

#define NATOMS 50
#define CHECK(call)                                                                                     \
    do {                                                                                                \
        int rc_ = (call);                                                                               \
        if (rc_ != MDZ_OK) {                                                                            \
            fprintf(stderr, "%s:%d: %s -> %d: %s\n", __FILE__, __LINE__, #call, rc_, mdz_last_error()); \
            return 1;                                                                                   \
        }                                                                                               \
    } while (0)
#define EXPECT(cond)                                                            \
    do {                                                                        \
        if (!(cond)) {                                                          \
            fprintf(stderr, "%s:%d: expected %s\n", __FILE__, __LINE__, #cond); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

static double pos(int frame, int k) { return 1.0 + 0.25 * k + 0.01 * frame + 0.001 * sin(k); }

static int write_frames(mdz_writer* w, int first, int count) {
    double x[NATOMS * 3], v[NATOMS * 3], f[NATOMS * 3];
    int32_t img[NATOMS * 3];
    double box[9] = {20.0, 0, 0, 1.5, 21.0, 0, -0.5, 0.25, 22.0};
    double origin[3] = {-1.25, 0.5, 3.0};
    for (int s = first; s < first + count; s++) {
        for (int k = 0; k < NATOMS * 3; k++) {
            x[k] = pos(s, k);
            v[k] = 0.001 * k - 0.05;
            f[k] = 0.5 * sin(0.1 * k + s);
            img[k] = (k % 7) - 3;
        }
        mdz_frame fr;
        mdz_frame_init(&fr);
        fr.step = 1000LL * s;
        fr.time = 2.0 * s;
        fr.has_time = 1;
        fr.box = box;
        fr.origin = origin;
        fr.positions = x;
        fr.velocities = v;
        fr.forces = f;
        fr.image = img;
        CHECK(mdz_writer_write_frame(w, &fr));
    }
    return 0;
}

static int file_exists(const char* path) {
    FILE* f = fopen(path, "rb");
    if (f) fclose(f);
    return f != NULL;
}

/* Bad arguments and inconsistent frames must return error codes. */
static int misuse(const char* path) {
    double x[NATOMS * 3], v[NATOMS * 3];
    for (int k = 0; k < NATOMS * 3; k++) {
        x[k] = pos(0, k);
        v[k] = 0.0;
    }
    mdz_writer* w = NULL;
    mdz_reader* r = NULL;
    mdz_frame fr = {0};
    mdz_frame_out out = {0};

    /* NULL handles and arguments. */
    EXPECT(mdz_writer_open(path, NATOMS, NULL, NULL) == MDZ_ERR_INVALID);
    EXPECT(mdz_writer_open(NULL, NATOMS, NULL, &w) == MDZ_ERR_INVALID && w == NULL);
    EXPECT(mdz_writer_open(path, -1, NULL, &w) == MDZ_ERR_INVALID && w == NULL);
    EXPECT(mdz_writer_write_frame(NULL, &fr) == MDZ_ERR_INVALID);
    EXPECT(mdz_writer_flush(NULL) == MDZ_ERR_INVALID);
    EXPECT(mdz_writer_set_atoms(NULL, NULL, NULL) == MDZ_ERR_INVALID);
    EXPECT(mdz_writer_close(NULL) == MDZ_OK);
    EXPECT(mdz_writer_frames(NULL) == -1);
    EXPECT(mdz_reader_open(NULL, &r) == MDZ_ERR_INVALID && r == NULL);
    EXPECT(mdz_reader_open(path, NULL) == MDZ_ERR_INVALID);
    mdz_reader_close(NULL);
    EXPECT(mdz_reader_natoms(NULL) == -1 && mdz_reader_nframes(NULL) == -1);
    EXPECT(mdz_reader_complete(NULL) == 0 && mdz_reader_units(NULL) == NULL);
    EXPECT(mdz_reader_has(NULL, "box") == 0);
    EXPECT(mdz_reader_set_units(NULL, "real") == MDZ_ERR_INVALID);
    EXPECT(mdz_reader_read_frame(NULL, 0, &out) == MDZ_ERR_INVALID);
    mdz_writer_options_init(NULL);
    mdz_frame_init(NULL);
    mdz_frame_out_init(NULL);

    /* An options struct smaller than the one in mdz.h is rejected. */
    mdz_writer_options opts;
    mdz_writer_options_init(&opts);
    opts.struct_size = 4;
    EXPECT(mdz_writer_open(path, NATOMS, &opts, &w) == MDZ_ERR_INVALID && w == NULL);
    EXPECT(strstr(mdz_last_error(), "struct_size") != NULL);

    /* A writer closed with no frames leaves no file behind. NULL options
     * select the defaults (real units). */
    CHECK(mdz_writer_open(path, NATOMS, NULL, &w));
    EXPECT(mdz_writer_frames(w) == 0);
    EXPECT(strlen(mdz_last_error()) == 0); /* success clears the error */
    CHECK(mdz_writer_close(w));
    EXPECT(!file_exists(path));

    /* Each frame is validated as it is written. */
    mdz_writer_options_init(&opts);
    opts.gof_size = 4;
    CHECK(mdz_writer_open(path, NATOMS, &opts, &w));
    int64_t big_ids[NATOMS];
    for (int i = 0; i < NATOMS; i++) big_ids[i] = (int64_t)1 << 40;
    EXPECT(mdz_writer_set_atoms(w, big_ids, NULL) == MDZ_ERR_INVALID);
    EXPECT(strstr(mdz_last_error(), "32 bits") != NULL);
    mdz_frame_init(&fr);
    EXPECT(mdz_writer_write_frame(w, &fr) == MDZ_ERR_INVALID); /* no positions */
    fr.struct_size = 8;
    fr.positions = x;
    EXPECT(mdz_writer_write_frame(w, &fr) == MDZ_ERR_INVALID); /* struct_size too small */
    mdz_frame_init(&fr);
    double origin[3] = {0, 0, 0};
    fr.positions = x;
    fr.origin = origin;
    EXPECT(mdz_writer_write_frame(w, &fr) == MDZ_ERR_INVALID); /* origin without a box */
    EXPECT(mdz_writer_frames(w) == 0);
    mdz_frame_init(&fr);
    fr.positions = x;
    fr.step = 0;
    CHECK(mdz_writer_write_frame(w, &fr));
    fr.velocities = v; /* the first frame had none */
    fr.step = 1;
    EXPECT(mdz_writer_write_frame(w, &fr) == MDZ_ERR_INVALID);
    EXPECT(strstr(mdz_last_error(), "velocities") != NULL);
    fr.velocities = NULL;
    CHECK(mdz_writer_write_frame(w, &fr));
    EXPECT(mdz_writer_frames(w) == 2);
    /* Lossy positions must be finite. The error is reported when the group
     * is encoded, by the write that completes the group. */
    x[0] = NAN;
    fr.step = 2;
    CHECK(mdz_writer_write_frame(w, &fr));
    fr.step = 3;
    EXPECT(mdz_writer_write_frame(w, &fr) == MDZ_ERR_INVALID);
    printf("expected error: %s\n", mdz_last_error());
    x[0] = pos(0, 0);
    CHECK(mdz_writer_close(w));
    remove(path);

    /* Appending with a different atom count is rejected. */
    mdz_writer_options_init(&opts);
    CHECK(mdz_writer_open(path, NATOMS, &opts, &w));
    mdz_frame_init(&fr);
    fr.positions = x;
    CHECK(mdz_writer_write_frame(w, &fr));
    CHECK(mdz_writer_close(w));
    opts.append = 1;
    EXPECT(mdz_writer_open(path, NATOMS - 1, &opts, &w) == MDZ_ERR_INVALID && w == NULL);
    /* The rejected append leaves the file intact. */
    CHECK(mdz_reader_open(path, &r));
    EXPECT(mdz_reader_nframes(r) == 1 && mdz_reader_complete(r) == 1);
    EXPECT(strcmp(mdz_reader_units(r), "real") == 0);
    EXPECT(mdz_reader_has(r, "velocities") == 0 && mdz_reader_has(r, "box") == 0);
    EXPECT(mdz_reader_has(r, "positions") == 1 && mdz_reader_has(r, NULL) == 0);
    /* Requests for data the file lacks, or with too small a struct_size. */
    double buf[NATOMS * 3];
    int32_t img[NATOMS * 3];
    int64_t ids[NATOMS];
    mdz_frame_out_init(&out);
    out.velocities = buf;
    EXPECT(mdz_reader_read_frame(r, 0, &out) == MDZ_ERR_INVALID);
    mdz_frame_out_init(&out);
    out.image = img;
    EXPECT(mdz_reader_read_frame(r, 0, &out) == MDZ_ERR_INVALID);
    mdz_frame_out_init(&out);
    out.struct_size = 8;
    EXPECT(mdz_reader_read_frame(r, 0, &out) == MDZ_ERR_INVALID);
    mdz_frame_out_init(&out);
    EXPECT(mdz_reader_read_frame(r, -1, &out) == MDZ_ERR_RANGE);
    CHECK(mdz_reader_read_frame(r, 0, &out)); /* no arrays requested */
    EXPECT(out.has_step && out.step == 0 && !out.has_time && !out.has_box);
    EXPECT(mdz_reader_atom_ids(r, ids) == MDZ_ERR_INVALID);
    EXPECT(mdz_reader_set_units(r, "furlongs") == MDZ_ERR_INVALID);
    EXPECT(strcmp(mdz_reader_units(r), "real") == 0);
    mdz_reader_close(r);

    /* A file that is not an mdz file. */
    FILE* junk = fopen(path, "wb");
    fputs("this is not a trajectory", junk);
    fclose(junk);
    EXPECT(mdz_reader_open(path, &r) == MDZ_ERR_IO && r == NULL);
    remove(path);
    return 0;
}

/* Frame s of a run where atoms come and go. It holds n = vcount(s) atoms with
 * ids s/2 + 1 .. s/2 + n, and frame 5 is empty. */
#define VMAX 40
static int64_t vcount(int s) { return s == 5 ? 0 : 12 + (s * 5) % 9; }
static double vpos(int s, int64_t id, int k) { return 0.5 * (double)id + 0.1 * k + 0.003 * s; }

static int variable_atoms(const char* path) {
    mdz_writer_options opts;
    mdz_writer_options_init(&opts);
    opts.units = "metal";
    opts.lossless = 1;
    opts.gof_size = 4;
    opts.variable_atoms = 1;
    mdz_writer* w = NULL;
    /* natoms is ignored in variable-atom mode, so 0 is accepted. */
    CHECK(mdz_writer_open(path, 0, &opts, &w));
    int64_t ids[VMAX];
    int32_t types[VMAX];
    double x[VMAX * 3], v[VMAX * 3];
    EXPECT(mdz_writer_set_atoms(w, ids, NULL) == MDZ_ERR_INVALID);
    mdz_frame fr;
    mdz_frame_init(&fr);
    fr.natoms = 3;
    fr.positions = x;
    EXPECT(mdz_writer_write_frame(w, &fr) == MDZ_ERR_INVALID); /* no ids */
    for (int s = 0; s < 11; s++) {
        const int64_t n = vcount(s);
        for (int64_t a = 0; a < n; a++) {
            ids[a] = s / 2 + a + 1;
            types[a] = (int32_t)(ids[a] % 3) + 1;
            for (int k = 0; k < 3; k++) {
                x[3 * a + k] = vpos(s, ids[a], k);
                v[3 * a + k] = 0.01 * (double)ids[a] - 0.1 * k;
            }
        }
        mdz_frame_init(&fr);
        fr.step = 10 * s;
        fr.natoms = n;
        /* the empty frame passes NULL per-atom arrays */
        fr.positions = n ? x : NULL;
        fr.velocities = n ? v : NULL;
        fr.ids = n ? ids : NULL;
        fr.types = n ? types : NULL;
        CHECK(mdz_writer_write_frame(w, &fr));
    }
    EXPECT(mdz_writer_frames(w) == 11);
    CHECK(mdz_writer_close(w));

    /* Appending continues the file. Appending without variable_atoms is
     * rejected. */
    opts.append = 1;
    opts.variable_atoms = 0;
    EXPECT(mdz_writer_open(path, 12, &opts, &w) == MDZ_ERR_INVALID && w == NULL);
    EXPECT(strstr(mdz_last_error(), "variable_atoms") != NULL);
    opts.variable_atoms = 1;
    CHECK(mdz_writer_open(path, 0, &opts, &w));
    EXPECT(mdz_writer_frames(w) == 11);
    for (int s = 11; s < 14; s++) {
        const int64_t n = vcount(s);
        for (int64_t a = 0; a < n; a++) {
            ids[a] = s / 2 + a + 1;
            types[a] = (int32_t)(ids[a] % 3) + 1;
            for (int k = 0; k < 3; k++) {
                x[3 * a + k] = vpos(s, ids[a], k);
                v[3 * a + k] = 0.01 * (double)ids[a] - 0.1 * k;
            }
        }
        mdz_frame_init(&fr);
        fr.step = 10 * s;
        fr.natoms = n;
        fr.positions = x;
        fr.velocities = v;
        fr.ids = ids;
        fr.types = types;
        CHECK(mdz_writer_write_frame(w, &fr));
    }
    CHECK(mdz_writer_close(w));

    mdz_reader* r = NULL;
    CHECK(mdz_reader_open(path, &r));
    EXPECT(mdz_reader_variable_atoms(r) == 1);
    EXPECT(mdz_reader_natoms(r) == 0);
    EXPECT(mdz_reader_nframes(r) == 14);
    EXPECT(mdz_reader_frame_natoms(r, 14) == -1 && mdz_reader_frame_natoms(r, -1) == -1);
    EXPECT(mdz_reader_frame_natoms(NULL, 0) == -1);
    EXPECT(mdz_reader_atom_ids(r, ids) == MDZ_ERR_INVALID);
    EXPECT(mdz_reader_atom_types(r, types) == MDZ_ERR_INVALID);
    EXPECT(mdz_reader_bodies(r, NULL) == 0);
    for (int s = 0; s < 14; s++) {
        const int64_t n = vcount(s);
        EXPECT(mdz_reader_frame_natoms(r, s) == n);
        int64_t ids_back[VMAX];
        int32_t types_back[VMAX];
        mdz_frame_out out;
        mdz_frame_out_init(&out);
        out.positions = x;
        out.velocities = v;
        out.ids = ids_back;
        out.types = types_back;
        CHECK(mdz_reader_read_frame(r, s, &out));
        EXPECT(out.natoms == n);
        EXPECT(out.has_step && out.step == 10 * s);
        for (int64_t a = 0; a < n; a++) {
            EXPECT(ids_back[a] == s / 2 + a + 1);
            EXPECT(types_back[a] == (int32_t)(ids_back[a] % 3) + 1);
            for (int k = 0; k < 3; k++) {
                EXPECT(x[3 * a + k] == (double)(float)vpos(s, ids_back[a], k));
                EXPECT(v[3 * a + k] == (double)(float)(0.01 * (double)ids_back[a] - 0.1 * k));
            }
        }
    }
    mdz_reader_close(r);
    remove(path);

    /* A fixed-count writer rejects a per-frame natoms that differs, and
     * per-frame ids. */
    mdz_writer_options_init(&opts);
    CHECK(mdz_writer_open(path, NATOMS, &opts, &w));
    double xf[NATOMS * 3];
    for (int k = 0; k < NATOMS * 3; k++) xf[k] = pos(0, k);
    mdz_frame_init(&fr);
    fr.positions = xf;
    fr.natoms = NATOMS - 1;
    EXPECT(mdz_writer_write_frame(w, &fr) == MDZ_ERR_INVALID);
    EXPECT(strstr(mdz_last_error(), "variable_atoms") != NULL);
    fr.natoms = NATOMS;
    fr.ids = ids;
    EXPECT(mdz_writer_write_frame(w, &fr) == MDZ_ERR_INVALID);
    fr.ids = NULL;
    CHECK(mdz_writer_write_frame(w, &fr));
    CHECK(mdz_writer_close(w));
    /* A fixed-count file cannot be appended to in variable-atom mode. */
    opts.append = 1;
    opts.variable_atoms = 1;
    EXPECT(mdz_writer_open(path, 0, &opts, &w) == MDZ_ERR_INVALID && w == NULL);
    /* Fixed-count files report the same count for every frame, and ids
     * 0..natoms-1 when none were stored. */
    CHECK(mdz_reader_open(path, &r));
    EXPECT(mdz_reader_variable_atoms(r) == 0);
    EXPECT(mdz_reader_frame_natoms(r, 0) == NATOMS);
    {
        int64_t ids_back[NATOMS];
        mdz_frame_out out;
        mdz_frame_out_init(&out);
        out.ids = ids_back;
        CHECK(mdz_reader_read_frame(r, 0, &out));
        EXPECT(out.natoms == NATOMS && ids_back[0] == 0 && ids_back[NATOMS - 1] == NATOMS - 1);
        out.types = types;
        EXPECT(mdz_reader_read_frame(r, 0, &out) == MDZ_ERR_INVALID); /* no types stored */
    }
    mdz_reader_close(r);
    remove(path);

    /* Rigid mode cannot be combined with variable_atoms. */
    mdz_writer_options_init(&opts);
    opts.variable_atoms = 1;
    opts.rigid = 1;
    EXPECT(mdz_writer_open(path, 0, &opts, &w) == MDZ_ERR_INVALID && w == NULL);
    return 0;
}

/* Rigid triangles that drift and rotate. Atom a belongs to body a / 3. */
#define NMOL 16
static double tri(int s, int a, int k) {
    const int m = a / 3, j = a % 3;
    const double ang = 0.3 * s + m + 2.0943951023931953 * j;
    const double c[3] = {0.6 * (m % 4) + 0.01 * s, 0.6 * (m / 4), 0.002 * s * m};
    const double off[3] = {0.08 * cos(ang), 0.08 * sin(ang), 0.0};
    return 10.0 * (c[k] + off[k]); /* Angstrom */
}

static int rigid_bodies(const char* path) {
    mdz_writer_options opts;
    mdz_writer_options_init(&opts);
    opts.units = "metal";
    opts.gof_size = 5;
    opts.precision = 0.001;
    mdz_writer* w = NULL;
    int32_t bodies[NMOL * 3];
    for (int a = 0; a < NMOL * 3; a++) bodies[a] = a / 3;
    CHECK(mdz_writer_open(path, NMOL * 3, &opts, &w));
    EXPECT(mdz_writer_set_bodies(w, bodies) == MDZ_ERR_INVALID); /* not rigid */
    CHECK(mdz_writer_close(w));
    opts.rigid = 1;
    CHECK(mdz_writer_open(path, NMOL * 3, &opts, &w));
    EXPECT(mdz_writer_set_bodies(w, NULL) == MDZ_ERR_INVALID);
    EXPECT(mdz_writer_set_bodies(NULL, bodies) == MDZ_ERR_INVALID);
    bodies[0] = -2;
    EXPECT(mdz_writer_set_bodies(w, bodies) == MDZ_ERR_INVALID);
    bodies[0] = 0;
    CHECK(mdz_writer_set_bodies(w, bodies));
    double x[NMOL * 9];
    for (int s = 0; s < 12; s++) {
        for (int a = 0; a < NMOL * 3; a++)
            for (int k = 0; k < 3; k++) x[3 * a + k] = tri(s, a, k);
        mdz_frame fr;
        mdz_frame_init(&fr);
        fr.step = s;
        fr.positions = x;
        CHECK(mdz_writer_write_frame(w, &fr));
        if (s == 0) EXPECT(mdz_writer_set_bodies(w, bodies) == MDZ_ERR_INVALID); /* too late */
    }
    CHECK(mdz_writer_close(w));

    mdz_reader* r = NULL;
    CHECK(mdz_reader_open(path, &r));
    int32_t back[NMOL * 3];
    EXPECT(mdz_reader_bodies(r, back) == NMOL);
    EXPECT(memcmp(back, bodies, sizeof back) == 0);
    EXPECT(mdz_reader_bodies(NULL, back) == -1);
    for (int s = 0; s < 12; s++) {
        mdz_frame_out out;
        mdz_frame_out_init(&out);
        out.positions = x;
        CHECK(mdz_reader_read_frame(r, s, &out));
        for (int a = 0; a < NMOL * 3; a++)
            for (int k = 0; k < 3; k++) EXPECT(fabs(x[3 * a + k] - tri(s, a, k)) <= 4 * 0.001 + 1e-5);
    }
    mdz_reader_close(r);

    /* The same trajectory with the bodies detected instead of given. */
    CHECK(mdz_writer_open(path, NMOL * 3, &opts, &w));
    for (int s = 0; s < 12; s++) {
        for (int a = 0; a < NMOL * 3; a++)
            for (int k = 0; k < 3; k++) x[3 * a + k] = tri(s, a, k);
        mdz_frame fr;
        mdz_frame_init(&fr);
        fr.positions = x;
        CHECK(mdz_writer_write_frame(w, &fr));
    }
    CHECK(mdz_writer_close(w));
    CHECK(mdz_reader_open(path, &r));
    EXPECT(mdz_reader_bodies(r, NULL) == NMOL);
    mdz_reader_close(r);
    remove(path);
    return 0;
}

/* General columns: a per-atom energy stored exactly, a 6-wide stress at a
 * precision of 0.01 and a 2-wide integer flag. */
static double col_pe(int s, int a) { return -3.0 + 0.001 * a + 0.0001 * s; }
static double col_stress(int s, int a, int k) { return 100.0 * sin(0.1 * a + k) + s; }
static int32_t col_flag(int s, int a, int k) { return (int32_t)(a * 7 + s * 3 + k); }

static int fill_columns(int s, int n, double* pe, double* stress, int32_t* flag) {
    for (int a = 0; a < n; a++) {
        pe[a] = col_pe(s, a);
        for (int k = 0; k < 6; k++) stress[6 * a + k] = col_stress(s, a, k);
        for (int k = 0; k < 2; k++) flag[2 * a + k] = col_flag(s, a, k);
    }
    return 0;
}

static int general_columns(const char* path) {
    const mdz_column cols[3] = {{"c_pe", MDZ_FLOAT, 1, 0.0}, {"stress", MDZ_FLOAT, 6, 0.01}, {"flag", MDZ_INT, 2, 0.0}};
    mdz_writer_options opts;
    mdz_writer* w = NULL;

    /* Bad declarations are refused at open. */
    mdz_column bad[2] = {{"positions", MDZ_FLOAT, 3, 0.0}, {"x", MDZ_FLOAT, 1, 0.0}};
    mdz_writer_options_init(&opts);
    opts.columns = bad;
    opts.ncolumns = 1;
    EXPECT(mdz_writer_open(path, NATOMS, &opts, &w) == MDZ_ERR_INVALID && w == NULL);
    EXPECT(strstr(mdz_last_error(), "reserved") != NULL);
    bad[0].name = "x";
    opts.ncolumns = 2;
    EXPECT(mdz_writer_open(path, NATOMS, &opts, &w) == MDZ_ERR_INVALID);
    EXPECT(strstr(mdz_last_error(), "twice") != NULL);
    bad[1].name = "y";
    bad[1].width = 0;
    EXPECT(mdz_writer_open(path, NATOMS, &opts, &w) == MDZ_ERR_INVALID);
    bad[1].width = 1;
    bad[1].type = MDZ_INT;
    bad[1].precision = 0.5;
    EXPECT(mdz_writer_open(path, NATOMS, &opts, &w) == MDZ_ERR_INVALID);
    opts.columns = NULL;
    EXPECT(mdz_writer_open(path, NATOMS, &opts, &w) == MDZ_ERR_INVALID);

    mdz_writer_options_init(&opts);
    opts.units = "metal";
    opts.gof_size = 4;
    opts.columns = cols;
    opts.ncolumns = 3;
    CHECK(mdz_writer_open(path, NATOMS, &opts, &w));
    double x[NATOMS * 3], pe[NATOMS], stress[NATOMS * 6];
    int32_t flag[NATOMS * 2];
    const void* ptrs[3] = {pe, stress, flag};
    mdz_frame fr;
    mdz_frame_init(&fr);
    fr.positions = x;
    for (int k = 0; k < NATOMS * 3; k++) x[k] = pos(0, k);
    EXPECT(mdz_writer_write_frame(w, &fr) == MDZ_ERR_INVALID); /* no columns */
    const void* missing[3] = {pe, NULL, flag};
    fr.columns = missing;
    EXPECT(mdz_writer_write_frame(w, &fr) == MDZ_ERR_INVALID);
    EXPECT(strstr(mdz_last_error(), "stress") != NULL);
    for (int s = 0; s < 10; s++) {
        for (int k = 0; k < NATOMS * 3; k++) x[k] = pos(s, k);
        fill_columns(s, NATOMS, pe, stress, flag);
        mdz_frame_init(&fr);
        fr.step = s;
        fr.positions = x;
        fr.columns = ptrs;
        CHECK(mdz_writer_write_frame(w, &fr));
    }
    CHECK(mdz_writer_close(w));

    /* Appending needs the same columns. */
    mdz_writer_options_init(&opts);
    opts.units = "metal";
    opts.gof_size = 4;
    opts.append = 1;
    opts.columns = cols;
    opts.ncolumns = 2;
    EXPECT(mdz_writer_open(path, NATOMS, &opts, &w) == MDZ_ERR_INVALID && w == NULL);
    EXPECT(strstr(mdz_last_error(), "general columns") != NULL);
    opts.ncolumns = 3;
    CHECK(mdz_writer_open(path, NATOMS, &opts, &w));
    for (int s = 10; s < 12; s++) {
        for (int k = 0; k < NATOMS * 3; k++) x[k] = pos(s, k);
        fill_columns(s, NATOMS, pe, stress, flag);
        mdz_frame_init(&fr);
        fr.step = s;
        fr.positions = x;
        fr.columns = ptrs;
        CHECK(mdz_writer_write_frame(w, &fr));
    }
    CHECK(mdz_writer_close(w));

    mdz_reader* r = NULL;
    CHECK(mdz_reader_open(path, &r));
    EXPECT(mdz_reader_nframes(r) == 12);
    EXPECT(mdz_reader_ncolumns(r) == 3);
    mdz_column info;
    EXPECT(mdz_reader_column(r, 3, &info) == MDZ_ERR_RANGE);
    int ipe = -1, istress = -1, iflag = -1;
    for (int i = 0; i < 3; i++) {
        CHECK(mdz_reader_column(r, i, &info));
        if (strcmp(info.name, "c_pe") == 0) {
            ipe = i;
            EXPECT(info.type == MDZ_FLOAT && info.width == 1 && info.precision == 0.0);
        } else if (strcmp(info.name, "stress") == 0) {
            istress = i;
            EXPECT(info.type == MDZ_FLOAT && info.width == 6 && fabs(info.precision - 0.01) < 1e-9);
        } else if (strcmp(info.name, "flag") == 0) {
            iflag = i;
            EXPECT(info.type == MDZ_INT && info.width == 2);
        }
    }
    EXPECT(ipe >= 0 && istress >= 0 && iflag >= 0);
    EXPECT(mdz_reader_has(r, "stress") == 1);
    double pe_back[NATOMS], stress_back[NATOMS * 6];
    int32_t flag_back[NATOMS * 2];
    void* out_ptrs[3];
    out_ptrs[ipe] = pe_back;
    out_ptrs[istress] = stress_back;
    out_ptrs[iflag] = flag_back;
    mdz_frame_out out;
    for (int s = 0; s < 12; s++) {
        mdz_frame_out_init(&out);
        out.columns = out_ptrs;
        CHECK(mdz_reader_read_frame(r, s, &out));
        for (int a = 0; a < NATOMS; a++) {
            EXPECT(pe_back[a] == (double)(float)col_pe(s, a));
            for (int k = 0; k < 6; k++) EXPECT(fabs(stress_back[6 * a + k] - col_stress(s, a, k)) <= 0.005 + 1e-4);
            for (int k = 0; k < 2; k++) EXPECT(flag_back[2 * a + k] == col_flag(s, a, k));
        }
    }
    /* A NULL entry skips that column. */
    out_ptrs[istress] = NULL;
    mdz_frame_out_init(&out);
    out.columns = out_ptrs;
    CHECK(mdz_reader_read_frame(r, 0, &out));
    mdz_reader_close(r);
    remove(path);

    /* With a varying atom count. */
    mdz_writer_options_init(&opts);
    opts.gof_size = 3;
    opts.variable_atoms = 1;
    opts.columns = cols;
    opts.ncolumns = 3;
    CHECK(mdz_writer_open(path, 0, &opts, &w));
    int64_t ids[NATOMS];
    for (int s = 0; s < 7; s++) {
        const int n = 20 + 3 * s;
        for (int a = 0; a < n; a++) ids[a] = a + 1;
        for (int k = 0; k < n * 3; k++) x[k] = pos(s, k);
        fill_columns(s, n, pe, stress, flag);
        mdz_frame_init(&fr);
        fr.step = s;
        fr.natoms = n;
        fr.ids = ids;
        fr.positions = x;
        fr.columns = ptrs;
        CHECK(mdz_writer_write_frame(w, &fr));
    }
    CHECK(mdz_writer_close(w));
    CHECK(mdz_reader_open(path, &r));
    EXPECT(mdz_reader_ncolumns(r) == 3);
    out_ptrs[istress] = stress_back;
    for (int s = 0; s < 7; s++) {
        mdz_frame_out_init(&out);
        out.columns = out_ptrs;
        CHECK(mdz_reader_read_frame(r, s, &out));
        EXPECT(out.natoms == 20 + 3 * s);
        for (int a = 0; a < out.natoms; a++) {
            EXPECT(pe_back[a] == (double)(float)col_pe(s, a));
            for (int k = 0; k < 6; k++) EXPECT(fabs(stress_back[6 * a + k] - col_stress(s, a, k)) <= 0.005 + 1e-4);
            for (int k = 0; k < 2; k++) EXPECT(flag_back[2 * a + k] == col_flag(s, a, k));
        }
    }
    mdz_reader_close(r);
    remove(path);
    return 0;
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "c_api_test.mdz";
    printf("mdz %s\n", mdz_version());

    /* Lossless, metal units, 7-frame groups. */
    mdz_writer_options opts;
    mdz_writer_options_init(&opts);
    opts.units = "metal";
    opts.lossless = 1;
    opts.gof_size = 7;
    opts.boundary = "pp pp ff";
    opts.unwrapped = 1;
    mdz_writer* w = NULL;
    CHECK(mdz_writer_open(path, NATOMS, &opts, &w));
    int64_t ids[NATOMS];
    int32_t types[NATOMS];
    for (int i = 0; i < NATOMS; i++) {
        ids[i] = 100 + i;
        types[i] = 1 + i % 3;
    }
    CHECK(mdz_writer_set_atoms(w, ids, types));
    if (write_frames(w, 0, 16)) return 1;
    EXPECT(mdz_writer_frames(w) == 16);
    CHECK(mdz_writer_close(w));

    /* Reopen and append 9 more frames, as after a restart. */
    opts.append = 1;
    CHECK(mdz_writer_open(path, NATOMS, &opts, &w));
    EXPECT(mdz_writer_frames(w) == 16);
    CHECK(mdz_writer_set_atoms(w, ids, types));
    if (write_frames(w, 16, 9)) return 1;
    CHECK(mdz_writer_close(w));

    /* Read everything back and compare exactly. */
    mdz_reader* r = NULL;
    CHECK(mdz_reader_open(path, &r));
    EXPECT(mdz_reader_natoms(r) == NATOMS);
    EXPECT(mdz_reader_nframes(r) == 25);
    EXPECT(mdz_reader_complete(r) == 1);
    EXPECT(strcmp(mdz_reader_units(r), "metal") == 0);
    EXPECT(mdz_reader_has(r, "velocities") && mdz_reader_has(r, "image"));
    EXPECT(mdz_reader_has(r, "box"));
    int64_t ids_back[NATOMS];
    int32_t types_back[NATOMS];
    CHECK(mdz_reader_atom_ids(r, ids_back));
    CHECK(mdz_reader_atom_types(r, types_back));
    EXPECT(memcmp(ids, ids_back, sizeof ids) == 0);
    EXPECT(memcmp(types, types_back, sizeof types) == 0);

    double x[NATOMS * 3], v[NATOMS * 3], f[NATOMS * 3];
    int32_t img[NATOMS * 3];
    for (int s = 0; s < 25; s++) {
        mdz_frame_out out;
        mdz_frame_out_init(&out);
        out.positions = x;
        out.velocities = v;
        out.forces = f;
        out.image = img;
        CHECK(mdz_reader_read_frame(r, s, &out));
        EXPECT(out.has_step && out.step == 1000LL * s);
        EXPECT(out.has_time && fabs(out.time - 2.0 * s) < 1e-9);
        EXPECT(out.has_box && out.box[3] == 1.5 && out.origin[0] == -1.25);
        for (int k = 0; k < NATOMS * 3; k++) {
            /* Values are stored as float32, so they match the input rounded
             * to float32 exactly. */
            EXPECT(x[k] == (double)(float)pos(s, k));
            EXPECT(v[k] == (double)(float)(0.001 * k - 0.05));
            EXPECT(img[k] == (k % 7) - 3);
        }
    }
    mdz_frame_out out;
    mdz_frame_out_init(&out);
    EXPECT(mdz_reader_read_frame(r, 25, &out) == MDZ_ERR_RANGE);

    /* Convert on read from metal to real units (velocities A/ps to A/fs). */
    CHECK(mdz_reader_set_units(r, "real"));
    out.velocities = v;
    CHECK(mdz_reader_read_frame(r, 0, &out));
    EXPECT(fabs(v[0] - (-0.05 / 1000.0)) < 1e-9);
    mdz_reader_close(r);

    /* Errors are returned as codes with a message. */
    EXPECT(mdz_reader_open("/nonexistent/file.mdz", &r) == MDZ_ERR_IO);
    EXPECT(strlen(mdz_last_error()) > 0);
    opts.units = "furlongs";
    opts.append = 0;
    EXPECT(mdz_writer_open(path, NATOMS, &opts, &w) == MDZ_ERR_INVALID);
    EXPECT(w == NULL);

    /* If the writer never closes, as in a crash, every flushed group is
     * still readable. */
    opts.units = "real";
    opts.lossless = 0;
    opts.precision = 0.001;
    char crashed[1024];
    snprintf(crashed, sizeof crashed, "%s.crash", path);
    CHECK(mdz_writer_open(crashed, NATOMS, &opts, &w));
    if (write_frames(w, 0, 10)) return 1;
    CHECK(mdz_writer_flush(w));
    /* Simulate the process dying by copying the file as it is on disk. */
    char copy[1100];
    snprintf(copy, sizeof copy, "%s.copy", crashed);
    {
        FILE *in = fopen(crashed, "rb"), *o = fopen(copy, "wb");
        char buf[65536];
        size_t n;
        while ((n = fread(buf, 1, sizeof buf, in)) > 0) fwrite(buf, 1, n, o);
        fclose(in);
        fclose(o);
    }
    CHECK(mdz_writer_close(w));
    CHECK(mdz_reader_open(copy, &r));
    EXPECT(mdz_reader_complete(r) == 0);
    EXPECT(mdz_reader_nframes(r) == 10);
    out.positions = x;
    out.velocities = NULL;
    CHECK(mdz_reader_read_frame(r, 9, &out));
    for (int k = 0; k < NATOMS * 3; k++) /* adaptive precision allows up to 4 quantization steps */
        EXPECT(fabs(x[k] - pos(9, k)) <= 4 * 0.001 + 1e-5);
    mdz_reader_close(r);
    remove(copy);
    remove(crashed);
    remove(path);

    if (misuse(path)) return 1;
    if (variable_atoms(path)) return 1;
    if (rigid_bodies(path)) return 1;
    if (general_columns(path)) return 1;
    printf("all C API checks passed\n");
    return 0;
}
