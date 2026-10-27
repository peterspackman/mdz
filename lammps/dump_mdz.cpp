/* ----------------------------------------------------------------------
   dump mdz: write LAMMPS trajectories as compressed, crash-tolerant,
   appendable .mdz files through the mdz C API (mdz/mdz.h).

   This file is distributed under the terms of the GNU General Public
   License, like LAMMPS itself.

   Usage:
     dump ID group-ID mdz N file.mdz [field ...]
     dump_modify ID keyword value ...

   The fields are those of dump custom. Without any, they are
   "id type x y z ix iy iz". Positions (x y z, or xu yu zu for unwrapped
   ones) are required, and id is added if it is not listed. x y z, vx vy vz
   and fx fy fz are stored to a precision; every other field, such as q,
   mol, c_ID, f_ID[2] or v_name, is stored exactly (as single precision)
   under its own name unless given a precision.

   Keywords (besides the standard dump_modify ones such as append, pbc,
   every, first, delay, skip, region and thresh):
     precision P         position step in length units (0 = default,
                         a maximum error of 0.005 Angstrom)
     velocity/precision P, force/precision P
     precision/column F P  quantization step for field F, in its own units
                         (0 = exact, the default)
     lossless yes/no     store every value exactly (as single precision)
     gof N               frames per compressed group (default 20)
     variable yes/no     let the group's atom count change from frame to
                         frame (atoms lost, deleted, deposited; default no)
     rigid no/yes/mol    model rigid bodies: yes detects them, mol takes
                         each molecule as a body, and needs the mol field
                         (default no)

   Values are stored in the run's unit style, which the file records.
   Frames are compressed one group at a time. If a run dies, the file
   holds every complete group and can be read as is. After a restart,
   "dump_modify ID append yes" continues the same file.

   Without "variable yes", a frame whose atoms differ from those of the
   first frame (a different count or different IDs) cannot be stored. It
   is skipped with a warning and the run continues, so that losing an
   atom does not stop the simulation.
------------------------------------------------------------------------- */

#include "dump_mdz.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "group.h"
#include "update.h"

#include "mdz/mdz.h"
#ifdef MDZ_PLUGIN_SERIAL
#include "mdz/mdz.hpp"
#endif

#include <algorithm>
#include <climits>
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

DumpMDZArgs::DumpMDZArgs(int narg, char** arg) : strings(arg, arg + narg) {
    if (narg == 5)
        for (const char* f : {"id", "type", "x", "y", "z", "ix", "iy", "iz"}) strings.emplace_back(f);
    if (std::find(strings.begin() + std::min(5, narg), strings.end(), "id") == strings.end())
        strings.emplace_back("id");
    for (auto& s : strings) argv.push_back(&s[0]);
}

DumpMDZ::DumpMDZ(LAMMPS* lmp, int narg, char** arg)
    : DumpMDZArgs(narg, arg),
      DumpCustom(lmp, (int)argv.size(), argv.data()),
      natoms(0),
      nframe(0),
      nfilled(0),
      frame_open(0),
      skip_frame(0),
      nskipped(0),
      lossless_flag(0),
      variable_flag(0),
      rigid_flag(0),
      precision(0.0),
      vel_precision(0.0),
      force_precision(0.0),
      gof_size(0),
      unwrapped(0),
      step(0),
      time(0.0),
      writer(nullptr),
      nwritten(0) {
    if (binary || compressed || multifile || multiproc)
        error->all(FLERR, "Dump mdz writes one .mdz file: no '*', '%', .bin or .gz in its name");

    // records reach proc 0 as numbers, not formatted text
    buffer_allow = 0;
    buffer_flag = 0;
    sort_flag = 1;
    sortcol = 0;
    flush_flag = 0;
    map_fields();
#ifdef MDZ_PLUGIN_SERIAL
    // built with MDZ_PARALLEL=off, so mdz runs only on the writing rank's thread
    mdz::set_num_threads(1);
#endif
}

/* ---------------------------------------------------------------------- */

DumpMDZ::~DumpMDZ() {
    if (nskipped)
        error->warning(FLERR, "Dump mdz skipped {} frames of {} whose atoms differed from the first frame's", nskipped,
                       filename);
    if (writer && mdz_writer_close(writer) != MDZ_OK)
        error->warning(FLERR, "Dump mdz could not finish {}: {}", filename, mdz_last_error());
    writer = nullptr;
}

/* ----------------------------------------------------------------------
   sort the fields into the ones mdz has a place for and general columns
------------------------------------------------------------------------- */

void DumpMDZ::map_fields() {
    f_id = f_type = f_mol = -1;
    for (int k = 0; k < 3; k++) f_x[k] = f_v[k] = f_f[k] = f_image[k] = -1;
    std::map<std::string, int> at;
    for (int i = 0; i < nfield; i++) {
        if (!at.emplace(earg[i], i).second) error->all(FLERR, "Dump mdz field {} is listed twice", earg[i]);
        if (vtype[i] == Dump::STRING || vtype[i] == Dump::STRING2)
            error->all(FLERR, "Dump mdz cannot store the text field {}", earg[i]);
    }
    auto triple = [&](const char* a, const char* b, const char* c, int* out) {
        const char* names[3] = {a, b, c};
        int found = 0;
        for (int k = 0; k < 3; k++) found += at.count(names[k]);
        if (found != 3) return false;
        for (int k = 0; k < 3; k++) out[k] = at[names[k]];
        return true;
    };

    for (const char* s : {"xs", "ys", "zs", "xsu", "ysu", "zsu"})
        if (at.count(s)) error->all(FLERR, "Dump mdz stores real coordinates: use x y z or xu yu zu, not {}", s);
    f_id = at["id"];
    if (at.count("type")) f_type = at["type"];
    if (triple("xu", "yu", "zu", f_x)) {
        unwrapped = 1;
        if (at.count("x") || at.count("y") || at.count("z"))
            error->all(FLERR, "Dump mdz takes x y z or xu yu zu, not both");
    } else if (!triple("x", "y", "z", f_x)) {
        error->all(FLERR, "Dump mdz needs positions: list x y z, or xu yu zu for unwrapped ones");
    }
    triple("vx", "vy", "vz", f_v);
    triple("fx", "fy", "fz", f_f);
    triple("ix", "iy", "iz", f_image);
    if (at.count("mol")) f_mol = at["mol"];

    extra.clear();
    for (int i = 0; i < nfield; i++) {
        bool used = i == f_id || i == f_type;
        for (int k = 0; k < 3; k++) used = used || i == f_x[k] || i == f_v[k] || i == f_f[k] || i == f_image[k];
        if (used) continue;
        const std::string name = earg[i];
        const int is_int = vtype[i] == Dump::INT || vtype[i] == Dump::BIGINT;
        extra.push_back(Column{name, i, is_int});
    }
}

/* ---------------------------------------------------------------------- */

void DumpMDZ::init_style() {
    if (sort_flag == 0 || sortcol != 0) error->all(FLERR, "Dump mdz requires sorting by atom ID");
    if (flush_flag) error->all(FLERR, "Dump mdz does not support dump_modify flush: it writes a frame group at a time");
    if (triclinic_general)
        error->all(FLERR, "Dump mdz stores the restricted triclinic box: not with triclinic/general");
    if (domain->dimension != 3) error->all(FLERR, "Dump mdz requires a 3d simulation");
    if (rigid_flag == 2 && f_mol < 0) error->all(FLERR, "Dump mdz rigid mol needs the mol field in the dump command");
    if (rigid_flag && variable_flag)
        error->all(FLERR, "Dump mdz rigid bodies need a fixed atom count: not with variable yes");
    for (const auto& p : column_precision) {
        auto it = std::find_if(extra.begin(), extra.end(), [&](const Column& c) { return c.name == p.first; });
        if (it == extra.end())
            error->all(FLERR, "Dump mdz precision/column {}: not a general field of this dump", p.first);
        if (it->is_int && p.second != 0.0)
            error->all(FLERR, "Dump mdz precision/column {}: an integer field is stored exactly", p.first);
    }

    domain->boundary_string(boundstr);
    if (!singlefile_opened) {
        // count atoms at the first run, so the dump can be defined before atoms exist
        natoms = group->count(igroup);
        if (natoms == 0 && !variable_flag) error->all(FLERR, "Dump mdz group {} has no atoms", group->names[igroup]);
    }
    // finds the computes, fixes and variables the fields use, then opens the file
    DumpCustom::init_style();
}

/* ---------------------------------------------------------------------- */

void DumpMDZ::check(int rc, const char* what) {
    if (rc != MDZ_OK) error->one(FLERR, "Dump mdz {} {}: {}", what, filename, mdz_last_error());
}

void DumpMDZ::openfile() {
    if (singlefile_opened) return;
    singlefile_opened = 1;
    if (me != 0) return;

    std::vector<mdz_column> columns(extra.size());
    for (size_t c = 0; c < extra.size(); c++) {
        columns[c].name = extra[c].name.c_str();
        columns[c].type = extra[c].is_int ? MDZ_INT : MDZ_FLOAT;
        columns[c].width = 1;
        auto p = column_precision.find(extra[c].name);
        columns[c].precision = p == column_precision.end() ? 0.0 : p->second;
    }

    mdz_writer_options opts;
    mdz_writer_options_init(&opts);
    opts.units = update->unit_style;
    opts.precision = precision;
    opts.velocity_precision = vel_precision;
    opts.force_precision = force_precision;
    opts.lossless = lossless_flag;
    opts.gof_size = gof_size;
    opts.append = append_flag;
    opts.boundary = boundstr;
    opts.unwrapped = unwrapped;
    opts.variable_atoms = variable_flag;
    opts.rigid = rigid_flag != 0;
    opts.columns = columns.data();
    opts.ncolumns = (uint32_t)columns.size();
    check(mdz_writer_open(filename, variable_flag ? 0 : natoms, &opts, &writer), "cannot open");
    if (append_flag && mdz_writer_frames(writer) > 0)
        utils::logmesg(lmp, "Dump mdz appending to {} after {} frames\n", filename, mdz_writer_frames(writer));
    size_frame(natoms);
}

/* ----------------------------------------------------------------------
   size the buffers for a frame of n atoms. They are never empty, so a
   frame with no atoms still passes non-null arrays to the C API.
------------------------------------------------------------------------- */

void DumpMDZ::size_frame(bigint n) {
    const bigint m = n > 0 ? n : 1;
    xs.resize(3 * m);
    if (f_image[0] >= 0) images.resize(3 * m);
    if (f_v[0] >= 0) vs.resize(3 * m);
    if (f_f[0] >= 0) fs.resize(3 * m);
    frame_ids.resize(m);
    frame_types.resize(m);
    extra_float.resize(extra.size());
    extra_int.resize(extra.size());
    for (size_t c = 0; c < extra.size(); c++) {
        if (extra[c].is_int)
            extra_int[c].resize(m);
        else
            extra_float[c].resize(m);
    }
}

/* ----------------------------------------------------------------------
   drop the current frame because its atoms do not match a fixed-count file
------------------------------------------------------------------------- */

void DumpMDZ::skip(const std::string& why) {
    skip_frame = 1;
    if (nskipped++ == 0)
        error->warning(FLERR,
                       "Dump mdz {} skips frames whose atoms differ from the first frame's ({}); "
                       "use dump_modify {} variable yes to store them",
                       filename, why, id);
}

/* ---------------------------------------------------------------------- */

void DumpMDZ::write_header(bigint n) {
    nframe = n;
    nfilled = 0;
    frame_open = 1;
    skip_frame = 0;
    if (variable_flag) {
        size_frame(n);
    } else if (n != natoms) {
        skip(fmt::format("group {} had {} atoms, now {}", group->names[igroup], natoms, n));
    }

    step = update->ntimestep;
    time = update->atime + (update->ntimestep - update->atimestep) * update->dt;

    // LAMMPS restricted triclinic cell: a = (xprd,0,0), b = (xy,yprd,0),
    // c = (xz,yz,zprd), with its lower corner at boxlo. Orthogonal boxes have
    // zero tilt.
    for (int k = 0; k < 3; k++) origin[k] = domain->boxlo[k];
    const int tri = domain->triclinic;
    double xy = tri ? domain->xy : 0.0, xz = tri ? domain->xz : 0.0, yz = tri ? domain->yz : 0.0;
    double cell[9] = {domain->xprd, 0.0, 0.0, xy, domain->yprd, 0.0, xz, yz, domain->zprd};
    memcpy(box, cell, sizeof(box));
}

/* ----------------------------------------------------------------------
   called on proc 0 with chunks of the snapshot, sorted by atom ID
------------------------------------------------------------------------- */

void DumpMDZ::write_data(int n, double* mybuf) {
    if (!frame_open) return;
    if (skip_frame) {
        nfilled += n;
        if (nfilled >= nframe) frame_open = 0;
        return;
    }
    for (int i = 0; i < n; i++, nfilled++) {
        if (nfilled >= nframe) error->one(FLERR, "Dump mdz received more atoms than expected");
        const double* rec = mybuf + (bigint)i * size_one;
        const bigint a = nfilled;
        frame_ids[a] = (int64_t)rec[f_id];
        frame_types[a] = f_type >= 0 ? (int)rec[f_type] : 0;
        for (int k = 0; k < 3; k++) {
            xs[3 * a + k] = rec[f_x[k]];
            if (f_image[0] >= 0) images[3 * a + k] = (int)rec[f_image[k]];
            if (f_v[0] >= 0) vs[3 * a + k] = rec[f_v[k]];
            if (f_f[0] >= 0) fs[3 * a + k] = rec[f_f[k]];
        }
        for (size_t c = 0; c < extra.size(); c++) {
            const double v = rec[extra[c].field];
            if (!extra[c].is_int) {
                extra_float[c][a] = v;
            } else {
                if (v < INT_MIN || v > INT_MAX)
                    error->one(FLERR, "Dump mdz field {} value {} does not fit in 32 bits", extra[c].name, v);
                extra_int[c][a] = (int)v;
            }
        }
    }
    // called once per proc, so a frame with no atoms is complete on the first call
    if (nfilled == nframe) {
        frame_open = 0;
        write_frame();
    }
}

void DumpMDZ::write_frame() {
    mdz_frame frame;
    mdz_frame_init(&frame);
    // With a varying atom count, IDs and types are stored with every frame.
    // Otherwise they are stored once per file and must not change. When
    // appending, they are checked against those already in the file.
    if (variable_flag) {
        frame.natoms = nframe;
        frame.ids = frame_ids.data();
        frame.types = f_type >= 0 ? frame_types.data() : nullptr;
    } else if (nwritten == 0) {
        check(mdz_writer_set_atoms(writer, frame_ids.data(), f_type >= 0 ? frame_types.data() : nullptr),
              "cannot write");
        first_ids.assign(frame_ids.begin(), frame_ids.begin() + nframe);
        if (rigid_flag == 2 && mdz_writer_frames(writer) == 0) {
            // map molecule IDs to body indices 0, 1, .... Atoms with molecule ID 0
            // belong to no body. An appended file keeps its existing bodies.
            size_t mol = 0;
            while (extra[mol].field != f_mol) mol++;
            std::map<int, int32_t> body_of_mol;
            std::vector<int32_t> bodies(nframe);
            for (bigint a = 0; a < nframe; a++) {
                const int m = extra_int[mol][a];
                if (m == 0) {
                    bodies[a] = -1;
                } else {
                    auto it = body_of_mol.emplace(m, (int32_t)body_of_mol.size()).first;
                    bodies[a] = it->second;
                }
            }
            check(mdz_writer_set_bodies(writer, bodies.data()), "cannot write");
        }
    } else if (!std::equal(first_ids.begin(), first_ids.end(), frame_ids.begin())) {
        skip("the atom IDs changed");
        return;
    }

    std::vector<const void*> columns(extra.size());
    for (size_t c = 0; c < extra.size(); c++)
        columns[c] = extra[c].is_int ? (const void*)extra_int[c].data() : (const void*)extra_float[c].data();

    frame.step = step;
    frame.time = time;
    frame.has_time = 1;
    frame.origin = origin;
    frame.box = box;
    frame.positions = xs.data();
    frame.velocities = f_v[0] >= 0 ? vs.data() : nullptr;
    frame.forces = f_f[0] >= 0 ? fs.data() : nullptr;
    frame.image = f_image[0] >= 0 ? images.data() : nullptr;
    frame.columns = columns.data();
    check(mdz_writer_write_frame(writer, &frame), "cannot write");
    nwritten++;
}

/* ---------------------------------------------------------------------- */

int DumpMDZ::modify_param(int narg, char** arg) {
    // the per-frame contents are fixed once the file is open
    auto settable = [&]() {
        if (singlefile_opened) error->all(FLERR, "Dump mdz: dump_modify {} must come before the first run", arg[0]);
    };
    auto need = [&](int n) {
        if (narg < n) error->all(FLERR, "Illegal dump_modify {} command: missing value", arg[0]);
    };
    auto positive = [&](double value) {
        if (value < 0.0) error->all(FLERR, "Dump mdz {} must be >= 0", arg[0]);
        return value;
    };

    if (strcmp(arg[0], "precision") == 0) {
        need(2);
        settable();
        precision = positive(utils::numeric(FLERR, arg[1], false, lmp));
        return 2;
    } else if (strcmp(arg[0], "velocity/precision") == 0) {
        need(2);
        settable();
        vel_precision = positive(utils::numeric(FLERR, arg[1], false, lmp));
        return 2;
    } else if (strcmp(arg[0], "force/precision") == 0) {
        need(2);
        settable();
        force_precision = positive(utils::numeric(FLERR, arg[1], false, lmp));
        return 2;
    } else if (strcmp(arg[0], "precision/column") == 0) {
        need(3);
        settable();
        column_precision[arg[1]] = positive(utils::numeric(FLERR, arg[2], false, lmp));
        return 3;
    } else if (strcmp(arg[0], "lossless") == 0) {
        need(2);
        settable();
        lossless_flag = utils::logical(FLERR, arg[1], false, lmp);
        return 2;
    } else if (strcmp(arg[0], "gof") == 0) {
        need(2);
        settable();
        gof_size = utils::inumeric(FLERR, arg[1], false, lmp);
        if (gof_size < 1) error->all(FLERR, "Dump mdz gof must be at least 1");
        return 2;
    } else if (strcmp(arg[0], "variable") == 0) {
        need(2);
        settable();
        variable_flag = utils::logical(FLERR, arg[1], false, lmp);
        return 2;
    } else if (strcmp(arg[0], "rigid") == 0) {
        need(2);
        settable();
        if (strcmp(arg[1], "mol") == 0)
            rigid_flag = 2;
        else
            rigid_flag = utils::logical(FLERR, arg[1], false, lmp);
        return 2;
    }
    for (const char* gone : {"velocities", "forces", "image", "unwrap"})
        if (strcmp(arg[0], gone) == 0)
            error->all(FLERR,
                       "Dump mdz has no dump_modify {}: list the fields in the dump command "
                       "instead (vx vy vz, fx fy fz, ix iy iz, or xu yu zu)",
                       arg[0]);
    return DumpCustom::modify_param(narg, arg);
}

/* ---------------------------------------------------------------------- */

double DumpMDZ::memory_usage() {
    double bytes = DumpCustom::memory_usage();
    bytes += (double)(xs.capacity() + vs.capacity() + fs.capacity()) * sizeof(double);
    bytes += (double)images.capacity() * sizeof(int) + frame_types.capacity() * sizeof(int);
    bytes += (double)(frame_ids.capacity() + first_ids.capacity()) * sizeof(int64_t);
    for (const auto& v : extra_float) bytes += (double)v.capacity() * sizeof(double);
    for (const auto& v : extra_int) bytes += (double)v.capacity() * sizeof(int);
    return bytes;
}
