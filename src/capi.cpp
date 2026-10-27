// The C API declared in include/mdz/mdz.h, implemented over the C++ library.
// The header documents the contract.
//
// Every entry point catches every exception. A failure, or a bug in mdz, is
// reported as a negative MDZ_ERR_* code with a message from mdz_last_error.
// No exception may cross into C code or abort the host process, which is
// often a long simulation such as a LAMMPS run.

#include "mdz/mdz.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "mdz/error.hpp"
#include "mdz/frame_batch.hpp"
#include "mdz/mdz.hpp"
#include "mdz/metadata.hpp"
#include "mdz/trajectory.hpp"
#include "mdz/units.hpp"
#include "mdz/writer.hpp"

namespace {

// Attribute keys for LAMMPS dump settings, matching those used by the lammps
// format reader and writer.
constexpr const char* kAttrUnits = "lammps.units";
constexpr const char* kAttrBoundary = "lammps.boundary";
constexpr const char* kAttrCoords = "lammps.coords";
constexpr const char* kAttrBox = "lammps.box";

thread_local std::string g_last_error;

// An error with its C code, for failures the C layer detects itself.
struct CError {
    int code;
    std::string msg;
};

[[noreturn]] void invalid(std::string msg) { throw CError{MDZ_ERR_INVALID, std::move(msg)}; }

int code_of(mdz::ErrorKind k) noexcept {
    switch (k) {
        case mdz::ErrorKind::InvalidInput:
            return MDZ_ERR_INVALID;
        case mdz::ErrorKind::Corrupt:
            return MDZ_ERR_IO;
        case mdz::ErrorKind::Io:
            return MDZ_ERR_IO;
        case mdz::ErrorKind::OutOfRange:
            return MDZ_ERR_RANGE;
    }
    return MDZ_ERR_INTERNAL;
}

// Copying the message can fail to allocate, in which case it is left empty.
void set_error(const char* msg) noexcept {
    try {
        g_last_error = msg;
    } catch (...) {
        g_last_error.clear();
    }
}

// Run `f`, turning every exception into a C status code.
template <class F>
int guard(F&& f) noexcept {
    try {
        f();
        g_last_error.clear();
        return MDZ_OK;
    } catch (const CError& e) {
        set_error(e.msg.c_str());
        return e.code;
    } catch (const mdz::Error& e) {
        set_error(e.what());
        return code_of(e.kind());
    } catch (const std::bad_alloc&) {
        set_error("out of memory");
        return MDZ_ERR_INTERNAL;
    } catch (const std::exception& e) {
        try {
            g_last_error = std::string("internal error in mdz (please report): ") + e.what();
        } catch (...) {
            g_last_error.clear();
        }
        return MDZ_ERR_INTERNAL;
    } catch (...) {
        set_error("internal error in mdz (please report): unknown exception");
        return MDZ_ERR_INTERNAL;
    }
}

const mdz::LammpsUnits& units_of(const char* name) {
    const std::string_view s = name ? name : "real";
    if (const mdz::LammpsUnits* u = mdz::LammpsUnits::from_style(s)) return *u;
    std::string all;
    for (std::string_view st : mdz::LammpsUnits::styles()) {
        if (!all.empty()) all += ", ";
        all += st;
    }
    invalid("unknown LAMMPS unit style '" + std::string(s) + "' (expected one of " + all + ")");
}

std::size_t atom_count(std::int64_t natoms) {
    if (natoms < 0) invalid("natoms is negative");
    // natoms x 3 values of 8 bytes must be addressable.
    if (static_cast<std::uint64_t>(natoms) > std::numeric_limits<std::size_t>::max() / 24)
        invalid("natoms is too large");
    return static_cast<std::size_t>(natoms);
}

bool file_exists(const std::string& path) {
    if (std::FILE* f = std::fopen(path.c_str(), "rb")) {
        std::fclose(f);
        return true;
    }
    return false;
}

std::vector<float> to_f32(const double* p, std::size_t n) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<float>(p[i]);
    return v;
}

// Whether a caller's struct is at least as large as the one in mdz.h. Fields
// past sizeof(T) in a larger struct, from a newer header, are ignored.
template <class T>
bool big_enough(const T& s) noexcept {
    return s.struct_size >= sizeof(T);
}

std::int32_t id32(std::int64_t id) {
    if (id < std::numeric_limits<std::int32_t>::min() || id > std::numeric_limits<std::int32_t>::max())
        invalid("atom id " + std::to_string(id) + " does not fit in 32 bits");
    return static_cast<std::int32_t>(id);
}

// The number of rigid bodies in a per-atom body index, counting only indices
// shared by two or more atoms.
std::size_t count_bodies(const std::vector<std::int32_t>& bodies) {
    std::map<std::int32_t, std::size_t> size;
    for (std::int32_t b : bodies)
        if (b >= 0) ++size[b];
    std::size_t n = 0;
    for (const auto& [b, k] : size) n += k >= 2;
    return n;
}

// The optional fields a frame carries. The first frame fixes the layout.
struct Layout {
    bool velocities, forces, image, box, time, types;
    bool operator==(const Layout& o) const noexcept {
        return velocities == o.velocities && forces == o.forces && image == o.image && box == o.box && time == o.time &&
               types == o.types;
    }
};

// A general column, as declared to the writer or found by the reader.
struct Column {
    std::string name;
    int type;
    std::uint32_t width;
    double precision;
};

// Names that mdz_frame and mdz_frame_out already have fields for, or that
// per-frame columns use.
bool is_reserved(std::string_view name) {
    for (std::string_view r : {"positions", "velocities", "forces", "image", "box", "origin", "step", "time"})
        if (name == r) return true;
    return name.substr(0, 4) == "mdz.";
}

std::vector<Column> declared_columns(const mdz_writer_options& o) {
    if (o.ncolumns > 0 && !o.columns)
        invalid("opts.columns is null but opts.ncolumns is " + std::to_string(o.ncolumns));
    std::vector<Column> out;
    for (std::uint32_t i = 0; i < o.ncolumns; ++i) {
        const mdz_column& c = o.columns[i];
        const std::string at = "opts.columns[" + std::to_string(i) + "]";
        if (!c.name || !*c.name) invalid(at + " has no name");
        const std::string name = c.name;
        if (name.size() > 255) invalid(at + ": the name is longer than 255 bytes");
        if (is_reserved(name))
            invalid(at + ": '" + name +
                    "' is a reserved name (positions, velocities, forces, image, box, origin, "
                    "step, time and mdz.*)");
        for (const Column& d : out)
            if (d.name == name) invalid(at + ": '" + name + "' is declared twice");
        if (c.type != MDZ_FLOAT && c.type != MDZ_INT) invalid(at + " '" + name + "' has an unknown type");
        if (c.width < 1 || c.width > 255) invalid(at + " '" + name + "': width must be 1 to 255");
        if (!std::isfinite(c.precision) || c.precision < 0.0)
            invalid(at + " '" + name + "': precision must be 0 or positive and finite");
        if (c.type == MDZ_INT && c.precision != 0.0)
            invalid(at + " '" + name + "': an MDZ_INT column has no precision");
        out.push_back(Column{name, c.type, c.width, c.precision});
    }
    return out;
}

std::string describe(const Layout& l) {
    std::string s = "positions";
    if (l.velocities) s += ", velocities";
    if (l.forces) s += ", forces";
    if (l.image) s += ", image";
    if (l.box) s += ", box";
    if (l.time) s += ", time";
    if (l.types) s += ", types";
    return s;
}

}  // namespace

struct mdz_writer {
    std::optional<mdz::Writer> inner;
    std::string path;
    // True when appending to an existing file. That file must be given a
    // footer again on close, even if no frames were added.
    bool appended = false;
    std::size_t natoms = 0;  // 0 with variable atoms
    bool variable = false, rigid = false;
    const mdz::LammpsUnits* units = nullptr;
    mdz::NamedMap<std::string> attributes;
    mdz::Shared<std::int32_t> ids, types, bodies;
    std::int64_t frames = 0;
    std::optional<Layout> layout;
    std::vector<Column> columns;
};

struct mdz_reader {
    mdz::CompressedTrajectory ct;
    bool complete = true;
    const mdz::LammpsUnits* units = nullptr;
    std::string units_c;
    // The last decoded group, converted to the output units, and the atom
    // offset at which each of its frames starts in the per-atom columns.
    std::size_t cached_group = std::numeric_limits<std::size_t>::max();
    mdz::FrameBatch cache;
    std::vector<std::size_t> cache_offsets;
    std::vector<Column> columns;  // general columns
    // Atom counts of every frame when the count varies. Filled on first use.
    mutable std::optional<std::vector<std::size_t>> counts;
    // Static ids and types of a fixed-count file. Filled on first use.
    std::optional<std::vector<std::int64_t>> ids;
    std::optional<std::vector<std::int32_t>> types;
    bool atoms_loaded = false;

    std::size_t frame_natoms(std::size_t i) const {
        if (!ct.variable_atoms) return ct.natoms;
        if (!counts) counts = ct.frame_natoms();
        return (*counts)[i];
    }
};

extern "C" {

const char* mdz_version(void) {
    static const std::string v(mdz::version());
    return v.c_str();
}

const char* mdz_last_error(void) { return g_last_error.c_str(); }

// --- Writer ---

void mdz_writer_options_init(mdz_writer_options* opts) {
    if (!opts) return;
    std::memset(opts, 0, sizeof *opts);
    opts->struct_size = sizeof *opts;
}

void mdz_frame_init(mdz_frame* frame) {
    if (!frame) return;
    std::memset(frame, 0, sizeof *frame);
    frame->struct_size = sizeof *frame;
}

int mdz_writer_open(const char* path, int64_t natoms, const mdz_writer_options* opts, mdz_writer** out) {
    return guard([&] {
        if (!out) invalid("out is null");
        *out = nullptr;
        if (!path) invalid("path is null");
        mdz_writer_options defaults;
        mdz_writer_options_init(&defaults);
        if (opts && !big_enough(*opts)) invalid("opts.struct_size is too small: call mdz_writer_options_init");
        if (opts) defaults = *opts;
        defaults.struct_size = sizeof defaults;
        const mdz_writer_options& o = defaults;
        const bool variable = o.variable_atoms != 0;
        const std::size_t n = variable ? 0 : atom_count(natoms);
        if (variable && o.rigid)
            invalid("rigid bodies need a fixed atom count: opts.rigid cannot be used with opts.variable_atoms");
        const mdz::LammpsUnits& units = units_of(o.units);

        // Precisions arrive in the style's units. CompressOptions takes
        // internal units, and converts back when the data is stored in the
        // style.
        mdz::CompressOptions copts;
        copts.gof_size = o.gof_size == 0 ? 20 : o.gof_size;
        copts.anchor_step = 1;
        copts.lossless = o.lossless != 0;
        copts.variable_atoms = variable;
        copts.rigid = o.rigid != 0;
        if (o.precision > 0.0) copts.precision = static_cast<float>(o.precision * units.length);
        if (o.velocity_precision > 0.0)
            copts.column_precision["velocities"] = static_cast<float>(o.velocity_precision * units.velocity);
        if (o.force_precision > 0.0)
            copts.column_precision["forces"] = static_cast<float>(o.force_precision * units.force);
        std::vector<Column> columns = declared_columns(o);
        for (const Column& c : columns)
            if (c.type == MDZ_FLOAT) copts.column_precision[c.name] = static_cast<float>(c.precision);

        auto w = std::make_unique<mdz_writer>();
        w->path = path;
        w->appended = o.append != 0 && file_exists(w->path);
        w->natoms = n;
        w->variable = variable;
        w->rigid = copts.rigid;
        w->units = &units;
        w->columns = std::move(columns);
        if (w->appended) {
            w->inner.emplace(mdz::Writer::append(w->path, copts));
            const mdz::FrameBatch* meta = w->inner->appended_metadata();
            if (meta && meta->variable() != variable)
                invalid(w->path + (variable ? " has a fixed atom count; append to it without opts.variable_atoms"
                                            : " has an atom count that varies; append to it with opts.variable_atoms"));
            if (meta && !variable && meta->natoms != n)
                invalid(w->path + " holds frames of " + std::to_string(meta->natoms) + " atoms, not " +
                        std::to_string(n));
            if (meta) {
                // The widths and precisions are checked against the file's
                // when the first group is written.
                std::vector<std::pair<std::string, int>> have, want;
                for (const auto& kv : meta->columns)
                    if (!is_reserved(kv.first)) have.emplace_back(kv.first, MDZ_FLOAT);
                for (const auto& kv : meta->int_columns)
                    if (!is_reserved(kv.first)) have.emplace_back(kv.first, MDZ_INT);
                for (const Column& c : w->columns) want.emplace_back(c.name, c.type);
                std::sort(have.begin(), have.end());
                std::sort(want.begin(), want.end());
                if (have != want) {
                    std::string list;
                    for (const auto& h : have) list += (list.empty() ? "" : ", ") + h.first;
                    invalid(w->path + " has the general columns [" + list +
                            "]; opts.columns must declare the same ones to append to it");
                }
            }
        } else {
            w->inner.emplace(mdz::Writer::create(w->path, copts));
        }
        w->frames = static_cast<std::int64_t>(w->inner->frames_written());
        w->attributes[kAttrUnits] = std::string(units.style);
        w->attributes[kAttrBoundary] = o.boundary ? o.boundary : "pp pp pp";
        w->attributes[kAttrCoords] = o.unwrapped ? "unwrapped" : "wrapped";
        w->attributes[kAttrBox] = "triclinic";
        *out = w.release();
    });
}

int mdz_writer_set_atoms(mdz_writer* w, const int64_t* ids, const int32_t* types) {
    return guard([&] {
        if (!w) invalid("writer is null");
        if (w->variable)
            invalid(
                "a writer whose atom count varies takes each frame's ids and types with the frame "
                "(mdz_frame.ids, mdz_frame.types), not mdz_writer_set_atoms");
        const std::size_t n = w->natoms;
        if (ids) {
            std::vector<std::int32_t> v(n);
            for (std::size_t i = 0; i < n; ++i) v[i] = id32(ids[i]);
            w->ids = mdz::share(std::move(v));
        }
        if (types) w->types = mdz::share(std::vector<std::int32_t>(types, types + n));
    });
}

int mdz_writer_set_bodies(mdz_writer* w, const int32_t* bodies) {
    return guard([&] {
        if (!w) invalid("writer is null");
        if (!bodies) invalid("bodies is null");
        if (!w->rigid) invalid("rigid bodies are used only by a writer opened with opts.rigid");
        if (w->layout) invalid("mdz_writer_set_bodies must come before the first frame");
        std::vector<std::int32_t> v(bodies, bodies + w->natoms);
        for (std::int32_t b : v)
            if (b < -1) invalid("body index " + std::to_string(b) + " is not a body (>= 0) or -1 (no body)");
        w->bodies = mdz::share(std::move(v));
    });
}

int mdz_writer_write_frame(mdz_writer* w, const mdz_frame* frame) {
    return guard([&] {
        if (!w) invalid("writer is null");
        if (!frame) invalid("frame is null");
        if (!big_enough(*frame)) invalid("frame.struct_size is too small: call mdz_frame_init");
        const mdz_frame& f = *frame;
        std::size_t n = w->natoms;
        if (w->variable) {
            n = atom_count(f.natoms);
            if (!f.ids && n > 0) invalid("frame.ids is required when the atom count varies");
        } else {
            if (f.natoms != 0 && static_cast<std::uint64_t>(f.natoms) != n)
                invalid("frame.natoms is " + std::to_string(f.natoms) + " but the writer was opened for " +
                        std::to_string(n) + " atoms (open it with opts.variable_atoms to let the count change)");
            if (f.ids || f.types)
                invalid(
                    "frame.ids and frame.types are for a writer whose atom count varies "
                    "(opts.variable_atoms); give fixed ones to mdz_writer_set_atoms");
        }
        if (!f.positions && n > 0) invalid("frame.positions is required");
        if (f.origin && !f.box) invalid("frame.origin given without frame.box");
        Layout layout{f.velocities != nullptr, f.forces != nullptr, f.image != nullptr,
                      f.box != nullptr,        f.has_time != 0,     f.types != nullptr};
        if (n == 0 && w->layout) {
            // A frame of no atoms may leave its per-atom arrays NULL.
            layout.velocities = w->layout->velocities;
            layout.forces = w->layout->forces;
            layout.image = w->layout->image;
            layout.types = w->layout->types;
        }
        // Check here rather than when the group is encoded, so that the
        // error names the frame at fault and the frames before it are kept.
        if (w->layout && !(*w->layout == layout))
            invalid("frame carries " + describe(layout) + "; the first frame carried " + describe(*w->layout));

        const std::size_t n3 = n * 3;
        auto values = [&](const double* p) { return p ? to_f32(p, n3) : std::vector<float>(); };
        mdz::FrameBatch b(n, 1);
        b.units = w->units;
        b.columns.emplace("positions", values(f.positions));
        if (layout.velocities) b.columns.emplace("velocities", values(f.velocities));
        if (layout.forces) b.columns.emplace("forces", values(f.forces));
        if (layout.image)
            b.int_columns.emplace(
                "image", f.image ? std::vector<std::int32_t>(f.image, f.image + n3) : std::vector<std::int32_t>());
        if (f.box) {
            mdz::Box m;
            for (std::size_t i = 0; i < 9; ++i) m[i] = f.box[i];
            mdz::Vec3 o{0.0, 0.0, 0.0};
            if (f.origin) o = {f.origin[0], f.origin[1], f.origin[2]};
            b.box = std::vector<mdz::Box>{m};
            b.origin = std::vector<mdz::Vec3>{o};
        }
        b.int_frame_props.emplace("step", std::vector<std::int64_t>{f.step});
        // Times are stored in ps whatever the unit style.
        if (f.has_time) b.frame_props.emplace("time", std::vector<double>{f.time * w->units->time});
        if (w->variable) {
            std::vector<std::int32_t> ids(n);
            for (std::size_t i = 0; i < n; ++i) ids[i] = id32(f.ids[i]);
            b.atom_ids = mdz::share(std::move(ids));
            if (layout.types)
                b.atom_types =
                    mdz::share(f.types ? std::vector<std::int32_t>(f.types, f.types + n) : std::vector<std::int32_t>());
        } else {
            b.atom_ids = w->ids;
            b.atom_types = w->types;
            b.bodies = w->bodies;
        }
        if (!w->columns.empty() && !f.columns && n > 0)
            invalid("frame.columns is required: the writer was opened with " + std::to_string(w->columns.size()) +
                    " columns");
        for (std::size_t c = 0; c < w->columns.size(); ++c) {
            const Column& col = w->columns[c];
            const void* p = f.columns ? f.columns[c] : nullptr;
            if (!p && n > 0) invalid("frame.columns[" + std::to_string(c) + "] ('" + col.name + "') is null");
            const std::size_t len = p ? n * col.width : 0;
            if (col.type == MDZ_FLOAT)
                b.columns.emplace(col.name, to_f32(static_cast<const double*>(p), len));
            else {
                const auto* q = static_cast<const std::int32_t*>(p);
                b.int_columns.emplace(col.name, std::vector<std::int32_t>(q, q + len));
            }
        }
        b.attributes = w->attributes;
        w->inner->write(b);
        w->layout = layout;
        w->frames += 1;
    });
}

int mdz_writer_flush(mdz_writer* w) {
    return guard([&] {
        if (!w) invalid("writer is null");
        w->inner->flush();
    });
}

int mdz_writer_close(mdz_writer* w) {
    if (!w) return MDZ_OK;
    std::unique_ptr<mdz_writer> owned(w);
    return guard([&] {
        if (!owned->appended && owned->inner->frames_written() == 0) {
            // Without frames there are no columns to describe, so remove the
            // file rather than leave an empty one behind.
            owned->inner.reset();
            std::remove(owned->path.c_str());
            return;
        }
        owned->inner->close();
    });
}

int64_t mdz_writer_frames(const mdz_writer* w) { return w ? w->frames : -1; }

// --- Reader ---

void mdz_frame_out_init(mdz_frame_out* frame) {
    if (!frame) return;
    std::memset(frame, 0, sizeof *frame);
    frame->struct_size = sizeof *frame;
}

int mdz_reader_open(const char* path, mdz_reader** out) {
    return guard([&] {
        if (!out) invalid("out is null");
        *out = nullptr;
        if (!path) invalid("path is null");
        auto r = std::make_unique<mdz_reader>();
        mdz::LoadStatus status;
        r->ct = mdz::load(path, &status);
        r->complete = status.complete;
        // Return frames in the unit style the file is stored in. Files in
        // mdz's internal units, such as those converted from other formats,
        // are read back as metal.
        const mdz::Metadata meta = r->ct.decoded_metadata();
        auto it = meta.attributes.find(mdz::kUnitsAttr);
        if (it == meta.attributes.end()) {
            r->units = &units_of("metal");
        } else {
            std::string_view sys = it->second;
            if (sys.substr(0, 7) == "lammps:") sys.remove_prefix(7);
            r->units = &units_of(std::string(sys).c_str());
        }
        r->units_c = std::string(r->units->style);
        for (const auto& c : r->ct.columns) {
            const mdz::ColumnDesc& d = c.desc;
            if (d.per_frame || is_reserved(d.name)) continue;
            if (d.dtype == mdz::Dtype::F32)
                r->columns.push_back(
                    Column{d.name, MDZ_FLOAT, d.width, d.lossy ? static_cast<double>(d.precision) : 0.0});
            else if (d.dtype == mdz::Dtype::I32)
                r->columns.push_back(Column{d.name, MDZ_INT, d.width, 0.0});
        }
        *out = r.release();
    });
}

void mdz_reader_close(mdz_reader* r) { delete r; }

int64_t mdz_reader_natoms(const mdz_reader* r) { return r ? static_cast<int64_t>(r->ct.natoms) : -1; }

int64_t mdz_reader_nframes(const mdz_reader* r) { return r ? static_cast<int64_t>(r->ct.nframes) : -1; }

int mdz_reader_variable_atoms(const mdz_reader* r) { return r && r->ct.variable_atoms ? 1 : 0; }

int64_t mdz_reader_frame_natoms(const mdz_reader* r, int64_t index) {
    int64_t n = -1;
    const int rc = guard([&] {
        if (!r) invalid("reader is null");
        if (index < 0 || static_cast<std::uint64_t>(index) >= r->ct.nframes)
            throw CError{MDZ_ERR_RANGE, "frame " + std::to_string(index) +
                                            " out of range (nframes = " + std::to_string(r->ct.nframes) + ")"};
        n = static_cast<int64_t>(r->frame_natoms(static_cast<std::size_t>(index)));
    });
    return rc == MDZ_OK ? n : -1;
}

int64_t mdz_reader_bodies(const mdz_reader* r, int32_t* bodies) {
    int64_t n = -1;
    const int rc = guard([&] {
        if (!r) invalid("reader is null");
        const mdz::Metadata meta = r->ct.decoded_metadata();
        n = 0;
        if (!meta.bodies) return;
        n = static_cast<int64_t>(count_bodies(*meta.bodies));
        if (bodies) std::copy(meta.bodies->begin(), meta.bodies->end(), bodies);
    });
    return rc == MDZ_OK ? n : -1;
}

int mdz_reader_complete(const mdz_reader* r) { return r && r->complete ? 1 : 0; }

const char* mdz_reader_units(const mdz_reader* r) { return r ? r->units_c.c_str() : nullptr; }

int mdz_reader_set_units(mdz_reader* r, const char* units) {
    return guard([&] {
        if (!r) invalid("reader is null");
        r->units = &units_of(units);
        r->units_c = std::string(r->units->style);
        r->cached_group = std::numeric_limits<std::size_t>::max();
        r->cache = mdz::FrameBatch();
    });
}

int mdz_reader_has(const mdz_reader* r, const char* column) {
    if (!r || !column) return 0;
    const std::string_view name = column;
    if (name == "box") return r->ct.has_column("box");
    return r->ct.has_column(name) ? 1 : 0;
}

int64_t mdz_reader_ncolumns(const mdz_reader* r) { return r ? static_cast<int64_t>(r->columns.size()) : -1; }

int mdz_reader_column(const mdz_reader* r, int64_t i, mdz_column* column) {
    return guard([&] {
        if (!r) invalid("reader is null");
        if (!column) invalid("column is null");
        if (i < 0 || static_cast<std::uint64_t>(i) >= r->columns.size())
            throw CError{MDZ_ERR_RANGE, "column " + std::to_string(i) +
                                            " out of range (ncolumns = " + std::to_string(r->columns.size()) + ")"};
        const Column& c = r->columns[static_cast<std::size_t>(i)];
        column->name = c.name.c_str();
        column->type = c.type;
        column->width = c.width;
        column->precision = c.precision;
    });
}

int mdz_reader_atom_ids(const mdz_reader* r, int64_t* ids) {
    return guard([&] {
        if (!r) invalid("reader is null");
        if (r->ct.variable_atoms)
            invalid("the file's atoms vary from frame to frame: read each frame's ids with mdz_frame_out.ids");
        const mdz::Metadata meta = r->ct.decoded_metadata();
        if (!meta.atom_ids) invalid("file has no atom ids");
        if (!ids) invalid("ids is null");
        for (std::size_t i = 0; i < meta.atom_ids->size(); ++i) ids[i] = (*meta.atom_ids)[i];
    });
}

int mdz_reader_atom_types(const mdz_reader* r, int32_t* types) {
    return guard([&] {
        if (!r) invalid("reader is null");
        if (r->ct.variable_atoms)
            invalid("the file's atoms vary from frame to frame: read each frame's types with mdz_frame_out.types");
        const mdz::Metadata meta = r->ct.decoded_metadata();
        if (!meta.atom_types) invalid("file has no atom types");
        if (!types) invalid("types is null");
        std::copy(meta.atom_types->begin(), meta.atom_types->end(), types);
    });
}

int mdz_reader_read_frame(mdz_reader* r, int64_t index, mdz_frame_out* frame) {
    return guard([&] {
        if (!r) invalid("reader is null");
        if (!frame) invalid("frame is null");
        mdz_frame_out& f = *frame;
        if (!big_enough(f)) invalid("frame.struct_size is too small: call mdz_frame_out_init");
        if (index < 0 || static_cast<std::uint64_t>(index) >= r->ct.nframes)
            throw CError{MDZ_ERR_RANGE, "frame " + std::to_string(index) +
                                            " out of range (nframes = " + std::to_string(r->ct.nframes) + ")"};
        const auto i = static_cast<std::size_t>(index);
        const std::size_t g = r->ct.gofs.overlapping(i, i + 1).start;
        const auto span = r->ct.gofs.span(g);
        if (r->cached_group != g) {
            r->cached_group = std::numeric_limits<std::size_t>::max();
            r->cache = mdz::decompress(r->ct, span.start, span.end);
            r->cache.convert_units(r->units);
            r->cache_offsets = r->cache.atom_offsets();
            r->cached_group = g;
        }
        const mdz::FrameBatch& b = r->cache;
        const std::size_t local = i - span.start;
        const std::size_t first = r->cache_offsets.at(local), n = r->cache_offsets.at(local + 1) - first;
        const std::size_t n3 = n * 3;
        auto copy = [&](double* dst, const char* name) {
            if (!dst) return;
            const std::vector<float>* src = b.column(name);
            if (!src) invalid(std::string("file has no ") + name);
            if (src->size() < (first + n) * 3)
                throw mdz::Error(mdz::ErrorKind::Corrupt, std::string(name) + " is short");
            const float* s = src->data() + first * 3;
            for (std::size_t k = 0; k < n3; ++k) dst[k] = static_cast<double>(s[k]);
        };
        // Per-atom ids and types come from columns when the count varies,
        // and from the static metadata otherwise.
        auto per_atom = [&](const char* column) -> const std::int32_t* {
            auto it = b.int_columns.find(column);
            if (it == b.int_columns.end()) return nullptr;
            if (it->second.size() < first + n)
                throw mdz::Error(mdz::ErrorKind::Corrupt, std::string(column) + " is short");
            return it->second.data() + first;
        };
        if ((f.ids || f.types) && !r->ct.variable_atoms && !r->atoms_loaded) {
            const mdz::Metadata meta = r->ct.decoded_metadata();
            if (meta.atom_ids) r->ids.emplace(meta.atom_ids->begin(), meta.atom_ids->end());
            r->types = meta.atom_types;
            r->atoms_loaded = true;
        }
        if (f.ids) {
            if (r->ct.variable_atoms) {
                const std::int32_t* ids = per_atom(mdz::columns::kId);
                if (!ids) throw mdz::Error(mdz::ErrorKind::Corrupt, "frame has no atom ids");
                std::copy_n(ids, n, f.ids);
            } else if (r->ids && r->ids->size() == n) {
                std::copy_n(r->ids->data(), n, f.ids);
            } else {
                for (std::size_t k = 0; k < n; ++k) f.ids[k] = static_cast<int64_t>(k);
            }
        }
        if (f.types) {
            const std::int32_t* types = r->ct.variable_atoms                ? per_atom(mdz::columns::kType)
                                        : r->types && r->types->size() == n ? r->types->data()
                                                                            : nullptr;
            if (!types) invalid("file has no atom types");
            std::copy_n(types, n, f.types);
        }
        copy(f.positions, "positions");
        copy(f.velocities, "velocities");
        copy(f.forces, "forces");
        if (f.image) {
            auto it = b.int_columns.find("image");
            if (it == b.int_columns.end()) invalid("file has no image flags");
            if (it->second.size() < (first + n) * 3) throw mdz::Error(mdz::ErrorKind::Corrupt, "image is short");
            std::copy_n(it->second.data() + first * 3, n3, f.image);
        }
        for (std::size_t c = 0; f.columns && c < r->columns.size(); ++c) {
            void* dst = f.columns[c];
            if (!dst) continue;
            const Column& col = r->columns[c];
            const std::size_t at = first * col.width, len = n * col.width;
            auto short_column = [&] {
                throw mdz::Error(mdz::ErrorKind::Corrupt, "column '" + col.name + "' is short");
            };
            if (col.type == MDZ_FLOAT) {
                const std::vector<float>* src = b.column(col.name);
                if (!src || src->size() < at + len) short_column();
                auto* out = static_cast<double*>(dst);
                for (std::size_t k = 0; k < len; ++k) out[k] = static_cast<double>((*src)[at + k]);
            } else {
                auto it = b.int_columns.find(col.name);
                if (it == b.int_columns.end() || it->second.size() < at + len) short_column();
                std::copy_n(it->second.data() + at, len, static_cast<std::int32_t*>(dst));
            }
        }
        f.natoms = static_cast<int64_t>(n);
        auto step = b.int_frame_props.find("step");
        f.has_step = step != b.int_frame_props.end() && local < step->second.size();
        f.step = f.has_step ? step->second[local] : 0;
        auto time = b.frame_props.find("time");
        f.has_time = time != b.frame_props.end() && local < time->second.size();
        f.time = f.has_time ? time->second[local] / r->units->time : 0.0;
        f.has_box = b.box && local < b.box->size();
        for (int k = 0; k < 9; ++k) f.box[k] = f.has_box ? (*b.box)[local][static_cast<std::size_t>(k)] : 0.0;
        const bool has_origin = b.origin && local < b.origin->size();
        for (int k = 0; k < 3; ++k) f.origin[k] = has_origin ? (*b.origin)[local][static_cast<std::size_t>(k)] : 0.0;
    });
}

}  // extern "C"
