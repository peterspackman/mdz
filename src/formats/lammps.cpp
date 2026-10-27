// LAMMPS dump files (dump atom and dump custom). Each frame is
//
//   ITEM: TIMESTEP
//   <step>
//   ITEM: NUMBER OF ATOMS
//   <natoms>
//   ITEM: BOX BOUNDS [xy xz yz | abc origin] <boundary flags>
//   <three lines of box geometry>
//   ITEM: ATOMS <column names>
//   <one line per atom>
//
// A frame may be preceded by "ITEM: TIME" and the simulation time, as written
// by `dump_modify time yes`.
//
// Units. A dump does not record its unit style, so the reader is given one
// (FormatOptions::lammps_units, default real). Values are kept in the dump's
// own units and the batch's `units` field records which, so a dump
// round-trips exactly. Times are converted to ps. The writer uses the
// requested style if there is one, then the batch's own style, then the
// lammps.units attribute, then real.
//
// Boxes. Orthogonal (lo hi per axis), restricted triclinic (xy xz yz) and
// general triclinic (abc origin) boxes are all read exactly into an origin and
// a matrix of cell vectors. For restricted triclinic boxes the dump gives the
// bounds of the tilted box's bounding box. The writer uses the simplest form
// that represents the matrix.
//
// Atoms are sorted by id. The set of atoms may change between frames, for
// example through deposition, delete_atoms or lost atoms. From the first
// change on, the reader returns variable-atom-count batches, with atom ids
// taken from the id column.
//
// Columns. Coordinates may be wrapped (x), unwrapped (xu) or scaled (xs, xsu).
// Scaled coordinates are stored as Cartesian and written back as fractions to
// 6 significant digits. Image flags become the integer column "image" (width
// 3). Every other column is carried through under its own name, as an integer
// for mol, proc and i_* columns and as a float otherwise.
//
// Attributes recorded on the batch are lammps.units, lammps.boundary,
// lammps.coords (wrapped, unwrapped, scaled, scaled-unwrapped), lammps.box
// (orthogonal, triclinic, general) and lammps.columns (the original column
// order).

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>

#include "internal.hpp"
#include "mdz/error.hpp"
#include "text.hpp"

namespace mdz::detail {
namespace {

constexpr const char* kAttrUnits = "lammps.units";
constexpr const char* kAttrBoundary = "lammps.boundary";
constexpr const char* kAttrCoords = "lammps.coords";
constexpr const char* kAttrBox = "lammps.box";
constexpr const char* kAttrColumns = "lammps.columns";
constexpr std::string_view kDefaultUnits = "real";

[[noreturn]] void corrupt_at(const text::LineReader& lr, const std::string& msg) {
    throw Error(ErrorKind::Corrupt, lr.name() + ":" + std::to_string(lr.line_number()) + ": " + msg);
}

[[noreturn]] void invalid(const std::string& msg) { throw Error(ErrorKind::InvalidInput, msg); }

std::string quoted(std::string_view s) { return "\"" + std::string(s) + "\""; }

// Returns true for per-atom columns stored as integers. Any other unrecognised
// column is stored as float.
bool is_int_column(std::string_view name) {
    return name == "mol" || name == "proc" || name == "procp1" || text::starts_with(name, "i_") ||
           text::starts_with(name, "i2_");
}

enum class BoxForm { Orthogonal, Triclinic, General };

const char* form_name(BoxForm f) {
    switch (f) {
        case BoxForm::Orthogonal:
            return "orthogonal";
        case BoxForm::Triclinic:
            return "triclinic";
        case BoxForm::General:
            return "general";
    }
    return "orthogonal";
}

// --- reader ---

// Destination of one token of an atom line.
struct Slot {
    enum Kind : std::uint8_t { Id, Type, Element, Vec, Image, Float, Int } kind;
    std::uint32_t k = 0;  // component or carried-column index
};

// How to read the atom lines under one ITEM: ATOMS header. Built once for
// each distinct header.
struct Plan {
    std::string header;
    std::vector<std::string> names;
    std::vector<Slot> slots;
    bool scaled = false, unwrapped = false;
    bool has_vel = false, has_force = false, has_image = false, has_type = false, has_element = false;
    std::vector<std::string> float_cols, int_cols;
    // The batch columns a frame produces, as (name, source) pairs. For float
    // columns, source -1, -2 and -3 mean positions, velocities and forces. For
    // integer columns, -1 means image. A source k >= 0 is the k-th carried
    // column. A later column with the same name replaces an earlier one.
    std::vector<std::pair<std::string, int>> float_out, int_out;
    std::string columns_attr;
};

// One parsed atom line, before sorting by id.
struct Atom {
    std::int64_t id;
    std::int32_t type;
    std::uint32_t el_off, el_len;
    double vec[9];  // coordinates, velocities, forces
    std::int32_t image[3];
};

// The atoms that frames are compared against to tell whether they changed.
struct ReferenceAtoms {
    std::size_t natoms;
    Shared<std::int32_t> ids;
    Shared<std::int32_t> types;
    Shared<std::string> elements;
    NamedMap<std::string> attributes;
};

struct LammpsBox {
    Vec3 origin;
    Box matrix;
    BoxForm form;
};

class LammpsReader final : public TrajectoryReader {
public:
    LammpsReader(std::unique_ptr<InputStream> in, const LammpsUnits* units)
        : in_(std::move(in)), lines_(*in_), units_(units) {}

    // The atom count of the first frame. Later frames may differ.
    std::size_t natoms() const override { return first_natoms_; }

    std::optional<FrameBatch> read_batch(std::size_t n) override {
        std::optional<FrameBatch> out;
        for (std::size_t i = 0; i < n; ++i)
            if (!read_frame(out)) break;
        return out;
    }

private:
    // Reads the next line and trims it. Returns false at end of input.
    bool line_trimmed(std::string_view& line) {
        if (!lines_.next(line)) return false;
        line = text::trim(line);
        return true;
    }

    std::string_view required_line(const char* what) {
        std::string_view line;
        if (!line_trimmed(line))
            throw Error(ErrorKind::Corrupt, lines_.name() + ": dump ends where " + what + " was expected");
        return line;
    }

    // Reads three lines of at least `per_line` numbers each. Only the first
    // four numbers of a line are kept.
    void number_lines(std::size_t per_line, double (&out)[3][4]) {
        for (auto& row : out) {
            const std::string_view line = required_line("box bounds");
            std::string_view rest = line;
            std::size_t n = 0;
            for (std::string_view tok = text::next_token(rest); !tok.empty(); tok = text::next_token(rest)) {
                double v;
                if (!text::parse(tok, v))
                    corrupt_at(lines_, "bad box bounds line " + quoted(line) + ": bad number " + quoted(tok));
                if (n < 4) row[n] = v;
                ++n;
            }
            if (n < per_line)
                corrupt_at(lines_,
                           "box bounds line " + quoted(line) + " needs " + std::to_string(per_line) + " numbers");
        }
    }

    // Reads ITEM: BOX BOUNDS and its three lines. The boundary flags are
    // stored in `boundary`.
    LammpsBox read_box(std::string_view line, std::string& boundary) {
        if (!text::starts_with(line, "ITEM: BOX BOUNDS"))
            corrupt_at(lines_, "expected ITEM: BOX BOUNDS, got: " + std::string(line));
        // Copy the header line, because reading the numbers invalidates it.
        box_header_.assign(line.data(), line.size());
        std::string_view rest = std::string_view(box_header_).substr(16);
        std::string_view toks[6];
        std::size_t ntok = 0;
        for (std::string_view t = text::next_token(rest); !t.empty(); t = text::next_token(rest)) {
            if (ntok < 6) toks[ntok] = t;
            ++ntok;
        }
        auto flags = [&](std::size_t skip) {
            // Dumps from LAMMPS versions before 2009 have no boundary flags.
            // Treat them as fully periodic.
            if (ntok != skip + 3) return std::string("pp pp pp");
            return std::string(toks[skip]) + " " + std::string(toks[skip + 1]) + " " + std::string(toks[skip + 2]);
        };
        double l[3][4] = {};
        LammpsBox b{};
        if (ntok >= 2 && toks[0] == "abc" && toks[1] == "origin") {
            // General triclinic. Each line holds a cell vector followed by one
            // component of the origin.
            number_lines(4, l);
            b.matrix = {l[0][0], l[0][1], l[0][2], l[1][0], l[1][1], l[1][2], l[2][0], l[2][1], l[2][2]};
            b.origin = {l[0][3], l[1][3], l[2][3]};
            b.form = BoxForm::General;
            boundary = flags(2);
            return b;
        }
        if (ntok >= 3 && toks[0] == "xy" && toks[1] == "xz" && toks[2] == "yz") {
            // Restricted triclinic. The bounds are those of the tilted box's
            // bounding box, so recover the box's own extent first.
            number_lines(3, l);
            const double xy = l[0][2], xz = l[1][2], yz = l[2][2];
            const double xlo = l[0][0] - std::fmin(std::fmin(std::fmin(0.0, xy), xz), xy + xz);
            const double xhi = l[0][1] - std::fmax(std::fmax(std::fmax(0.0, xy), xz), xy + xz);
            const double ylo = l[1][0] - std::fmin(0.0, yz);
            const double yhi = l[1][1] - std::fmax(0.0, yz);
            const double zlo = l[2][0], zhi = l[2][1];
            b.origin = {xlo, ylo, zlo};
            b.matrix = {xhi - xlo, 0.0, 0.0, xy, yhi - ylo, 0.0, xz, yz, zhi - zlo};
            b.form = BoxForm::Triclinic;
            boundary = flags(3);
            return b;
        }
        number_lines(2, l);
        b.origin = {l[0][0], l[1][0], l[2][0]};
        b.matrix = {l[0][1] - l[0][0], 0.0, 0.0, 0.0, l[1][1] - l[1][0], 0.0, 0.0, 0.0, l[2][1] - l[2][0]};
        b.form = BoxForm::Orthogonal;
        boundary = flags(0);
        return b;
    }

    // Returns the plan for an ITEM: ATOMS line. The last plan is cached,
    // since the header rarely changes between frames.
    const Plan& plan_for(std::string_view atoms_line) {
        if (plan_ && plan_->header == atoms_line) return *plan_;
        if (!text::starts_with(atoms_line, "ITEM: ATOMS"))
            corrupt_at(lines_, "expected ITEM: ATOMS, got: " + std::string(atoms_line));
        Plan p;
        p.header.assign(atoms_line.data(), atoms_line.size());
        std::string_view rest = atoms_line.substr(11);
        for (std::string_view t = text::next_token(rest); !t.empty(); t = text::next_token(rest))
            p.names.emplace_back(t);
        const std::size_t n = p.names.size();
        // A repeated column name resolves to its last occurrence.
        auto index = [&](std::string_view name) -> std::ptrdiff_t {
            for (std::size_t i = n; i-- > 0;)
                if (p.names[i] == name) return static_cast<std::ptrdiff_t>(i);
            return -1;
        };
        auto has = [&](std::string_view name) { return index(name) >= 0; };
        auto triple = [&](const char* a, const char* b, const char* c) { return has(a) && has(b) && has(c); };
        const char* const* coords;
        static const char *const kXu[] = {"xu", "yu", "zu"}, *const kX[] = {"x", "y", "z"},
                                 *const kXsu[] = {"xsu", "ysu", "zsu"}, *const kXs[] = {"xs", "ys", "zs"};
        if (triple("xu", "yu", "zu")) {
            coords = kXu;
            p.unwrapped = true;
        } else if (triple("x", "y", "z")) {
            coords = kX;
        } else if (triple("xsu", "ysu", "zsu")) {
            coords = kXsu;
            p.scaled = p.unwrapped = true;
        } else if (triple("xs", "ys", "zs")) {
            coords = kXs;
            p.scaled = true;
        } else {
            corrupt_at(lines_, "dump has no coordinates (x y z, xu yu zu, xs ys zs or xsu ysu zsu): " +
                                   std::string(atoms_line));
        }
        p.has_vel = triple("vx", "vy", "vz");
        p.has_force = triple("fx", "fy", "fz");
        p.has_image = triple("ix", "iy", "iz");
        p.has_type = has("type");
        p.has_element = has("element");

        std::vector<std::optional<Slot>> slot(n);
        auto set = [&](std::string_view name, Slot::Kind kind, std::uint32_t k) {
            const std::ptrdiff_t i = index(name);
            if (i >= 0) slot[static_cast<std::size_t>(i)] = Slot{kind, k};
        };
        set("id", Slot::Id, 0);
        set("type", Slot::Type, 0);
        set("element", Slot::Element, 0);
        for (std::uint32_t k = 0; k < 3; ++k) set(coords[k], Slot::Vec, k);
        static const char *const kV[] = {"vx", "vy", "vz"}, *const kF[] = {"fx", "fy", "fz"},
                                 *const kI[] = {"ix", "iy", "iz"};
        if (p.has_vel)
            for (std::uint32_t k = 0; k < 3; ++k) set(kV[k], Slot::Vec, 3 + k);
        if (p.has_force)
            for (std::uint32_t k = 0; k < 3; ++k) set(kF[k], Slot::Vec, 6 + k);
        if (p.has_image)
            for (std::uint32_t k = 0; k < 3; ++k) set(kI[k], Slot::Image, k);
        // Every remaining column is carried through by name.
        p.slots.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            if (slot[i]) {
                p.slots[i] = *slot[i];
            } else if (is_int_column(p.names[i])) {
                p.slots[i] = Slot{Slot::Int, static_cast<std::uint32_t>(p.int_cols.size())};
                p.int_cols.push_back(p.names[i]);
            } else {
                p.slots[i] = Slot{Slot::Float, static_cast<std::uint32_t>(p.float_cols.size())};
                p.float_cols.push_back(p.names[i]);
            }
        }
        auto put = [](std::vector<std::pair<std::string, int>>& v, const std::string& name, int src) {
            for (auto& e : v)
                if (e.first == name) {
                    e.second = src;
                    return;
                }
            v.emplace_back(name, src);
        };
        put(p.float_out, "positions", -1);
        if (p.has_vel) put(p.float_out, "velocities", -2);
        if (p.has_force) put(p.float_out, "forces", -3);
        for (std::size_t k = 0; k < p.float_cols.size(); ++k) put(p.float_out, p.float_cols[k], static_cast<int>(k));
        if (p.has_image) put(p.int_out, "image", -1);
        for (std::size_t k = 0; k < p.int_cols.size(); ++k) put(p.int_out, p.int_cols[k], static_cast<int>(k));
        for (std::size_t i = 0; i < n; ++i) {
            if (i) p.columns_attr += ' ';
            p.columns_attr += p.names[i];
        }
        plan_ = std::move(p);
        return *plan_;
    }

    // Appends one frame to *out, creating it on the first frame. Returns false
    // at end of input.
    bool read_frame(std::optional<FrameBatch>& out) {
        std::string_view line;
        do {
            if (!line_trimmed(line)) return false;
        } while (line.empty());
        // `dump_modify time yes` writes the simulation time before the step.
        std::optional<double> time;
        if (text::starts_with(line, "ITEM: TIME") && !text::starts_with(line, "ITEM: TIMESTEP")) {
            const std::string_view t = required_line("time");
            double v;
            if (!text::parse(t, v)) corrupt_at(lines_, "bad time " + quoted(t));
            time = v;
            line = required_line("ITEM: TIMESTEP");
        }
        if (!text::starts_with(line, "ITEM: TIMESTEP"))
            corrupt_at(lines_, "expected ITEM: TIMESTEP, got: " + std::string(line));
        const std::string_view step_line = required_line("timestep");
        std::int64_t step;
        if (!text::parse(step_line, step)) corrupt_at(lines_, "bad timestep " + quoted(step_line));

        line = required_line("ITEM: NUMBER OF ATOMS");
        if (!text::starts_with(line, "ITEM: NUMBER OF ATOMS"))
            corrupt_at(lines_, "expected ITEM: NUMBER OF ATOMS, got: " + std::string(line));
        const std::string_view count_line = required_line("atom count");
        std::uint64_t count;
        if (!text::parse(count_line, count)) corrupt_at(lines_, "bad atom count " + quoted(count_line));
        const std::size_t natoms = static_cast<std::size_t>(count);

        std::string boundary;
        const LammpsBox lbox = read_box(required_line("ITEM: BOX BOUNDS"), boundary);
        const Plan& plan = plan_for(required_line("ITEM: ATOMS"));
        const std::size_t ncols = plan.names.size();

        // Parse the atom lines in one pass, with one slot lookup per token.
        // Reservations are capped because the count may be corrupt.
        const std::size_t cap = std::min<std::size_t>(natoms, std::size_t(1) << 24);
        atoms_.clear();
        atoms_.reserve(cap);
        floats_.resize(plan.float_cols.size());
        ints_.resize(plan.int_cols.size());
        for (auto& v : floats_) {
            v.clear();
            v.reserve(cap);
        }
        for (auto& v : ints_) {
            v.clear();
            v.reserve(cap);
        }
        elem_arena_.clear();
        for (std::size_t li = 0; li < natoms; ++li) {
            if (!lines_.next(line))
                throw Error(ErrorKind::Corrupt, lines_.name() + ": frame at step " + std::to_string(step) +
                                                    " ends after " + std::to_string(li) + " of " +
                                                    std::to_string(natoms) + " atoms");
            Atom& a = atoms_.emplace_back();
            a.id = static_cast<std::int64_t>(li) + 1;
            a.type = 1;
            a.el_off = a.el_len = 0;
            std::fill(std::begin(a.vec), std::end(a.vec), 0.0);
            a.image[0] = a.image[1] = a.image[2] = 0;
            std::size_t i = 0;
            const char* p = line.data();
            const char* const e = p + line.size();
            for (;; ++i) {
                while (p != e && text::is_space(*p)) ++p;
                if (p == e) break;
                if (i >= ncols) {
                    p = text::token_end(p, e);
                    continue;
                }
                const char* q = nullptr;
                const Slot sl = plan.slots[i];
                switch (sl.kind) {
                    case Slot::Id:
                        q = text::scan_number(p, e, a.id);
                        break;
                    case Slot::Type:
                        q = text::scan_number(p, e, a.type);
                        break;
                    case Slot::Element:
                        q = text::token_end(p, e);
                        a.el_off = static_cast<std::uint32_t>(elem_arena_.size());
                        a.el_len = static_cast<std::uint32_t>(q - p);
                        elem_arena_.append(p, static_cast<std::size_t>(q - p));
                        break;
                    case Slot::Vec:
                        q = text::scan_number(p, e, a.vec[sl.k]);
                        break;
                    case Slot::Image:
                        q = text::scan_number(p, e, a.image[sl.k]);
                        break;
                    case Slot::Float: {
                        float v = 0;
                        q = text::scan_number(p, e, v);
                        floats_[sl.k].push_back(v);
                        break;
                    }
                    case Slot::Int: {
                        std::int32_t v = 0;
                        q = text::scan_number(p, e, v);
                        ints_[sl.k].push_back(v);
                        break;
                    }
                }
                if (!q) {
                    const std::string_view tok(p, static_cast<std::size_t>(text::token_end(p, e) - p));
                    corrupt_at(lines_, "atom line " + std::to_string(li + 1) + " of step " + std::to_string(step) +
                                           ", column '" + plan.names[i] + "': " + quoted(tok) + ": invalid number");
                }
                p = q;
            }
            if (i != ncols)
                corrupt_at(lines_, "atom line " + std::to_string(li + 1) + " of step " + std::to_string(step) +
                                       " has " + std::to_string(i) + " values for " + std::to_string(ncols) +
                                       " columns");
        }

        // Sort by id, keeping atoms with equal ids in input order. Most dumps
        // are already sorted.
        bool sorted = true;
        for (std::size_t i = 1; i < natoms && sorted; ++i) sorted = atoms_[i - 1].id <= atoms_[i].id;
        if (!sorted) {
            sort_order(natoms);
            // The atoms are reordered here. The carried columns are read
            // through order_ when they are copied into the batch.
            sorted_atoms_.resize(natoms);
            for (std::size_t i = 0; i < natoms; ++i) sorted_atoms_[i] = atoms_[order_[i]];
            atoms_.swap(sorted_atoms_);
        }
        auto at = [&](std::size_t i) -> const Atom& { return atoms_[i]; };
        auto element = [&](const Atom& a) { return std::string_view(elem_arena_).substr(a.el_off, a.el_len); };

        for (std::size_t i = 0; i < natoms; ++i) {
            const std::int64_t id = at(i).id;
            if (id < std::numeric_limits<std::int32_t>::min() || id > std::numeric_limits<std::int32_t>::max())
                corrupt_at(lines_,
                           "atom id " + std::to_string(id) + " does not fit in 32 bits, which mdz stores ids in");
        }

        // Usually the atoms match the reference frame and the frame is
        // appended to the batch. If atoms were inserted, deleted, lost or
        // retyped, this frame becomes the new reference and the reader returns
        // variable batches from then on.
        bool changed = !ref_;
        if (ref_) {
            const ReferenceAtoms& f = *ref_;
            changed = natoms != f.natoms || static_cast<bool>(f.types) != plan.has_type ||
                      static_cast<bool>(f.elements) != plan.has_element;
            for (std::size_t i = 0; !changed && i < natoms; ++i)
                changed = (*f.ids)[i] != static_cast<std::int32_t>(at(i).id) ||
                          (plan.has_type && (*f.types)[i] != at(i).type) ||
                          (plan.has_element && (*f.elements)[i] != element(at(i)));
            varying_ = varying_ || changed;
        }
        if (changed) {
            ReferenceAtoms f;
            f.natoms = natoms;
            std::vector<std::int32_t> ids(natoms);
            for (std::size_t i = 0; i < natoms; ++i) ids[i] = static_cast<std::int32_t>(at(i).id);
            f.ids = share(std::move(ids));
            if (plan.has_type) {
                std::vector<std::int32_t> types(natoms);
                for (std::size_t i = 0; i < natoms; ++i) types[i] = at(i).type;
                f.types = share(std::move(types));
            }
            if (plan.has_element) {
                std::vector<std::string> els;
                els.reserve(natoms);
                for (std::size_t i = 0; i < natoms; ++i) els.emplace_back(element(at(i)));
                f.elements = share(std::move(els));
            }
            if (ref_) {
                // Attributes describe the whole file, so keep those of the first
                // frame.
                f.attributes = std::move(ref_->attributes);
            } else {
                first_natoms_ = natoms;
                f.attributes[kAttrUnits] = std::string(units_->style);
                f.attributes[kAttrBoundary] = boundary;
                f.attributes[kAttrCoords] = plan.scaled ? (plan.unwrapped ? "scaled-unwrapped" : "scaled")
                                                        : (plan.unwrapped ? "unwrapped" : "wrapped");
                f.attributes[kAttrBox] = form_name(lbox.form);
                f.attributes[kAttrColumns] = plan.columns_attr;
            }
            ref_ = std::move(f);
        }

        // Values stay in the dump's units and the batch is tagged with them.
        // In variable mode each frame is built separately and then appended
        // to the batch.
        std::optional<FrameBatch> one;
        std::optional<FrameBatch>& target = varying_ ? one : out;
        if (!target) target.emplace(natoms, 0);
        FrameBatch& b = *target;
        const Box& m = lbox.matrix;
        const Vec3& o = lbox.origin;
        for (const auto& [name, src] : plan.float_out) {
            std::vector<float>& col = column_for(b.columns, name);
            const std::size_t base = col.size();
            if (src >= 0) {
                const std::vector<float>& v = floats_[static_cast<std::size_t>(src)];
                col.resize(base + v.size());
                for (std::size_t i = 0; i < v.size(); ++i) col[base + i] = v[sorted ? i : order_[i]];
                continue;
            }
            col.resize(base + natoms * 3);
            float* dst = col.data() + base;
            if (src == -1 && plan.scaled) {
                // Convert fractional to Cartesian as origin + s a + t b + u c.
                for (std::size_t i = 0; i < natoms; ++i) {
                    const double* v = at(i).vec;
                    for (int k = 0; k < 3; ++k)
                        dst[3 * i + k] = static_cast<float>(o[k] + v[0] * m[k] + v[1] * m[3 + k] + v[2] * m[6 + k]);
                }
            } else {
                const int off = src == -1 ? 0 : src == -2 ? 3 : 6;
                for (std::size_t i = 0; i < natoms; ++i) {
                    const double* v = at(i).vec + off;
                    for (int k = 0; k < 3; ++k) dst[3 * i + k] = static_cast<float>(v[k]);
                }
            }
        }
        for (const auto& [name, src] : plan.int_out) {
            std::vector<std::int32_t>& col = column_for(b.int_columns, name);
            const std::size_t base = col.size();
            if (src >= 0) {
                const std::vector<std::int32_t>& v = ints_[static_cast<std::size_t>(src)];
                col.resize(base + v.size());
                for (std::size_t i = 0; i < v.size(); ++i) col[base + i] = v[sorted ? i : order_[i]];
                continue;
            }
            col.resize(base + natoms * 3);
            for (std::size_t i = 0; i < natoms; ++i)
                for (int k = 0; k < 3; ++k) col[base + 3 * i + k] = at(i).image[k];
        }
        if (!b.box) b.box.emplace();
        b.box->push_back(m);
        if (!b.origin) b.origin.emplace();
        b.origin->push_back(o);
        b.int_frame_props["step"].push_back(step);
        // Times are stored in ps regardless of the unit style.
        if (time) b.frame_props["time"].push_back(*time * units_->time);
        if (!b.units) b.units = units_;
        if (!b.atom_ids) b.atom_ids = ref_->ids;
        if (!b.atom_types) b.atom_types = ref_->types;
        if (!b.elements) b.elements = ref_->elements;
        if (b.attributes.empty()) b.attributes = ref_->attributes;
        b.nframes += 1;
        if (varying_) append_varying(out, std::move(*one));
        return true;
    }

    // Sets order_ to the atom indices sorted by id, with equal ids in input
    // order.
    void sort_order(std::size_t n) {
        order_.resize(n);
        // When the ids are a permutation of a contiguous range, which is the
        // common case, place each atom directly. Fall back to sorting if an id
        // is repeated.
        std::int64_t lo = atoms_[0].id, hi = atoms_[0].id;
        for (const Atom& a : atoms_) lo = std::min(lo, a.id), hi = std::max(hi, a.id);
        if (static_cast<std::uint64_t>(hi) - static_cast<std::uint64_t>(lo) == n - 1) {
            constexpr std::size_t kNone = ~std::size_t(0);
            std::fill(order_.begin(), order_.end(), kNone);
            bool dup = false;
            for (std::size_t i = 0; i < n && !dup; ++i) {
                std::size_t& slot = order_[static_cast<std::size_t>(static_cast<std::uint64_t>(atoms_[i].id) -
                                                                    static_cast<std::uint64_t>(lo))];
                dup = slot != kNone;
                slot = i;
            }
            if (!dup) return;
        }
        keys_.resize(n);
        for (std::size_t i = 0; i < n; ++i) keys_[i] = {atoms_[i].id, i};
        std::sort(keys_.begin(), keys_.end());  // (id, index) pairs keep equal ids stable
        for (std::size_t i = 0; i < n; ++i) order_[i] = keys_[i].second;
    }

    template <class T>
    static std::vector<T>& column_for(NamedMap<std::vector<T>>& m, const std::string& name) {
        auto it = m.find(name);
        if (it == m.end()) it = m.emplace(name, std::vector<T>()).first;
        return it->second;
    }

    std::unique_ptr<InputStream> in_;
    text::LineReader lines_;
    const LammpsUnits* units_;
    // Taken from the most recent frame whose atoms differed from the frame
    // before it.
    std::optional<ReferenceAtoms> ref_;
    std::size_t first_natoms_ = 0;
    // Set once the atoms have changed.
    bool varying_ = false;
    std::optional<Plan> plan_;
    // Scratch buffers reused across frames.
    std::vector<Atom> atoms_;
    std::vector<std::vector<float>> floats_;
    std::vector<std::vector<std::int32_t>> ints_;
    std::vector<Atom> sorted_atoms_;
    std::vector<std::size_t> order_;
    std::vector<std::pair<std::int64_t, std::size_t>> keys_;
    std::string elem_arena_;
    std::string box_header_;
};

// --- writer ---

// Rounds v to `digits` significant digits by formatting it in scientific
// notation and parsing it back.
double round_sig(double v, int digits) {
    if (!std::isfinite(v)) return v;
    char b[64];
    auto r = std::to_chars(b, b + sizeof b, v, std::chars_format::scientific, digits - 1);
    double out;
    if (!text::parse(std::string_view(b, static_cast<std::size_t>(r.ptr - b)), out)) return v;
    return out;
}

// Rounds a box value to 15 significant digits. Converting to nm and back can
// introduce an error of about one ulp (for example, -1.41 comes back as
// -1.4099999999999997). Rounding removes it, so the value is written as read.
double g15(double v) { return round_sig(v, 15); }
// Rounds a fractional coordinate to 6 significant digits, the precision LAMMPS
// writes in dumps.
double g6(double v) { return round_sig(v, 6); }

// Writes one frame's ITEM: BOX BOUNDS block, in the simplest form that
// represents the box exactly.
void box_block(text::TextWriter& w, Vec3 origin, Box m, std::string_view boundary, bool general) {
    for (double& x : origin) x = g15(x);
    for (double& x : m) x = g15(x);
    auto num = [&](double v) -> text::TextWriter& { return w.put_shortest(v); };
    const bool restricted = m[1] == 0.0 && m[2] == 0.0 && m[5] == 0.0;
    if (general || !restricted) {
        w.put("ITEM: BOX BOUNDS abc origin ").put(boundary).put('\n');
        for (int r = 0; r < 3; ++r) {
            num(m[3 * r]).put(' ');
            num(m[3 * r + 1]).put(' ');
            num(m[3 * r + 2]).put(' ');
            num(origin[static_cast<std::size_t>(r)]).put('\n');
        }
        return;
    }
    const double xy = m[3], xz = m[6], yz = m[7];
    const double hi[3] = {g15(origin[0] + m[0]), g15(origin[1] + m[4]), g15(origin[2] + m[8])};
    if (xy == 0.0 && xz == 0.0 && yz == 0.0) {
        w.put("ITEM: BOX BOUNDS ").put(boundary).put('\n');
        for (int r = 0; r < 3; ++r) num(origin[static_cast<std::size_t>(r)]).put(' '), num(hi[r]).put('\n');
        return;
    }
    // Write the bounds of the bounding box. This is the inverse of the
    // reader's conversion.
    const double xlo_b = g15(origin[0] + std::fmin(std::fmin(std::fmin(0.0, xy), xz), xy + xz));
    const double xhi_b = g15(hi[0] + std::fmax(std::fmax(std::fmax(0.0, xy), xz), xy + xz));
    const double ylo_b = g15(origin[1] + std::fmin(0.0, yz));
    const double yhi_b = g15(hi[1] + std::fmax(0.0, yz));
    w.put("ITEM: BOX BOUNDS xy xz yz ").put(boundary).put('\n');
    num(xlo_b).put(' '), num(xhi_b).put(' '), num(xy).put('\n');
    num(ylo_b).put(' '), num(yhi_b).put(' '), num(xz).put('\n');
    num(origin[2]).put(' '), num(hi[2]).put(' '), num(yz).put('\n');
}

// Returns the inverse of a row-major 3x3 matrix, or nullopt if it is singular
// or the determinant is not finite.
std::optional<Box> invert3(const Box& m) {
    const double det =
        m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if (det == 0.0 || !std::isfinite(det)) return std::nullopt;
    const double d = 1.0 / det;
    return Box{(m[4] * m[8] - m[5] * m[7]) * d, (m[2] * m[7] - m[1] * m[8]) * d, (m[1] * m[5] - m[2] * m[4]) * d,
               (m[5] * m[6] - m[3] * m[8]) * d, (m[0] * m[8] - m[2] * m[6]) * d, (m[2] * m[3] - m[0] * m[5]) * d,
               (m[3] * m[7] - m[4] * m[6]) * d, (m[1] * m[6] - m[0] * m[7]) * d, (m[0] * m[4] - m[1] * m[3]) * d};
}

class LammpsWriter final : public TrajectoryWriter {
public:
    LammpsWriter(std::unique_ptr<OutputStream> out, const LammpsUnits* units)
        : out_(std::move(out)), w_(*out_), units_(units) {}
    ~LammpsWriter() override {
        try {
            finish();
        } catch (...) {
        }
    }

    void write_batch(const FrameBatch& batch) override {
        // Each frame of a dump stores its own atom count, so the frames of a
        // variable batch are written one at a time with their own atom ids.
        if (batch.variable()) {
            for (std::size_t f = 0; f < batch.nframes; ++f) write_batch(batch.frame(f));
            return;
        }
        auto attr = [&](const char* k) -> const std::string* {
            auto it = batch.attributes.find(k);
            return it == batch.attributes.end() ? nullptr : &it->second;
        };
        // Use the requested unit style, then the style the data is stored in,
        // then the style recorded when it was read, then real. Data already in
        // the chosen style is written without conversion, so it is exact.
        const LammpsUnits* units = units_ ? units_ : batch.units;
        if (!units) {
            const std::string* s = attr(kAttrUnits);
            const std::string_view style = s ? std::string_view(*s) : kDefaultUnits;
            units = LammpsUnits::from_style(style);
            if (!units) invalid("trajectory records unknown LAMMPS unit style '" + std::string(style) + "'");
        }
        // These factors match FrameBatch::convert_units. They are applied while
        // writing, which avoids copying the batch.
        const bool same = batch.units && *batch.units == *units;
        auto factor = [&](double LammpsUnits::* f) {
            if (same) return 1.0;
            return (batch.units ? batch.units->*f : 1.0) / (units->*f);
        };
        const double fl = factor(&LammpsUnits::length), fv = factor(&LammpsUnits::velocity),
                     ff = factor(&LammpsUnits::force);

        const std::size_t natoms = batch.natoms, nframes = batch.nframes, per_atom = natoms * nframes;
        const std::vector<float>* pos = batch.positions();
        if (!pos) invalid("no positions column");
        const std::vector<float>* vel = batch.column("velocities");
        const std::vector<float>* frc = batch.column("forces");
        for (const std::vector<float>* v : {pos, vel, frc})
            if (v && v->size() < per_atom * 3) invalid("a per-atom vector column is shorter than the batch");
        const std::vector<std::int32_t>* image = nullptr;
        if (auto it = batch.int_columns.find("image");
            it != batch.int_columns.end() && it->second.size() == per_atom * 3)
            image = &it->second;
        for (const auto* s : {batch.atom_ids.get(), batch.atom_types.get()})
            if (s && s->size() < natoms) invalid("per-atom metadata is shorter than the atom count");
        if (batch.elements && batch.elements->size() < natoms)
            invalid("per-atom metadata is shorter than the atom count");

        const std::string* bnd = attr(kAttrBoundary);
        const std::string_view boundary = bnd ? std::string_view(*bnd) : "pp pp pp";
        const std::string* box_attr = attr(kAttrBox);
        const bool general = box_attr && *box_attr == "general";
        const bool have_box0 = batch.box && !batch.box->empty();
        // Scaled coordinates are stored as Cartesian and converted back for
        // each frame, which needs that frame's box. Without a box in the first
        // frame, they are written unscaled.
        static const char *const kXu[] = {"xu", "yu", "zu"}, *const kX[] = {"x", "y", "z"},
                                 *const kXsu[] = {"xsu", "ysu", "zsu"}, *const kXs[] = {"xs", "ys", "zs"};
        const char* const* coord = kX;
        bool scaled = false;
        if (const std::string* c = attr(kAttrCoords)) {
            if (*c == "unwrapped")
                coord = kXu;
            else if (*c == "scaled" && have_box0)
                coord = kXs, scaled = true;
            else if (*c == "scaled-unwrapped" && have_box0)
                coord = kXsu, scaled = true;
            else if (*c == "scaled-unwrapped")
                coord = kXu;
        }
        std::vector<std::string_view> orig;
        const std::string* orig_attr = attr(kAttrColumns);
        if (orig_attr) {
            std::string_view rest = *orig_attr;
            for (std::string_view t = text::next_token(rest); !t.empty(); t = text::next_token(rest)) orig.push_back(t);
        }
        // id and type are each written unless the original column list is
        // known and does not include them.
        auto keep = [&](std::string_view name) {
            return !orig_attr || std::find(orig.begin(), orig.end(), name) != orig.end();
        };

        // Use the original column order if it covers exactly the available
        // columns. Otherwise use a canonical order.
        std::vector<std::string_view> available;
        if (keep("id")) available.push_back("id");
        if (keep("type")) available.push_back("type");
        if (batch.elements) available.push_back("element");
        for (int k = 0; k < 3; ++k) available.push_back(coord[k]);
        if (vel) available.insert(available.end(), {"vx", "vy", "vz"});
        if (frc) available.insert(available.end(), {"fx", "fy", "fz"});
        if (image) available.insert(available.end(), {"ix", "iy", "iz"});
        // Add the carried columns, skipping any whose name is already used,
        // such as an extxyz "id" column alongside the atom ids. A dump must not
        // name a column twice.
        auto fresh = [&](const std::string& k) {
            return std::find(available.begin(), available.end(), k) == available.end();
        };
        for (const auto& [k, v] : batch.columns)
            if (k != "positions" && k != "velocities" && k != "forces" && per_atom > 0 && v.size() == per_atom &&
                fresh(k))
                available.push_back(k);
        for (const auto& [k, v] : batch.int_columns)
            if (k != "image" && per_atom > 0 && v.size() == per_atom && fresh(k)) available.push_back(k);
        std::vector<std::string_view> columns = available;
        if (orig_attr) {
            std::vector<std::string_view> mapped;
            for (std::string_view c : orig) {
                // Map each original coordinate name to the one being written.
                if (c == "xs" || c == "xsu" || c == "x" || c == "xu")
                    c = coord[0];
                else if (c == "ys" || c == "ysu" || c == "y" || c == "yu")
                    c = coord[1];
                else if (c == "zs" || c == "zsu" || c == "z" || c == "zu")
                    c = coord[2];
                mapped.push_back(c);
            }
            std::vector<std::string_view> so = mapped, sa = available;
            std::sort(so.begin(), so.end());
            std::sort(sa.begin(), sa.end());
            if (so == sa) columns = mapped;
        }

        // Resolve the source of each column once per batch.
        struct Src {
            enum Kind { Id, Type, Element, Coord, Vel, Force, Image, Float, Int, Zero } kind;
            int k = 0;
            const std::vector<float>* f = nullptr;
            const std::vector<std::int32_t>* i = nullptr;
        };
        std::vector<Src> srcs;
        for (std::string_view c : columns) {
            Src s{Src::Zero};
            const bool vec3 = c.size() == 2 && c[1] >= 'x' && c[1] <= 'z';
            const int k = vec3 ? c[1] - 'x' : 0;
            if (c == "id")
                s.kind = Src::Id;
            else if (c == "type")
                s.kind = Src::Type;
            else if (c == "element")
                s.kind = Src::Element;
            else if (c == coord[0] || c == coord[1] || c == coord[2])
                s = {Src::Coord, c == coord[0] ? 0 : c == coord[1] ? 1 : 2};
            else if (vec3 && c[0] == 'v' && vel)
                s = {Src::Vel, k};
            else if (vec3 && c[0] == 'f' && frc)
                s = {Src::Force, k};
            else if (vec3 && c[0] == 'i' && image)
                s = {Src::Image, k};
            else if (auto* f = batch.column(c))
                s.kind = Src::Float, s.f = f;
            else if (auto it = batch.int_columns.find(c); it != batch.int_columns.end())
                s.kind = Src::Int, s.i = &it->second;
            if ((s.f && s.f->size() < per_atom) || (s.i && s.i->size() < per_atom))
                invalid("column '" + std::string(c) + "' is shorter than the batch");
            srcs.push_back(s);
        }

        const std::vector<std::int64_t>* steps = nullptr;
        if (auto it = batch.int_frame_props.find("step"); it != batch.int_frame_props.end()) steps = &it->second;
        const std::vector<double>* times = nullptr;
        if (auto it = batch.frame_props.find("time"); it != batch.frame_props.end()) times = &it->second;
        auto conv = [](float x, double f) { return f != 1.0 ? static_cast<float>(static_cast<double>(x) * f) : x; };
        // Gets frame f's box and origin in the output units. Returns false if
        // the frame has no box.
        auto frame_box = [&](std::size_t f, Vec3& o, Box& m) {
            if (!batch.box || f >= batch.box->size()) return false;
            m = (*batch.box)[f];
            o = batch.origin && f < batch.origin->size() ? (*batch.origin)[f] : Vec3{0.0, 0.0, 0.0};
            if (fl != 1.0) {
                for (double& x : m) x *= fl;
                for (double& x : o) x *= fl;
            }
            return true;
        };

        for (std::size_t f = 0; f < nframes; ++f) {
            const std::int64_t step = steps && f < steps->size() ? (*steps)[f] : step_;
            if (times && f < times->size()) w_.put("ITEM: TIME\n").put_shortest((*times)[f] / units->time).put('\n');
            w_.put("ITEM: TIMESTEP\n").put_int(step).put("\nITEM: NUMBER OF ATOMS\n").put_uint(natoms).put('\n');
            Vec3 o;
            Box m;
            const bool has_box = frame_box(f, o, m);
            if (has_box)
                box_block(w_, o, m, boundary, general);
            else
                w_.put("ITEM: BOX BOUNDS ").put(boundary).put("\n0 1\n0 1\n0 1\n");
            w_.put("ITEM: ATOMS");
            for (std::string_view c : columns) w_.put(' ').put(c);
            w_.put('\n');
            // For scaled output, convert Cartesian to fractional by solving
            // r - origin = s a + t b + u c with the inverse box matrix.
            Box inv{};
            if (scaled) {
                if (!has_box) invalid("frame " + std::to_string(f) + " has no box to scale coordinates by");
                auto i = invert3(m);
                if (!i) invalid("box matrix is singular");
                inv = *i;
            }
            const float* p = pos->data() + f * natoms * 3;
            for (std::size_t a = 0; a < natoms; ++a) {
                const std::size_t idx = f * natoms + a, base = idx * 3;
                double frac[3] = {0.0, 0.0, 0.0};
                if (scaled) {
                    double d[3];
                    for (int k = 0; k < 3; ++k)
                        d[k] = static_cast<double>(conv(p[3 * a + k], fl)) - o[static_cast<std::size_t>(k)];
                    for (int j = 0; j < 3; ++j) frac[j] = d[0] * inv[j] + d[1] * inv[3 + j] + d[2] * inv[6 + j];
                }
                bool first = true;
                for (const Src& s : srcs) {
                    if (!first) w_.put(' ');
                    first = false;
                    switch (s.kind) {
                        case Src::Id:
                            w_.put_int(batch.atom_ids ? (*batch.atom_ids)[a] : static_cast<std::int32_t>(a + 1));
                            break;
                        case Src::Type:
                            w_.put_int(batch.atom_types ? (*batch.atom_types)[a] : 1);
                            break;
                        case Src::Element:
                            w_.put(batch.elements ? std::string_view((*batch.elements)[a]) : "X");
                            break;
                        case Src::Coord:
                            // Positions are stored as f32 Cartesian, so digits
                            // past the 6th of a fraction are noise. Write 6,
                            // matching LAMMPS.
                            if (scaled)
                                w_.put_shortest(g6(frac[s.k]));
                            else
                                w_.put_shortest(conv(p[3 * a + static_cast<std::size_t>(s.k)], fl));
                            break;
                        case Src::Vel:
                            w_.put_shortest(conv((*vel)[base + static_cast<std::size_t>(s.k)], fv));
                            break;
                        case Src::Force:
                            w_.put_shortest(conv((*frc)[base + static_cast<std::size_t>(s.k)], ff));
                            break;
                        case Src::Image:
                            w_.put_int((*image)[base + static_cast<std::size_t>(s.k)]);
                            break;
                        case Src::Float:
                            w_.put_shortest((*s.f)[idx]);
                            break;
                        case Src::Int:
                            w_.put_int((*s.i)[idx]);
                            break;
                        case Src::Zero:
                            w_.put('0');
                            break;
                    }
                }
                w_.put('\n');
            }
            ++step_;
        }
    }

    void finish() override {
        if (finished_) return;
        finished_ = true;
        w_.flush();
        out_->close();
    }

private:
    std::unique_ptr<OutputStream> out_;
    text::TextWriter w_;
    const LammpsUnits* units_;
    std::int64_t step_ = 0;
    bool finished_ = false;
};

}  // namespace

std::unique_ptr<TrajectoryReader> make_lammps_reader(std::unique_ptr<InputStream> in, const FormatOptions& opts) {
    const LammpsUnits* u = opts.lammps_units ? opts.lammps_units : LammpsUnits::from_style(kDefaultUnits);
    return std::make_unique<LammpsReader>(std::move(in), u);
}

std::unique_ptr<TrajectoryWriter> make_lammps_writer(std::unique_ptr<OutputStream> out, std::size_t,
                                                     const FormatOptions& opts) {
    return std::make_unique<LammpsWriter>(std::move(out), opts.lammps_units);
}

}  // namespace mdz::detail
