// Extended XYZ. Each frame is an atom count, a comment line of key=value
// pairs, then one line per atom. The Properties key declares the per-atom
// columns (Properties=species:S:1:pos:R:3:...). The box is given by
// Lattice="ax ay az bx by bz cx cy cz" and is kept as the full matrix.
// Positions are in angstroms, velocities in angstrom/ps, forces in
// eV/angstrom and energies in eV.

#include <algorithm>
#include <optional>
#include <string>

#include "internal.hpp"
#include "mdz/error.hpp"
#include "text.hpp"

namespace mdz::detail {
namespace {

constexpr double kKjmolToEv = 1.0 / units::kEvToKjmol;

[[noreturn]] void corrupt_at(const text::LineReader& lr, const std::string& msg) {
    throw Error(ErrorKind::Corrupt, lr.name() + ":" + std::to_string(lr.line_number()) + ": " + msg);
}

std::size_t count_tokens(std::string_view s) {
    std::size_t n = 0;
    for (std::string_view t = text::next_token(s); !t.empty(); t = text::next_token(s)) ++n;
    return n;
}

// --- comment line ---

struct KeyValue {
    std::string_view key, value;
};

// Splits a comment line into key=value pairs. A value may be double-quoted.
// A key without '=' gets an empty value. The results are views into `line`.
void parse_comment(std::string_view line, std::vector<KeyValue>& out) {
    out.clear();
    std::size_t i = 0;
    const std::size_t n = line.size();
    auto blank = [](char c) { return c == ' ' || c == '\t'; };
    while (i < n) {
        while (i < n && blank(line[i])) ++i;
        const std::size_t k0 = i;
        while (i < n && line[i] != '=' && !blank(line[i])) ++i;
        const std::string_view key = line.substr(k0, i - k0);
        if (i < n && line[i] == '=') ++i;
        if (key.empty()) break;
        std::string_view value;
        if (i < n && line[i] == '"') {
            ++i;
            const std::size_t v0 = i;
            while (i < n && line[i] != '"') ++i;
            value = line.substr(v0, i - v0);
            if (i < n) ++i;  // skip the closing quote
        } else {
            const std::size_t v0 = i;
            while (i < n && !blank(line[i])) ++i;
            value = line.substr(v0, i - v0);
        }
        out.push_back({key, value});
    }
}

// Returns the value of `key`, or nullptr. If a key is repeated, the last value
// is used.
const std::string_view* find_kv(const std::vector<KeyValue>& kvs, std::string_view key) {
    for (auto it = kvs.rbegin(); it != kvs.rend(); ++it)
        if (it->key == key) return &it->value;
    return nullptr;
}

struct Property {
    std::string_view name;
    char dtype;
    std::size_t width;
};

// Parses "species:S:1:pos:R:3" into (name, type, width) triples. A missing
// type is taken as R and an unreadable width as 1. A trailing partial triple
// is ignored.
void parse_properties(std::string_view s, std::vector<Property>& out, std::vector<std::string_view>& parts) {
    out.clear();
    parts.clear();
    std::size_t start = 0;
    for (;;) {
        const std::size_t c = s.find(':', start);
        parts.push_back(s.substr(start, c == std::string_view::npos ? std::string_view::npos : c - start));
        if (c == std::string_view::npos) break;
        start = c + 1;
    }
    for (std::size_t i = 0; i + 2 < parts.size(); i += 3) {
        std::uint64_t w;
        const std::size_t width = text::parse(parts[i + 2], w) ? static_cast<std::size_t>(w) : 1;
        out.push_back({parts[i], parts[i + 1].empty() ? 'R' : parts[i + 1][0], width});
    }
}

// Parses a Lattice value into a box in nm. Unreadable tokens are skipped. If
// the remaining count is not exactly nine, returns nullopt.
std::optional<Box> parse_lattice(std::string_view s) {
    Box m{};
    std::size_t n = 0;
    for (std::string_view tok = text::next_token(s); !tok.empty(); tok = text::next_token(s)) {
        while (!tok.empty() && tok.front() == '"') tok.remove_prefix(1);
        while (!tok.empty() && tok.back() == '"') tok.remove_suffix(1);
        double v;
        if (!text::parse(tok, v)) continue;
        if (n < 9) m[n] = v * units::kAngstromToNm;
        ++n;
    }
    if (n != 9) return std::nullopt;
    return m;
}

class ExtxyzReader final : public TrajectoryReader {
public:
    explicit ExtxyzReader(std::unique_ptr<InputStream> in) : in_(std::move(in)), lines_(*in_) {}

    std::size_t natoms() const override { return first_natoms_.value_or(0); }

    std::optional<FrameBatch> read_batch(std::size_t n) override {
        std::optional<FrameBatch> out;
        for (std::size_t i = 0; i < n; ++i)
            if (!read_frame(out)) break;
        return out;
    }

private:
    // Destination for the tokens of one declared property.
    enum class Dest { Species, Id, Real, Int, Skip };
    struct Step {
        Dest dest;
        std::size_t width;
        std::vector<float>* real = nullptr;
        std::vector<std::int32_t>* ints = nullptr;
    };

    // Appends one frame to *out, creating it on the first frame. Returns false
    // at end of input.
    bool read_frame(std::optional<FrameBatch>& out) {
        std::string_view line;
        if (!lines_.next(line)) return false;
        std::uint64_t count;
        if (!text::parse(text::trim(line), count))
            corrupt_at(lines_, "bad atom count '" + std::string(text::trim(line)) + "'");
        const std::size_t natoms = static_cast<std::size_t>(count);
        if (!first_natoms_) first_natoms_ = natoms;

        // Copy the comment line, because reading the atom lines invalidates
        // its view.
        if (!lines_.next(line)) line = {};
        comment_.assign(line.data(), line.size());
        parse_comment(text::trim(comment_), kvs_);
        const std::string_view* props = find_kv(kvs_, "Properties");
        parse_properties(props ? *props : std::string_view("species:S:1:pos:R:3"), props_, parts_);

        // While each frame has the same atoms as the previous one, it is read
        // directly into the batch. Once the atoms change (a different count,
        // species or ids), each frame is built separately and the batch
        // becomes variable.
        const bool direct = !varying_ && (!seen_ || natoms == ref_natoms_);
        std::optional<FrameBatch> one;
        std::optional<FrameBatch>& dst = direct ? out : one;
        if (!dst) dst.emplace(natoms, 0);
        FrameBatch& b = *dst;

        if (const std::string_view* lat = find_kv(kvs_, "Lattice"))
            if (auto m = parse_lattice(*lat)) {
                if (!b.box) b.box.emplace();
                b.box->push_back(*m);
            }
        // Per-frame properties accept two spellings. The first one present is
        // used.
        auto either = [&](std::string_view a, std::string_view c) {
            const std::string_view* v = find_kv(kvs_, a);
            return v ? v : find_kv(kvs_, c);
        };
        if (const std::string_view* t = either("Time", "time")) {
            double v;
            if (text::parse(*t, v)) b.frame_props["time"].push_back(v);
        }
        if (const std::string_view* e = either("energy", "Energy")) {
            double v;
            if (text::parse(*e, v)) b.frame_props["energy"].push_back(units::energy_to_kjmol(v, "eV"));
        }
        if (const std::string_view* s = either("step", "Step")) {
            std::int64_t v;
            if (text::parse(*s, v)) b.int_frame_props["step"].push_back(v);
        }

        // Resolve each property to its destination once per frame. Species and
        // ids are collected in reused buffers so they can be compared with the
        // reference.
        arena_.clear();
        name_end_.clear();
        ids_v_.clear();
        bool has_species = false, has_ids = false;
        steps_.clear();
        std::size_t expected = 0;
        for (const Property& p : props_) {
            Step st{Dest::Skip, p.width};
            if (p.dtype == 'S') {
                if (p.name == "species" && !has_species) st.dest = Dest::Species, has_species = true;
            } else if ((p.dtype == 'I' || p.dtype == 'i') && p.name == "id" && p.width == 1 && !has_ids) {
                // Atom ids identify atoms across frames. They are also kept as
                // an "id" column so the file is written back as it was read.
                st.dest = Dest::Id;
                st.ints = &column_for(b.int_columns, p.name);
                has_ids = true;
            } else if (p.dtype == 'R' || p.dtype == 'r') {
                std::string_view col = p.name;
                if (col == "pos") col = "positions";
                else if (col == "vel" || col == "velo" || col == "velocities") col = "velocities";
                else if (col == "force" || col == "forces") col = "forces";
                st.dest = Dest::Real;
                st.real = &column_for(b.columns, col);
            } else if (p.dtype == 'I' || p.dtype == 'i') {
                // Integer properties are kept as integers so that they are
                // written back with type I.
                st.dest = Dest::Int;
                st.ints = &column_for(b.int_columns, p.name);
            }
            if (p.width <= (1u << 16) && natoms <= (std::size_t(1) << 24)) {
                if (st.real) st.real->reserve(st.real->size() + natoms * p.width);
                if (st.ints) st.ints->reserve(st.ints->size() + natoms * p.width);
            }
            expected = p.width > ~std::size_t(0) - expected ? ~std::size_t(0) : expected + p.width;
            steps_.push_back(st);
        }
        // Offsets where this frame's values start in the columns that need
        // unit conversion.
        std::vector<float>* pos = b.column("positions");
        std::vector<float>* vel = b.column("velocities");
        std::vector<float>* frc = b.column("forces");
        const std::size_t pos0 = pos ? pos->size() : 0, vel0 = vel ? vel->size() : 0, frc0 = frc ? frc->size() : 0;

        for (std::size_t atom = 0; atom < natoms; ++atom) {
            if (!lines_.next(line)) line = {};
            // A short line is an error, since it would misalign every column
            // after it.
            auto short_line = [&] {
                corrupt_at(lines_, "extxyz atom line " + std::to_string(atom) + " has " +
                                       std::to_string(count_tokens(line)) +
                                       " fields but the Properties header declares " + std::to_string(expected));
            };
            auto bad = [&](const char* what, const char* p, const char* e) {
                if (count_tokens(line) < expected) short_line();
                corrupt_at(lines_, std::string("bad ") + what + " value '" + std::string(p, text::token_end(p, e)) + "'");
            };
            const char* p = line.data();
            const char* const e = p + line.size();
            for (const Step& st : steps_) {
                for (std::size_t w = 0; w < st.width; ++w) {
                    while (p != e && text::is_space(*p)) ++p;
                    if (p == e) short_line();
                    const char* q = nullptr;
                    switch (st.dest) {
                        case Dest::Species:
                            q = text::token_end(p, e);
                            arena_.append(p, static_cast<std::size_t>(q - p));
                            name_end_.push_back(arena_.size());
                            break;
                        case Dest::Id: {
                            std::int32_t v = 0;
                            if (!(q = text::scan_number(p, e, v))) bad("integer", p, e);
                            st.ints->push_back(v);
                            ids_v_.push_back(v);
                            break;
                        }
                        case Dest::Real: {
                            float v = 0;
                            if (!(q = text::scan_number(p, e, v))) bad("real", p, e);
                            st.real->push_back(v);
                            break;
                        }
                        case Dest::Int: {
                            std::int32_t v = 0;
                            if (!(q = text::scan_number(p, e, v))) bad("integer", p, e);
                            st.ints->push_back(v);
                            break;
                        }
                        case Dest::Skip:
                            q = text::token_end(p, e);
                            break;
                    }
                    p = q;
                }
            }
        }

        // Convert angstrom to nm, angstrom/ps to nm/ps and eV/angstrom to
        // kJ/mol/nm.
        auto scale = [](std::vector<float>* v, std::size_t from, float f) {
            if (!v) return;
            for (std::size_t i = from; i < v->size(); ++i) (*v)[i] *= f;
        };
        const float ang_to_nm = static_cast<float>(units::kAngstromToNm);
        scale(pos, pos0, ang_to_nm);
        scale(vel, vel0, ang_to_nm);
        scale(frc, frc0, static_cast<float>(units::kEvToKjmol * units::kNmToAngstrom));

        auto name = [&](std::size_t a) {
            const std::size_t lo = a ? name_end_[a - 1] : 0;
            return std::string_view(arena_).substr(lo, name_end_[a] - lo);
        };
        const bool with_names = has_species && natoms > 0, with_ids = has_ids && natoms > 0;
        bool same = seen_ && natoms == ref_natoms_ && static_cast<bool>(names_) == with_names &&
                    static_cast<bool>(ids_) == with_ids;
        if (same && with_ids) same = *ids_ == ids_v_;
        for (std::size_t a = 0; same && with_names && a < natoms; ++a) same = (*names_)[a] == name(a);
        if (!same) {
            // The first frame sets the reference. A later change switches the
            // reader to variable mode for the rest of the file.
            names_.reset();
            if (with_names) {
                std::vector<std::string> names;
                names.reserve(natoms);
                for (std::size_t a = 0; a < natoms; ++a) names.emplace_back(name(a));
                names_ = share(std::move(names));
            }
            ids_ = with_ids ? share(ids_v_) : nullptr;
            ref_natoms_ = natoms;
            varying_ = varying_ || seen_;
        }
        seen_ = true;
        if (!varying_) {
            b.nframes += 1;
            if (!b.atom_names) b.atom_names = names_;
            if (!b.atom_ids) b.atom_ids = ids_;
            return true;
        }
        FrameBatch frame;
        if (direct) {
            // This frame was read into the batch before its atoms were found
            // to differ, so take it back out.
            b.nframes += 1;
            frame = pop_last_frame(b);
        } else {
            frame = std::move(b);
            frame.nframes = 1;
        }
        frame.atom_names = names_;
        frame.atom_ids = ids_;
        append_varying(out, std::move(frame));
        return true;
    }

    template <class T>
    static std::vector<T>& column_for(NamedMap<std::vector<T>>& m, std::string_view name) {
        auto it = m.find(name);
        if (it == m.end()) it = m.emplace(std::string(name), std::vector<T>()).first;
        return it->second;
    }

    std::unique_ptr<InputStream> in_;
    text::LineReader lines_;
    std::optional<std::size_t> first_natoms_;
    // Atom count, species and ids of the previous frame.
    std::size_t ref_natoms_ = 0;
    Shared<std::string> names_;
    Shared<std::int32_t> ids_;
    bool seen_ = false, varying_ = false;
    // Scratch buffers reused across frames.
    std::string arena_;
    std::vector<std::size_t> name_end_;
    std::vector<std::int32_t> ids_v_;
    std::string comment_;
    std::vector<KeyValue> kvs_;
    std::vector<Property> props_;
    std::vector<std::string_view> parts_;
    std::vector<Step> steps_;
};

class ExtxyzWriter final : public TrajectoryWriter {
public:
    explicit ExtxyzWriter(std::unique_ptr<OutputStream> out) : out_(std::move(out)), w_(*out_) {}
    ~ExtxyzWriter() override {
        try {
            finish();
        } catch (...) {
        }
    }

    void write_batch(const FrameBatch& in) override {
        // Each frame stores its own atom count, so the frames of a variable
        // batch are written one at a time.
        if (in.variable()) {
            ids_ = true;
            for (std::size_t f = 0; f < in.nframes; ++f) write_batch(in.frame(f));
            ids_ = false;
            return;
        }
        // extxyz has fixed units, so a batch in LAMMPS native units is
        // converted first.
        std::optional<FrameBatch> converted;
        if (in.units) converted = in.in_internal_units();
        const FrameBatch& batch = converted ? *converted : in;
        const std::size_t natoms = batch.natoms, nframes = batch.nframes;
        const std::size_t per_atom = natoms * nframes;
        auto vec3 = [&](const char* name) {
            const std::vector<float>* v = batch.column(name);
            if (v && v->size() < per_atom * 3)
                throw Error(ErrorKind::InvalidInput, std::string(name) + " column is shorter than the batch");
            return v;
        };
        const std::vector<float>* pos = vec3("positions");
        const std::vector<float>* vel = vec3("velocities");
        const std::vector<float>* frc = vec3("forces");

        // The standard columns come first, then the other float columns, then
        // the integer columns. Each group is in name order.
        struct Extra {
            const std::string* name;
            const std::vector<float>* f;
            const std::vector<std::int32_t>* i;
            std::size_t width;
        };
        std::vector<Extra> extras;
        for (const auto& [k, v] : batch.columns)
            if (k != "positions" && k != "velocities" && k != "forces")
                extras.push_back({&k, &v, nullptr, per_atom ? v.size() / per_atom : 0});
        // For the frames of a variable batch, atoms are identified by id. The
        // ids are written as the integer property "id", in name order among
        // the integer columns. This is skipped if the batch already has an
        // "id" column, as the reader produces.
        static const std::string kId = "id";
        std::vector<std::int32_t> ids;
        const bool write_ids = ids_ && batch.atom_ids && !batch.int_columns.count(kId);
        if (write_ids) {
            if (batch.atom_ids->size() < natoms)
                throw Error(ErrorKind::InvalidInput, "atom_ids is shorter than the atom count");
            ids.reserve(per_atom);
            for (std::size_t f = 0; f < nframes; ++f)
                ids.insert(ids.end(), batch.atom_ids->begin(), batch.atom_ids->begin() + static_cast<std::ptrdiff_t>(natoms));
        }
        bool ids_placed = !write_ids;
        for (const auto& [k, v] : batch.int_columns) {
            if (!ids_placed && kId < k) {
                extras.push_back({&kId, nullptr, &ids, 1});
                ids_placed = true;
            }
            extras.push_back({&k, nullptr, &v, per_atom ? v.size() / per_atom : 0});
        }
        if (!ids_placed) extras.push_back({&kId, nullptr, &ids, 1});

        const bool has_names = batch.atom_names || batch.elements;
        std::string props;
        auto add_prop = [&](std::string_view name, char type, std::size_t width) {
            if (!props.empty()) props += ':';
            props.append(name.data(), name.size());
            props += ':';
            props += type;
            props += ':';
            props += std::to_string(width);
        };
        if (has_names) add_prop("species", 'S', 1);
        if (pos) add_prop("pos", 'R', 3);
        if (vel) add_prop("vel", 'R', 3);
        if (frc) add_prop("force", 'R', 3);
        for (const Extra& e : extras) add_prop(*e.name, e.f ? 'R' : 'I', std::max<std::size_t>(e.width, 1));

        const std::vector<double>* time = nullptr;
        const std::vector<double>* energy = nullptr;
        if (auto it = batch.frame_props.find("time"); it != batch.frame_props.end()) time = &it->second;
        if (auto it = batch.frame_props.find("energy"); it != batch.frame_props.end()) energy = &it->second;
        const std::vector<std::string>* elements = batch.elements.get();
        const std::vector<std::string>* names = batch.atom_names.get();
        const double force_factor = kKjmolToEv / units::kNmToAngstrom;

        for (std::size_t f = 0; f < nframes; ++f) {
            w_.put_uint(natoms).put('\n');
            if (batch.box && f < batch.box->size()) {
                const Box& m = (*batch.box)[f];
                w_.put("Lattice=\"");
                for (int k = 0; k < 9; ++k) {
                    if (k) w_.put(' ');
                    w_.put_fixed(m[k] * units::kNmToAngstrom, 6);
                }
                w_.put("\" ");
            }
            w_.put("Properties=").put(props);
            if (time && f < time->size()) w_.put(" Time=").put_fixed((*time)[f], 6);
            if (energy && f < energy->size()) w_.put(" energy=").put_fixed((*energy)[f] * kKjmolToEv, 6);
            w_.put(" pbc=\"T T T\"\n");

            for (std::size_t a = 0; a < natoms; ++a) {
                const std::size_t idx = f * natoms + a;
                bool sep = false;
                auto part = [&]() -> text::TextWriter& {
                    if (sep) w_.put(' ');
                    sep = true;
                    return w_;
                };
                if (has_names) {
                    // Use the element if present, then the atom name, then "X".
                    std::string_view s = "X";
                    if (elements && a < elements->size()) s = (*elements)[a];
                    else if (names && a < names->size()) s = (*names)[a];
                    part().put(s);
                }
                auto put3 = [&](const std::vector<float>* v, double factor) {
                    if (!v) return;
                    const float* p = v->data() + idx * 3;
                    part().put_fixed(static_cast<double>(p[0]) * factor, 6);
                    w_.put(' ').put_fixed(static_cast<double>(p[1]) * factor, 6);
                    w_.put(' ').put_fixed(static_cast<double>(p[2]) * factor, 6);
                };
                put3(pos, units::kNmToAngstrom);
                put3(vel, units::kNmToAngstrom);
                put3(frc, force_factor);
                for (const Extra& e : extras) {
                    text::TextWriter& o = part();
                    const std::size_t base = idx * e.width;
                    for (std::size_t k = 0; k < e.width; ++k) {
                        if (k) o.put(' ');
                        if (e.f) o.put_fixed(static_cast<double>((*e.f)[base + k]), 6);
                        else o.put_int((*e.i)[base + k]);
                    }
                }
                w_.put('\n');
            }
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
    // True while writing the frames of a variable batch, whose atom ids are
    // written as an "id" column.
    bool ids_ = false;
    bool finished_ = false;
};

}  // namespace

std::unique_ptr<TrajectoryReader> make_extxyz_reader(std::unique_ptr<InputStream> in, const FormatOptions&) {
    return std::make_unique<ExtxyzReader>(std::move(in));
}

std::unique_ptr<TrajectoryWriter> make_extxyz_writer(std::unique_ptr<OutputStream> out, std::size_t,
                                                     const FormatOptions&) {
    return std::make_unique<ExtxyzWriter>(std::move(out));
}

}  // namespace mdz::detail
