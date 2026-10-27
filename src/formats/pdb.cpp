// PDB. Atoms are fixed-column ATOM/HETATM records, the cell is a CRYST1
// record, and each frame of a trajectory is wrapped in MODEL/ENDMDL.
// Coordinates are in angstroms.
//
// ATOM record columns (1-based):
//    1-6   "ATOM  " or "HETATM"     23-26  residue sequence number
//    7-11  serial number            31-54  x, y, z (f8.3)
//   13-16  atom name                55-60  occupancy (f6.2)
//   18-20  residue name             61-66  B-factor (f6.2)
//   22     chain id                 77-78  element (right-justified)
//
// Atoms are identified by serial number. The topology (names, elements,
// serials, residues, chains, HETATM flags) is read from the first frame and
// shared by every later frame with the same serials. A model with different
// atoms (a different count or different serials) gets its own topology, and
// the reader returns variable-atom-count batches from then on. Alternate
// locations and insertion codes are discarded.

#include <charconv>
#include <cmath>
#include <cstring>
#include <optional>
#include <string>

#include "internal.hpp"
#include "mdz/error.hpp"
#include "text.hpp"

namespace mdz::detail {
namespace {

[[noreturn]] void corrupt_at(const text::LineReader& lr, const std::string& msg) {
    throw Error(ErrorKind::Corrupt, lr.name() + ":" + std::to_string(lr.line_number()) + ": " + msg);
}

// Returns columns [start, end) of a line (0-based), clipped to its length.
std::string_view field_raw(std::string_view line, std::size_t start, std::size_t end) {
    if (start >= line.size()) return {};
    return line.substr(start, std::min(end, line.size()) - start);
}

std::string_view field(std::string_view line, std::size_t start, std::size_t end) {
    return text::trim(field_raw(line, start, end));
}

template <class T>
T field_or(std::string_view line, std::size_t start, std::size_t end, T fallback) {
    T v;
    return text::parse(field(line, start, end), v) ? v : fallback;
}

// Returns the first UTF-8 character of s, or an empty view if s is empty.
std::string_view first_char(std::string_view s) {
    if (s.empty()) return s;
    const auto c = static_cast<unsigned char>(s[0]);
    const std::size_t n = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
    return s.substr(0, std::min(n, s.size()));
}

// Counts UTF-8 characters rather than bytes. Column padding is based on this.
std::size_t char_count(std::string_view s) {
    std::size_t n = 0;
    for (char c : s) n += (static_cast<unsigned char>(c) & 0xC0) != 0x80;
    return n;
}

void put_aligned(text::TextWriter& w, std::string_view s, std::size_t width, bool left) {
    const std::size_t n = char_count(s);
    if (!left)
        for (std::size_t i = n; i < width; ++i) w.put(' ');
    w.put(s);
    if (left)
        for (std::size_t i = n; i < width; ++i) w.put(' ');
}

// Writes a right-aligned fixed-point number, like printf("%*.*f", width, prec, v).
void put_fixed_width(text::TextWriter& w, double v, int prec, int width) {
    char b[400];
    std::size_t n;
    if (std::isnan(v)) {
        std::memcpy(b, "NaN", n = 3);
    } else if (std::isinf(v)) {
        if (v < 0) std::memcpy(b, "-inf", n = 4);
        else std::memcpy(b, "inf", n = 3);
    } else {
        auto r = std::to_chars(b, b + sizeof b, v, std::chars_format::fixed, prec);
        n = static_cast<std::size_t>(r.ptr - b);
    }
    w.put_padded(std::string_view(b, n), width);
}

void put_int_width(text::TextWriter& w, std::int64_t v, int width) {
    char b[24];
    auto r = std::to_chars(b, b + sizeof b, v);
    w.put_padded(std::string_view(b, static_cast<std::size_t>(r.ptr - b)), width);
}

std::int32_t rem_euclid(std::int32_t v, std::int32_t m) {
    const std::int32_t r = v % m;
    return r < 0 ? r + m : r;
}

class PdbReader final : public TrajectoryReader {
public:
    explicit PdbReader(std::unique_ptr<InputStream> in) : in_(std::move(in)), lines_(*in_) {}

    std::size_t natoms() const override { return first_natoms_.value_or(0); }

    std::optional<FrameBatch> read_batch(std::size_t n) override {
        std::optional<FrameBatch> b = collect_frames(n, [&] { return read_frame(); });
        // Once the atoms have changed, every batch is marked variable.
        if (b && varying_) b->make_variable();
        return b;
    }

private:
    std::optional<FrameBatch> read_frame() {
        pos_.clear();
        occ_.clear();
        bfac_.clear();
        std::optional<Box> box;
        bool in_model = false, found_atoms = false;
        // Whether this frame has its own topology. The first frame always does.
        bool own = !meta_ready_;
        std::size_t atom = 0;
        names_v_.clear();
        elements_v_.clear();
        ids_v_.clear();
        res_names_v_.clear();
        res_ids_v_.clear();
        chains_v_.clear();
        hetatm_v_.clear();
        const float ang_to_nm = static_cast<float>(units::kAngstromToNm);
        std::string_view line;
        for (;;) {
            if (!lines_.next(line)) {
                if (found_atoms) break;
                return std::nullopt;
            }
            // The length thresholds below count the newline. A line one column
            // short of a field's end still has that field parsed, clipped.
            const std::size_t len = line.size() + 1;
            const std::string_view rec = line.substr(0, 6);
            if (rec == "MODEL ") {
                in_model = true;
            } else if (rec == "ENDMDL") {
                if (found_atoms) break;
            } else if (rec == "END   " || rec == "END") {
                if (found_atoms) break;
                if (!in_model) return std::nullopt;
            } else if (rec == "CRYST1") {
                if (len >= 54) {
                    const double a = field_or(line, 6, 15, 0.0), b = field_or(line, 15, 24, 0.0),
                                 c = field_or(line, 24, 33, 0.0);
                    const double alpha = field_or(line, 33, 40, 90.0), beta = field_or(line, 40, 47, 90.0),
                                 gamma = field_or(line, 47, 54, 90.0);
                    // A 1 x 1 x 1 cell is the conventional placeholder for "no
                    // cell".
                    if (a > 1.5 || b > 1.5 || c > 1.5)
                        box = cell_params_to_box({a * units::kAngstromToNm, gamma, b * units::kAngstromToNm, beta,
                                                  alpha, c * units::kAngstromToNm});
                }
            } else if (rec == "ATOM  " || rec == "HETATM") {
                found_atoms = true;
                // Coordinates are mandatory. A record without them is treated
                // as malformed rather than as an atom at the origin.
                for (std::size_t k = 0; k < 3; ++k) {
                    const std::string_view tok = field(line, 30 + 8 * k, 38 + 8 * k);
                    float v;
                    if (!text::parse(tok, v))
                        corrupt_at(lines_, "bad PDB coordinate '" + std::string(tok) + "' in '" +
                                               std::string(line) + "'");
                    pos_.push_back(v * ang_to_nm);
                }
                occ_.push_back(len >= 60 ? field_or(line, 54, 60, 1.0f) : 1.0f);
                bfac_.push_back(len >= 66 ? field_or(line, 60, 66, 0.0f) : 0.0f);
                // While the serials match the reference, the frame reuses the
                // reference topology and only the serial is parsed. From the
                // first mismatch on, the frame builds its own topology.
                if (!own && !(atom < ids_->size() && (*ids_)[atom] == serial_of(line))) {
                    take_reference(atom);
                    own = true;
                }
                if (own) {
                    Topology t;
                    topology(line, len, rec == "HETATM", t);
                    add_atom(t);
                }
                ++atom;
            }
        }

        const std::size_t natoms = pos_.size() / 3;
        if (natoms == 0) return std::nullopt;
        if (!first_natoms_) first_natoms_ = natoms;
        // A model with different atoms becomes the new reference, and batches
        // are variable from then on. This case handles a model with fewer
        // atoms than the reference whose serials all match it.
        if (!own && natoms < ids_->size()) {
            take_reference(natoms);
            own = true;
        }
        if (own) {
            varying_ = varying_ || meta_ready_;
            meta_ready_ = true;
            names_ = share(names_v_);
            elements_ = share(elements_v_);
            ids_ = share(ids_v_);
            res_names_ = share(res_names_v_);
            res_ids_ = share(res_ids_v_);
            chains_ = share(chains_v_);
            hetatm_ = share(hetatm_v_);
        }

        FrameBatch b(natoms, 1);
        b.columns.emplace("positions", pos_);
        b.columns.emplace("occupancy", occ_);
        b.columns.emplace("bfactor", bfac_);
        if (box) b.box.emplace(1, *box);
        b.atom_names = names_;
        b.elements = elements_;
        b.atom_ids = ids_;
        b.residue_names = res_names_;
        b.residue_ids = res_ids_;
        b.chain_ids = chains_;
        b.hetatm = hetatm_;
        return b;
    }

    // Topology of one ATOM/HETATM record, as views into the line. The chain id
    // is stored in chain_buf as one or two UTF-8 bytes.
    struct Topology {
        std::int32_t serial, resid;
        std::string_view name, element, resname, chain;
        std::uint8_t hetatm;
        char chain_buf[2];
    };

    static std::int32_t serial_of(std::string_view line) {
        // Serials wider than 32 bits wrap modulo 2^32. An unparsable serial
        // reads as 0.
        std::uint64_t serial;
        if (!text::parse(field(line, 6, 11), serial)) serial = 0;
        return static_cast<std::int32_t>(static_cast<std::uint32_t>(serial));
    }

    static void topology(std::string_view line, std::size_t len, bool hetatm, Topology& t) {
        t.serial = serial_of(line);
        t.name = text::trim(field_raw(line, 12, 16));
        t.element = len >= 78 ? field(line, 76, 78) : std::string_view();
        // If the element column is blank, guess it from the name.
        if (t.element.empty()) t.element = first_char(t.name);
        if (line.size() > 21) {
            // The chain id is one byte. Interpret it as Latin-1 and store it
            // as UTF-8.
            const auto c = static_cast<unsigned char>(line[21]);
            if (c < 0x80) {
                t.chain_buf[0] = static_cast<char>(c);
                t.chain = std::string_view(t.chain_buf, 1);
            } else {
                t.chain_buf[0] = static_cast<char>(0xC0 | (c >> 6));
                t.chain_buf[1] = static_cast<char>(0x80 | (c & 0x3F));
                t.chain = std::string_view(t.chain_buf, 2);
            }
        } else {
            t.chain = " ";
        }
        t.resname = field(line, 17, 20);
        t.resid = field_or<std::int32_t>(line, 22, 26, 0);
        t.hetatm = hetatm ? 1 : 0;
    }

    // Starts this frame's topology with a copy of the reference's first n
    // atoms, which the frame shares with it.
    void take_reference(std::size_t n) {
        const auto upto = [n](const auto& v) { return v->begin() + static_cast<std::ptrdiff_t>(n); };
        names_v_.assign(names_->begin(), upto(names_));
        elements_v_.assign(elements_->begin(), upto(elements_));
        ids_v_.assign(ids_->begin(), upto(ids_));
        res_names_v_.assign(res_names_->begin(), upto(res_names_));
        res_ids_v_.assign(res_ids_->begin(), upto(res_ids_));
        chains_v_.assign(chains_->begin(), upto(chains_));
        hetatm_v_.assign(hetatm_->begin(), upto(hetatm_));
    }

    void add_atom(const Topology& t) {
        names_v_.emplace_back(t.name);
        elements_v_.emplace_back(t.element);
        ids_v_.push_back(t.serial);
        res_names_v_.emplace_back(t.resname);
        res_ids_v_.push_back(t.resid);
        chains_v_.emplace_back(t.chain);
        hetatm_v_.push_back(t.hetatm);
    }

    std::unique_ptr<InputStream> in_;
    text::LineReader lines_;
    std::optional<std::size_t> first_natoms_;
    std::vector<float> pos_, occ_, bfac_;
    // The *_v_ vectors hold the topology of the frame being read. The Shared
    // pointers hold the reference topology that matching frames reuse.
    bool meta_ready_ = false, varying_ = false;
    std::vector<std::string> names_v_, elements_v_, res_names_v_, chains_v_;
    std::vector<std::int32_t> ids_v_, res_ids_v_;
    std::vector<std::uint8_t> hetatm_v_;
    Shared<std::string> names_, elements_, res_names_, chains_;
    Shared<std::int32_t> ids_, res_ids_;
    Shared<std::uint8_t> hetatm_;
};

class PdbWriter final : public TrajectoryWriter {
public:
    explicit PdbWriter(std::unique_ptr<OutputStream> out) : out_(std::move(out)), w_(*out_) {}
    ~PdbWriter() override {
        try {
            finish();
        } catch (...) {
        }
    }

    void write_batch(const FrameBatch& in) override {
        // The frames of a variable batch are written one at a time. Like any
        // multi-frame batch, they are wrapped in MODEL records.
        if (in.variable()) {
            models_ = models_ || in.nframes > 1;
            for (std::size_t f = 0; f < in.nframes; ++f) write_batch(in.frame(f));
            return;
        }
        // PDB has fixed units, so a batch in LAMMPS native units is converted
        // first.
        std::optional<FrameBatch> converted;
        if (in.units) converted = in.in_internal_units();
        const FrameBatch& batch = converted ? *converted : in;
        const std::size_t natoms = batch.natoms;
        const std::vector<float>* pos = batch.positions();
        if (!pos) throw Error(ErrorKind::InvalidInput, "no positions column");
        if (pos->size() < batch.nframes * natoms * 3)
            throw Error(ErrorKind::InvalidInput, "positions column is shorter than the batch");
        const std::vector<float>* occ = batch.column("occupancy");
        const std::vector<float>* bfac = batch.column("bfactor");
        const bool use_models = batch.nframes > 1 || model_ > 0 || models_;
        auto get = [](const auto& shared, std::size_t i) -> const typename std::decay_t<decltype(*shared)>::value_type* {
            return shared && i < shared->size() ? &(*shared)[i] : nullptr;
        };

        for (std::size_t f = 0; f < batch.nframes; ++f) {
            ++model_;
            if (batch.box && f < batch.box->size()) {
                const CellParams c = box_to_cell_params((*batch.box)[f]);
                w_.put("CRYST1");
                put_fixed_width(w_, c[0] * units::kNmToAngstrom, 3, 9);
                put_fixed_width(w_, c[2] * units::kNmToAngstrom, 3, 9);
                put_fixed_width(w_, c[5] * units::kNmToAngstrom, 3, 9);
                put_fixed_width(w_, c[4], 2, 7);
                put_fixed_width(w_, c[3], 2, 7);
                put_fixed_width(w_, c[1], 2, 7);
                w_.put(" P 1           1\n");
            }
            if (use_models) {
                w_.put("MODEL     ");
                put_int_width(w_, static_cast<std::int64_t>(model_), 4);
                w_.put('\n');
            }
            for (std::size_t a = 0; a < natoms; ++a) {
                const std::size_t idx = f * natoms + a;
                const float* p = pos->data() + idx * 3;
                const std::string* nm = get(batch.atom_names, a);
                const std::string_view name = nm ? std::string_view(*nm) : "X";
                const std::uint8_t* het = get(batch.hetatm, a);
                w_.put(het && *het ? "HETATM" : "ATOM  ");
                // The serial field is 5 columns wide, so larger serials wrap
                // modulo 100000.
                const std::int32_t* id = get(batch.atom_ids, a);
                put_int_width(w_, rem_euclid(id ? *id : static_cast<std::int32_t>(a + 1), 100000), 5);
                w_.put(' ');
                // Names of up to 3 characters start in the second column of
                // the name field (column 14).
                if (char_count(name) <= 3) {
                    w_.put(' ');
                    put_aligned(w_, name, 3, true);
                } else {
                    put_aligned(w_, name, 4, true);
                }
                w_.put(' ');
                const std::string* rn = get(batch.residue_names, a);
                put_aligned(w_, rn ? std::string_view(*rn) : "UNK", 3, false);
                w_.put(' ');
                const std::string* ch = get(batch.chain_ids, a);
                const std::string_view chain = ch ? first_char(*ch) : std::string_view();
                w_.put(chain.empty() ? std::string_view("A") : chain);
                const std::int32_t* rs = get(batch.residue_ids, a);
                put_int_width(w_, rem_euclid(rs ? *rs : 1, 10000), 4);
                w_.put("    ");
                for (int k = 0; k < 3; ++k)
                    put_fixed_width(w_, static_cast<double>(p[k]) * units::kNmToAngstrom, 3, 8);
                put_fixed_width(w_, occ && idx < occ->size() ? (*occ)[idx] : 1.0f, 2, 6);
                put_fixed_width(w_, bfac && idx < bfac->size() ? (*bfac)[idx] : 0.0f, 2, 6);
                w_.put("          ");
                // Use the stored element, or guess it from the name.
                const std::string* el = get(batch.elements, a);
                std::string_view element;
                if (el && !el->empty()) element = *el;
                else element = char_count(name) <= 2 ? name : first_char(name);
                put_aligned(w_, element, 2, false);
                w_.put('\n');
            }
            if (use_models) w_.put("ENDMDL\n");
        }
    }

    void finish() override {
        if (finished_) return;
        finished_ = true;
        w_.put("END\n");
        w_.flush();
        out_->close();
    }

private:
    std::unique_ptr<OutputStream> out_;
    text::TextWriter w_;
    std::size_t model_ = 0;
    // Whether to write MODEL records even for a batch of one frame.
    bool models_ = false;
    bool finished_ = false;
};

}  // namespace

std::unique_ptr<TrajectoryReader> make_pdb_reader(std::unique_ptr<InputStream> in, const FormatOptions&) {
    return std::make_unique<PdbReader>(std::move(in));
}

std::unique_ptr<TrajectoryWriter> make_pdb_writer(std::unique_ptr<OutputStream> out, std::size_t,
                                                  const FormatOptions&) {
    return std::make_unique<PdbWriter>(std::move(out));
}

}  // namespace mdz::detail
