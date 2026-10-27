// Plain XYZ. Each frame is an atom count, a comment line (ignored on read),
// then one "element x y z" line per atom with coordinates in angstroms. Only
// positions and atom names are stored.

#include <algorithm>
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

class XyzReader final : public TrajectoryReader {
public:
    explicit XyzReader(std::unique_ptr<InputStream> in) : in_(std::move(in)), lines_(*in_) {}

    std::size_t natoms() const override { return first_natoms_.value_or(0); }

    std::optional<FrameBatch> read_batch(std::size_t n) override {
        std::optional<FrameBatch> out;
        for (std::size_t i = 0; i < n; ++i)
            if (!read_frame(out)) break;
        return out;
    }

private:
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
        lines_.next(line);  // skip the comment line

        // While each frame has the same atoms as the previous one, it is read
        // directly into the batch. Once the atoms change (a different count or
        // different names), each frame is built separately and the batch
        // becomes variable.
        const bool direct = !varying_ && (!seen_ || natoms == ref_natoms_);
        std::optional<FrameBatch> one;
        std::optional<FrameBatch>& dst = direct ? out : one;
        if (!dst) dst.emplace(natoms, 0);
        std::vector<float>& pos = dst->columns["positions"];
        // Cap the reservation, since the count may be corrupt.
        pos.reserve(pos.size() + std::min<std::size_t>(natoms, std::size_t(1) << 24) * 3);
        // Compare each name with the reference names as it is read. From the
        // first mismatch, or once past the end of the reference, this frame
        // keeps its own copy of the names.
        const std::size_t ref = names_ ? names_->size() : 0;
        bool own = !seen_;
        own_names_.clear();
        const float ang_to_nm = static_cast<float>(units::kAngstromToNm);
        for (std::size_t a = 0; a < natoms; ++a) {
            if (!lines_.next(line)) line = {};
            std::string_view rest = line;
            const std::string_view el = text::next_token(rest);
            if (el.empty()) corrupt_at(lines_, "xyz line needs element x y z");
            for (int k = 0; k < 3; ++k) {
                const std::string_view tok = text::next_token(rest);
                if (tok.empty()) corrupt_at(lines_, "xyz line needs element x y z");
                float v;
                if (!text::parse(tok, v)) corrupt_at(lines_, "bad xyz coordinate '" + std::string(tok) + "'");
                pos.push_back(v * ang_to_nm);
            }
            if (!own && !(a < ref && (*names_)[a] == el)) {
                if (names_) own_names_.assign(names_->begin(), names_->begin() + static_cast<std::ptrdiff_t>(a));
                own = true;
            }
            if (own) own_names_.emplace_back(el);
        }
        if (!own && natoms != ref) {
            // Fewer atoms than the reference, but all names match it.
            own_names_.assign(names_->begin(), names_->begin() + static_cast<std::ptrdiff_t>(natoms));
            own = true;
        }
        if (own) {
            // The first frame sets the reference. A later change switches the
            // reader to variable mode for the rest of the file.
            names_ = natoms ? share(own_names_) : nullptr;
            ref_natoms_ = natoms;
            varying_ = varying_ || seen_;
        }
        seen_ = true;
        if (!varying_) {
            out->nframes += 1;
            if (!out->atom_names) out->atom_names = names_;
            return true;
        }
        FrameBatch frame;
        if (direct) {
            // This frame was read into the batch before its atoms were found
            // to differ, so take it back out.
            out->nframes += 1;
            frame = pop_last_frame(*out);
        } else {
            frame = std::move(*one);
            frame.nframes = 1;
        }
        frame.atom_names = names_;
        append_varying(out, std::move(frame));
        return true;
    }

    std::unique_ptr<InputStream> in_;
    text::LineReader lines_;
    std::optional<std::size_t> first_natoms_;
    // Atom count and names of the previous frame.
    std::size_t ref_natoms_ = 0;
    Shared<std::string> names_;
    bool seen_ = false, varying_ = false;
    // Scratch space for a frame's names when they differ from the reference.
    std::vector<std::string> own_names_;
};

class XyzWriter final : public TrajectoryWriter {
public:
    explicit XyzWriter(std::unique_ptr<OutputStream> out) : out_(std::move(out)), w_(*out_) {}
    ~XyzWriter() override {
        try {
            finish();
        } catch (...) {
        }
    }

    void write_batch(const FrameBatch& in) override {
        // Each frame stores its own atom count, so the frames of a variable
        // batch are written one at a time.
        if (in.variable()) {
            for (std::size_t f = 0; f < in.nframes; ++f) write_batch(in.frame(f));
            return;
        }
        // XYZ has fixed units, so a batch in LAMMPS native units is converted
        // first.
        std::optional<FrameBatch> converted;
        if (in.units) converted = in.in_internal_units();
        const FrameBatch& batch = converted ? *converted : in;
        const std::size_t natoms = batch.natoms;
        const std::vector<float>* pos = batch.positions();
        if (!pos) throw Error(ErrorKind::InvalidInput, "no positions column");
        if (pos->size() < batch.nframes * natoms * 3)
            throw Error(ErrorKind::InvalidInput, "positions column is shorter than the batch");
        const std::vector<std::string>* elements = batch.elements.get();
        const std::vector<std::string>* names = batch.atom_names.get();
        for (std::size_t f = 0; f < batch.nframes; ++f) {
            w_.put_uint(natoms).put('\n').put("Generated by mdz\n");
            const float* p = pos->data() + f * natoms * 3;
            for (std::size_t a = 0; a < natoms; ++a) {
                // The first column is an element symbol. Use the element if
                // present, then the atom name, then "X".
                std::string_view name = "X";
                if (elements && a < elements->size()) name = (*elements)[a];
                else if (names && a < names->size()) name = (*names)[a];
                w_.put(name);
                for (int k = 0; k < 3; ++k)
                    w_.put(' ').put_fixed(static_cast<double>(p[3 * a + k]) * units::kNmToAngstrom, 6);
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
    bool finished_ = false;
};

}  // namespace

std::unique_ptr<TrajectoryReader> make_xyz_reader(std::unique_ptr<InputStream> in, const FormatOptions&) {
    return std::make_unique<XyzReader>(std::move(in));
}

std::unique_ptr<TrajectoryWriter> make_xyz_writer(std::unique_ptr<OutputStream> out, std::size_t,
                                                  const FormatOptions&) {
    return std::make_unique<XyzWriter>(std::move(out));
}

}  // namespace mdz::detail
