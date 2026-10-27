// XTC (GROMACS) is a sequence of self-contained XDR (big-endian) frames. There
// is no global header and no frame count. Each frame is
//
//   magic (1995), natoms, step (i32), time (f32), box (9 x f32, nm),
//   natoms again, then the coordinates in xdr3dfcoord (see xdr3d.hpp)
//
// The coordinate codec is in xdr3d.cpp. This file handles the frame headers.

#include <algorithm>
#include <cstring>
#include <string>

#include "bytes.hpp"
#include "internal.hpp"
#include "mdz/error.hpp"
#include "xdr3d.hpp"

namespace mdz::detail {
namespace {

constexpr std::int32_t kMagic = 1995;
// Writer precision. 1000 means 0.001 nm, which is the GROMACS default.
constexpr float kPrecision = 1000.0f;
// Frame header: magic, natoms, step, time, box, natoms.
constexpr std::size_t kHeaderBytes = 14 * 4;
// Coordinate header before the opaque payload. It holds precision, minint
// (3 words), maxint (3 words), smallidx and the payload length.
constexpr std::size_t kCoordHeaderBytes = 9 * 4;
constexpr std::size_t kReadChunk = 4 << 20;
constexpr std::size_t kWriteChunk = 4 << 20;

inline std::uint32_t load_be32(const std::uint8_t* p) noexcept {
    return (static_cast<std::uint32_t>(p[0]) << 24) | (static_cast<std::uint32_t>(p[1]) << 16) |
           (static_cast<std::uint32_t>(p[2]) << 8) | p[3];
}
inline std::int32_t load_be_i32(const std::uint8_t* p) noexcept { return static_cast<std::int32_t>(load_be32(p)); }
inline float load_be_f32(const std::uint8_t* p) noexcept {
    std::uint32_t u = load_be32(p);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline void put_be32(Bytes& b, std::uint32_t v) {
    const std::uint8_t q[4] = {static_cast<std::uint8_t>(v >> 24), static_cast<std::uint8_t>(v >> 16),
                               static_cast<std::uint8_t>(v >> 8), static_cast<std::uint8_t>(v)};
    b.insert(b.end(), q, q + 4);
}
inline void put_be_i32(Bytes& b, std::int32_t v) { put_be32(b, static_cast<std::uint32_t>(v)); }
inline void put_be_f32(Bytes& b, float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    put_be32(b, u);
}

// A buffered window over the stream. Frames are variable-length, so bytes are
// read in large chunks and frames are decoded in place.
class Window {
public:
    explicit Window(InputStream& in) : in_(in) {}

    // Makes n bytes available from the current position. Returns false if the
    // stream ends first. The buffer grows only as data arrives, so a corrupt
    // length cannot force a large allocation.
    bool ensure(std::size_t n) {
        while (end_ - begin_ < n) {
            if (eof_) return false;
            // Move the unread tail to the front. Then grow the buffer to at
            // least one chunk, and to at most one chunk past the data already
            // held. Growth doubles so that a large frame is not copied
            // repeatedly.
            if (begin_ > 0) {
                std::memmove(buf_.data(), buf_.data() + begin_, end_ - begin_);
                end_ -= begin_;
                begin_ = 0;
            }
            const std::size_t target = std::max(kReadChunk, std::min(n, end_ + kReadChunk));
            if (buf_.size() < target)
                buf_.resize(std::min(std::max(target, 2 * buf_.size()), std::max(n, kReadChunk)));
            const std::size_t got = in_.read(buf_.data() + end_, buf_.size() - end_);
            if (got == 0) eof_ = true;
            end_ += got;
        }
        return true;
    }
    const std::uint8_t* data() const noexcept { return buf_.data() + begin_; }
    void consume(std::size_t n) noexcept { begin_ += n; }

private:
    InputStream& in_;
    std::vector<std::uint8_t> buf_;
    std::size_t begin_ = 0, end_ = 0;
    bool eof_ = false;
};

class XtcReader final : public TrajectoryReader {
public:
    explicit XtcReader(std::unique_ptr<InputStream> in) : in_(std::move(in)), win_(*in_) {
        // Peek at natoms in the first frame's header without consuming it.
        if (!win_.ensure(8)) bad("unexpected end of XTC file");
        const std::int32_t magic = load_be_i32(win_.data());
        if (magic != kMagic) bad("bad XTC magic: " + std::to_string(magic));
        natoms_ = first_natoms_ = read_natoms(win_.data() + 4);
    }

    // The atom count of the first frame. Later frames may differ.
    std::size_t natoms() const override { return first_natoms_; }

    std::optional<FrameBatch> read_batch(std::size_t n) override {
        FrameBatch b(natoms_, 0);
        // Once the atom count has changed, every batch records per-frame
        // counts. XTC has no atom ids, so atoms are identified only by their
        // position in the frame.
        if (varying_) b.frame_natoms.emplace();
        std::size_t at = 0;  // number of position values filled so far
        std::vector<float>& pos = b.columns["positions"];
        std::vector<double>& time = b.frame_props["time"];
        std::vector<std::int64_t>& step = b.int_frame_props["step"];
        std::vector<Box> boxes;
        std::size_t f = 0;
        for (; f < n; ++f) {
            // Fewer than 4 bytes left is treated as end of file.
            if (!win_.ensure(4)) break;
            if (!win_.ensure(kHeaderBytes)) bad("unexpected end of XTC file");
            const std::uint8_t* h = win_.data();
            const std::int32_t magic = load_be_i32(h);
            if (magic != kMagic)
                bad("bad XTC magic: " + std::to_string(magic) + " (expected " + std::to_string(kMagic) + ")");
            const std::size_t natoms = read_natoms(h + 4);
            const std::int32_t s = load_be_i32(h + 8);
            const float t = load_be_f32(h + 12);
            Box box;
            for (std::size_t k = 0; k < 9; ++k) box[k] = static_cast<double>(load_be_f32(h + 16 + 4 * k));
            const std::size_t natoms2 = read_natoms(h + 52);
            if (natoms2 != natoms)
                bad("natoms mismatch: " + std::to_string(natoms) + " vs " + std::to_string(natoms2));
            if (natoms != natoms_ || varying_) {
                if (!b.variable()) b.frame_natoms.emplace(f, natoms_);
                b.natoms = 0;
                varying_ = true;
                natoms_ = natoms;
            }
            win_.consume(kHeaderBytes);

            const std::size_t len = coord_bytes(natoms);
            if (f == 0) {
                // The coordinate bytes for this atom count are now in the
                // buffer, so it is safe to reserve based on it.
                const std::size_t hint = std::min<std::size_t>(n, 4096);
                pos.reserve(hint * natoms * 3);
                time.reserve(hint);
                step.reserve(hint);
                boxes.reserve(hint);
            }
            pos.resize(at + natoms * 3);
            const std::size_t used = xdr3d_decode(ByteSpan(win_.data(), len), natoms, lip_, pos.data() + at);
            at += natoms * 3;
            if (b.variable()) b.frame_natoms->push_back(natoms);
            if (used != len) bad("XTC frame payload has trailing bytes");
            win_.consume(len);
            time.push_back(static_cast<double>(t));
            step.push_back(s);
            boxes.push_back(box);
        }
        if (f == 0) return std::nullopt;
        b.nframes = f;
        b.box = std::move(boxes);
        return b;
    }

private:
    [[noreturn]] void bad(const std::string& what) const { throw Error(ErrorKind::Corrupt, in_->name() + ": " + what); }

    std::size_t read_natoms(const std::uint8_t* p) const {
        const std::int32_t v = load_be_i32(p);
        if (v < 0) bad("bad XTC atom count " + std::to_string(v));
        return static_cast<std::size_t>(v);
    }

    // Makes the coordinate block of a frame with natoms atoms available in the
    // window and returns its length in bytes. Frames of 9 atoms or fewer are
    // stored uncompressed.
    std::size_t coord_bytes(std::size_t natoms) {
        if (natoms <= 9) {
            if (!win_.ensure(natoms * 12)) bad("unexpected end of XTC file");
            return natoms * 12;
        }
        if (!win_.ensure(kCoordHeaderBytes)) bad("unexpected end of XTC file");
        const std::int32_t nbytes = load_be_i32(win_.data() + kCoordHeaderBytes - 4);
        if (nbytes < 0) bad("negative XDR opaque length");
        const std::size_t padded = (static_cast<std::size_t>(nbytes) + 3) & ~std::size_t{3};
        // Every atom takes at least one bit. Check this before sizing anything
        // by natoms. xdr3d_decode repeats the check.
        if (natoms > sat_mul(static_cast<std::size_t>(nbytes), 8)) bad("XTC payload too short for its atom count");
        if (!win_.ensure(kCoordHeaderBytes + padded)) bad("XDR opaque data truncated");
        return kCoordHeaderBytes + padded;
    }

    std::unique_ptr<InputStream> in_;
    Window win_;
    // Atom counts of the previous frame and of the first frame.
    std::size_t natoms_ = 0, first_natoms_ = 0;
    bool varying_ = false;
    std::vector<std::int32_t> lip_;
};

class XtcWriter final : public TrajectoryWriter {
public:
    explicit XtcWriter(std::unique_ptr<OutputStream> out) : out_(std::move(out)) {}

    ~XtcWriter() override {
        try {
            finish();
        } catch (...) {
        }
    }

    void write_batch(const FrameBatch& batch) override {
        if (finished_) throw Error(ErrorKind::InvalidInput, "XTC writer is already finished");
        // XTC has fixed units, so a batch in LAMMPS native units is converted
        // first.
        std::optional<FrameBatch> converted;
        const FrameBatch* b = &batch;
        if (batch.units) b = &converted.emplace(batch.in_internal_units());

        const std::vector<float>* pos = b->positions();
        if (!pos) throw Error(ErrorKind::InvalidInput, "no positions in batch");
        // Every frame stores its own atom count, so batches, and frames within
        // a variable batch, may have different counts.
        if (pos->size() != b->total_atoms() * 3)
            throw Error(ErrorKind::InvalidInput, "XTC writer: batch does not hold " +
                                                     (b->variable() ? std::string("its frames' atoms")
                                                                    : std::to_string(b->natoms) + " atoms per frame"));
        for (std::size_t f = 0; f < b->nframes; ++f)
            if (b->atoms_in(f) > 0x7fffffff)
                throw Error(ErrorKind::InvalidInput, "XTC holds at most 2147483647 atoms per frame");

        auto ti = b->frame_props.find("time");
        auto si = b->int_frame_props.find("step");
        const std::vector<double>* times = ti == b->frame_props.end() ? nullptr : &ti->second;
        const std::vector<std::int64_t>* steps = si == b->int_frame_props.end() ? nullptr : &si->second;

        const float* p = pos->data();
        for (std::size_t f = 0; f < b->nframes; ++f) {
            const std::size_t natoms = b->atoms_in(f);
            // Frames without a step or time (for example from DCD or xyz input)
            // use the global frame index instead, so the output does not
            // depend on how frames were batched.
            const std::size_t g = written_ + f;
            std::int32_t step;
            if (steps && f < steps->size()) {
                const std::int64_t s = (*steps)[f];
                if (s < INT32_MIN || s > INT32_MAX)
                    throw Error(ErrorKind::InvalidInput,
                                "step " + std::to_string(s) + " does not fit XTC's 32-bit step field");
                step = static_cast<std::int32_t>(s);
            } else {
                step = static_cast<std::int32_t>(static_cast<std::uint32_t>(g));
            }
            const float time = times && f < times->size() ? static_cast<float>((*times)[f])
                                                          : static_cast<float>(g) * 0.001f;

            const std::size_t start = buf_.size();
            put_be_i32(buf_, kMagic);
            put_be_i32(buf_, static_cast<std::int32_t>(natoms));
            put_be_i32(buf_, step);
            put_be_f32(buf_, time);
            // A frame without a box gets a zero box.
            const bool has_box = b->box && f < b->box->size();
            for (std::size_t k = 0; k < 9; ++k) put_be_f32(buf_, has_box ? static_cast<float>((*b->box)[f][k]) : 0.0f);
            put_be_i32(buf_, static_cast<std::int32_t>(natoms));
            try {
                xdr3d_encode(p, natoms, kPrecision, buf_);
            } catch (...) {
                // Leave no partial frame behind.
                buf_.resize(start);
                written_ += f;
                throw;
            }
            p += natoms * 3;
            if (buf_.size() >= kWriteChunk) drain();
        }
        written_ += b->nframes;
    }

    void finish() override {
        if (finished_) return;
        finished_ = true;
        drain();
        out_->close();
    }

private:
    void drain() {
        if (!buf_.empty()) out_->write(buf_.data(), buf_.size());
        buf_.clear();
    }

    std::unique_ptr<OutputStream> out_;
    // Frames written so far across all batches. Used for the fallback step and
    // time.
    std::size_t written_ = 0;
    bool finished_ = false;
    Bytes buf_;
};

}  // namespace

std::unique_ptr<TrajectoryReader> make_xtc_reader(std::unique_ptr<InputStream> in, const FormatOptions&) {
    return std::make_unique<XtcReader>(std::move(in));
}

std::unique_ptr<TrajectoryWriter> make_xtc_writer(std::unique_ptr<OutputStream> out, std::size_t,
                                                  const FormatOptions&) {
    return std::make_unique<XtcWriter>(std::move(out));
}

}  // namespace mdz::detail
