// DCD (CHARMM/NAMD/OpenMM) is a sequence of Fortran unformatted records. Each
// record is bracketed by its payload length as a 4-byte integer, in the byte
// order of the machine that wrote the file. Coordinates and cell lengths are
// in angstroms.
//
// Layout:
//   record 1 (84 bytes): "CORD", nframes, istart, nsavc, nstep, 4 unused,
//                        nfixed, delta (f32), has_unitcell, 8 unused, version
//   record 2:            ntitle, ntitle x 80-byte titles
//   record 3 (4 bytes):  natoms
//   per frame:           [48-byte record of 6 f64: A, gamma, B, beta, alpha, C]
//                        then the X, Y and Z records, natoms f32 each
//
// The frame count is stored at byte 8, before any frame. On a seekable stream
// the writer patches it in finish(). On other streams the writer holds every
// frame in memory until it knows the count.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

#include "bytes.hpp"
#include "internal.hpp"
#include "mdz/error.hpp"

namespace mdz::detail {
namespace {

constexpr float kAngstromToNm = 0.1f;
constexpr float kNmToAngstrom = 10.0f;
constexpr double kPi = 3.14159265358979323846;
// CHARMM's AKMA time unit, in ps. The header's timestep is in these units.
constexpr double kAkmaPs = 0.0488882129;
// Bytes read from the stream at a time.
constexpr std::size_t kReadChunk = 8 << 20;
// Bytes of encoded frames the writer buffers before writing them out.
constexpr std::size_t kWriteChunk = 4 << 20;

inline std::uint32_t bswap32(std::uint32_t v) noexcept {
    return (v >> 24) | ((v >> 8) & 0xff00u) | ((v << 8) & 0xff0000u) | (v << 24);
}
inline std::uint64_t bswap64(std::uint64_t v) noexcept {
    return (static_cast<std::uint64_t>(bswap32(static_cast<std::uint32_t>(v))) << 32) |
           bswap32(static_cast<std::uint32_t>(v >> 32));
}

[[noreturn]] void bad(const std::string& name, const std::string& what) {
    throw Error(ErrorKind::Corrupt, name + ": " + what);
}

// Reads up to n bytes, stopping early only at end of input. Returns the number
// of bytes read.
std::size_t read_full(InputStream& in, std::uint8_t* buf, std::size_t n) {
    std::size_t have = 0;
    while (have < n) {
        std::size_t got = in.read(buf + have, n - have);
        if (got == 0) break;
        have += got;
    }
    return have;
}

void read_exact(InputStream& in, std::uint8_t* buf, std::size_t n) {
    if (read_full(in, buf, n) != n) bad(in.name(), "unexpected end of DCD file");
}

// Reads exactly n bytes into buf, or throws Corrupt. buf grows at most 64 MiB
// at a time as data arrives, so a corrupt atom count cannot force a large
// allocation before the file has supplied the bytes.
void read_exact_grow(InputStream& in, std::vector<std::uint8_t>& buf, std::size_t n) {
    std::size_t have = 0;
    while (have < n) {
        const std::size_t want = std::min(n, have + (std::size_t{64} << 20));
        if (buf.size() < want) buf.resize(want);
        const std::size_t got = read_full(in, buf.data() + have, want - have);
        have += got;
        if (have < want) bad(in.name(), "unexpected end of DCD file");
    }
}

// Decodes values in the file's byte order.
struct Order {
    bool swap;
    std::uint32_t u32(const std::uint8_t* p) const noexcept {
        std::uint32_t v = load_le<std::uint32_t>(p);
        return swap ? bswap32(v) : v;
    }
    std::int32_t i32(const std::uint8_t* p) const noexcept { return static_cast<std::int32_t>(u32(p)); }
    double f64(const std::uint8_t* p) const noexcept {
        std::uint64_t v = load_le<std::uint64_t>(p);
        if (swap) v = bswap64(v);
        double d;
        std::memcpy(&d, &v, 8);
        return d;
    }
};

// Converts one coordinate record (X, Y or Z) to nm and stores it in every
// third float of out.
template <bool Swap>
void scatter_channel(const std::uint8_t* src, std::size_t natoms, float* out) noexcept {
    for (std::size_t i = 0; i < natoms; ++i) {
        std::uint32_t u = load_le<std::uint32_t>(src + 4 * i);
        if (Swap) u = bswap32(u);
        float v;
        std::memcpy(&v, &u, 4);
        out[3 * i] = v * kAngstromToNm;
    }
}

class DcdReader final : public TrajectoryReader {
public:
    explicit DcdReader(std::unique_ptr<InputStream> in) : in_(std::move(in)) { read_header(); }

    std::size_t natoms() const override { return natoms_; }
    std::optional<std::size_t> nframes() const override { return nframes_; }

    std::optional<FrameBatch> read_batch(std::size_t n) override {
        const std::size_t remaining = nframes_ - read_;
        if (remaining == 0 || n == 0) return std::nullopt;
        const std::size_t count = std::min(n, remaining);
        const std::size_t n3 = natoms_ * 3;

        FrameBatch b(natoms_, count);
        std::vector<float>& pos = b.columns["positions"];
        // Cap the reservation. The header's counts are not trusted until the
        // frames have actually been read.
        pos.reserve(std::min(sat_mul(count, n3), std::size_t{64} << 20));
        std::vector<Box> boxes;
        if (has_cell_) boxes.reserve(std::min<std::size_t>(count, 1 << 16));

        const std::size_t per_read = std::max<std::size_t>(1, kReadChunk / frame_bytes_);
        for (std::size_t done = 0; done < count;) {
            const std::size_t m = std::min(per_read, count - done);
            read_exact_grow(*in_, buf_, m * frame_bytes_);
            pos.resize((done + m) * n3);
            for (std::size_t f = 0; f < m; ++f)
                decode_frame(buf_.data() + f * frame_bytes_, pos.data() + (done + f) * n3, boxes);
            done += m;
        }
        read_ += count;
        if (has_cell_) b.box = std::move(boxes);
        return b;
    }

private:
    void read_header() {
        std::uint8_t h[92];
        read_exact(*in_, h, 4);
        const std::uint32_t le = load_le<std::uint32_t>(h);
        const std::uint32_t be = bswap32(le);
        if (le == 84)
            order_.swap = false;
        else if (be == 84)
            order_.swap = true;
        else
            bad(in_->name(), "cannot detect DCD endianness: first record marker is " + std::to_string(le) + " (LE) / " +
                                 std::to_string(be) + " (BE), expected 84");
        read_exact(*in_, h + 4, 4);
        if (std::memcmp(h + 4, "CORD", 4) != 0) bad(in_->name(), "expected CORD magic");
        // The rest of record 1 and its end marker.
        read_exact(*in_, h + 8, 84);
        const std::int32_t nframes = order_.i32(h + 8);
        if (nframes < 0) bad(in_->name(), "DCD header declares negative frame count (" + std::to_string(nframes) + ")");
        nframes_ = static_cast<std::size_t>(nframes);
        // Offsets 12 to 40 hold istart, nsavc, nstep and four unused words.
        const std::int32_t nfixed = order_.i32(h + 40);
        if (nfixed != 0)
            bad(in_->name(), "DCD files with fixed atoms (nfixed=" + std::to_string(nfixed) + ") are not supported");
        // delta at 44, unused at 52..84, version at 84, end marker at 88.
        has_cell_ = order_.i32(h + 48) != 0;

        // Record 2 holds the titles, which are read and discarded.
        std::uint8_t t[8];
        read_exact(*in_, t, 8);
        const std::int32_t ntitle = order_.i32(t + 4);
        if (ntitle < 0 || ntitle > 10000)
            bad(in_->name(), "DCD header declares implausible title count (" + std::to_string(ntitle) + ")");
        std::vector<std::uint8_t> titles(static_cast<std::size_t>(ntitle) * 80);
        read_exact(*in_, titles.data(), titles.size());

        // The end marker of record 2, then record 3 (natoms) with its markers.
        std::uint8_t a[16];
        read_exact(*in_, a, 16);
        const std::int32_t natoms = order_.i32(a + 8);
        if (natoms <= 0)
            bad(in_->name(), "DCD header declares non-positive atom count (" + std::to_string(natoms) + ")");
        natoms_ = static_cast<std::size_t>(natoms);
        coord_marker_ = static_cast<std::uint32_t>(natoms_ * 4);
        frame_bytes_ = (has_cell_ ? 56 : 0) + 3 * (8 + natoms_ * 4);
    }

    void decode_frame(const std::uint8_t* p, float* out, std::vector<Box>& boxes) {
        if (has_cell_) {
            const std::uint32_t marker = order_.u32(p);
            if (marker != 48) bad(in_->name(), "expected 48-byte unitcell record, got " + std::to_string(marker));
            CellParams uc;
            for (std::size_t k = 0; k < 6; ++k) uc[k] = order_.f64(p + 4 + 8 * k);
            // The record is [A, gamma, B, beta, alpha, C]. Convert the lengths
            // to nm.
            uc[0] *= static_cast<double>(kAngstromToNm);
            uc[2] *= static_cast<double>(kAngstromToNm);
            uc[5] *= static_cast<double>(kAngstromToNm);
            // CHARMM stores the cosine of each angle. Values in [-1, 1] are
            // taken as cosines and converted to degrees.
            for (std::size_t k : {1, 3, 4})
                if (uc[k] >= -1.0 && uc[k] <= 1.0) uc[k] = std::acos(uc[k]) * (180.0 / kPi);
            boxes.push_back(cell_params_to_box(uc));
            p += 56;
        }
        for (std::size_t c = 0; c < 3; ++c) {
            const std::uint32_t marker = order_.u32(p);
            if (marker != coord_marker_)
                bad(in_->name(),
                    "expected " + std::to_string(coord_marker_) + "-byte coord record, got " + std::to_string(marker));
            if (order_.swap)
                scatter_channel<true>(p + 4, natoms_, out + c);
            else
                scatter_channel<false>(p + 4, natoms_, out + c);
            p += 8 + natoms_ * 4;
        }
    }

    std::unique_ptr<InputStream> in_;
    Order order_{false};
    std::size_t natoms_ = 0;
    std::size_t nframes_ = 0;
    bool has_cell_ = false;
    std::uint32_t coord_marker_ = 0;
    std::size_t frame_bytes_ = 0;
    std::size_t read_ = 0;
    std::vector<std::uint8_t> buf_;
};

// The header's step numbering: the first frame's step, the steps between
// frames, and the length of a step in AKMA units (0 if unknown). The reader
// does not use these.
struct Timing {
    std::int32_t istart = 0;
    std::int32_t nsavc = 1;
    float delta = 0;
};

// Works out the timing from the first two frames' step and time, where the
// batch has them.
Timing timing_of(const std::vector<std::int64_t>& steps, const std::vector<double>& times) {
    Timing t;
    auto fits = [](std::int64_t v) { return v >= 0 && v <= 0x7fffffff; };
    if (!steps.empty() && fits(steps[0])) t.istart = static_cast<std::int32_t>(steps[0]);
    if (steps.size() == 2 && steps[1] > steps[0] && fits(steps[1] - steps[0]))
        t.nsavc = static_cast<std::int32_t>(steps[1] - steps[0]);
    if (times.size() == 2 && times[1] > times[0]) {
        const double dt = (times[1] - times[0]) / t.nsavc;
        if (std::isfinite(dt)) t.delta = static_cast<float>(dt / kAkmaPs);
    }
    return t;
}

// Writes the three header records. The writer always emits little-endian.
// nframes is stored at byte 8, and nframes * nsavc as nstep at byte 20.
void put_header(Bytes& h, std::size_t natoms, std::size_t nframes, bool has_cell, const Timing& t) {
    auto i32 = [&](std::int64_t v) { put_le(h, static_cast<std::int32_t>(v)); };
    i32(84);
    put_str(h, "CORD");
    i32(static_cast<std::int64_t>(nframes));
    i32(t.istart);
    i32(t.nsavc);
    i32(std::min<std::int64_t>(static_cast<std::int64_t>(nframes) * t.nsavc, 0x7fffffff));  // nstep
    for (int k = 0; k < 4; ++k) i32(0);
    i32(0);  // nfixed
    put_le(h, t.delta);
    i32(has_cell ? 1 : 0);
    for (int k = 0; k < 8; ++k) i32(0);
    i32(24);  // version
    i32(84);

    i32(84);
    i32(1);  // ntitle
    char title[80] = "Created by mdz";
    put_bytes(h, title, 80);
    i32(84);

    i32(4);
    i32(static_cast<std::int64_t>(natoms));
    i32(4);
}

class DcdWriter final : public TrajectoryWriter {
public:
    DcdWriter(std::unique_ptr<OutputStream> out, std::size_t natoms)
        : out_(std::move(out)), natoms_(natoms), seekable_(out_->seekable()) {}

    ~DcdWriter() override {
        try {
            finish();
        } catch (...) {
        }
    }

    void write_batch(const FrameBatch& batch) override {
        if (finished_) throw Error(ErrorKind::InvalidInput, "DCD writer is already finished");
        // The header holds a single atom count for the whole file.
        if (batch.variable())
            throw Error(ErrorKind::InvalidInput,
                        "DCD holds one atom count per file and cannot store frames whose atom "
                        "count varies; write xtc, extxyz, xyz, pdb or a LAMMPS dump instead");
        if (started_ && batch.natoms != natoms_)
            throw Error(ErrorKind::InvalidInput, "DCD holds one atom count per file: this batch has " +
                                                     std::to_string(batch.natoms) + " atoms per frame, the file " +
                                                     std::to_string(natoms_));
        // DCD has fixed units, so a batch in LAMMPS native units is converted
        // first.
        std::optional<FrameBatch> converted;
        const FrameBatch* b = &batch;
        if (batch.units) b = &converted.emplace(batch.in_internal_units());

        const std::vector<float>* pos = b->positions();
        if (!pos) throw Error(ErrorKind::InvalidInput, "no positions in batch");
        if (!started_) {
            if (natoms_ == 0) natoms_ = b->natoms;
            if (natoms_ > 0x3fffffff)
                throw Error(ErrorKind::InvalidInput, "DCD holds at most 1073741823 atoms per frame");
        }
        if (b->natoms != natoms_ || pos->size() != b->nframes * natoms_ * 3)
            throw Error(ErrorKind::InvalidInput,
                        "DCD writer: batch does not hold " + std::to_string(natoms_) + " atoms per frame");
        if (nframes_ + b->nframes > 0x7fffffff)
            throw Error(ErrorKind::InvalidInput, "DCD holds at most 2147483647 frames");
        if (!started_) {
            // The header declares whether frames carry a unit cell, so the
            // first batch decides it for the whole file.
            has_cell_ = b->box.has_value();
            if (seekable_) {
                Bytes h;
                put_header(h, natoms_, 0, has_cell_, Timing{});
                out_->write(h.data(), h.size());
            }
            started_ = true;
        }

        if (has_cell_ && (!b->box || b->box->size() < b->nframes))
            throw Error(ErrorKind::InvalidInput, "frame " + std::to_string(nframes_ + (b->box ? b->box->size() : 0)) +
                                                     " has no box, but the DCD header declares unit cells");

        // Keep the first two frames' step and time for the header.
        auto keep = [&](auto& kept, const auto& props, const char* name) {
            auto it = props.find(name);
            // Stop once two are kept, or if an earlier frame lacked it.
            if (it == props.end() || kept.size() >= 2 || kept.size() < nframes_) return;
            for (std::size_t f = 0; f < b->nframes && kept.size() < 2 && f < it->second.size(); ++f)
                kept.push_back(it->second[f]);
        };
        keep(steps_, b->int_frame_props, "step");
        keep(times_, b->frame_props, "time");

        const std::size_t n3 = natoms_ * 3;
        const std::uint32_t rec = static_cast<std::uint32_t>(natoms_ * 4);
        for (std::size_t f = 0; f < b->nframes; ++f) {
            if (has_cell_) {
                const CellParams uc = box_to_cell_params((*b->box)[f]);
                const double rad = kPi / 180.0;
                put_le<std::uint32_t>(buf_, 48);
                // CHARMM and NAMD store the cosine of each angle.
                put_le(buf_, uc[0] * static_cast<double>(kNmToAngstrom));
                put_le(buf_, std::cos(uc[1] * rad));
                put_le(buf_, uc[2] * static_cast<double>(kNmToAngstrom));
                put_le(buf_, std::cos(uc[3] * rad));
                put_le(buf_, std::cos(uc[4] * rad));
                put_le(buf_, uc[5] * static_cast<double>(kNmToAngstrom));
                put_le<std::uint32_t>(buf_, 48);
            }
            const float* frame = pos->data() + f * n3;
            for (std::size_t c = 0; c < 3; ++c) {
                put_le(buf_, rec);
                const std::size_t at = buf_.size();
                buf_.resize(at + natoms_ * 4);
                std::uint8_t* dst = buf_.data() + at;
                for (std::size_t i = 0; i < natoms_; ++i) store_le(dst + 4 * i, frame[3 * i + c] * kNmToAngstrom);
                put_le(buf_, rec);
            }
            if (seekable_ && buf_.size() >= kWriteChunk) drain();
        }
        nframes_ += b->nframes;
    }

    void finish() override {
        if (finished_) return;
        finished_ = true;
        if (seekable_ && started_) {
            drain();
            // Rewrite the header written up front, now that the frame count
            // and timing are known. Its size does not change.
            Bytes h;
            put_header(h, natoms_, nframes_, has_cell_, timing_of(steps_, times_));
            out_->seek(0);
            out_->write(h.data(), h.size());
        } else {
            // Nothing has been written yet, either because the stream is not
            // seekable or because no frames arrived. Write the header with the
            // final count, then the buffered frames.
            Bytes h;
            put_header(h, natoms_, nframes_, has_cell_, timing_of(steps_, times_));
            out_->write(h.data(), h.size());
            drain();
        }
        out_->close();
    }

private:
    void drain() {
        if (!buf_.empty()) out_->write(buf_.data(), buf_.size());
        buf_.clear();
    }

    std::unique_ptr<OutputStream> out_;
    std::size_t natoms_;
    bool seekable_;
    bool started_ = false;
    bool has_cell_ = false;
    bool finished_ = false;
    std::size_t nframes_ = 0;
    // Step and time of the first two frames, where every frame so far had them.
    std::vector<std::int64_t> steps_;
    std::vector<double> times_;
    // Encoded frames not yet written. On a seekable stream this is at most
    // about kWriteChunk bytes. Otherwise it holds every frame until finish().
    Bytes buf_;
};

}  // namespace

std::unique_ptr<TrajectoryReader> make_dcd_reader(std::unique_ptr<InputStream> in, const FormatOptions&) {
    return std::make_unique<DcdReader>(std::move(in));
}

std::unique_ptr<TrajectoryWriter> make_dcd_writer(std::unique_ptr<OutputStream> out, std::size_t natoms,
                                                  const FormatOptions&) {
    return std::make_unique<DcdWriter>(std::move(out), natoms);
}

}  // namespace mdz::detail
