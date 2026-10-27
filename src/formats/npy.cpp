// NumPy .npy files holding positions in nm. The array is float32 in C order,
// with shape (nframes, natoms, 3), or (natoms, 3) for a single frame.
//
//   "\x93NUMPY", major, minor, header length (u16 in v1, u32 in v2),
//   a Python dict literal padded with spaces to a 64-byte boundary and ended
//   by '\n', then the raw little-endian values.
//
// The header holds the frame count. On a seekable stream the writer reserves
// room for the largest possible count and patches the header in finish(). On
// other streams it holds every frame in memory until it knows the count.

#include <algorithm>
#include <cstring>
#include <string>
#include <string_view>

#include "bytes.hpp"
#include "internal.hpp"
#include "mdz/error.hpp"

namespace mdz::detail {
namespace {

// Bytes read from the stream at a time.
constexpr std::size_t kReadChunk = 8 << 20;

[[noreturn]] void bad(const std::string& name, const std::string& what) {
    throw Error(ErrorKind::Corrupt, name + ": " + what);
}

std::size_t read_full(InputStream& in, void* buf, std::size_t n) {
    auto* p = static_cast<std::uint8_t*>(buf);
    std::size_t have = 0;
    while (have < n) {
        std::size_t got = in.read(p + have, n - have);
        if (got == 0) break;
        have += got;
    }
    return have;
}

bool valid_utf8(std::string_view s) noexcept {
    for (std::size_t i = 0; i < s.size();) {
        const auto c = static_cast<unsigned char>(s[i]);
        std::size_t n;
        std::uint32_t cp;
        if (c < 0x80) {
            ++i;
            continue;
        } else if ((c & 0xe0) == 0xc0) {
            n = 1;
            cp = c & 0x1fu;
        } else if ((c & 0xf0) == 0xe0) {
            n = 2;
            cp = c & 0x0fu;
        } else if ((c & 0xf8) == 0xf0) {
            n = 3;
            cp = c & 0x07u;
        } else {
            return false;
        }
        if (i + n >= s.size()) return false;
        for (std::size_t k = 1; k <= n; ++k) {
            const auto d = static_cast<unsigned char>(s[i + k]);
            if ((d & 0xc0) != 0x80) return false;
            cp = (cp << 6) | (d & 0x3fu);
        }
        // Reject overlong forms, surrogates and values past U+10FFFF.
        static constexpr std::uint32_t kMin[4] = {0, 0x80, 0x800, 0x10000};
        if (cp < kMin[n] || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) return false;
        i += n + 1;
    }
    return true;
}

// Parses a whole token as an unsigned integer: an optional '+', then digits.
// Returns false on overflow.
bool parse_dim(std::string_view s, std::size_t& out) noexcept {
    if (!s.empty() && s[0] == '+') s.remove_prefix(1);
    if (s.empty()) return false;
    std::size_t v = 0;
    for (char c : s) {
        if (c < '0' || c > '9') return false;
        const std::size_t d = static_cast<std::size_t>(c - '0');
        if (v > (SIZE_MAX - d) / 10) return false;
        v = v * 10 + d;
    }
    out = v;
    return true;
}

std::string_view trim(std::string_view s) noexcept {
    auto space = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; };
    while (!s.empty() && space(s.front())) s.remove_prefix(1);
    while (!s.empty() && space(s.back())) s.remove_suffix(1);
    return s;
}

std::string shape_string(const std::vector<std::size_t>& shape) {
    std::string s = "(";
    for (std::size_t i = 0; i < shape.size(); ++i) s += (i ? ", " : "") + std::to_string(shape[i]);
    return s + (shape.size() == 1 ? ",)" : ")");
}

class NpyReader final : public TrajectoryReader {
public:
    explicit NpyReader(std::unique_ptr<InputStream> in) : in_(std::move(in)) {
        read_header();
        if (nframes_ == 0) check_end();
    }

    std::size_t natoms() const override { return natoms_; }
    std::optional<std::size_t> nframes() const override { return nframes_; }

    std::optional<FrameBatch> read_batch(std::size_t n) override {
        const std::size_t remaining = nframes_ - read_;
        if (remaining == 0 || n == 0) return std::nullopt;
        const std::size_t count = std::min(n, remaining);
        const std::size_t n3 = natoms_ * 3;
        FrameBatch b(natoms_, count);
        std::vector<float>& pos = b.columns["positions"];
        // The values are already little-endian float32, so read them directly
        // into the column. Reading a chunk at a time means a shape larger
        // than the data cannot allocate more memory than the data itself.
        const std::size_t total = count * n3;
        pos.reserve(std::min(total, kReadChunk / 4));
        for (std::size_t have = 0; have < total;) {
            const std::size_t want = std::min(total - have, kReadChunk / 4);
            pos.resize(have + want);
            const std::size_t got = read_full(*in_, pos.data() + have, want * 4);
            data_bytes_ += got;
            if (got != want * 4) mismatch();
            have += want;
        }
        read_ += count;
        if (read_ == nframes_) check_end();
        return b;
    }

private:
    void read_header() {
        const std::string& name = in_->name();
        std::uint8_t pre[12];
        if (read_full(*in_, pre, 10) != 10 || std::memcmp(pre, "\x93NUMPY", 6) != 0) bad(name, "not a valid .npy file");
        std::size_t len;
        if (pre[6] == 1) {
            len = load_le<std::uint16_t>(pre + 8);
        } else if (pre[6] == 2) {
            if (read_full(*in_, pre + 10, 2) != 2) bad(name, "npy v2 header truncated");
            len = load_le<std::uint32_t>(pre + 8);
        } else {
            bad(name, "unsupported .npy version");
        }
        std::string header(len, '\0');
        if (read_full(*in_, header.data(), len) != len) bad(name, "npy header extends past end of file");
        if (!valid_utf8(header)) bad(name, "npy header is not valid UTF-8");

        // Only little-endian or native float32 is accepted. Native is
        // little-endian, which bytes.hpp checks at build time.
        if (header.find("'<f4'") == std::string::npos && header.find("'f4'") == std::string::npos &&
            header.find("'=f4'") == std::string::npos)
            bad(name, "only float32 arrays supported, got header: " + header);
        if (header.find("'fortran_order': True") != std::string::npos) bad(name, "Fortran order not supported");

        const std::string_view key = "'shape': (";
        const std::size_t at = header.find(key);
        if (at == std::string::npos) bad(name, "no shape in header");
        const std::size_t open = at + key.size();
        const std::size_t close = header.find(')', open);
        if (close == std::string::npos) bad(name, "malformed shape");
        std::string_view dims = std::string_view(header).substr(open, close - open);
        while (true) {
            const std::size_t comma = dims.find(',');
            std::string_view d = trim(dims.substr(0, comma));
            if (!d.empty()) {
                std::size_t v;
                if (!parse_dim(d, v)) bad(name, "bad npy shape dimension '" + std::string(d) + "'");
                shape_.push_back(v);
            }
            if (comma == std::string_view::npos) break;
            dims.remove_prefix(comma + 1);
        }
        expected_ = 1;
        for (std::size_t d : shape_) expected_ = checked_mul(expected_, d, "npy shape");

        if (shape_.size() == 3 && shape_[2] == 3) {
            nframes_ = shape_[0];
            natoms_ = shape_[1];
        } else if (shape_.size() == 2 && shape_[1] == 3) {
            nframes_ = 1;
            natoms_ = shape_[0];
        } else {
            bad(name, "expected shape (nframes, natoms, 3) or (natoms, 3), got " + shape_string(shape_));
        }
    }

    // Called after the last frame. Throws if any bytes follow the data.
    void check_end() {
        std::uint8_t probe[4096];
        while (std::size_t got = read_full(*in_, probe, sizeof probe)) {
            data_bytes_ += got;
            if (got < sizeof probe) break;
        }
        if (data_bytes_ != expected_ * 4) mismatch();
    }

    // Throws because the data does not match the shape. Reads the rest of the
    // stream first so the message can report the actual length.
    [[noreturn]] void mismatch() {
        std::uint8_t probe[4096];
        while (std::size_t got = read_full(*in_, probe, sizeof probe)) data_bytes_ += got;
        if (data_bytes_ % 4 != 0) bad(in_->name(), "npy data length is not a multiple of 4 (truncated float32 array)");
        bad(in_->name(), "npy data has " + std::to_string(data_bytes_ / 4) + " float32 values but shape " +
                             shape_string(shape_) + " requires " + std::to_string(expected_));
    }

    std::unique_ptr<InputStream> in_;
    std::vector<std::size_t> shape_;
    std::size_t expected_ = 0;
    std::size_t natoms_ = 0;
    std::size_t nframes_ = 0;
    std::size_t read_ = 0;
    std::size_t data_bytes_ = 0;
};

// Builds the v1 header for shape (nframes, natoms, 3). It is padded to a
// 64-byte boundary and to at least min_total bytes, which must be a multiple
// of 64.
Bytes npy_header(std::size_t nframes, std::size_t natoms, std::size_t min_total = 0) {
    std::string dict =
        "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_string({nframes, natoms, 3}) + ", }";
    std::size_t total = 10 + dict.size() + 1;
    total += (64 - total % 64) % 64;
    total = std::max(total, min_total);
    dict.append(total - 10 - dict.size() - 1, ' ');
    dict += '\n';
    if (dict.size() > 0xffff) throw Error(ErrorKind::InvalidInput, "npy header length does not fit");
    Bytes h;
    put_str(h, "\x93NUMPY");
    put_u8(h, 1);
    put_u8(h, 0);
    put_le(h, static_cast<std::uint16_t>(dict.size()));
    put_str(h, dict);
    return h;
}

class NpyWriter final : public TrajectoryWriter {
public:
    NpyWriter(std::unique_ptr<OutputStream> out, std::size_t natoms)
        : out_(std::move(out)), natoms_(natoms), seekable_(out_->seekable()) {}

    ~NpyWriter() override {
        try {
            finish();
        } catch (...) {
        }
    }

    void write_batch(const FrameBatch& batch) override {
        if (finished_) throw Error(ErrorKind::InvalidInput, "npy writer is already finished");
        // The header holds a single atom count for the whole file.
        if (batch.variable())
            throw Error(ErrorKind::InvalidInput,
                        "npy holds one atom count per file and cannot store frames whose atom "
                        "count varies; write xtc, extxyz, xyz, pdb or a LAMMPS dump instead");
        if (started_ && batch.natoms != natoms_)
            throw Error(ErrorKind::InvalidInput, "npy holds one atom count per file: this batch has " +
                                                     std::to_string(batch.natoms) + " atoms per frame, the file " +
                                                     std::to_string(natoms_));
        // npy stores nm, so a batch in LAMMPS native units is converted first.
        std::optional<FrameBatch> converted;
        const FrameBatch* b = &batch;
        if (batch.units) b = &converted.emplace(batch.in_internal_units());

        const std::vector<float>* pos = b->positions();
        if (!pos) throw Error(ErrorKind::InvalidInput, "no positions column");
        if (!started_ && natoms_ == 0) natoms_ = b->natoms;
        if (b->natoms != natoms_ || pos->size() != b->nframes * natoms_ * 3)
            throw Error(ErrorKind::InvalidInput,
                        "npy writer: batch does not hold " + std::to_string(natoms_) + " atoms per frame");
        if (!started_) {
            if (seekable_) {
                // Reserve room for any frame count. finish() patches it.
                Bytes h = npy_header(SIZE_MAX, natoms_);
                reserved_ = h.size();
                out_->write(h.data(), h.size());
            }
            started_ = true;
        }
        if (seekable_)
            out_->write(pos->data(), pos->size() * 4);
        else
            buf_.insert(buf_.end(), pos->begin(), pos->end());
        nframes_ += b->nframes;
    }

    void finish() override {
        if (finished_) return;
        finished_ = true;
        if (seekable_ && started_) {
            Bytes h = npy_header(nframes_, natoms_, reserved_);
            out_->seek(0);
            out_->write(h.data(), h.size());
        } else {
            Bytes h = npy_header(nframes_, natoms_);
            out_->write(h.data(), h.size());
            if (!buf_.empty()) out_->write(buf_.data(), buf_.size() * 4);
        }
        out_->close();
    }

private:
    std::unique_ptr<OutputStream> out_;
    std::size_t natoms_;
    bool seekable_;
    bool started_ = false;
    bool finished_ = false;
    std::size_t nframes_ = 0;
    std::size_t reserved_ = 0;
    // All frames, buffered when the stream cannot seek back to the header.
    std::vector<float> buf_;
};

}  // namespace

std::unique_ptr<TrajectoryReader> make_npy_reader(std::unique_ptr<InputStream> in, const FormatOptions&) {
    return std::make_unique<NpyReader>(std::move(in));
}

std::unique_ptr<TrajectoryWriter> make_npy_writer(std::unique_ptr<OutputStream> out, std::size_t natoms,
                                                  const FormatOptions&) {
    return std::make_unique<NpyWriter>(std::move(out), natoms);
}

}  // namespace mdz::detail
