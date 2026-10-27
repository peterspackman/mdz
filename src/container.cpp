#include "container.hpp"

#include <cmath>
#include <cstring>
#include <string>

#include "crc32.hpp"
#include "zstd.hpp"
#include "variable.hpp"

namespace mdz::detail {
namespace {

constexpr char kMagic[4] = {'M', 'D', 'Z', 4};
// Decompressed metadata is capped, since its zstd frame comes from the file.
constexpr std::size_t kMaxMetadataBytes = std::size_t(1) << 30;
constexpr std::uint16_t kVersion = 4;
// Header flags. A reader refuses a file with any flag it does not know.
constexpr std::uint16_t kFlagVariableAtoms = 1;
constexpr std::uint16_t kKnownFlags = kFlagVariableAtoms;
constexpr std::size_t kHeaderSize = 24;
constexpr char kRecordMagic[4] = {'G', 'O', 'F', 1};
constexpr char kIndexMagic[4] = {'I', 'D', 'X', 1};
constexpr char kTrailerMagic[8] = {'M', 'D', 'Z', '4', 'E', 'N', 'D', 0};
constexpr std::size_t kTrailerSize = 16;
constexpr std::size_t kCompressMetadataFrom = 64;
constexpr std::uint8_t kZstdMagic[4] = {0x28, 0xB5, 0x2F, 0xFD};

[[noreturn]] void too_big(const std::string& what) {
    throw Error(ErrorKind::InvalidInput, what + " does not fit the .mdz v4 format");
}

bool has_magic(ByteSpan b, std::size_t off, const char* m, std::size_t n) {
    return off <= b.size() && b.size() - off >= n && std::memcmp(b.data() + off, m, n) == 0;
}

Bytes encode_schema(const Schema& s) {
    Bytes out;
    put_le<std::uint32_t>(out, s.gof_size);
    if (s.columns.size() > 0xFFFF) too_big("column count");
    put_le<std::uint16_t>(out, static_cast<std::uint16_t>(s.columns.size()));
    for (const auto& d : s.columns) {
        if (d.name.size() > 255) too_big("column name");
        if (d.unit.size() > 255) too_big("column unit");
        if (d.width > 255) too_big("column width");
        put_u8(out, static_cast<std::uint8_t>(d.name.size()));
        put_str(out, d.name);
        put_u8(out, static_cast<std::uint8_t>(d.dtype));
        put_u8(out, static_cast<std::uint8_t>(d.width));
        put_u8(out, d.per_frame ? 1 : 0);
        put_u8(out, d.lossy ? 1 : 0);
        put_le<float>(out, d.precision);
        put_u8(out, d.pred_order);
        put_u8(out, d.codec_flags);
        put_u8(out, d.anchor_step);
        put_u8(out, static_cast<std::uint8_t>(d.unit.size()));
        put_str(out, d.unit);
    }
    // Metadata of kCompressMetadataFrom bytes or more is stored as a zstd
    // frame. An uncompressed block starts with its version byte, which never
    // matches the zstd magic, so the reader can tell the two apart.
    Bytes packed;
    if (s.metadata && s.metadata->size() >= kCompressMetadataFrom) packed = zstd_compress(*s.metadata, 9);
    const Bytes* meta = packed.empty() ? (s.metadata ? &*s.metadata : nullptr) : &packed;
    const std::size_t mlen = meta ? meta->size() : 0;
    if (mlen > 0xFFFFFFFFu) too_big("metadata");
    put_le<std::uint32_t>(out, static_cast<std::uint32_t>(mlen));
    if (mlen) put_bytes(out, meta->data(), mlen);
    return out;
}

Schema decode_schema(std::size_t natoms, ByteSpan bytes) {
    ByteCursor c(bytes);
    Schema s;
    s.natoms = natoms;
    s.gof_size = c.u32();
    const std::size_t ncols = c.u16();
    s.columns.reserve(std::min(ncols, bytes.size()));
    for (std::size_t i = 0; i < ncols; ++i) {
        ColumnDesc d;
        d.name = std::string(c.str(c.u8()));
        const std::uint8_t dt = c.u8();
        if (!dtype_from_u8(dt, d.dtype)) corrupt("unknown column dtype byte " + std::to_string(dt));
        d.width = c.u8();
        d.per_frame = c.u8() != 0;
        d.lossy = c.u8() != 0;
        d.precision = c.f32();
        d.pred_order = c.u8();
        d.codec_flags = c.u8();
        d.anchor_step = c.u8();
        d.unit = std::string(c.str(c.u8()));
        s.columns.push_back(std::move(d));
    }
    const std::size_t mlen = c.u32();
    ByteSpan meta = c.take(mlen);
    if (c.remaining() != 0) corrupt("schema has trailing bytes");
    if (mlen >= 4 && std::memcmp(meta.data(), kZstdMagic, 4) == 0) {
        Bytes raw;
        zstd_decompress_bounded(meta, kMaxMetadataBytes, raw);
        s.metadata = std::move(raw);
    } else if (mlen) {
        s.metadata = Bytes(meta.begin(), meta.end());
    }
    return s;
}

struct ParsedRecord {
    std::size_t total;    // bytes the record occupies
    std::size_t nframes;
    ByteSpan body;
    std::uint32_t crc;
    std::vector<ByteSpan> blobs;
};

// Parse the record at pos, which must end by limit. Does not check the CRC.
ParsedRecord parse_record(ByteSpan bytes, std::size_t pos, std::size_t limit, std::size_t ncols,
                          std::size_t expected_first) {
    ByteCursor c(bytes.first(limit), pos);
    if (std::memcmp(c.take(4).data(), kRecordMagic, 4) != 0) corrupt("no group record at byte " + std::to_string(pos));
    const std::size_t body_len = c.u32();
    ParsedRecord r;
    r.body = c.take(body_len);
    r.crc = c.u32();
    r.total = c.offset() - pos;
    ByteCursor b(r.body);
    const std::uint64_t first = b.u64();
    if (first != expected_first)
        corrupt("group record at byte " + std::to_string(pos) + " starts at frame " + std::to_string(first) +
                ", expected " + std::to_string(expected_first));
    r.nframes = b.u32();
    if (r.nframes == 0) corrupt("group record at byte " + std::to_string(pos) + " holds no frames");
    r.blobs.reserve(ncols);
    for (std::size_t i = 0; i < ncols; ++i) {
        const std::size_t len = b.u32();
        r.blobs.push_back(b.take(len));
    }
    if (b.remaining() != 0) corrupt("group record at byte " + std::to_string(pos) + " has trailing bytes");
    return r;
}

// Read the footer's offset and index, if the file ends in a valid footer and
// trailer. Returns false for any inconsistency, and the file is then scanned
// as unclosed.
bool read_footer(ByteSpan bytes, std::size_t data_start, std::size_t& offset, std::vector<IndexEntry>& entries) {
    if (bytes.size() < kTrailerSize) return false;
    const std::size_t tail = bytes.size() - kTrailerSize;
    if (tail < data_start || std::memcmp(bytes.data() + tail + 8, kTrailerMagic, 8) != 0) return false;
    const std::uint64_t off = load_le<std::uint64_t>(bytes.data() + tail);
    if (off < data_start || off > tail) return false;
    try {
        ByteCursor c(bytes.first(tail), static_cast<std::size_t>(off));
        if (std::memcmp(c.take(4).data(), kIndexMagic, 4) != 0) return false;
        const std::uint64_t body_len = c.u64();
        if (body_len > c.remaining()) return false;
        ByteSpan body = c.take(static_cast<std::size_t>(body_len));
        if (crc32(body.data(), body.size()) != c.u32() || c.remaining() != 0) return false;
        ByteCursor b(body);
        const std::uint64_t n = b.u64();
        if (n > b.remaining() / 20 || n * 20 != b.remaining()) return false;
        entries.resize(static_cast<std::size_t>(n));
        for (auto& e : entries) {
            e.offset = b.u64();
            e.first_frame = b.u64();
            e.nframes = b.u32();
        }
    } catch (const Error&) {
        return false;
    }
    offset = static_cast<std::size_t>(off);
    return true;
}

void validate_layout(const CompressedTrajectory& ct) {
    const CompressedColumn* pos = ct.column("positions");
    if (!pos) corrupt("no positions column");
    if (pos->desc.dtype != Dtype::F32 || pos->desc.width != 3 || pos->desc.per_frame)
        corrupt("positions column has the wrong shape");
    for (const auto& col : ct.columns) {
        const ColumnDesc& d = col.desc;
        if (d.width == 0) corrupt("column '" + d.name + "' has width 0");
        const std::size_t rows = d.per_frame ? 1 : ct.natoms;
        if (sat_mul(sat_mul(sat_mul(rows, d.width), ct.nframes), 8) == SIZE_MAX)
            corrupt("column '" + d.name + "' is too large to address");
        if (d.lossy && !(std::isfinite(d.precision) && d.precision > 0.0f))
            corrupt("lossy column '" + d.name + "' has precision " + std::to_string(d.precision));
    }
    if (ct.variable_atoms) {
        const CompressedColumn* m = ct.column(kMembershipColumn);
        if (!m || m->desc.per_frame || m->desc.width != 1 || m->desc.dtype != Dtype::I64)
            corrupt("a file with a varying atom count needs a membership column");
    }
}

}  // namespace

void GroupChecks::verify(std::size_t gi) {
    std::uint8_t s = state[gi].load(std::memory_order_acquire);
    if (s == 0) {
        s = crc32(bodies[gi].data(), bodies[gi].size()) == crcs[gi] ? 1 : 2;
        state[gi].store(s, std::memory_order_release);
    }
    if (s == 2) corrupt("group record " + std::to_string(gi) + " fails its checksum");
}

Schema Schema::of(const CompressedTrajectory& ct) {
    Schema s;
    s.natoms = ct.natoms;
    s.gof_size = ct.gof_size;
    for (const auto& c : ct.columns) s.columns.push_back(c.desc);
    s.metadata = ct.metadata;
    if (s.metadata && s.metadata->empty()) s.metadata.reset();
    s.variable_atoms = ct.variable_atoms;
    return s;
}

Bytes encode_header(const Schema& s) {
    Bytes body = encode_schema(s);
    Bytes out;
    out.reserve(kHeaderSize + body.size());
    put_str(out, std::string_view(kMagic, 4));
    put_le<std::uint16_t>(out, kVersion);
    put_le<std::uint16_t>(out, s.variable_atoms ? kFlagVariableAtoms : 0);
    put_le<std::uint64_t>(out, s.natoms);
    if (body.size() > 0xFFFFFFFFu) too_big("schema");
    put_le<std::uint32_t>(out, static_cast<std::uint32_t>(body.size()));
    put_le<std::uint32_t>(out, crc32(body.data(), body.size()));
    put_bytes(out, body.data(), body.size());
    return out;
}

void encode_record(std::uint64_t first_frame, std::size_t nframes, const std::vector<ByteSpan>& blobs, Bytes& out) {
    if (nframes > 0xFFFFFFFFu) too_big("group frame count");
    std::size_t body_len = 12;
    for (const auto& b : blobs) body_len += 4 + b.size();
    if (body_len > 0xFFFFFFFFu) too_big("group record (over 4 GiB; use smaller groups)");
    const std::size_t start = out.size();
    out.reserve(start + body_len + 12);
    put_str(out, std::string_view(kRecordMagic, 4));
    put_le<std::uint32_t>(out, static_cast<std::uint32_t>(body_len));
    const std::size_t body_at = out.size();
    put_le<std::uint64_t>(out, first_frame);
    put_le<std::uint32_t>(out, static_cast<std::uint32_t>(nframes));
    for (const auto& b : blobs) {
        put_le<std::uint32_t>(out, static_cast<std::uint32_t>(b.size()));
        put_bytes(out, b);
    }
    put_le<std::uint32_t>(out, crc32(out.data() + body_at, body_len));
}

void encode_footer(std::uint64_t offset, const std::vector<IndexEntry>& index, Bytes& out) {
    Bytes body;
    body.reserve(8 + index.size() * 20);
    put_le<std::uint64_t>(body, index.size());
    for (const auto& e : index) {
        put_le<std::uint64_t>(body, e.offset);
        put_le<std::uint64_t>(body, e.first_frame);
        put_le<std::uint32_t>(body, e.nframes);
    }
    put_str(out, std::string_view(kIndexMagic, 4));
    put_le<std::uint64_t>(out, body.size());
    put_bytes(out, body.data(), body.size());
    put_le<std::uint32_t>(out, crc32(body.data(), body.size()));
    put_le<std::uint64_t>(out, offset);
    put_str(out, std::string_view(kTrailerMagic, 8));
}

Loaded load_container(std::shared_ptr<const void> keep, ByteSpan bytes) {
    if (!has_magic(bytes, 0, kMagic, 4)) {
        if (bytes.size() >= 4 && std::memcmp(bytes.data(), "MDZ", 3) == 0)
            throw Error(ErrorKind::Corrupt, ".mdz container version " + std::to_string(bytes[3]) +
                                                " is not supported (this build reads version 4)");
        corrupt("not an .mdz file");
    }
    ByteCursor c(bytes, 4);
    const std::uint16_t version = c.u16();
    if (version != kVersion)
        throw Error(ErrorKind::Corrupt, ".mdz container version " + std::to_string(version) +
                                            " is not supported (this build reads version 4)");
    const std::uint16_t flags = c.u16();
    if (flags & ~kKnownFlags)
        throw Error(ErrorKind::Corrupt, "file uses header flags " + std::to_string(flags) +
                                            " that this build does not understand");
    const std::uint64_t natoms64 = c.u64();
    if (natoms64 > SIZE_MAX / 64) corrupt("atom count too large");
    const std::size_t schema_len = c.u32();
    const std::uint32_t schema_crc = c.u32();
    ByteSpan schema_bytes = c.take(schema_len);
    if (crc32(schema_bytes.data(), schema_bytes.size()) != schema_crc) corrupt("schema fails its checksum");
    Schema schema = decode_schema(static_cast<std::size_t>(natoms64), schema_bytes);
    const std::size_t ncols = schema.columns.size();
    const std::size_t data_start = c.offset();

    std::size_t footer_at = 0;
    std::vector<IndexEntry> footer;
    const bool complete = read_footer(bytes, data_start, footer_at, footer);

    Loaded L;
    GofTable gofs;
    std::vector<std::vector<ByteSpan>> blobs(ncols);
    std::shared_ptr<GroupChecks> checks;
    std::size_t pos = data_start;

    if (complete) {
        // Take record positions from the footer. Each record's framing is
        // checked now, and its CRC the first time the group is decoded.
        checks = std::make_shared<GroupChecks>(footer.size());
        checks->bodies.reserve(footer.size());
        checks->crcs.reserve(footer.size());
        for (const auto& e : footer) {
            if (e.offset != pos) corrupt("footer index disagrees with the records");
            ParsedRecord r = parse_record(bytes, pos, footer_at, ncols, gofs.nframes());
            if (e.first_frame != gofs.nframes() || e.nframes != r.nframes)
                corrupt("footer index disagrees with the records");
            gofs.push(r.nframes);
            for (std::size_t k = 0; k < ncols; ++k) blobs[k].push_back(r.blobs[k]);
            checks->bodies.push_back(r.body);
            checks->crcs.push_back(r.crc);
            L.index.push_back(e);
            pos += r.total;
        }
        if (pos != footer_at) corrupt("footer index disagrees with the records");
    } else {
        // Unclosed file: scan the records, stopping at the first one that is
        // malformed or fails its CRC.
        while (pos < bytes.size()) {
            ParsedRecord r;
            try {
                r = parse_record(bytes, pos, bytes.size(), ncols, gofs.nframes());
            } catch (const Error&) {
                break;
            }
            if (crc32(r.body.data(), r.body.size()) != r.crc) break;
            L.index.push_back({pos, gofs.nframes(), static_cast<std::uint32_t>(r.nframes)});
            if (!gofs.push(r.nframes)) corrupt("frame count overflows");
            for (std::size_t k = 0; k < ncols; ++k) blobs[k].push_back(r.blobs[k]);
            pos += r.total;
        }
    }

    L.status.complete = complete;
    L.status.data_end = pos;
    L.status.trailing_bytes = complete ? 0 : bytes.size() - pos;

    CompressedTrajectory& ct = L.ct;
    ct.natoms = schema.natoms;
    ct.nframes = gofs.nframes();
    ct.gof_size = schema.gof_size;
    ct.gofs = std::move(gofs);
    ct.metadata = std::move(schema.metadata);
    ct.variable_atoms = flags & kFlagVariableAtoms;
    if (ct.variable_atoms) ct.atom_counts = new_atom_counts();
    ct.columns.reserve(ncols);
    for (std::size_t k = 0; k < ncols; ++k)
        ct.columns.emplace_back(std::move(schema.columns[k]), keep, std::move(blobs[k]), checks);
    validate_layout(ct);
    return L;
}

void check_same_schema(const Schema& file, const Schema& chunk) {
    auto names = [](const Schema& s) {
        std::string out = "[";
        for (std::size_t i = 0; i < s.columns.size(); ++i) out += (i ? ", \"" : "\"") + s.columns[i].name + "\"";
        return out + "]";
    };
    bool same_names = file.columns.size() == chunk.columns.size();
    for (std::size_t i = 0; same_names && i < file.columns.size(); ++i)
        same_names = file.columns[i].name == chunk.columns[i].name;
    if (!same_names)
        throw Error(ErrorKind::InvalidInput, "frames do not match the file's columns: file has " + names(file) +
                                                 ", these frames " + names(chunk));
    for (std::size_t i = 0; i < file.columns.size(); ++i) {
        const ColumnDesc &f = file.columns[i], &c = chunk.columns[i];
        if (f != c)
            throw Error(ErrorKind::InvalidInput,
                        "column '" + f.name + "' is stored with different settings than the file uses (precision " +
                            std::to_string(f.precision) + " vs " + std::to_string(c.precision) + ", codec flags " +
                            std::to_string(f.codec_flags) + " vs " + std::to_string(c.codec_flags) +
                            "); append with the options the file was created with");
    }
    if (file.variable_atoms != chunk.variable_atoms)
        throw Error(ErrorKind::InvalidInput,
                    file.variable_atoms ? "the file's atom count varies; these frames were encoded with a fixed one"
                                        : "the file has a fixed atom count; frames whose count varies need "
                                          "CompressOptions::variable_atoms set when the file is created");
    if (file.natoms != chunk.natoms)
        throw Error(ErrorKind::InvalidInput, "frames have " + std::to_string(chunk.natoms) + " atoms; the file has " +
                                                 std::to_string(file.natoms));
    if (file.metadata != chunk.metadata)
        throw Error(ErrorKind::InvalidInput, "static metadata (atom names, types, ids, ...) changed between frames");
}

}  // namespace mdz::detail
