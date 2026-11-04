#pragma once
// The .mdz v4 container. A file is a header and schema, followed by one
// self-contained record per group of frames. A closed file ends with a footer
// index and a trailer. docs/FORMAT.md gives the byte layout.

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "bytes.hpp"
#include "mdz/trajectory.hpp"

namespace mdz::detail {

struct Schema {
    std::size_t natoms = 0;
    std::uint32_t gof_size = 0;
    std::vector<ColumnDesc> columns;
    std::optional<Bytes> metadata;
    bool variable_atoms = false;

    static Schema of(const CompressedTrajectory& ct);
};

struct IndexEntry {
    std::uint64_t offset;
    std::uint64_t first_frame;
    std::uint32_t nframes;
    friend bool operator==(const IndexEntry& a, const IndexEntry& b) noexcept {
        return a.offset == b.offset && a.first_frame == b.first_frame && a.nframes == b.nframes;
    }
};

// Record CRCs of a closed file. Each group's CRC is checked the first time
// the group is used, so opening a file does not read every record. verify()
// may be called from several threads at once.
struct GroupChecks {
    std::vector<ByteSpan> bodies;
    std::vector<std::uint32_t> crcs;
    std::unique_ptr<std::atomic<std::uint8_t>[]> state;  // 0 unchecked, 1 good, 2 bad

    explicit GroupChecks(std::size_t n) : state(new std::atomic<std::uint8_t>[n]) {
        for (std::size_t i = 0; i < n; ++i) state[i].store(0, std::memory_order_relaxed);
    }
    void verify(std::size_t gi);
};

inline constexpr std::size_t kHeaderSize = 24;
inline constexpr std::size_t kTrailerSize = 16;

// The fixed-size header at the start of a file.
struct Header {
    std::size_t natoms = 0;
    bool variable_atoms = false;
    std::size_t schema_len = 0;
    std::uint32_t schema_crc = 0;
};

// Parse the first kHeaderSize bytes. Throws Error(Corrupt).
Header decode_header(ByteSpan bytes);
// Parse the schema that follows the header, checking its CRC. Throws
// Error(Corrupt).
Schema decode_schema(const Header& h, ByteSpan bytes);
// The footer offset stored in `trailer`, the last kTrailerSize bytes of a
// file, or nullopt if they are not a trailer.
std::optional<std::uint64_t> decode_trailer(ByteSpan trailer);
// Parse a footer, the bytes from its magic up to the trailer. Returns nullopt
// if it is malformed or fails its CRC.
std::optional<std::vector<IndexEntry>> decode_footer(ByteSpan footer);

struct ParsedRecord {
    std::size_t total;  // bytes the record occupies
    std::size_t nframes;
    ByteSpan body;
    std::uint32_t crc;
    std::vector<ByteSpan> blobs;
};

// Parse the record at `pos` in `bytes`, which must end by `limit` and start
// at frame `expected_first`. Does not check the CRC. Throws Error(Corrupt).
ParsedRecord parse_record(ByteSpan bytes, std::size_t pos, std::size_t limit, std::size_t ncols,
                          std::size_t expected_first);

// A trajectory over the given groups, whose blobs `keep` keeps alive. Throws
// Error(Corrupt) if the columns are not a valid layout. A subset of a file's
// columns may leave out positions.
CompressedTrajectory assemble(Schema schema, GofTable gofs, std::vector<std::vector<ByteSpan>> blobs,
                              std::shared_ptr<const void> keep, std::shared_ptr<GroupChecks> checks,
                              bool require_positions = true);

// Build the rigid-body model from the metadata's bodies, needed to decode
// groups coded with it. Throws Error(Corrupt).
void attach_rigid_model(CompressedTrajectory& ct);

// The header and schema bytes. Throws Error(InvalidInput) if a field does
// not fit the format.
Bytes encode_header(const Schema& s);
// Append one group record to `out`. Throws Error(InvalidInput) if the record
// does not fit the format, as when it exceeds 4 GiB.
void encode_record(std::uint64_t first_frame, std::size_t nframes, const std::vector<ByteSpan>& blobs, Bytes& out);
// Append the footer index and trailer to `out`. `offset` is the file offset
// at which the footer starts.
void encode_footer(std::uint64_t offset, const std::vector<IndexEntry>& index, Bytes& out);

struct Loaded {
    CompressedTrajectory ct;
    LoadStatus status;
    std::vector<IndexEntry> index;
};

// Parse a whole file held in `bytes`, which `keep` keeps alive for the
// returned columns. The records of a closed file must all be well formed. An
// unclosed file yields every group before the first damaged record. Throws
// Error(Corrupt).
Loaded load_container(std::shared_ptr<const void> keep, ByteSpan bytes);

// Throws Error(InvalidInput) if frames for `chunk` cannot be appended to a
// file with schema `file`.
void check_same_schema(const Schema& file, const Schema& chunk);

}  // namespace mdz::detail
