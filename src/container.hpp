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
