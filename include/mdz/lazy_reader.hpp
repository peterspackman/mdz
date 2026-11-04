#pragma once
// Reading a .mdz file a piece at a time, for callers that fetch byte ranges
// themselves (HTTP range requests, browser Blobs). Each step names the bytes
// it needs; the caller reads them and passes them in. Nothing here does I/O.
//
//   head = read(0, kHeaderSize); head = read(0, head_size(head));
//   r = LazyReader::open(head, file_size);
//   if (auto f = r.footer_range(read(file_size - kTrailerSize, file_size)))
//       closed = r.read_footer(f->first, read(f->first, f->second));
//   if (!closed)  // scan the records of an unclosed file
//       while (n = r.record_size(read(r.scan_offset(), +kRecordHeadSize)))
//           if (!r.add_record(read(r.scan_offset(), +n))) break;
//   batch = r.decode_group(gi, read(g.offset, g.offset + g.size));

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "mdz/column.hpp"
#include "mdz/frame_batch.hpp"
#include "mdz/metadata.hpp"
#include "mdz/span.hpp"

namespace mdz {

class LazyReader {
public:
    static constexpr std::size_t kHeaderSize = 24;
    static constexpr std::size_t kTrailerSize = 16;
    static constexpr std::size_t kRecordHeadSize = 8;

    // A group record: `size` bytes at `offset`, holding frames
    // [first_frame, first_frame + nframes).
    struct Group {
        std::uint64_t offset = 0;
        std::uint64_t size = 0;
        std::uint64_t first_frame = 0;
        std::size_t nframes = 0;
    };

    // The length of the header and schema, from the first kHeaderSize bytes.
    // Throws Error(Corrupt).
    static std::uint64_t head_size(ByteSpan header);
    // Parse the first head_size() bytes of a file of `file_size` bytes.
    // Throws Error(Corrupt).
    static LazyReader open(ByteSpan head, std::uint64_t file_size);

    LazyReader(LazyReader&&) noexcept;
    LazyReader& operator=(LazyReader&&) noexcept;
    ~LazyReader();

    // The byte range [first, second) of the footer, given the file's last
    // kTrailerSize bytes, or nullopt if the file was not closed.
    std::optional<std::pair<std::uint64_t, std::uint64_t>> footer_range(ByteSpan trailer) const;
    // Take the groups from the footer at `offset`. Returns false, and changes
    // nothing, if the footer is not valid; scan the records instead.
    bool read_footer(std::uint64_t offset, ByteSpan footer);

    // Scanning an unclosed file. scan_offset() is where the next record
    // would start. record_size() gives its length from its first
    // kRecordHeadSize bytes, or 0 if no record starts there. add_record()
    // takes the whole record and returns false if it is damaged, which ends
    // the scan.
    std::uint64_t scan_offset() const noexcept;
    std::uint64_t record_size(ByteSpan record_head) const;
    bool add_record(ByteSpan record);

    // Atoms per frame, or 0 if the count varies.
    std::size_t natoms() const noexcept;
    std::size_t nframes() const noexcept;
    bool variable_atoms() const noexcept;
    // Whether the groups came from the footer of a closed file.
    bool complete() const noexcept;
    std::uint32_t gof_size() const noexcept;
    const std::vector<ColumnDesc>& columns() const noexcept;
    // Throws Error(Corrupt) if the metadata block is malformed.
    Metadata metadata() const;
    // The factor that converts stored lengths to nm.
    double length_to_nm() const noexcept;
    // The quantization step of positions in nm, or 0 if they are exact.
    float precision() const noexcept;

    const std::vector<Group>& groups() const noexcept;
    // The group holding `frame`. Throws Error(OutOfRange).
    std::size_t group_of(std::size_t frame) const;

    // Decode group `gi` from its record bytes. Values are in stored units.
    // `columns` limits decoding to the columns named, plus positions; empty
    // means every column. Throws Error(Corrupt) for a damaged record and
    // Error(OutOfRange) for a bad index.
    FrameBatch decode_group(std::size_t gi, ByteSpan record, const std::vector<std::string>& columns = {}) const;

    struct Impl;

private:
    explicit LazyReader(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace mdz
