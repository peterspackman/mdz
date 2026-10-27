#pragma once
// The in-memory form of a compressed trajectory: column descriptors, one
// compressed blob per column per group of frames, and the metadata block.
// This header also declares the functions that compress frames into one,
// decode frames from one, and save and load .mdz files.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "mdz/column.hpp"
#include "mdz/frame_batch.hpp"
#include "mdz/metadata.hpp"
#include "mdz/options.hpp"
#include "mdz/span.hpp"

namespace mdz {

namespace detail {
struct GroupChecks;  // record CRCs of a loaded file, verified on first use
}

class CompressedColumn {
public:
    ColumnDesc desc;

    // A column that owns its blobs, one per group.
    CompressedColumn(ColumnDesc d, std::vector<std::vector<std::uint8_t>> blobs);
    // A column whose blobs point into `backing`, typically a loaded file,
    // which the column keeps alive.
    CompressedColumn(ColumnDesc d, std::shared_ptr<const void> backing, std::vector<ByteSpan> blobs,
                     std::shared_ptr<detail::GroupChecks> checks);

    std::size_t n_gofs() const noexcept { return blobs_.size(); }
    // The blob of group gi, valid while this column exists. For a loaded file,
    // the group's record CRC is checked the first time it is requested.
    // Throws Error(Corrupt) if the check fails.
    ByteSpan blob(std::size_t gi) const;
    // Total bytes of all blobs.
    std::size_t compressed_size() const noexcept;

private:
    std::shared_ptr<const void> backing_;
    std::vector<ByteSpan> blobs_;
    std::shared_ptr<detail::GroupChecks> checks_;
};

namespace detail {
struct AtomCounts;  // per-frame atom counts of a file whose count varies
struct RigidModel;  // rigid bodies, for groups coded with them
}

// In files whose atom count varies, the column recording for each group
// which atoms (by id) are present in which frames.
inline constexpr const char* kMembershipColumn = "mdz.atoms";

struct CompressedTrajectory {
    // Atoms per frame, or 0 when the count varies (see variable_atoms).
    std::size_t natoms = 0;
    std::size_t nframes = 0;
    // Nominal frames per group. The actual group boundaries are in `gofs`.
    std::uint32_t gof_size = 0;
    GofTable gofs;
    std::vector<CompressedColumn> columns;
    // The encoded Metadata block, if any.
    std::optional<std::vector<std::uint8_t>> metadata;
    // Whether the atom count varies between frames. If so, each group's
    // atoms are listed by id in its kMembershipColumn blob.
    bool variable_atoms = false;
    // Per-frame atom counts of a file whose count varies, computed on first
    // use.
    std::shared_ptr<detail::AtomCounts> atom_counts;
    // The rigid-body model built from the metadata's `bodies`, if any.
    std::shared_ptr<const detail::RigidModel> rigid;

    // The column named `name`, or nullptr.
    const CompressedColumn* column(std::string_view name) const noexcept;
    bool has_column(std::string_view name) const noexcept { return column(name) != nullptr; }
    // Compressed bytes of every blob plus the metadata block.
    std::size_t compressed_size() const noexcept;
    // Uncompressed bytes of every column, the reference for compression
    // ratios.
    std::size_t raw_size() const;
    // The atom count of each frame in [start, end), clamped to nframes.
    std::vector<std::size_t> frame_natoms(std::size_t start = 0, std::size_t end = SIZE_MAX) const;
    // The decoded metadata block, empty if there is none. Throws
    // Error(Corrupt) if it is malformed.
    Metadata decoded_metadata() const;
    // The factor that converts stored lengths to nm. This is 1 unless the
    // file stores a LAMMPS unit style.
    double length_to_nm() const;
};

// A decoded column. Only the vector matching `dtype` (F32, F64, I32 or I64)
// holds values.
struct DecodedColumn {
    Dtype dtype = Dtype::F32;
    std::vector<float> f32;
    std::vector<double> f64;
    std::vector<std::int32_t> i32;
    std::vector<std::int64_t> i64;
    // Number of values in the active vector.
    std::size_t size() const noexcept;
};

// Passed as a frame count or end frame to mean "through the last frame".
inline constexpr std::size_t kAllFrames = std::numeric_limits<std::size_t>::max();

// Compress a batch. The columns are written in this order: positions
// (required), velocities, forces, the other per-atom float columns by name,
// the per-atom int columns, box, origin, the frame properties by name, and
// the integer frame properties (as I64). Among the other float columns, the
// standard scalars (charge, occupancy, bfactor) are quantized at the default
// precision. Any other column is stored exactly unless column_precision gives
// it a precision. Throws Error(InvalidInput) for invalid options or data the
// format cannot hold.
CompressedTrajectory compress(const FrameBatch& batch, const CompressOptions& opts = {});

// Compress bare coordinates, nframes * natoms * 3 floats in nm, with optional
// per-frame boxes (nullptr for none). The result has the same columns that
// compress() would produce for the same data. Throws Error(InvalidInput).
CompressedTrajectory compress_positions(const float* positions, std::size_t natoms, std::size_t nframes,
                                        const std::vector<Box>* boxes, const CompressOptions& opts = {});

// Decode frames [start, end) of one column of `ct`, with the range clamped to
// nframes. Only the overlapping groups are decompressed. Values are in stored
// units. Throws Error(Corrupt) for damaged data.
DecodedColumn decode_column(const CompressedTrajectory& ct, const CompressedColumn& col, std::size_t start = 0,
                            std::size_t end = kAllFrames);

// Frames [start, end) with every column and the static metadata, with the
// range clamped to nframes. Values are in stored units. Throws
// Error(Corrupt) for damaged data.
FrameBatch decompress(const CompressedTrajectory& ct, std::size_t start = 0, std::size_t end = kAllFrames);

// The positions of every frame, converted to nm and concatenated frame by
// frame. Throws Error(Corrupt) for damaged data.
std::vector<float> decompress_positions(const CompressedTrajectory& ct);

// How much of a file could be loaded.
struct LoadStatus {
    // Whether the file ends in a valid footer, which means its writer closed
    // it cleanly.
    bool complete = true;
    // The byte offset just past the last valid record. Appending resumes here.
    std::uint64_t data_end = 0;
    // Bytes after the last valid record that do not form a footer, usually
    // left by an interrupted write. Always 0 for a complete file.
    std::uint64_t trailing_bytes = 0;
};

// Write `ct` as a complete .mdz file, creating or truncating `path`, and
// return the number of bytes written. Throws Error(Io), or
// Error(InvalidInput) if `ct` cannot be represented in the format.
std::uint64_t save(const std::string& path, const CompressedTrajectory& ct);
// The bytes save() would write.
std::vector<std::uint8_t> to_bytes(const CompressedTrajectory& ct);

// Load a .mdz file. The file is memory-mapped where possible, and the
// returned trajectory keeps the mapping alive. A file whose writer never
// closed it loads with every complete group, and `status`, if given, reports
// what was found. Throws Error(Io) if the file cannot be read and
// Error(Corrupt) if it is not a valid .mdz file.
CompressedTrajectory load(const std::string& path, LoadStatus* status = nullptr);
CompressedTrajectory load_bytes(std::vector<std::uint8_t> bytes, LoadStatus* status = nullptr);

}  // namespace mdz
