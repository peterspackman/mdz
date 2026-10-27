#pragma once
// Random access to a .mdz file. Reading a range of frames decompresses only
// the groups of frames that overlap it.

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "mdz/trajectory.hpp"

namespace mdz {

namespace detail {
struct GroupCache;
}

// Const member functions may be called from several threads at once.
class Reader {
public:
    using Box32 = std::array<float, 9>;

    // Open a file (memory-mapped where possible) or an in-memory copy of one.
    // A file whose writer never closed it opens with every complete group;
    // see status(). Throws Error(Io) if the file cannot be read and
    // Error(Corrupt) if it is not a valid .mdz file.
    static Reader open(const std::string& path);
    static Reader from_bytes(std::vector<std::uint8_t> bytes);

    // Atoms per frame, or 0 if the count varies (see frame_natoms).
    std::size_t natoms() const noexcept { return ct_.natoms; }
    bool variable_atoms() const noexcept { return ct_.variable_atoms; }
    // The atom count of each frame in [first, first + count), with the range
    // clamped to the end of the file.
    std::vector<std::size_t> frame_natoms(std::size_t first = 0, std::size_t count = kAllFrames) const;
    // The ids of the atoms in `frame`, in the order their positions are
    // returned. For a fixed atom count these are the atom_ids metadata, or
    // 0..natoms-1 if the file has none. Throws Error(OutOfRange).
    std::vector<std::int32_t> atom_ids(std::size_t frame) const;
    std::size_t nframes() const noexcept { return ct_.nframes; }
    // The quantization step of positions in nm, or 0 if they are exact.
    float precision() const noexcept { return precision_; }
    std::uint32_t gof_size() const noexcept { return ct_.gof_size; }
    std::size_t n_gofs() const noexcept { return ct_.gofs.size(); }
    const GofTable& gofs() const noexcept { return ct_.gofs; }
    const LoadStatus& status() const noexcept { return status_; }
    // Whether the file was closed cleanly by its writer.
    bool complete() const noexcept { return status_.complete; }
    const CompressedTrajectory& trajectory() const noexcept { return ct_; }
    // Throws Error(Corrupt) if the metadata block is malformed.
    Metadata metadata() const { return ct_.decoded_metadata(); }
    // Column names in file order, with "positions" first.
    std::vector<std::string> column_names() const;

    // Write the positions of frames [first, first + count) to `out`, in nm.
    // `out` must hold 3 floats per atom of every frame in the range, which
    // is count * natoms() * 3 for a fixed atom count. Throws
    // Error(OutOfRange) if the range extends past the end.
    void read_positions(std::size_t first, std::size_t count, float* out) const;
    // As above, but the range is clamped to the end of the file.
    std::vector<float> read_positions(std::size_t first, std::size_t count) const;
    std::vector<float> read_frame(std::size_t frame) const { return read_positions(frame, 1); }

    // Per-frame boxes as row-major 3x3 matrices in nm, if the file has them.
    const std::optional<std::vector<Box32>>& boxes() const noexcept { return boxes_; }

    // Every column and the static metadata for frames [first, first + count),
    // in stored units. The range is clamped to the end of the file.
    FrameBatch read_batch(std::size_t first, std::size_t count) const;
    // One column for frames [first, first + count), in stored units. The range
    // is clamped to the end of the file. Throws Error(InvalidInput) if there is
    // no column named `name`.
    DecodedColumn read_column(std::string_view name, std::size_t first = 0, std::size_t count = kAllFrames) const;

private:
    explicit Reader(CompressedTrajectory ct, LoadStatus status);

    CompressedTrajectory ct_;
    LoadStatus status_;
    float precision_ = 0.0f;
    double length_to_nm_ = 1.0;
    std::optional<std::vector<Box32>> boxes_;
    // The last group read in part, so that reading a group one frame at a
    // time does not decode it once per frame.
    std::shared_ptr<detail::GroupCache> cache_;
};

}  // namespace mdz
