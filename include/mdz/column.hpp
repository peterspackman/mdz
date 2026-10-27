#pragma once
// Column descriptors and the group-of-frames table. A column is one named
// field of a trajectory, with its own element type, width and compression
// settings.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace mdz {

enum class Dtype : std::uint8_t { F32 = 0, F64 = 1, I32 = 2, U32 = 3, U8 = 4, I64 = 5 };

// Set `out` from its on-disk byte. Returns false for an unknown value.
bool dtype_from_u8(std::uint8_t v, Dtype& out) noexcept;
// Size of one element in bytes.
std::size_t dtype_size(Dtype d) noexcept;
// Short name, e.g. "f32".
std::string_view dtype_name(Dtype d) noexcept;

// Codec flag bits in ColumnDesc::codec_flags.
inline constexpr std::uint8_t kFlagAdaptive = 1;
inline constexpr std::uint8_t kFlagBFrames = 2;
inline constexpr std::uint8_t kFlagSpatialDelta = 4;

struct ColumnDesc {
    std::string name;
    Dtype dtype = Dtype::F32;
    // Values per atom, or per frame for per-frame columns.
    std::uint32_t width = 1;
    bool per_frame = false;
    bool lossy = false;
    // Quantization step in the column's unit; 0 = stored exactly.
    float precision = 0.0f;
    // Recorded for inspection only. Decoders do not rely on these.
    std::uint8_t pred_order = 0;
    std::uint8_t codec_flags = 0;  // kFlag* bits
    std::uint8_t anchor_step = 1;
    std::string unit;

    // Uncompressed size in bytes of this column over natoms atoms and nframes
    // frames.
    std::size_t raw_size(std::size_t natoms, std::size_t nframes) const noexcept;

    // Standard per-atom F32 vectors (width 3) in internal units. A precision
    // of 0 stores the values exactly.
    static ColumnDesc positions(float precision);
    static ColumnDesc velocities(float precision);
    static ColumnDesc forces(float precision);
    // Per-atom F32 scalar (width 1).
    static ColumnDesc scalar(std::string_view name, std::string_view unit, float precision);
    // Exact per-frame F64 scalar.
    static ColumnDesc frame_scalar(std::string_view name, std::string_view unit);
    // Exact per-frame F64 vector of `width` values (box, origin, ...).
    static ColumnDesc frame_vector(std::string_view name, std::uint32_t width, std::string_view unit);
    // Exact per-atom I32 column.
    static ColumnDesc int_column(std::string_view name);
};

bool operator==(const ColumnDesc& a, const ColumnDesc& b) noexcept;
inline bool operator!=(const ColumnDesc& a, const ColumnDesc& b) noexcept { return !(a == b); }

// The start and length of each group of frames (GoF). Groups are contiguous
// and together cover frames [0, nframes). A file compressed in one call has
// uniform groups, except possibly a shorter last one. A streamed file may also
// have a short group wherever its writer flushed, so frames must always be
// located through this table rather than by dividing by the group size.
class GofTable {
public:
    struct Span {
        std::size_t start;
        std::size_t end;
        std::size_t size() const noexcept { return end - start; }
    };

    GofTable() = default;
    // Groups of gof_size frames (treated as 1 if 0), the last one possibly
    // shorter.
    static GofTable uniform(std::size_t nframes, std::size_t gof_size);

    // Append a group of n frames. Returns false, and changes nothing, if n is
    // 0 or the total frame count would overflow.
    bool push(std::size_t n);

    std::size_t size() const noexcept { return starts_.size(); }
    bool empty() const noexcept { return starts_.empty(); }
    std::size_t nframes() const noexcept { return nframes_; }
    // Frames [start, end) of group i. Requires i < size().
    Span span(std::size_t i) const noexcept { return {starts_[i], i + 1 < starts_.size() ? starts_[i + 1] : nframes_}; }
    // Indices [start, end) of the groups overlapping frames [start, end). The
    // frame range is clamped to nframes(). Returns {0, 0} if it is empty.
    Span overlapping(std::size_t start, std::size_t end) const noexcept;
    // Frames in the largest group.
    std::size_t max_len() const noexcept;

    friend bool operator==(const GofTable& a, const GofTable& b) noexcept {
        return a.nframes_ == b.nframes_ && a.starts_ == b.starts_;
    }

private:
    std::vector<std::size_t> starts_;
    std::size_t nframes_ = 0;
};

}  // namespace mdz
