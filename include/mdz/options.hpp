#pragma once
// Compression settings.

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>

namespace mdz {

enum class CodecMode : std::uint8_t {
    // Polynomial prediction from earlier frames, with adaptive quantization.
    // Best for strongly correlated frames, as from frequent saves.
    Temporal = 0,
    // Each frame coded independently, in the style of XTC. Best for weakly
    // correlated frames.
    Intra = 1,
    // Bit-exact coding of prediction residuals over the float bit patterns.
    Lossless = 2,
    // Atom positions predicted from rigid-body motion. This mode is reported
    // for groups coded that way but cannot be requested here; set
    // CompressOptions::rigid instead.
    Rigid = 3,
    // Choose temporal or intra for each group by trial encoding.
    Auto = 255,
};

// Parse "auto", "temporal", "intra" (or "intraframe" or "spatial") and
// "lossless" (or "exact"). Throws Error(InvalidInput) for anything else.
CodecMode parse_codec_mode(std::string_view s);
// The canonical name of a mode, as accepted by parse_codec_mode.
std::string_view codec_mode_name(CodecMode m) noexcept;

struct CompressOptions {
    // Frames per independently decodable group, from 1 to 65535. This sets
    // the granularity of random access, and how many frames a crash can lose
    // while streaming.
    std::uint32_t gof_size = 20;
    // Base quantization step in each column's internal unit (nm, nm/ps or
    // kJ/mol/nm). With adaptive precision the maximum error can reach 4 times
    // this step. The default bounds position errors by 0.0005 nm (0.005 A),
    // matching XTC.
    float precision = 0.000125f;
    // Per-column precision overrides, keyed by column name. 0 stores a
    // column exactly, except for positions, velocities and forces, which are
    // always quantized.
    std::map<std::string, float, std::less<>> column_precision;
    // Order of the temporal predictor. 0 chooses one per group.
    std::uint32_t predictor_order = 0;
    int zstd_level = 3;
    bool adaptive = true;
    // Spacing of B-frame anchors, from 1 to 255. 1 disables B-frames.
    std::uint32_t anchor_step = 1;
    CodecMode codec = CodecMode::Auto;
    // Recorded in the metadata so the file can be converted back to this
    // format.
    std::optional<std::string> source_format;
    bool spatial_delta = true;
    // Store every float exactly. Overrides precision, column_precision,
    // codec, adaptive, anchor_step and spatial_delta.
    bool lossless = false;
    // Allow the atom count to vary between frames, with atoms matched by the
    // "mdz.id" column. A batch whose count varies implies this. Set it in
    // advance when streaming frames whose count may change later.
    bool variable_atoms = false;
    // Model rigid bodies, taken from FrameBatch::bodies or detected if the
    // batch has none. Each group is coded both with and without them, and the
    // smaller result is kept. Applies to lossy columns only, and cannot be
    // combined with a varying atom count.
    bool rigid = false;

    // The precision used for `column`: its override if any, else `precision`.
    // Returns 0 when lossless is set.
    float precision_for(std::string_view column) const;
    // A copy with `lossless` applied to the settings it overrides.
    CompressOptions resolved() const;
    // Throws Error(InvalidInput) if a setting is outside the range the format
    // can store, or a precision is not positive and finite.
    void validate() const;
};

}  // namespace mdz
