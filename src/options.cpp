#include "mdz/options.hpp"

#include <cmath>
#include <string>

#include "codec.hpp"
#include "mdz/error.hpp"

namespace mdz {

CodecMode parse_codec_mode(std::string_view s) {
    if (s == "auto") return CodecMode::Auto;
    if (s == "temporal") return CodecMode::Temporal;
    if (s == "intra" || s == "intraframe" || s == "spatial") return CodecMode::Intra;
    if (s == "lossless" || s == "exact") return CodecMode::Lossless;
    throw Error(ErrorKind::InvalidInput,
                "unknown codec mode '" + std::string(s) + "' (expected auto, temporal, intra or lossless)");
}

std::string_view codec_mode_name(CodecMode m) noexcept {
    switch (m) {
        case CodecMode::Temporal: return "temporal";
        case CodecMode::Intra: return "intra";
        case CodecMode::Lossless: return "lossless";
        case CodecMode::Rigid: return "rigid";
        case CodecMode::Auto: return "auto";
    }
    return "?";
}

float CompressOptions::precision_for(std::string_view column) const {
    if (lossless) return 0.0f;
    auto it = column_precision.find(column);
    return it != column_precision.end() ? it->second : precision;
}

CompressOptions CompressOptions::resolved() const {
    if (!lossless) return *this;
    CompressOptions o = *this;
    o.precision = 0.0f;
    o.column_precision.clear();
    o.adaptive = false;
    o.anchor_step = 1;
    o.spatial_delta = false;
    o.codec = CodecMode::Lossless;
    return o;
}

void CompressOptions::validate() const {
    if (lossless) {
        detail::validate_params(1.0f, gof_size, anchor_step);
        return;
    }
    detail::validate_params(precision, gof_size, anchor_step);
    for (const auto& [name, p] : column_precision) {
        // Positions, velocities and forces are always quantized. Other
        // columns may be given 0, to store them exactly.
        const bool xyz = name == "positions" || name == "velocities" || name == "forces";
        if (!std::isfinite(p) || p < 0.0f || (xyz && p == 0.0f))
            throw Error(ErrorKind::InvalidInput, "precision for column '" + name + "' must be " +
                                                     (xyz ? "positive" : "zero or positive") + " and finite (got " +
                                                     std::to_string(p) + ")");
    }
}

}  // namespace mdz
