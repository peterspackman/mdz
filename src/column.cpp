#include "mdz/column.hpp"

#include <algorithm>

namespace mdz {

bool dtype_from_u8(std::uint8_t v, Dtype& out) noexcept {
    if (v > 5) return false;
    out = static_cast<Dtype>(v);
    return true;
}

std::size_t dtype_size(Dtype d) noexcept {
    switch (d) {
        case Dtype::F64:
        case Dtype::I64:
            return 8;
        case Dtype::U8:
            return 1;
        default:
            return 4;
    }
}

std::string_view dtype_name(Dtype d) noexcept {
    switch (d) {
        case Dtype::F32: return "f32";
        case Dtype::F64: return "f64";
        case Dtype::I32: return "i32";
        case Dtype::U32: return "u32";
        case Dtype::U8: return "u8";
        case Dtype::I64: return "i64";
    }
    return "?";
}

std::size_t ColumnDesc::raw_size(std::size_t natoms, std::size_t nframes) const noexcept {
    std::size_t elems = per_frame ? nframes * width : nframes * natoms * width;
    return elems * dtype_size(dtype);
}

namespace {
ColumnDesc make(std::string_view name, Dtype dt, std::uint32_t width, bool per_frame, float precision,
                std::string_view unit) {
    ColumnDesc d;
    d.name = std::string(name);
    d.dtype = dt;
    d.width = width;
    d.per_frame = per_frame;
    d.lossy = precision > 0.0f;
    d.precision = precision;
    d.unit = std::string(unit);
    return d;
}
}  // namespace

ColumnDesc ColumnDesc::positions(float p) { return make("positions", Dtype::F32, 3, false, p, "nm"); }
ColumnDesc ColumnDesc::velocities(float p) { return make("velocities", Dtype::F32, 3, false, p, "nm/ps"); }
ColumnDesc ColumnDesc::forces(float p) { return make("forces", Dtype::F32, 3, false, p, "kJ/mol/nm"); }
ColumnDesc ColumnDesc::scalar(std::string_view name, std::string_view unit, float p) {
    return make(name, Dtype::F32, 1, false, p, unit);
}
ColumnDesc ColumnDesc::frame_scalar(std::string_view name, std::string_view unit) {
    return make(name, Dtype::F64, 1, true, 0.0f, unit);
}
ColumnDesc ColumnDesc::frame_vector(std::string_view name, std::uint32_t width, std::string_view unit) {
    return make(name, Dtype::F64, width, true, 0.0f, unit);
}
ColumnDesc ColumnDesc::int_column(std::string_view name) { return make(name, Dtype::I32, 1, false, 0.0f, ""); }

bool operator==(const ColumnDesc& a, const ColumnDesc& b) noexcept {
    // Precision is compared as a float, so 0.0 and -0.0 are equal.
    return a.name == b.name && a.dtype == b.dtype && a.width == b.width && a.per_frame == b.per_frame &&
           a.lossy == b.lossy && a.precision == b.precision && a.pred_order == b.pred_order &&
           a.codec_flags == b.codec_flags && a.anchor_step == b.anchor_step && a.unit == b.unit;
}

GofTable GofTable::uniform(std::size_t nframes, std::size_t gof_size) {
    GofTable t;
    const std::size_t g = std::max<std::size_t>(gof_size, 1);
    for (std::size_t s = 0; s < nframes; s += g) t.starts_.push_back(s);
    t.nframes_ = nframes;
    return t;
}

bool GofTable::push(std::size_t n) {
    if (n == 0 || nframes_ > SIZE_MAX - n) return false;
    starts_.push_back(nframes_);
    nframes_ += n;
    return true;
}

GofTable::Span GofTable::overlapping(std::size_t start, std::size_t end) const noexcept {
    start = std::min(start, nframes_);
    end = std::min(end, nframes_);
    if (start >= end) return {0, 0};
    std::size_t first = static_cast<std::size_t>(std::upper_bound(starts_.begin(), starts_.end(), start) - starts_.begin()) - 1;
    std::size_t last = static_cast<std::size_t>(std::lower_bound(starts_.begin(), starts_.end(), end) - starts_.begin());
    return {first, last};
}

std::size_t GofTable::max_len() const noexcept {
    std::size_t m = 0;
    for (std::size_t i = 0; i < starts_.size(); ++i) m = std::max(m, span(i).size());
    return m;
}

}  // namespace mdz
