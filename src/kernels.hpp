#pragma once
// The hot elementwise loops: quantization, zigzag, varint packing and byte
// planes. They are written so the compiler can vectorize them on any target.
// Each must match the scalar definition in its comment bit for bit, because
// the file format depends on the exact output.

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace mdz::detail {

// Rounds y half away from zero and converts to i32 with saturation. NaN maps
// to 0, and out-of-range values map to INT32_MIN or INT32_MAX. The result is
// the same as std::round followed by a saturating cast, computed without
// branches. Truncation is exact for every float in i32 range, and the
// fraction y - trunc(y) is exactly representable, so comparing it with +-0.5
// rounds exactly.
inline std::int32_t round_to_i32(float y) noexcept {
    float c = y < -2147483648.0f ? -2147483648.0f : (y > 2147483520.0f ? 2147483520.0f : y);
    c = (y == y) ? c : 0.0f;
    std::int32_t t = static_cast<std::int32_t>(c);
    float f = c - static_cast<float>(t);
    t += static_cast<std::int32_t>(f >= 0.5f) - static_cast<std::int32_t>(f <= -0.5f);
    return y >= 2147483648.0f ? INT32_MAX : t;
}

inline std::uint32_t zigzag(std::int32_t v) noexcept {
    auto u = static_cast<std::uint32_t>(v);
    return (u << 1) ^ static_cast<std::uint32_t>(v >> 31);
}

inline std::int32_t unzigzag(std::uint32_t v) noexcept {
    return static_cast<std::int32_t>((v >> 1) ^ (0u - (v & 1u)));
}

inline std::uint32_t float_bits(float v) noexcept {
    std::uint32_t u;
    std::memcpy(&u, &v, 4);
    return u;
}

inline float bits_float(std::uint32_t u) noexcept {
    float v;
    std::memcpy(&v, &u, 4);
    return v;
}

// out[i] = round(v[i] * (1 / precision))
void quantize_uniform(const float* v, std::size_t n, float precision, std::int32_t* out) noexcept;
// Per-atom step: out[3a+k] = round(v[3a+k] * (1 / prec[a])).
void quantize_adaptive(const float* v, const float* prec, std::size_t natoms, std::int32_t* out) noexcept;
// out[i] = float(q[i]) * precision
void dequantize_uniform(const std::int32_t* q, std::size_t n, float precision, float* out) noexcept;
void dequantize_adaptive(const std::int32_t* q, const float* prec, std::size_t natoms, float* out) noexcept;

// Writes one frame of quantized residuals (given in atom order) as the
// temporal codec stores them. Values are channel-major (every x, then every
// y, then every z), optionally delta coded along each channel, zigzagged, and
// written as LEB128 varints. `out` must have room for 15 bytes per atom.
// Returns the bytes written.
std::size_t pack_residual_frame(const std::int32_t* q, std::size_t natoms, bool spatial, std::uint8_t* out) noexcept;

// The inverse of pack_residual_frame. Reads one frame into q in atom order.
// Returns the bytes consumed, or 0 if the stream ends early or holds a varint
// wider than 32 bits. A frame uses at least one byte per value, so 0 is never
// a valid size when natoms > 0.
std::size_t read_residual_frame(const std::uint8_t* data, std::size_t len, std::size_t natoms, bool spatial,
                                std::int32_t* q) noexcept;

// Splits `words` into four byte planes, least significant first:
// out[p * n + i] is byte p of words[i].
void byte_planes(const std::uint32_t* words, std::size_t n, std::uint8_t* out) noexcept;

}  // namespace mdz::detail
