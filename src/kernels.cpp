#include "kernels.hpp"

namespace mdz::detail {

void quantize_uniform(const float* v, std::size_t n, float precision, std::int32_t* out) noexcept {
    const float inv = 1.0f / precision;
    for (std::size_t i = 0; i < n; ++i) out[i] = round_to_i32(v[i] * inv);
}

void quantize_adaptive(const float* v, const float* prec, std::size_t natoms, std::int32_t* out) noexcept {
    for (std::size_t a = 0; a < natoms; ++a) {
        const float inv = 1.0f / prec[a];
        out[3 * a] = round_to_i32(v[3 * a] * inv);
        out[3 * a + 1] = round_to_i32(v[3 * a + 1] * inv);
        out[3 * a + 2] = round_to_i32(v[3 * a + 2] * inv);
    }
}

void dequantize_uniform(const std::int32_t* q, std::size_t n, float precision, float* out) noexcept {
    for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<float>(q[i]) * precision;
}

void dequantize_adaptive(const std::int32_t* q, const float* prec, std::size_t natoms, float* out) noexcept {
    for (std::size_t a = 0; a < natoms; ++a) {
        const float p = prec[a];
        out[3 * a] = static_cast<float>(q[3 * a]) * p;
        out[3 * a + 1] = static_cast<float>(q[3 * a + 1]) * p;
        out[3 * a + 2] = static_cast<float>(q[3 * a + 2]) * p;
    }
}

std::size_t pack_residual_frame(const std::int32_t* q, std::size_t natoms, bool spatial, std::uint8_t* out) noexcept {
    std::uint8_t* o = out;
    for (std::size_t c = 0; c < 3; ++c) {
        std::int32_t prev = 0;
        for (std::size_t a = 0; a < natoms; ++a) {
            std::int32_t v = q[a * 3 + c];
            std::int32_t d =
                spatial ? static_cast<std::int32_t>(static_cast<std::uint32_t>(v) - static_cast<std::uint32_t>(prev))
                        : v;
            prev = v;
            std::uint32_t z = zigzag(d);
            // Most residuals fit in one byte, so handle that case before the loop.
            if (z < 0x80) {
                *o++ = static_cast<std::uint8_t>(z);
                continue;
            }
            while (z >= 0x80) {
                *o++ = static_cast<std::uint8_t>(z | 0x80);
                z >>= 7;
            }
            *o++ = static_cast<std::uint8_t>(z);
        }
    }
    return static_cast<std::size_t>(o - out);
}

std::size_t read_residual_frame(const std::uint8_t* data, std::size_t len, std::size_t natoms, bool spatial,
                                std::int32_t* q) noexcept {
    std::size_t pos = 0;
    for (std::size_t c = 0; c < 3; ++c) {
        std::uint32_t acc = 0;
        for (std::size_t a = 0; a < natoms; ++a) {
            if (pos >= len) return 0;
            std::uint8_t b = data[pos++];
            std::uint32_t v = b & 0x7Fu;
            if (b & 0x80) {
                unsigned shift = 7;
                for (;;) {
                    if (pos >= len) return 0;
                    b = data[pos++];
                    if (shift == 28 && b > 0x0F) return 0;  // more than 32 bits, or a sixth byte
                    v |= static_cast<std::uint32_t>(b & 0x7Fu) << shift;
                    if (!(b & 0x80)) break;
                    shift += 7;
                }
            }
            std::int32_t z = unzigzag(v);
            if (spatial) {
                acc += static_cast<std::uint32_t>(z);
                q[a * 3 + c] = static_cast<std::int32_t>(acc);
            } else {
                q[a * 3 + c] = z;
            }
        }
    }
    return pos;
}

void byte_planes(const std::uint32_t* words, std::size_t n, std::uint8_t* out) noexcept {
    for (unsigned p = 0; p < 4; ++p) {
        const unsigned shift = 8 * p;
        std::uint8_t* dst = out + p * n;
        for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<std::uint8_t>(words[i] >> shift);
    }
}

}  // namespace mdz::detail
