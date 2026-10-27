#include "crc32.hpp"

#include <array>
#include <cstring>

namespace mdz::detail {
namespace {

// Slicing-by-8: eight table lookups per 8 input bytes.
struct Tables {
    std::array<std::array<std::uint32_t, 256>, 8> t{};
    Tables() {
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[0][i] = c;
        }
        for (std::uint32_t i = 0; i < 256; ++i)
            for (int s = 1; s < 8; ++s) t[s][i] = (t[s - 1][i] >> 8) ^ t[0][t[s - 1][i] & 0xFF];
    }
};

const Tables& tables() {
    static const Tables tb;
    return tb;
}

}  // namespace

std::uint32_t crc32(const void* data, std::size_t n) noexcept {
    const auto& t = tables().t;
    const auto* p = static_cast<const std::uint8_t*>(data);
    std::uint32_t c = 0xFFFFFFFFu;
    while (n >= 8) {
        std::uint32_t lo, hi;
        std::memcpy(&lo, p, 4);
        std::memcpy(&hi, p + 4, 4);
        lo ^= c;
        c = t[7][lo & 0xFF] ^ t[6][(lo >> 8) & 0xFF] ^ t[5][(lo >> 16) & 0xFF] ^ t[4][lo >> 24] ^
            t[3][hi & 0xFF] ^ t[2][(hi >> 8) & 0xFF] ^ t[1][(hi >> 16) & 0xFF] ^ t[0][hi >> 24];
        p += 8;
        n -= 8;
    }
    while (n--) c = t[0][(c ^ *p++) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

}  // namespace mdz::detail
