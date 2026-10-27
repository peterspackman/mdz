#pragma once
// CRC-32 (the IEEE polynomial, as zlib and gzip use), for the v4 container's
// record and schema checksums.

#include <cstddef>
#include <cstdint>

namespace mdz::detail {

std::uint32_t crc32(const void* data, std::size_t n) noexcept;

}  // namespace mdz::detail
