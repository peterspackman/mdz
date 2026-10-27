#pragma once
// zstd with per-thread contexts, and decompression that refuses to expand a
// blob past what its position in the file allows.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "mdz/span.hpp"

namespace mdz::detail {

// The zstd version compiled in, as ZSTD_versionNumber (1.5.7 = 10507).
unsigned zstd_version() noexcept;

// Compress `src` at `level`, replacing the contents of `out`.
void zstd_compress(ByteSpan src, int level, std::vector<std::uint8_t>& out);
std::vector<std::uint8_t> zstd_compress(ByteSpan src, int level);
// Compressed size only (for trial encodes); uses a per-thread scratch buffer.
std::size_t zstd_compressed_size(ByteSpan src, int level);

// Decompress a zstd frame into `out`, producing at most `cap` bytes. A blob
// that would expand past `cap` is treated as corrupt, since it is either
// damaged or a decompression bomb. Throws Error(Corrupt).
void zstd_decompress_bounded(ByteSpan blob, std::size_t cap, std::vector<std::uint8_t>& out);

}  // namespace mdz::detail
