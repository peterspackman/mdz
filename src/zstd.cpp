#include "zstd.hpp"

#include <zstd.h>

#include <memory>
#include <string>

#include "bytes.hpp"

namespace mdz::detail {
namespace {

struct CCtxDeleter {
    void operator()(ZSTD_CCtx* c) const noexcept { ZSTD_freeCCtx(c); }
};
struct DCtxDeleter {
    void operator()(ZSTD_DCtx* c) const noexcept { ZSTD_freeDCtx(c); }
};

ZSTD_CCtx* cctx() {
    thread_local std::unique_ptr<ZSTD_CCtx, CCtxDeleter> c(ZSTD_createCCtx());
    if (!c) throw Error(ErrorKind::Io, "zstd: out of memory");
    return c.get();
}

ZSTD_DCtx* dctx() {
    thread_local std::unique_ptr<ZSTD_DCtx, DCtxDeleter> c(ZSTD_createDCtx());
    if (!c) throw Error(ErrorKind::Io, "zstd: out of memory");
    return c.get();
}

// The largest expansion a zstd frame can achieve per input byte. A 128 KiB
// RLE block costs 4 bytes: a 3-byte block header and the repeated byte.
constexpr std::size_t kZstdMaxRatio = 32 * 1024;

[[noreturn]] void too_big(std::size_t cap) {
    corrupt("blob decompresses past its " + std::to_string(cap) + "-byte limit");
}

std::size_t compress_into(ByteSpan src, int level, std::vector<std::uint8_t>& out) {
    ZSTD_CCtx* c = cctx();
    // Reset to the parameters of a fresh context at `level`, so the output
    // matches any other zstd 1.5.7 build compressing the same input.
    ZSTD_CCtx_reset(c, ZSTD_reset_session_and_parameters);
    ZSTD_CCtx_setParameter(c, ZSTD_c_compressionLevel, level);
    out.resize(ZSTD_compressBound(src.size()));
    std::size_t n = ZSTD_compress2(c, out.data(), out.size(), src.data(), src.size());
    if (ZSTD_isError(n)) throw Error(ErrorKind::InvalidInput, std::string("zstd: ") + ZSTD_getErrorName(n));
    return n;
}

}  // namespace

unsigned zstd_version() noexcept { return ZSTD_versionNumber(); }

void zstd_compress(ByteSpan src, int level, std::vector<std::uint8_t>& out) {
    out.resize(compress_into(src, level, out));
}

std::vector<std::uint8_t> zstd_compress(ByteSpan src, int level) {
    std::vector<std::uint8_t> out;
    zstd_compress(src, level, out);
    return out;
}

std::size_t zstd_compressed_size(ByteSpan src, int level) {
    thread_local std::vector<std::uint8_t> scratch;
    return compress_into(src, level, scratch);
}

void zstd_decompress_bounded(ByteSpan blob, std::size_t cap, std::vector<std::uint8_t>& out) {
    unsigned long long size = ZSTD_getFrameContentSize(blob.data(), blob.size());
    if (size == ZSTD_CONTENTSIZE_ERROR) corrupt("zstd: not a zstd frame");
    ZSTD_DCtx* d = dctx();
    if (size != ZSTD_CONTENTSIZE_UNKNOWN) {
        // Trust the header only as far as both the cap and what this many
        // compressed bytes could physically expand to.
        if (size > cap || size > sat_mul(blob.size(), kZstdMaxRatio)) too_big(cap);
        out.resize(static_cast<std::size_t>(size));
        std::size_t n = ZSTD_decompressDCtx(d, out.data(), out.size(), blob.data(), blob.size());
        if (ZSTD_isError(n)) corrupt(std::string("zstd: ") + ZSTD_getErrorName(n));
        if (n != size) corrupt("zstd frame size does not match its header");
        return;
    }
    // The header has no content size, so stream the output and stop as soon
    // as it passes the cap.
    ZSTD_DCtx_reset(d, ZSTD_reset_session_only);
    out.clear();
    ZSTD_inBuffer in{blob.data(), blob.size(), 0};
    std::size_t chunk = ZSTD_DStreamOutSize();
    for (;;) {
        std::size_t have = out.size();
        out.resize(have + chunk);
        ZSTD_outBuffer o{out.data() + have, chunk, 0};
        std::size_t r = ZSTD_decompressStream(d, &o, &in);
        if (ZSTD_isError(r)) corrupt(std::string("zstd: ") + ZSTD_getErrorName(r));
        out.resize(have + o.pos);
        if (out.size() > cap) too_big(cap);
        if (r == 0) break;
        if (in.pos == in.size && o.pos == 0) corrupt("zstd: truncated frame");
    }
}

}  // namespace mdz::detail
