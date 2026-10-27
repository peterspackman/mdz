#pragma once
// Bounds-checked reads over untrusted bytes, and little-endian appends.
//
// Everything a decoder reads from a file is untrusted: lengths, counts and
// offsets all come from the file. Every read here throws Error(Corrupt) rather
// than running past the end, and decoders size allocations from bytes that
// exist rather than from counts a file merely claims.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "mdz/error.hpp"
#include "mdz/span.hpp"

// The file format, .npy and raw float32 I/O all copy values straight to and
// from memory, which is only right on a little-endian host.
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "mdz needs a little-endian target"
#endif

namespace mdz::detail {

[[noreturn]] void corrupt(std::string_view what);

template <class T>
inline T load_le(const std::uint8_t* p) noexcept {
    T v;
    std::memcpy(&v, p, sizeof(T));  // little-endian host, checked above
    return v;
}

template <class T>
inline void store_le(std::uint8_t* p, T v) noexcept {
    std::memcpy(p, &v, sizeof(T));
}

class ByteCursor {
public:
    ByteCursor(ByteSpan data, std::size_t off = 0) noexcept : data_(data), off_(off) {}

    std::size_t offset() const noexcept { return off_; }
    std::size_t remaining() const noexcept { return off_ < data_.size() ? data_.size() - off_ : 0; }
    ByteSpan rest() const noexcept { return {data_.data() + off_, remaining()}; }

    ByteSpan take(std::size_t n) {
        if (n > remaining()) corrupt("unexpected end of data");
        ByteSpan s{data_.data() + off_, n};
        off_ += n;
        return s;
    }
    void skip(std::size_t n) { take(n); }
    std::uint8_t u8() { return take(1)[0]; }
    std::uint16_t u16() { return load_le<std::uint16_t>(take(2).data()); }
    std::uint32_t u32() { return load_le<std::uint32_t>(take(4).data()); }
    std::uint64_t u64() { return load_le<std::uint64_t>(take(8).data()); }
    float f32() { return load_le<float>(take(4).data()); }
    std::string_view str(std::size_t n) {
        ByteSpan s = take(n);
        return {reinterpret_cast<const char*>(s.data()), s.size()};
    }

private:
    ByteSpan data_;
    std::size_t off_;
};

using Bytes = std::vector<std::uint8_t>;

inline void put_u8(Bytes& b, std::uint8_t v) { b.push_back(v); }
template <class T>
inline void put_le(Bytes& b, T v) {
    std::size_t n = b.size();
    b.resize(n + sizeof(T));
    store_le(b.data() + n, v);
}
inline void put_bytes(Bytes& b, const void* p, std::size_t n) {
    const auto* c = static_cast<const std::uint8_t*>(p);
    b.insert(b.end(), c, c + n);
}
inline void put_bytes(Bytes& b, ByteSpan s) { put_bytes(b, s.data(), s.size()); }
inline void put_str(Bytes& b, std::string_view s) { put_bytes(b, s.data(), s.size()); }

// a + b and a * b, throwing Error(Corrupt) on overflow.
std::size_t checked_add(std::size_t a, std::size_t b, std::string_view what);
std::size_t checked_mul(std::size_t a, std::size_t b, std::string_view what);
inline std::size_t sat_mul(std::size_t a, std::size_t b) noexcept {
    return (b != 0 && a > SIZE_MAX / b) ? SIZE_MAX : a * b;
}
inline std::size_t sat_add(std::size_t a, std::size_t b) noexcept { return a > SIZE_MAX - b ? SIZE_MAX : a + b; }

}  // namespace mdz::detail
