#include "xdr3d.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

#include "bytes.hpp"
#include "mdz/error.hpp"

namespace mdz::detail {
namespace {

constexpr std::size_t kFirstIdx = 9;
constexpr std::uint32_t kMagicInts[] = {
    0,       0,       0,       0,       0,       0,       0,       0,       0,       8,       10,      12,
    16,      20,      25,      32,      40,      50,      64,      80,      101,     128,     161,     203,
    256,     322,     406,     512,     645,     812,     1024,    1290,    1625,    2048,    2580,    3250,
    4096,    5060,    6501,    8192,    10321,   13003,   16384,   20642,   26007,   32768,   41285,   52015,
    65536,   82570,   104031,  131072,  165140,  208063,  262144,  330280,  416127,  524287,  660561,  832255,
    1048576, 1321122, 1664510, 2097152, 2642245, 3329021, 4194304, 5284491, 6658042, 8388607, 10568983, 13316085,
    16777216};
constexpr std::size_t kNumMagic = sizeof(kMagicInts) / sizeof(kMagicInts[0]);

std::uint32_t sizeofint(std::uint32_t size) noexcept {
    if (size <= 1) return 0;
    std::uint32_t v = size - 1;
    std::uint32_t bits = 0;
    while (v) {
        ++bits;
        v >>= 1;
    }
    return bits;
}

std::uint32_t sizeofints(const std::uint32_t* sizes, std::size_t n) noexcept {
    std::uint32_t bytes[32] = {1};
    std::size_t num_bytes = 1;
    for (std::size_t s = 0; s < n; ++s) {
        std::uint64_t carry = 0;
        for (std::size_t b = 0; b < num_bytes; ++b) {
            std::uint64_t tmp = static_cast<std::uint64_t>(bytes[b]) * sizes[s] + carry;
            bytes[b] = static_cast<std::uint32_t>(tmp & 0xff);
            carry = tmp >> 8;
        }
        while (carry > 0) {
            bytes[num_bytes++] = static_cast<std::uint32_t>(carry & 0xff);
            carry >>= 8;
        }
    }
    std::uint32_t top = bytes[num_bytes - 1];
    std::uint32_t top_bits = 0;
    while (top) {
        ++top_bits;
        top >>= 1;
    }
    return static_cast<std::uint32_t>(num_bytes - 1) * 8 + top_bits;
}

std::uint32_t abs_diff(std::int32_t a, std::int32_t b) noexcept {
    return a > b ? static_cast<std::uint32_t>(a) - static_cast<std::uint32_t>(b)
                 : static_cast<std::uint32_t>(b) - static_cast<std::uint32_t>(a);
}

std::int32_t wadd(std::int32_t a, std::int32_t b) noexcept {
    return static_cast<std::int32_t>(static_cast<std::uint32_t>(a) + static_cast<std::uint32_t>(b));
}
std::int32_t wsub(std::int32_t a, std::int32_t b) noexcept {
    return static_cast<std::int32_t>(static_cast<std::uint32_t>(a) - static_cast<std::uint32_t>(b));
}

void put_be32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    std::uint8_t b[4] = {static_cast<std::uint8_t>(v >> 24), static_cast<std::uint8_t>(v >> 16),
                         static_cast<std::uint8_t>(v >> 8), static_cast<std::uint8_t>(v)};
    out.insert(out.end(), b, b + 4);
}
void put_be_f32(std::vector<std::uint8_t>& out, float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    put_be32(out, u);
}

// --- bit writer (GROMACS encodebits / encodeints) ---

struct BitWriter {
    std::vector<std::uint8_t> data;
    std::size_t cnt = 0;
    int lastbits = 0;
    std::uint32_t lastbyte = 0;

    explicit BitWriter(std::size_t capacity) : data(capacity, 0) {}

    void set(std::size_t i, std::uint8_t v) {
        if (i < data.size()) data[i] = v;
    }

    void encodebits(std::uint32_t num_of_bits, std::uint32_t num) {
        while (num_of_bits >= 8) {
            // encodeints pads with zero bits through a single call that can
            // ask for more than 32 bits. Every bit above bit 31 is zero.
            std::uint32_t sh = num_of_bits - 8;
            std::uint32_t byte = (sh >= 32 ? 0u : (num >> sh)) & 0xff;
            lastbyte = (lastbyte << 8) | byte;
            set(cnt, static_cast<std::uint8_t>(lastbyte >> lastbits));
            ++cnt;
            num_of_bits -= 8;
        }
        if (num_of_bits > 0) {
            lastbyte = (lastbyte << num_of_bits) | (num & ((1u << num_of_bits) - 1));
            lastbits += static_cast<int>(num_of_bits);
            if (lastbits >= 8) {
                lastbits -= 8;
                set(cnt, static_cast<std::uint8_t>(lastbyte >> lastbits));
                ++cnt;
            }
        }
        if (lastbits > 0) set(cnt, static_cast<std::uint8_t>(lastbyte << (8 - lastbits)));
    }

    void encodeints(std::size_t num_of_ints, std::uint32_t num_of_bits, const std::uint32_t* sizes,
                    const std::uint32_t* nums) {
        std::uint32_t bytes[32] = {};
        std::size_t num_of_bytes = 0;
        std::uint32_t tmp = nums[0];
        do {
            bytes[num_of_bytes++] = tmp & 0xff;
            tmp >>= 8;
        } while (tmp != 0);
        for (std::size_t i = 1; i < num_of_ints; ++i) {
            tmp = nums[i];
            std::size_t bytecnt = 0;
            while (bytecnt < num_of_bytes) {
                tmp += bytes[bytecnt] * sizes[i];
                bytes[bytecnt] = tmp & 0xff;
                tmp >>= 8;
                ++bytecnt;
            }
            while (tmp != 0) {
                bytes[bytecnt++] = tmp & 0xff;
                tmp >>= 8;
            }
            num_of_bytes = bytecnt;
        }
        if (num_of_bits >= num_of_bytes * 8) {
            for (std::size_t b = 0; b < num_of_bytes; ++b) encodebits(8, bytes[b]);
            encodebits(num_of_bits - static_cast<std::uint32_t>(num_of_bytes) * 8, 0);
        } else {
            for (std::size_t b = 0; b + 1 < num_of_bytes; ++b) encodebits(8, bytes[b]);
            encodebits(num_of_bits - static_cast<std::uint32_t>(num_of_bytes - 1) * 8, bytes[num_of_bytes - 1]);
        }
    }

    std::size_t length() const { return cnt + (lastbits > 0 ? 1 : 0); }
};

// --- bit reader (GROMACS decodebits / decodeints) ---

struct BitReader {
    const std::uint8_t* data;
    std::size_t len;
    std::size_t cnt = 0;
    std::uint32_t lastbits = 0;
    std::uint32_t lastbyte = 0;

    std::uint32_t next_byte() noexcept { return cnt < len ? data[cnt] : 0u; }

    std::int32_t decodebits(std::uint32_t num_of_bits) noexcept {
        std::int32_t mask = static_cast<std::int32_t>((num_of_bits >= 32 ? 0u : (1u << num_of_bits)) - 1u);
        std::int32_t num = 0;
        while (num_of_bits >= 8) {
            lastbyte = (lastbyte << 8) | next_byte();
            ++cnt;
            num |= static_cast<std::int32_t>(
                static_cast<std::uint32_t>(static_cast<std::int32_t>(lastbyte >> lastbits) & 0xff)
                << (num_of_bits - 8));
            num_of_bits -= 8;
        }
        if (num_of_bits > 0) {
            if (lastbits < num_of_bits) {
                lastbits += 8;
                lastbyte = (lastbyte << 8) | next_byte();
                ++cnt;
            }
            lastbits -= num_of_bits;
            num |= static_cast<std::int32_t>(lastbyte >> lastbits) & static_cast<std::int32_t>((1u << num_of_bits) - 1);
        }
        return num & mask;
    }

    void decodeints(std::size_t num_of_ints, std::uint32_t num_of_bits, const std::uint32_t* sizes,
                    std::int32_t* nums) noexcept {
        std::int32_t bytes[32] = {};
        std::size_t num_of_bytes = 0;
        std::uint32_t bits_left = num_of_bits;
        while (bits_left > 8) {
            bytes[num_of_bytes++] = decodebits(8);
            bits_left -= 8;
        }
        if (bits_left > 0) bytes[num_of_bytes++] = decodebits(bits_left);
        for (std::size_t i = num_of_ints - 1; i > 0; --i) {
            std::int32_t num = 0;
            const auto s = static_cast<std::int32_t>(sizes[i]);
            for (std::size_t j = num_of_bytes; j-- > 0;) {
                num = static_cast<std::int32_t>(static_cast<std::uint32_t>(num) << 8) | bytes[j];
                // A corrupt stream can make s zero or overflow the quotient.
                // The result is meaningless either way, but it must not trap.
                std::int32_t p = s > 0 ? num / s : 0;
                bytes[j] = p;
                num = wsub(num, static_cast<std::int32_t>(static_cast<std::uint32_t>(p) * static_cast<std::uint32_t>(s)));
            }
            nums[i] = num;
        }
        nums[0] = static_cast<std::int32_t>(static_cast<std::uint32_t>(bytes[0]) |
                                            (static_cast<std::uint32_t>(bytes[1]) << 8) |
                                            (static_cast<std::uint32_t>(bytes[2]) << 16) |
                                            (static_cast<std::uint32_t>(bytes[3]) << 24));
    }
};

[[noreturn]] void bad(const char* what) { throw Error(ErrorKind::Corrupt, what); }

std::int32_t read_be_i32(ByteCursor& c) {
    ByteSpan b = c.take(4);
    return static_cast<std::int32_t>((static_cast<std::uint32_t>(b[0]) << 24) | (static_cast<std::uint32_t>(b[1]) << 16) |
                                     (static_cast<std::uint32_t>(b[2]) << 8) | static_cast<std::uint32_t>(b[3]));
}
float read_be_f32(ByteCursor& c) {
    std::uint32_t u = static_cast<std::uint32_t>(read_be_i32(c));
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

std::string fmt_coord(float v) {
    if (std::isnan(v)) return "NaN";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    return std::to_string(v);
}

}  // namespace

void xdr3d_encode(const float* coords, std::size_t natoms, float precision, std::vector<std::uint8_t>& out) {
    if (natoms <= 9) {
        for (std::size_t i = 0; i < natoms * 3; ++i) put_be_f32(out, coords[i]);
        return;
    }
    put_be_f32(out, precision);

    std::vector<std::int32_t> ic(natoms * 3);
    std::int32_t minint[3] = {INT32_MAX, INT32_MAX, INT32_MAX};
    std::int32_t maxint[3] = {INT32_MIN, INT32_MIN, INT32_MIN};
    std::int32_t mindiff = INT32_MAX;
    std::int32_t old[3] = {0, 0, 0};
    for (std::size_t i = 0; i < natoms; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            float v = coords[i * 3 + j];
            if (!std::isfinite(v))
                throw Error(ErrorKind::InvalidInput, "cannot encode non-finite coordinate " + fmt_coord(v) + " to XTC");
            float scaled = v * precision;
            // 2147483520 is the largest float below 2^31.
            if (std::fabs(scaled) > 2147483520.0f)
                throw Error(ErrorKind::InvalidInput,
                            "coordinate " + fmt_coord(v) +
                                " nm overflows XTC's 32-bit integer range at this precision; rescale the data or "
                                "lower the precision");
            float r = v >= 0.0f ? scaled + 0.5f : scaled - 0.5f;
            // r lies in (-2^31 - 1, 2^31 + 1). Clamp values that rounding pushed
            // outside the i32 range.
            std::int32_t lint = r >= 2147483648.0f ? INT32_MAX : (r <= -2147483904.0f ? INT32_MIN : static_cast<std::int32_t>(r));
            ic[i * 3 + j] = lint;
            if (lint < minint[j]) minint[j] = lint;
            if (lint > maxint[j]) maxint[j] = lint;
        }
        if (i > 0) {
            std::uint64_t d = static_cast<std::uint64_t>(abs_diff(old[0], ic[i * 3])) +
                              abs_diff(old[1], ic[i * 3 + 1]) + abs_diff(old[2], ic[i * 3 + 2]);
            auto diff = static_cast<std::int32_t>(d > INT32_MAX ? INT32_MAX : d);
            if (diff < mindiff) mindiff = diff;
        }
        old[0] = ic[i * 3];
        old[1] = ic[i * 3 + 1];
        old[2] = ic[i * 3 + 2];
    }
    for (std::int32_t v : minint) put_be32(out, static_cast<std::uint32_t>(v));
    for (std::int32_t v : maxint) put_be32(out, static_cast<std::uint32_t>(v));

    std::uint32_t sizeint[3], bitsizeint[3];
    bool use_separate = false;
    for (std::size_t j = 0; j < 3; ++j) {
        std::int64_t range = static_cast<std::int64_t>(maxint[j]) - minint[j] + 1;
        if (range < 0 || range > static_cast<std::int64_t>(UINT32_MAX))
            throw Error(ErrorKind::InvalidInput, "coordinates span too wide a range for XTC at this precision");
        sizeint[j] = static_cast<std::uint32_t>(range);
        bitsizeint[j] = sizeofint(sizeint[j]);
        if (sizeint[j] > 0xffffff) use_separate = true;
    }
    std::uint32_t bitsize = use_separate ? 0 : sizeofints(sizeint, 3);

    std::size_t smallidx = kFirstIdx;
    while (smallidx < kNumMagic && static_cast<std::int32_t>(kMagicInts[smallidx]) < mindiff) ++smallidx;
    if (smallidx >= kNumMagic) smallidx = kNumMagic - 1;
    put_be32(out, static_cast<std::uint32_t>(smallidx));

    const std::size_t maxidx = std::min(kNumMagic, smallidx + 8);
    const std::size_t minidx = std::max(maxidx >= 8 ? maxidx - 8 : 0, kFirstIdx);
    std::size_t tmp = smallidx > 0 ? smallidx - 1 : 0;
    if (kFirstIdx > tmp) tmp = kFirstIdx;
    std::int32_t smaller = static_cast<std::int32_t>(kMagicInts[tmp]) / 2;
    std::int32_t smallnum = static_cast<std::int32_t>(kMagicInts[smallidx]) / 2;
    std::uint32_t sizesmall[3] = {kMagicInts[smallidx], kMagicInts[smallidx], kMagicInts[smallidx]};
    const std::int32_t larger = static_cast<std::int32_t>(kMagicInts[maxidx - 1]) / 2;

    BitWriter bits(natoms * 3 * 4 + 1000);
    std::int32_t prevcoord[3] = {0, 0, 0};
    std::int32_t prevrun = -1;
    std::size_t i = 0;
    while (i < natoms) {
        std::int32_t is_smaller;
        if (smallidx < maxidx && i >= 1 && abs_diff(ic[i * 3], prevcoord[0]) < static_cast<std::uint32_t>(larger) &&
            abs_diff(ic[i * 3 + 1], prevcoord[1]) < static_cast<std::uint32_t>(larger) &&
            abs_diff(ic[i * 3 + 2], prevcoord[2]) < static_cast<std::uint32_t>(larger)) {
            is_smaller = 1;
        } else if (smallidx > minidx) {
            is_smaller = -1;
        } else {
            is_smaller = 0;
        }

        bool is_small = false;
        if (i + 1 < natoms && abs_diff(ic[i * 3], ic[(i + 1) * 3]) < static_cast<std::uint32_t>(smallnum) &&
            abs_diff(ic[i * 3 + 1], ic[(i + 1) * 3 + 1]) < static_cast<std::uint32_t>(smallnum) &&
            abs_diff(ic[i * 3 + 2], ic[(i + 1) * 3 + 2]) < static_cast<std::uint32_t>(smallnum)) {
            for (std::size_t j = 0; j < 3; ++j) std::swap(ic[i * 3 + j], ic[(i + 1) * 3 + j]);
            is_small = true;
        }

        const std::uint32_t tmpcoord[3] = {static_cast<std::uint32_t>(wsub(ic[i * 3], minint[0])),
                                           static_cast<std::uint32_t>(wsub(ic[i * 3 + 1], minint[1])),
                                           static_cast<std::uint32_t>(wsub(ic[i * 3 + 2], minint[2]))};
        if (bitsize == 0) {
            bits.encodebits(bitsizeint[0], tmpcoord[0]);
            bits.encodebits(bitsizeint[1], tmpcoord[1]);
            bits.encodebits(bitsizeint[2], tmpcoord[2]);
        } else {
            bits.encodeints(3, bitsize, sizeint, tmpcoord);
        }
        prevcoord[0] = ic[i * 3];
        prevcoord[1] = ic[i * 3 + 1];
        prevcoord[2] = ic[i * 3 + 2];
        ++i;

        std::int32_t run = 0;
        std::uint32_t runbuf[30] = {};
        std::int32_t is_smaller_eff = is_smaller;
        if (!is_small && is_smaller == -1) is_smaller_eff = 0;

        while (is_small && run < 8 * 3) {
            std::int64_t tmpsum = 0;
            for (std::size_t j = 0; j < 3; ++j) {
                std::int64_t d = static_cast<std::int64_t>(ic[i * 3 + j]) - prevcoord[j];
                tmpsum += d * d;
            }
            if (is_smaller_eff == -1 && tmpsum >= static_cast<std::int64_t>(smaller) * smaller) is_smaller_eff = 0;
            for (std::size_t j = 0; j < 3; ++j)
                runbuf[run + j] = static_cast<std::uint32_t>(wadd(wsub(ic[i * 3 + j], prevcoord[j]), smallnum));
            prevcoord[0] = ic[i * 3];
            prevcoord[1] = ic[i * 3 + 1];
            prevcoord[2] = ic[i * 3 + 2];
            run += 3;
            ++i;
            is_small = i < natoms && abs_diff(ic[i * 3], prevcoord[0]) < static_cast<std::uint32_t>(smallnum) &&
                       abs_diff(ic[i * 3 + 1], prevcoord[1]) < static_cast<std::uint32_t>(smallnum) &&
                       abs_diff(ic[i * 3 + 2], prevcoord[2]) < static_cast<std::uint32_t>(smallnum);
        }

        if (run != prevrun || is_smaller_eff != 0) {
            prevrun = run;
            bits.encodebits(1, 1);
            bits.encodebits(5, static_cast<std::uint32_t>(run + is_smaller_eff + 1));
        } else {
            bits.encodebits(1, 0);
        }
        for (std::int32_t k = 0; k < run; k += 3)
            bits.encodeints(3, static_cast<std::uint32_t>(smallidx), sizesmall, runbuf + k);

        if (is_smaller_eff != 0) {
            std::int64_t si = static_cast<std::int64_t>(smallidx) + is_smaller_eff;
            smallidx = si < 0 ? 0 : static_cast<std::size_t>(si);
            if (smallidx >= kNumMagic) smallidx = kNumMagic - 1;
            if (is_smaller_eff < 0) {
                smallnum = smaller;
                if (smallidx > kFirstIdx) smaller = static_cast<std::int32_t>(kMagicInts[smallidx - 1]) / 2;
            } else {
                smaller = smallnum;
                smallnum = static_cast<std::int32_t>(kMagicInts[smallidx]) / 2;
            }
            sizesmall[0] = sizesmall[1] = sizesmall[2] = kMagicInts[smallidx];
        }
    }

    std::size_t len = bits.length();
    bits.data.resize(len);
    put_be32(out, static_cast<std::uint32_t>(len));
    out.insert(out.end(), bits.data.begin(), bits.data.end());
    std::size_t pad = (4 - len % 4) % 4;
    out.insert(out.end(), pad, 0);
}

std::size_t xdr3d_decode(ByteSpan in, std::size_t natoms, std::vector<std::int32_t>& lip, float* out) {
    ByteCursor cur(in);
    if (natoms <= 9) {
        for (std::size_t i = 0; i < natoms * 3; ++i) out[i] = read_be_f32(cur);
        return cur.offset();
    }
    float precision = read_be_f32(cur);
    if (!(std::isfinite(precision) && precision > 0.0f)) bad("bad xtc precision");
    const float inv_precision = 1.0f / precision;

    std::int32_t minint[3], maxint[3];
    for (auto& v : minint) v = read_be_i32(cur);
    for (auto& v : maxint) v = read_be_i32(cur);
    std::int32_t si = read_be_i32(cur);
    if (si < static_cast<std::int32_t>(kFirstIdx) || si >= static_cast<std::int32_t>(kNumMagic))
        bad("XTC smallidx out of range");
    std::size_t smallidx = static_cast<std::size_t>(si);

    std::size_t tmp = smallidx > 0 ? smallidx - 1 : 0;
    if (kFirstIdx > tmp) tmp = kFirstIdx;
    std::int32_t smaller = static_cast<std::int32_t>(kMagicInts[tmp]) / 2;
    std::int32_t smallnum = static_cast<std::int32_t>(kMagicInts[smallidx]) / 2;
    std::uint32_t sizesmall[3] = {kMagicInts[smallidx], kMagicInts[smallidx], kMagicInts[smallidx]};

    std::uint32_t sizeint[3], bitsizeint[3];
    bool use_decodebits = false;
    for (std::size_t j = 0; j < 3; ++j) {
        std::int64_t range = static_cast<std::int64_t>(maxint[j]) - minint[j] + 1;
        if (range <= 0 || range > static_cast<std::int64_t>(UINT32_MAX)) bad("XTC coordinate range is invalid");
        sizeint[j] = static_cast<std::uint32_t>(range);
        bitsizeint[j] = sizeofint(sizeint[j]);
        if (sizeint[j] > 0xffffff) use_decodebits = true;
    }
    const std::uint32_t bitsize = use_decodebits ? 0 : sizeofints(sizeint, 3);

    std::int32_t nbytes = read_be_i32(cur);
    if (nbytes < 0) bad("negative XDR opaque length");
    std::size_t len = static_cast<std::size_t>(nbytes);
    std::size_t padded = (len + 3) & ~static_cast<std::size_t>(3);
    if (padded > cur.remaining()) bad("XDR opaque data truncated");
    ByteSpan data = cur.take(padded).first(len);
    // Every atom costs at least one bit (its run flag).
    if (natoms > sat_mul(len, 8)) bad("XTC payload too short for its atom count");

    BitReader bits{data.data(), data.size()};
    if (lip.size() != natoms * 3) lip.assign(natoms * 3, 0);
    std::size_t nout = 0;
    const std::size_t total = natoms * 3;
    auto put = [&](std::int32_t a, std::int32_t b, std::int32_t c) {
        if (nout + 3 > total) bad("XTC decode produced too many values");
        out[nout] = static_cast<float>(a) * inv_precision;
        out[nout + 1] = static_cast<float>(b) * inv_precision;
        out[nout + 2] = static_cast<float>(c) * inv_precision;
        nout += 3;
    };

    std::int32_t prevcoord[3];
    std::int32_t run = 0;
    std::size_t i = 0;
    std::int32_t* L = lip.data();
    while (i < natoms) {
        const std::size_t tc = i * 3;
        if (use_decodebits) {
            L[tc] = bits.decodebits(bitsizeint[0]);
            L[tc + 1] = bits.decodebits(bitsizeint[1]);
            L[tc + 2] = bits.decodebits(bitsizeint[2]);
        } else {
            bits.decodeints(3, bitsize, sizeint, L + tc);
        }
        ++i;
        L[tc] = wadd(L[tc], minint[0]);
        L[tc + 1] = wadd(L[tc + 1], minint[1]);
        L[tc + 2] = wadd(L[tc + 2], minint[2]);
        prevcoord[0] = L[tc];
        prevcoord[1] = L[tc + 1];
        prevcoord[2] = L[tc + 2];

        std::int32_t flag = bits.decodebits(1);
        std::int32_t is_smaller = 0;
        if (flag == 1) {
            run = bits.decodebits(5);
            is_smaller = run % 3;
            run -= is_smaller;
            is_smaller -= 1;
        }

        if (run > 0) {
            for (std::int32_t k = 0; k < run; k += 3) {
                if (i >= natoms) break;
                const std::size_t t2 = i * 3;
                bits.decodeints(3, static_cast<std::uint32_t>(smallidx), sizesmall, L + t2);
                ++i;
                for (std::size_t j = 0; j < 3; ++j) L[t2 + j] = wsub(wadd(L[t2 + j], prevcoord[j]), smallnum);
                if (k == 0) {
                    // Swap this atom with the large atom before it. The encoder
                    // swapped them so that the run could start one atom earlier.
                    std::int32_t t0 = L[t2], t1 = L[t2 + 1], t2v = L[t2 + 2];
                    L[t2] = prevcoord[0];
                    L[t2 + 1] = prevcoord[1];
                    L[t2 + 2] = prevcoord[2];
                    prevcoord[0] = t0;
                    prevcoord[1] = t1;
                    prevcoord[2] = t2v;
                    put(prevcoord[0], prevcoord[1], prevcoord[2]);
                } else {
                    prevcoord[0] = L[t2];
                    prevcoord[1] = L[t2 + 1];
                    prevcoord[2] = L[t2 + 2];
                }
                put(L[t2], L[t2 + 1], L[t2 + 2]);
            }
        } else {
            put(L[tc], L[tc + 1], L[tc + 2]);
        }

        std::int64_t ns = static_cast<std::int64_t>(smallidx) + is_smaller;
        smallidx = ns < 0 ? 0 : static_cast<std::size_t>(ns);
        if (smallidx < kFirstIdx) bad("XTC smallidx stepped out of range");
        if (smallidx >= kNumMagic) smallidx = kNumMagic - 1;
        if (is_smaller < 0) {
            smallnum = smaller;
            smaller = smallidx > kFirstIdx ? static_cast<std::int32_t>(kMagicInts[smallidx - 1]) / 2 : 0;
        } else if (is_smaller > 0) {
            smaller = smallnum;
            smallnum = static_cast<std::int32_t>(kMagicInts[smallidx]) / 2;
        }
        sizesmall[0] = sizesmall[1] = sizesmall[2] = kMagicInts[smallidx];
    }
    if (nout != total)
        throw Error(ErrorKind::Corrupt, "XTC decode produced " + std::to_string(nout) + " floats, expected " +
                                            std::to_string(total));
    return cur.offset();
}

}  // namespace mdz::detail
