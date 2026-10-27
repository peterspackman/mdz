#include "text.hpp"

#include <charconv>
#include <cmath>
#include <cstring>

#include <fast_float/fast_float.h>

namespace mdz::text {

LineReader::LineReader(InputStream& in, std::size_t chunk) : in_(in), chunk_(chunk) { buf_.resize(chunk); }

bool LineReader::fill() {
    if (eof_) return false;
    // Move the unread tail to the front of the buffer, then append a chunk.
    if (begin_ > 0) {
        std::memmove(buf_.data(), buf_.data() + begin_, end_ - begin_);
        end_ -= begin_;
        begin_ = 0;
    }
    if (buf_.size() - end_ < chunk_) buf_.resize(end_ + chunk_);
    std::size_t got = in_.read(buf_.data() + end_, buf_.size() - end_);
    if (got == 0) eof_ = true;
    end_ += got;
    return got > 0;
}

bool LineReader::next(std::string_view& line) {
    if (peeked_) {
        peeked_ = false;
        line = peek_line_;
        ++lineno_;
        return true;
    }
    for (;;) {
        const char* start = buf_.data() + begin_;
        const void* nl = std::memchr(start, '\n', end_ - begin_);
        if (nl) {
            std::size_t len = static_cast<const char*>(nl) - start;
            begin_ += len + 1;
            if (len && start[len - 1] == '\r') --len;
            line = std::string_view(start, len);
            ++lineno_;
            return true;
        }
        if (!fill()) {
            if (begin_ == end_) return false;
            // The last line has no terminator.
            const char* s = buf_.data() + begin_;
            std::size_t len = end_ - begin_;
            begin_ = end_;
            if (len && s[len - 1] == '\r') --len;
            line = std::string_view(s, len);
            ++lineno_;
            return true;
        }
    }
}

bool LineReader::peek(std::string_view& line) {
    if (peeked_) {
        line = peek_line_;
        return true;
    }
    // Read the line with next() so that it is complete in the buffer, then
    // keep it for the following next() call. The line number is not advanced
    // until then.
    if (!next(line)) return false;
    --lineno_;
    peek_line_ = line;
    peeked_ = true;
    return true;
}

namespace {

// Strips a single leading '+'. Returns false if s is then empty or starts with
// another sign.
bool strip_plus(std::string_view& s) noexcept {
    if (!s.empty() && s.front() == '+') {
        s.remove_prefix(1);
        if (!s.empty() && (s.front() == '-' || s.front() == '+')) return false;
    }
    return !s.empty();
}

template <class T>
bool parse_float(std::string_view s, T& out) noexcept {
    if (!strip_plus(s)) return false;
    auto r = fast_float::from_chars(s.data(), s.data() + s.size(), out);
    // Accept out-of-range values, which fast_float returns as +-inf or +-0.
    return (r.ec == std::errc() || r.ec == std::errc::result_out_of_range) && r.ptr == s.data() + s.size();
}

template <class T>
bool parse_int(std::string_view s, T& out) noexcept {
    if (!strip_plus(s)) return false;
    auto r = std::from_chars(s.data(), s.data() + s.size(), out);
    return r.ec == std::errc() && r.ptr == s.data() + s.size();
}

// Writes the shortest round-trip digits of v to out in plain decimal notation
// and returns the number of characters written.
template <class T>
std::size_t shortest_plain(T v, char* out) {
    if (std::isnan(v)) {
        std::memcpy(out, "NaN", 3);
        return 3;
    }
    if (std::isinf(v)) {
        if (v < 0) {
            std::memcpy(out, "-inf", 4);
            return 4;
        }
        std::memcpy(out, "inf", 3);
        return 3;
    }
    char sci[64];
    auto r = std::to_chars(sci, sci + sizeof sci, v, std::chars_format::scientific);
    std::string_view s(sci, static_cast<std::size_t>(r.ptr - sci));
    std::size_t o = 0;
    if (s.front() == '-') {
        out[o++] = '-';
        s.remove_prefix(1);
    }
    const std::size_t e = s.find('e');
    char digits[40];
    std::size_t nd = 0;
    for (std::size_t i = 0; i < e; ++i)
        if (s[i] != '.') digits[nd++] = s[i];
    int exp = 0;
    std::from_chars(s.data() + e + 1 + (s[e + 1] == '+'), s.data() + s.size(), exp);
    // The value is 0.d1d2d3... x 10^(exp+1).
    const int point = exp + 1;  // digits before the decimal point
    if (nd == 1 && digits[0] == '0') {
        out[o++] = '0';
        return o;
    }
    if (point <= 0) {
        out[o++] = '0';
        out[o++] = '.';
        for (int i = 0; i < -point; ++i) out[o++] = '0';
        std::memcpy(out + o, digits, nd);
        o += nd;
    } else if (static_cast<std::size_t>(point) >= nd) {
        std::memcpy(out + o, digits, nd);
        o += nd;
        for (std::size_t i = nd; i < static_cast<std::size_t>(point); ++i) out[o++] = '0';
    } else {
        std::memcpy(out + o, digits, static_cast<std::size_t>(point));
        o += static_cast<std::size_t>(point);
        out[o++] = '.';
        std::memcpy(out + o, digits + point, nd - static_cast<std::size_t>(point));
        o += nd - static_cast<std::size_t>(point);
    }
    return o;
}

}  // namespace

bool parse(std::string_view s, float& out) noexcept { return parse_float(s, out); }
bool parse(std::string_view s, double& out) noexcept { return parse_float(s, out); }
bool parse(std::string_view s, std::int32_t& out) noexcept { return parse_int(s, out); }
bool parse(std::string_view s, std::int64_t& out) noexcept { return parse_int(s, out); }
bool parse(std::string_view s, std::uint64_t& out) noexcept { return parse_int(s, out); }

TextWriter& TextWriter::put_int(std::int64_t v) {
    char b[24];
    auto r = std::to_chars(b, b + sizeof b, v);
    return put(std::string_view(b, static_cast<std::size_t>(r.ptr - b)));
}

TextWriter& TextWriter::put_uint(std::uint64_t v) {
    char b[24];
    auto r = std::to_chars(b, b + sizeof b, v);
    return put(std::string_view(b, static_cast<std::size_t>(r.ptr - b)));
}

TextWriter& TextWriter::put_fixed(double v, int prec) {
    char b[512];
    if (std::isnan(v)) return put("NaN");
    if (std::isinf(v)) return put(v < 0 ? "-inf" : "inf");
    auto r = std::to_chars(b, b + sizeof b, v, std::chars_format::fixed, prec);
    return put(std::string_view(b, static_cast<std::size_t>(r.ptr - b)));
}

TextWriter& TextWriter::put_shortest(double v) {
    char b[400];
    return put(std::string_view(b, shortest_plain(v, b)));
}

TextWriter& TextWriter::put_shortest(float v) {
    char b[80];
    return put(std::string_view(b, shortest_plain(v, b)));
}

TextWriter& TextWriter::put_padded(std::string_view s, int width, bool left) {
    const int pad = width - static_cast<int>(s.size());
    if (!left)
        for (int i = 0; i < pad; ++i) buf_.push_back(' ');
    buf_.append(s.data(), s.size());
    if (left)
        for (int i = 0; i < pad; ++i) buf_.push_back(' ');
    maybe_flush();
    return *this;
}

std::string fixed(double v, int prec) {
    if (std::isnan(v)) return "NaN";
    if (std::isinf(v)) return v < 0 ? "-inf" : "inf";
    char b[512];
    auto r = std::to_chars(b, b + sizeof b, v, std::chars_format::fixed, prec);
    return std::string(b, static_cast<std::size_t>(r.ptr - b));
}

std::string shortest(double v) {
    char b[400];
    return std::string(b, shortest_plain(v, b));
}

std::string shortest(float v) {
    char b[80];
    return std::string(b, shortest_plain(v, b));
}

}  // namespace mdz::text
