#pragma once
// Text I/O helpers for the trajectory formats. LineReader returns lines as
// string_views into a large read buffer. Numbers are parsed with fast_float and
// formatted with std::to_chars into a buffered writer. None of this allocates
// per line or per number.

#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

#include <fast_float/fast_float.h>

#include "mdz/error.hpp"
#include "mdz/formats.hpp"

namespace mdz::text {

// Reads lines from a stream and strips the "\n" or "\r\n" terminator. A
// returned view stays valid until the next call to next() or peek().
class LineReader {
public:
    explicit LineReader(InputStream& in, std::size_t chunk = 1 << 20);
    // Returns false at end of input.
    bool next(std::string_view& line);
    // Returns the next line without consuming it.
    bool peek(std::string_view& line);
    // 1-based number of the line last returned by next(), for error messages.
    std::size_t line_number() const noexcept { return lineno_; }
    const std::string& name() const noexcept { return in_.name(); }

private:
    bool fill();
    InputStream& in_;
    std::vector<char> buf_;
    std::size_t begin_ = 0, end_ = 0;
    std::size_t chunk_;
    bool eof_ = false;
    std::size_t lineno_ = 0;
    bool peeked_ = false;
    std::string_view peek_line_;
};

namespace detail {
constexpr std::array<bool, 256> space_table() {
    std::array<bool, 256> t{};
    for (unsigned char c : {' ', '\t', '\r', '\n', '\v', '\f'}) t[c] = true;
    return t;
}
inline constexpr std::array<bool, 256> kSpace = space_table();
}  // namespace detail

inline bool is_space(char c) noexcept { return detail::kSpace[static_cast<unsigned char>(c)]; }

inline std::string_view trim(std::string_view s) noexcept {
    std::size_t a = 0, b = s.size();
    while (a < b && is_space(s[a])) ++a;
    while (b > a && is_space(s[b - 1])) --b;
    return s.substr(a, b - a);
}

// Returns the next whitespace-separated token of s and advances s past it.
// Returns an empty view when s has no more tokens.
inline std::string_view next_token(std::string_view& s) noexcept {
    std::size_t i = 0;
    while (i < s.size() && is_space(s[i])) ++i;
    std::size_t j = i;
    while (j < s.size() && !is_space(s[j])) ++j;
    std::string_view tok = s.substr(i, j - i);
    s.remove_prefix(j);
    return tok;
}

inline bool starts_with(std::string_view s, std::string_view p) noexcept { return s.substr(0, p.size()) == p; }

// Parse a whole token. A single leading '+' is accepted, but "+-" is not.
// Float parses also accept "inf" and "nan", and out-of-range values become
// +-inf or +-0. Returns false on anything else, including trailing characters.
bool parse(std::string_view s, float& out) noexcept;
bool parse(std::string_view s, double& out) noexcept;
bool parse(std::string_view s, std::int32_t& out) noexcept;
bool parse(std::string_view s, std::int64_t& out) noexcept;
bool parse(std::string_view s, std::uint64_t& out) noexcept;

// Parses the number starting at p, in place, for the per-atom loops that
// would otherwise split each line into tokens first. The same values as
// parse() are accepted. Returns the end of the token, or nullptr if the token
// at p is not exactly one number.
template <class T>
const char* scan_number(const char* p, const char* e, T& v) noexcept {
    if (*p == '+') {
        ++p;
        if (p == e || *p == '-') return nullptr;
    }
    const char* end;
    if constexpr (std::is_floating_point_v<T>) {
        auto r = fast_float::from_chars(p, e, v);
        if (r.ec != std::errc() && r.ec != std::errc::result_out_of_range) return nullptr;
        end = r.ptr;
    } else {
        auto r = std::from_chars(p, e, v);
        if (r.ec != std::errc()) return nullptr;
        end = r.ptr;
    }
    if (end != e && !is_space(*end)) return nullptr;
    return end;
}

// Returns the end of the token starting at p.
inline const char* token_end(const char* p, const char* e) noexcept {
    while (p != e && !is_space(*p)) ++p;
    return p;
}

// As parse(), but throws Error(Corrupt) on failure. The message names `what`
// and, if `where` is given, the file and line number.
template <class T>
T parse_or_throw(std::string_view tok, const char* what, const LineReader* where = nullptr) {
    T v;
    if (!parse(tok, v)) {
        std::string msg = "bad " + std::string(what) + " '" + std::string(tok) + "'";
        if (where) msg = where->name() + ":" + std::to_string(where->line_number()) + ": " + msg;
        throw Error(ErrorKind::Corrupt, msg);
    }
    return v;
}

// Buffered text output.
class TextWriter {
public:
    explicit TextWriter(OutputStream& out, std::size_t flush_at = 1 << 20) : out_(out), flush_at_(flush_at) {
        buf_.reserve(flush_at + 4096);
    }
    ~TextWriter() = default;

    TextWriter& put(std::string_view s) {
        buf_.append(s.data(), s.size());
        maybe_flush();
        return *this;
    }
    TextWriter& put(char c) {
        buf_.push_back(c);
        return *this;
    }
    TextWriter& put_int(std::int64_t v);
    TextWriter& put_uint(std::uint64_t v);
    // Fixed notation with `prec` digits after the point, correctly rounded.
    // Matches printf("%.*f", prec, v).
    TextWriter& put_fixed(double v, int prec);
    // The shortest digits that round-trip, never in exponent notation. For
    // example, 1 gives "1", 0.1 gives "0.1" and 1e-7 gives "0.0000001".
    TextWriter& put_shortest(double v);
    TextWriter& put_shortest(float v);
    // Pads s with spaces to `width` columns, right-aligned unless `left`.
    TextWriter& put_padded(std::string_view s, int width, bool left = false);

    void flush() {
        if (!buf_.empty()) out_.write(buf_.data(), buf_.size());
        buf_.clear();
    }

private:
    void maybe_flush() {
        if (buf_.size() >= flush_at_) flush();
    }
    OutputStream& out_;
    std::size_t flush_at_;
    std::string buf_;
};

// Same formatting as TextWriter, returned as a std::string.
std::string fixed(double v, int prec);
std::string shortest(double v);
std::string shortest(float v);

}  // namespace mdz::text
