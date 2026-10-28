#pragma once
// Conversions between UTF-8 std::string paths and std::filesystem::path.
// fs::u8path is deprecated in C++20, and path::u8string returns
// std::u8string there, so both are wrapped to build as C++17 or later.

#include <filesystem>
#include <string>

namespace cli {

inline std::filesystem::path utf8_path(const std::string& s) {
#if defined(__cpp_lib_char8_t)
    return std::filesystem::path(std::u8string(s.begin(), s.end()));
#else
    return std::filesystem::u8path(s);
#endif
}

inline std::string utf8_string(const std::filesystem::path& p) {
#if defined(__cpp_lib_char8_t)
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
#else
    return p.u8string();
#endif
}

}  // namespace cli
