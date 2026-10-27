#include "file_io.hpp"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <system_error>

#include "mdz/error.hpp"

namespace mdz::detail {
namespace {
[[noreturn]] void fail(const std::string& path, const char* what) {
    const int e = errno;
    throw Error(ErrorKind::Io, path + ": " + what + (e ? std::string(": ") + std::strerror(e) : std::string()));
}
}  // namespace

OutFile::OutFile(const std::string& path, const char* mode) : path_(path) {
    errno = 0;
    f_ = std::fopen(path.c_str(), mode);
    if (!f_) fail(path, "cannot open for writing");
    std::setvbuf(f_, nullptr, _IOFBF, 1 << 20);
}

OutFile::~OutFile() {
    if (f_) std::fclose(f_);
}

void OutFile::write(const void* data, std::size_t n) {
    if (n && std::fwrite(data, 1, n, f_) != n) fail(path_, "write failed");
}

void OutFile::seek(std::uint64_t off) {
#if defined(_WIN32)
    if (_fseeki64(f_, static_cast<long long>(off), SEEK_SET) != 0) fail(path_, "seek failed");
#else
    if (fseeko(f_, static_cast<off_t>(off), SEEK_SET) != 0) fail(path_, "seek failed");
#endif
}

void OutFile::flush() {
    if (std::fflush(f_) != 0) fail(path_, "write failed");
}

void OutFile::close() {
    if (!f_) return;
    std::FILE* f = f_;
    f_ = nullptr;
    if (std::fclose(f) != 0) fail(path_, "write failed");
}

void resize_file(const std::string& path, std::uint64_t size) {
    std::error_code ec;
    std::filesystem::resize_file(std::filesystem::u8path(path), size, ec);
    if (ec) {
        std::string msg = path + ": cannot resize: " + ec.message();
#ifdef _WIN32
        // ERROR_USER_MAPPED_FILE: Windows will not shorten a file that is
        // memory-mapped, which every open mdz reader of it is.
        if (ec.value() == 1224) msg += " (close every program and mdz reader that has the file open, then try again)";
#endif
        throw Error(ErrorKind::Io, msg);
    }
}

}  // namespace mdz::detail
