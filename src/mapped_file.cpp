#include "mapped_file.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>

#include "mdz/error.hpp"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace mdz::detail {
namespace {

[[noreturn]] void io_error(const std::string& path, const char* what) {
    throw Error(ErrorKind::Io, path + ": " + what + (errno ? std::string(": ") + std::strerror(errno) : std::string()));
}

}  // namespace

std::shared_ptr<const MappedFile> MappedFile::from_bytes(std::vector<std::uint8_t> bytes) {
    std::shared_ptr<MappedFile> f(new MappedFile());
    f->owned_ = std::move(bytes);
    f->data_ = f->owned_.data();
    f->size_ = f->owned_.size();
    return f;
}

#if defined(_WIN32)

std::shared_ptr<const MappedFile> MappedFile::open(const std::string& path) {
    int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring wpath(static_cast<std::size_t>(wlen), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);
    HANDLE h = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) throw Error(ErrorKind::Io, path + ": cannot open");
    LARGE_INTEGER sz;
    GetFileSizeEx(h, &sz);
    std::shared_ptr<MappedFile> f(new MappedFile());
    f->size_ = static_cast<std::size_t>(sz.QuadPart);
    if (f->size_ == 0) {
        CloseHandle(h);
        return f;
    }
    HANDLE m = CreateFileMappingW(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
    CloseHandle(h);
    if (!m) throw Error(ErrorKind::Io, path + ": cannot map");
    void* p = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(m);
    if (!p) throw Error(ErrorKind::Io, path + ": cannot map");
    f->data_ = static_cast<const std::uint8_t*>(p);
    f->map_ = p;
    return f;
}

MappedFile::~MappedFile() {
    if (map_) UnmapViewOfFile(map_);
}

#else

std::shared_ptr<const MappedFile> MappedFile::open(const std::string& path) {
    errno = 0;
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) io_error(path, "cannot open");
    struct stat st;
    if (fstat(fd, &st) != 0) {
        ::close(fd);
        io_error(path, "cannot stat");
    }
    std::shared_ptr<MappedFile> f(new MappedFile());
    f->size_ = static_cast<std::size_t>(st.st_size);
    if (f->size_ == 0) {
        ::close(fd);
        return f;
    }
    void* p = mmap(nullptr, f->size_, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
        // Not mappable (a pipe, some network filesystems): read it instead.
        f->owned_.resize(f->size_);
        std::size_t got = 0;
        while (got < f->size_) {
            ssize_t n = ::read(fd, f->owned_.data() + got, f->size_ - got);
            if (n <= 0) {
                ::close(fd);
                io_error(path, "read failed");
            }
            got += static_cast<std::size_t>(n);
        }
        ::close(fd);
        f->data_ = f->owned_.data();
        return f;
    }
    ::close(fd);
    f->data_ = static_cast<const std::uint8_t*>(p);
    f->map_ = p;
    return f;
}

MappedFile::~MappedFile() {
    if (map_) munmap(map_, size_);
}

#endif

}  // namespace mdz::detail
