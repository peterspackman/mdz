#pragma once
// Buffered binary output to a file. Write, flush and close failures throw
// Error(Io). flush() hands buffered bytes to the OS but does not fsync.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "mdz/span.hpp"

namespace mdz::detail {

class OutFile {
public:
    // mode is "wb" to create or truncate, or "r+b" to update in place.
    // Throws Error(Io).
    OutFile(const std::string& path, const char* mode);
    ~OutFile();
    OutFile(const OutFile&) = delete;
    OutFile& operator=(const OutFile&) = delete;

    void write(const void* data, std::size_t n);
    void write(ByteSpan b) { write(b.data(), b.size()); }
    void seek(std::uint64_t off);
    void flush();
    // Close the file. The destructor also closes it but ignores errors.
    void close();
    const std::string& path() const noexcept { return path_; }

private:
    std::string path_;
    std::FILE* f_ = nullptr;
};

// Shrink or grow a file to `size` bytes. Throws Error(Io).
void resize_file(const std::string& path, std::uint64_t size);

}  // namespace mdz::detail
