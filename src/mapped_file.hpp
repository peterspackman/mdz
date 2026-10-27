#pragma once
// A read-only view of a whole file: memory-mapped where the OS allows, read
// into memory otherwise.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "mdz/span.hpp"

namespace mdz::detail {

class MappedFile {
public:
    // Throws Error(Io).
    static std::shared_ptr<const MappedFile> open(const std::string& path);
    static std::shared_ptr<const MappedFile> from_bytes(std::vector<std::uint8_t> bytes);
    ~MappedFile();
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    ByteSpan bytes() const noexcept { return {data_, size_}; }

private:
    MappedFile() = default;
    const std::uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
    void* map_ = nullptr;  // platform mapping handle, if mapped
    std::vector<std::uint8_t> owned_;
};

}  // namespace mdz::detail
