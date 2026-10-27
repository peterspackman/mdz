#include <cerrno>
#include <cstdio>
#include <cstring>

#include "mdz/error.hpp"
#include "file_io.hpp"
#include "mdz/formats.hpp"

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

namespace mdz {

void OutputStream::seek(std::uint64_t) { throw Error(ErrorKind::Io, name() + ": cannot seek in this stream"); }

namespace {

[[noreturn]] void io_fail(const std::string& name, const char* what) {
    const int e = errno;
    throw Error(ErrorKind::Io, name + ": " + what + (e ? std::string(": ") + std::strerror(e) : std::string()));
}

class FileInput final : public InputStream {
public:
    FileInput(std::FILE* f, std::string name, bool owned) : f_(f), name_(std::move(name)), owned_(owned) {}
    ~FileInput() override {
        if (owned_) std::fclose(f_);
    }
    std::size_t read(void* buf, std::size_t n) override {
        std::size_t got = std::fread(buf, 1, n, f_);
        if (got < n && std::ferror(f_)) io_fail(name_, "read failed");
        return got;
    }
    const std::string& name() const override { return name_; }

private:
    std::FILE* f_;
    std::string name_;
    bool owned_;
};

class FileOutput final : public OutputStream {
public:
    FileOutput(std::FILE* f, std::string name, bool owned) : f_(f), name_(std::move(name)), owned_(owned) {
        std::setvbuf(f_, nullptr, _IOFBF, 1 << 20);
    }
    ~FileOutput() override {
        if (f_ && owned_) std::fclose(f_);
        else if (f_) std::fflush(f_);
    }
    void write(const void* d, std::size_t n) override {
        if (n && std::fwrite(d, 1, n, f_) != n) io_fail(name_, "write failed");
    }
    void flush() override {
        if (std::fflush(f_) != 0) io_fail(name_, "write failed");
    }
    void close() override {
        if (!f_) return;
        std::FILE* f = f_;
        f_ = nullptr;
        if (owned_ ? std::fclose(f) != 0 : std::fflush(f) != 0) io_fail(name_, "write failed");
    }
    bool seekable() const override { return owned_; }
    void seek(std::uint64_t off) override {
#if defined(_WIN32)
        if (_fseeki64(f_, static_cast<long long>(off), SEEK_SET) != 0) io_fail(name_, "seek failed");
#else
        if (fseeko(f_, static_cast<off_t>(off), SEEK_SET) != 0) io_fail(name_, "seek failed");
#endif
    }
    const std::string& name() const override { return name_; }

private:
    std::FILE* f_;
    std::string name_;
    bool owned_;
};

class MemoryInput final : public InputStream {
public:
    MemoryInput(std::vector<std::uint8_t> b, std::string name) : b_(std::move(b)), name_(std::move(name)) {}
    std::size_t read(void* buf, std::size_t n) override {
        n = std::min(n, b_.size() - pos_);
        if (n) std::memcpy(buf, b_.data() + pos_, n);
        pos_ += n;
        return n;
    }
    const std::string& name() const override { return name_; }

private:
    std::vector<std::uint8_t> b_;
    std::size_t pos_ = 0;
    std::string name_;
};

class MemoryOutput final : public OutputStream {
public:
    MemoryOutput(std::vector<std::uint8_t>* out, std::string name) : out_(out), name_(std::move(name)) {}
    void write(const void* d, std::size_t n) override {
        if (pos_ == out_->size()) {
            const auto* c = static_cast<const std::uint8_t*>(d);
            out_->insert(out_->end(), c, c + n);
        } else {
            if (pos_ + n > out_->size()) out_->resize(pos_ + n);
            std::memcpy(out_->data() + pos_, d, n);
        }
        pos_ += n;
    }
    void flush() override {}
    void close() override {}
    bool seekable() const override { return true; }
    void seek(std::uint64_t off) override {
        if (off > out_->size()) out_->resize(static_cast<std::size_t>(off));
        pos_ = static_cast<std::size_t>(off);
    }
    const std::string& name() const override { return name_; }

private:
    std::vector<std::uint8_t>* out_;
    std::size_t pos_ = 0;
    std::string name_;
};

}  // namespace

std::unique_ptr<InputStream> open_input(const std::string& path) {
    if (path == "-") {
#if defined(_WIN32)
        _setmode(_fileno(stdin), _O_BINARY);
#endif
        return std::make_unique<FileInput>(stdin, "<stdin>", false);
    }
    errno = 0;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) io_fail(path, "cannot open");
    std::setvbuf(f, nullptr, _IOFBF, 1 << 20);
    return std::make_unique<FileInput>(f, path, true);
}

std::unique_ptr<OutputStream> open_output(const std::string& path) {
    if (path == "-") {
#if defined(_WIN32)
        _setmode(_fileno(stdout), _O_BINARY);
#endif
        return std::make_unique<FileOutput>(stdout, "<stdout>", false);
    }
    errno = 0;
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) io_fail(path, "cannot open for writing");
    return std::make_unique<FileOutput>(f, path, true);
}

namespace detail {

// Truncate an existing file to `offset` bytes and write on from there.
std::unique_ptr<OutputStream> open_output_at(const std::string& path, std::uint64_t offset) {
    resize_file(path, offset);
    errno = 0;
    std::FILE* f = std::fopen(path.c_str(), "r+b");
    if (!f) io_fail(path, "cannot open for writing");
    auto out = std::make_unique<FileOutput>(f, path, true);
    out->seek(offset);
    return out;
}

}  // namespace detail

std::unique_ptr<InputStream> memory_input(std::vector<std::uint8_t> bytes, std::string name) {
    return std::make_unique<MemoryInput>(std::move(bytes), std::move(name));
}

std::unique_ptr<OutputStream> memory_output(std::vector<std::uint8_t>* out, std::string name) {
    return std::make_unique<MemoryOutput>(out, std::move(name));
}

}  // namespace mdz
