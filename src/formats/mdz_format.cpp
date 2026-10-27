// Exposes .mdz through the same reader/writer interface as the other formats,
// so conversion code does not need to special-case it.

#include <algorithm>

#include "internal.hpp"
#include "mdz/reader.hpp"
#include "mdz/writer.hpp"

namespace mdz::detail {
namespace {

class MdzFormatReader final : public TrajectoryReader {
public:
    explicit MdzFormatReader(Reader r) : r_(std::move(r)) {}
    std::size_t natoms() const override { return r_.natoms(); }
    std::optional<std::size_t> nframes() const override { return r_.nframes(); }
    std::optional<FrameBatch> read_batch(std::size_t n) override {
        if (next_ >= r_.nframes()) return std::nullopt;
        const std::size_t count = std::min(n, r_.nframes() - next_);
        FrameBatch b = r_.read_batch(next_, count);
        next_ += count;
        return b;
    }

private:
    Reader r_;
    std::size_t next_ = 0;
};

class MdzFormatWriter final : public TrajectoryWriter {
public:
    explicit MdzFormatWriter(Writer w) : w_(std::move(w)) {}
    void write_batch(const FrameBatch& b) override { w_.write(b); }
    void finish() override { w_.close(); }

private:
    Writer w_;
};

}  // namespace

std::unique_ptr<TrajectoryReader> make_mdz_reader(std::unique_ptr<InputStream> in, const FormatOptions&) {
    std::vector<std::uint8_t> bytes;
    std::vector<std::uint8_t> chunk(1 << 20);
    while (std::size_t n = in->read(chunk.data(), chunk.size())) bytes.insert(bytes.end(), chunk.begin(), chunk.begin() + static_cast<std::ptrdiff_t>(n));
    return std::make_unique<MdzFormatReader>(Reader::from_bytes(std::move(bytes)));
}

std::unique_ptr<TrajectoryReader> make_mdz_reader_path(const std::string& path) {
    return std::make_unique<MdzFormatReader>(Reader::open(path));
}

std::unique_ptr<TrajectoryWriter> make_mdz_writer(std::unique_ptr<OutputStream> out, std::size_t,
                                                  const FormatOptions& opts) {
    return std::make_unique<MdzFormatWriter>(Writer::create(std::move(out), opts.compress));
}

}  // namespace mdz::detail
