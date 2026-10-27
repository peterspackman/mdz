// Fuzzes every trajectory format reader with arbitrary input. The first byte
// selects the format.

#include "mdz/mdz.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (size < 1) return 0;
    const auto& fs = mdz::formats();
    const mdz::FormatInfo& f = fs[data[0] % fs.size()];
    try {
        auto r = f.open_reader(mdz::memory_input(std::vector<std::uint8_t>(data + 1, data + size)), {});
        for (int i = 0; i < 64; ++i)
            if (!r->read_batch(7)) break;
    } catch (const mdz::Error&) {
    }
    return 0;
}
