// Fuzzes whole .mdz files. Each input is parsed, decoded through both the
// batch and the random-access paths, and then opened for appending.

#include <cstdio>
#include <filesystem>
#include <fstream>

#include "mdz/mdz.hpp"

// Decoding is bounded by the size the header declares. A small file can
// legitimately declare a large trajectory, because a static structure over
// many frames compresses to almost nothing. Such inputs are skipped because
// decoding them is slow and would exceed the fuzzer's time and memory limits.
constexpr std::size_t kMaxDeclaredValues = 1 << 22;

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    std::vector<std::uint8_t> bytes(data, data + size);
    try {
        auto ct = mdz::load_bytes(bytes);
        if (std::max<std::size_t>(ct.natoms, 1) * ct.nframes * 9 > kMaxDeclaredValues) return 0;
        try {
            mdz::decompress(ct);
        } catch (const mdz::Error&) {
        }
        try {
            mdz::decompress_positions(ct);
        } catch (const mdz::Error&) {
        }
        try {
            mdz::decompress(ct, 1, ct.nframes > 1 ? ct.nframes - 1 : 0);
        } catch (const mdz::Error&) {
        }
    } catch (const mdz::Error&) {
        return 0;
    }
    try {
        auto r = mdz::Reader::from_bytes(bytes);
        for (std::size_t f = 0; f < r.nframes(); f += 3) {
            try {
                r.read_frame(f);
            } catch (const mdz::Error&) {
            }
        }
        try {
            r.read_positions(r.nframes() / 2, r.nframes() - r.nframes() / 2);
        } catch (const mdz::Error&) {
        }
    } catch (const mdz::Error&) {
    }
    // Opening for append also parses the file. It truncates after the last
    // valid record and writes a new footer, so read the result back as well.
    const auto path = std::filesystem::temp_directory_path() / "mdz_fuzz_append.mdz";
    {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    }
    try {
        auto w = mdz::Writer::append(path.string());
        w.close();
        mdz::load(path.string());
    } catch (const mdz::Error&) {
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return 0;
}
