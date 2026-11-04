// Fuzzes whole .mdz files. Each input is parsed, decoded through the batch,
// random-access and lazy paths, and then opened for appending.

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>

#include "mdz/mdz.hpp"

// Decoding is bounded by the size the header declares. A small file can
// legitimately declare a large trajectory, because a static structure over
// many frames compresses to almost nothing. Such inputs are skipped because
// decoding them is slow and would exceed the fuzzer's time and memory limits.
constexpr std::size_t kMaxDeclaredValues = 1 << 22;

// Open the file in byte ranges, as LazyReader's callers do, and decode each
// group.
void fuzz_lazy(const std::uint8_t* data, std::size_t size) {
    auto range = [&](std::uint64_t lo, std::uint64_t hi) {
        hi = std::min<std::uint64_t>(hi, size);
        lo = std::min(lo, hi);
        return mdz::ByteSpan(data + lo, static_cast<std::size_t>(hi - lo));
    };
    try {
        auto r = mdz::LazyReader::open(range(0, mdz::LazyReader::head_size(range(0, 24))), size);
        bool closed = false;
        if (auto f = r.footer_range(range(size >= 16 ? size - 16 : 0, size)))
            closed = r.read_footer(f->first, range(f->first, f->second));
        if (!closed)
            while (std::uint64_t n = r.record_size(range(r.scan_offset(), r.scan_offset() + 8)))
                if (!r.add_record(range(r.scan_offset(), r.scan_offset() + n))) break;
        for (std::size_t gi = 0; gi < r.groups().size(); ++gi) {
            const auto& g = r.groups()[gi];
            if (std::max<std::size_t>(r.natoms(), 1) * g.nframes * 9 > kMaxDeclaredValues) continue;
            try {
                r.decode_group(gi, range(g.offset, g.offset + g.size));
            } catch (const mdz::Error&) {
            }
        }
    } catch (const mdz::Error&) {
    }
}

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
        fuzz_lazy(data, size);
        return 0;
    }
    fuzz_lazy(data, size);
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
