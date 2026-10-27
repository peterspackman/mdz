// Writes a seed corpus for the fuzz targets. The seeds are valid inputs from
// every codec, which gives the fuzzer realistic starting points to mutate.
//
//     gen_corpus <dir>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>

#include "mdz/mdz.hpp"
#include "zstd.hpp"

namespace fs = std::filesystem;

namespace {

mdz::FrameBatch batch(std::size_t natoms, std::size_t nframes) {
    mdz::FrameBatch b(natoms, nframes);
    auto& p = b.columns["positions"];
    auto& v = b.columns["velocities"];
    for (std::size_t i = 0; i < natoms * nframes * 3; ++i) {
        p.push_back(std::sin(float(i) * 0.37f) * 2.0f + float(i / (natoms * 3)) * 0.001f);
        v.push_back(std::cos(float(i) * 0.11f));
    }
    auto& q = b.columns["charge"];
    for (std::size_t i = 0; i < natoms * nframes; ++i) q.push_back(float(i) * 0.01f);
    b.box.emplace(nframes, mdz::Box{3, 0, 0, 0, 3, 0, 0, 0, 3});
    for (std::size_t f = 0; f < nframes; ++f) {
        b.frame_props["time"].push_back(double(f));
        b.int_frame_props["step"].push_back(std::int64_t(f) * 10);
    }
    std::vector<std::string> names;
    std::vector<std::int32_t> ids;
    for (std::size_t i = 0; i < natoms; ++i) {
        names.push_back("A" + std::to_string(i));
        ids.push_back(std::int32_t(i + 1));
    }
    b.atom_names = mdz::share(std::move(names));
    b.atom_ids = mdz::share(std::move(ids));
    return b;
}

void put(const fs::path& p, const std::vector<std::uint8_t>& b) {
    fs::create_directories(p.parent_path());
    std::ofstream(p, std::ios::binary).write(reinterpret_cast<const char*>(b.data()), std::streamsize(b.size()));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    const fs::path root = argv[1];
    int k = 0;
    for (std::size_t natoms : {3, 12}) {
        for (auto codec : {mdz::CodecMode::Temporal, mdz::CodecMode::Intra, mdz::CodecMode::Lossless}) {
            for (unsigned anchor : {1u, 3u}) {
                mdz::CompressOptions o;
                o.gof_size = 4;
                o.codec = codec;
                o.lossless = codec == mdz::CodecMode::Lossless;
                o.anchor_step = o.lossless ? 1 : anchor;
                auto ct = mdz::compress(batch(natoms, 9), o);
                put(root / "mdz_file" / ("seed" + std::to_string(k)), mdz::to_bytes(ct));
                // Seed for fuzz_gof_payload: the decompressed payload of the
                // first group, prefixed by its atom and frame counts.
                std::vector<std::uint8_t> raw;
                mdz::detail::zstd_decompress_bounded(ct.columns[0].blob(0), 1 << 20, raw);
                raw.insert(raw.begin(), {std::uint8_t(natoms), std::uint8_t(ct.gofs.span(0).size() - 1)});
                put(root / "gof_payload" / ("seed" + std::to_string(k)), raw);
                if (ct.metadata) put(root / "metadata" / ("seed" + std::to_string(k)), *ct.metadata);
                ++k;
            }
        }
    }
    return 0;
}
