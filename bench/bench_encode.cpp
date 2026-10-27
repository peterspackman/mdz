// Compression and decompression benchmark for every codec. The input is either
// a synthetic trajectory (smooth drift plus thermal noise, which compresses
// much like real MD) or a raw float32 file.
//
//     ./bench_encode                 # NATOMS, NFRAMES, GOF_SIZE, THREADS
//     ./bench_encode traj.f32 769    # raw float32 dump and its atom count

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "mdz/mdz.hpp"

namespace {

struct Rng {
    std::uint64_t s;
    float next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<float>(s >> 40) / static_cast<float>(1u << 24) - 0.5f;
    }
};

std::vector<float> make_traj(std::size_t natoms, std::size_t nframes) {
    Rng rng{0x12345678};
    const std::size_t n3 = natoms * 3;
    std::vector<float> data(nframes * n3), pos(n3), vel(n3);
    for (std::size_t i = 0; i < n3; ++i) {
        pos[i] = rng.next() * 5.0f;
        vel[i] = rng.next() * 0.002f;
    }
    for (std::size_t f = 0; f < nframes; ++f)
        for (std::size_t i = 0; i < n3; ++i) {
            pos[i] += vel[i] + rng.next() * 0.0004f;
            data[f * n3 + i] = pos[i];
        }
    return data;
}

std::size_t env(const char* name, std::size_t def) {
    const char* v = std::getenv(name);
    return v ? std::strtoull(v, nullptr, 10) : def;
}

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<float> data;
    std::size_t natoms;
    if (argc < 2) {
        natoms = env("NATOMS", 20000);
        const std::size_t nframes = env("NFRAMES", 200);
        std::printf("synthetic: %zu atoms x %zu frames\n", natoms, nframes);
        data = make_traj(natoms, nframes);
    } else {
        if (argc < 3) {
            std::fprintf(stderr, "a raw float32 file needs an atom count as the second argument\n");
            return 1;
        }
        natoms = std::strtoull(argv[2], nullptr, 10);
        std::ifstream f(argv[1], std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        data.resize(bytes.size() / 4);
        std::memcpy(data.data(), bytes.data(), data.size() * 4);
    }
    if (std::size_t t = env("THREADS", 0)) mdz::set_num_threads(static_cast<unsigned>(t));

    const std::size_t n3 = natoms * 3, nframes = data.size() / n3;
    const double raw_mb = static_cast<double>(data.size() * 4) / 1e6;
    const std::size_t gof = env("GOF_SIZE", 20);
    std::printf("%zu atoms x %zu frames = %.2f MB raw, group of %zu, %u thread(s)\n\n", natoms, nframes, raw_mb, gof,
                mdz::num_threads());
    std::printf("%-10s %8s %11s %12s %11s %12s\n", "codec", "ratio", "compress", "decompress", "random", "max error");
    std::printf("%-10s %8s %11s %12s %11s %12s\n", "", "", "MB/s", "MB/s", "us/frame", "nm");

    const std::string tmp = (std::filesystem::temp_directory_path() / "mdz_bench_cpp.mdz").string();
    const std::pair<const char*, mdz::CodecMode> modes[] = {{"auto", mdz::CodecMode::Auto},
                                                            {"temporal", mdz::CodecMode::Temporal},
                                                            {"intra", mdz::CodecMode::Intra},
                                                            {"lossless", mdz::CodecMode::Lossless}};
    for (const auto& [name, mode] : modes) {
        mdz::CompressOptions o;
        o.precision = 0.000125f;
        o.gof_size = static_cast<std::uint32_t>(gof);
        o.codec = mode;
        o.lossless = mode == mdz::CodecMode::Lossless;

        auto t = std::chrono::steady_clock::now();
        auto ct = mdz::compress_positions(data.data(), natoms, nframes, nullptr, o);
        const double enc = seconds_since(t);
        const std::size_t size = ct.compressed_size();
        mdz::save(tmp, ct);
        auto reader = mdz::Reader::open(tmp);

        std::vector<float> back(data.size());
        t = std::chrono::steady_clock::now();
        reader.read_positions(0, nframes, back.data());
        const double dec = seconds_since(t);
        float max_err = 0;
        for (std::size_t i = 0; i < data.size(); ++i) max_err = std::max(max_err, std::fabs(data[i] - back[i]));

        const int probes = 2000;
        std::uint32_t seed = 12345;
        std::vector<float> frame(n3);
        t = std::chrono::steady_clock::now();
        for (int k = 0; k < probes; ++k) {
            seed = seed * 1664525u + 1013904223u;
            reader.read_positions((seed >> 8) % nframes, 1, frame.data());
        }
        const double random_us = seconds_since(t) * 1e6 / probes;

        std::printf("%-10s %7.1fx %11.1f %12.1f %11.1f %12.2e\n", name, raw_mb / (static_cast<double>(size) / 1e6),
                    raw_mb / enc, raw_mb / dec, random_us, static_cast<double>(max_err));
    }
    std::filesystem::remove(tmp);
    std::printf("\ncompress and decompress are the codec alone, in memory. random is %d reads at\n"
                "uniformly random frames, so nearly every one decodes a whole group.\n",
                2000);
    return 0;
}
