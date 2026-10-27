// Measures size, error and speed of mdz on real trajectories, alongside XTC
// and zstd over the raw floats. Frames are loaded once. Compressed output is
// written to a scratch directory and read back through mmap, so timings
// measure the codecs rather than disk I/O. Peak memory is the trajectory, one
// decode buffer and one compressed copy.
//
//     bench_files [--threads 1,16] [--repeat 3] [--tmp DIR] [--bodies-from COL]
//                 [--lammps-units STYLE] traj.dcd [more...]
//
// --bodies-from makes the mdz-rigid method take its rigid bodies from a
// per-atom integer column, such as the mol column of a LAMMPS dump, instead of
// detecting them. A value of 0 means the atom is in no body.
//
// Output is one JSON object per (file, method, threads) on stdout.
// bench/benchmark.py runs this program and tabulates the results.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "mdz/mdz.hpp"
#include "zstd.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double secs(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

// Minimum time over `repeat` runs of f, which is the least affected by
// other activity on the machine.
template <class F>
double best_of(int repeat, F&& f) {
    double best = 1e300;
    for (int i = 0; i < repeat; ++i) {
        auto t = Clock::now();
        f();
        best = std::min(best, secs(t));
    }
    return best;
}

double max_err_angstrom(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, double(std::fabs(a[i] - b[i])));
    return m * 10.0;
}

struct Method {
    const char* name;
    mdz::CompressOptions opts;
};

std::vector<Method> methods() {
    std::vector<Method> m;
    auto with = [](auto f) {
        mdz::CompressOptions o;
        f(o);
        return o;
    };
    // The CLI defaults: at most 0.005 A error (comparable to XTC), with the
    // codec chosen per group.
    m.push_back({"mdz", {}});
    m.push_back({"mdz-temporal", with([](auto& o) { o.codec = mdz::CodecMode::Temporal; })});
    m.push_back({"mdz-intra", with([](auto& o) { o.codec = mdz::CodecMode::Intra; })});
    m.push_back({"mdz-lossless", with([](auto& o) { o.lossless = true; })});
    // Rigid bodies detected from the frames (water molecules in the test data).
    m.push_back({"mdz-rigid", with([](auto& o) { o.rigid = true; })});
    // Equivalent to --max-error 0.001 A and 0.05 A, tighter and looser than XTC.
    m.push_back({"mdz-0.001A", with([](auto& o) { o.precision = 0.0001f / 4; })});
    m.push_back({"mdz-0.05A", with([](auto& o) { o.precision = 0.005f / 4; })});
    return m;
}

void emit(const std::string& file, const char* method, unsigned threads, std::size_t natoms, std::size_t nframes,
          std::size_t bytes, double enc, double dec, double random_us, double seq_us, double err) {
    std::printf(
        "{\"file\": \"%s\", \"method\": \"%s\", \"threads\": %u, \"natoms\": %zu, \"nframes\": %zu, "
        "\"raw_bytes\": %zu, \"bytes\": %zu, \"compress_s\": %.6f, \"decompress_s\": %.6f, "
        "\"random_us\": %.2f, \"sequential_us\": %.2f, \"max_error_A\": %.6g}\n",
        file.c_str(), method, threads, natoms, nframes, natoms * nframes * 12, bytes, enc, dec, random_us, seq_us,
        err);
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<unsigned> thread_counts = {1, std::max(1u, mdz::num_threads())};
    int repeat = 3;
    std::string tmp = std::filesystem::temp_directory_path().string();
    std::string bodies_from;
    mdz::FormatOptions fopts;
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--threads" && i + 1 < argc) {
            thread_counts.clear();
            for (char* t = std::strtok(argv[++i], ","); t; t = std::strtok(nullptr, ","))
                thread_counts.push_back(static_cast<unsigned>(std::atoi(t)));
        } else if (a == "--repeat" && i + 1 < argc) {
            repeat = std::atoi(argv[++i]);
        } else if (a == "--tmp" && i + 1 < argc) {
            tmp = argv[++i];
        } else if (a == "--bodies-from" && i + 1 < argc) {
            bodies_from = argv[++i];
        } else if (a == "--lammps-units" && i + 1 < argc) {
            fopts.lammps_units = mdz::LammpsUnits::from_style(argv[++i]);
        } else {
            files.push_back(a);
        }
    }

    for (const auto& file : files) {
        // Convert to internal units, which is what both the mdz reader and the
        // XTC reader return.
        mdz::FrameBatch b = mdz::open_reader(file, {}, fopts)->read_all()->in_internal_units();
        b.atom_names.reset();  // keep only coordinates and box, as XTC stores
        b.frame_props.clear();
        b.int_frame_props.clear();
        if (!bodies_from.empty()) {
            auto it = b.int_columns.find(bodies_from);
            if (it == b.int_columns.end()) {
                std::fprintf(stderr, "%s: no integer column '%s'\n", file.c_str(), bodies_from.c_str());
                return 1;
            }
            std::vector<std::int32_t> bodies(it->second.begin(), it->second.begin() + std::ptrdiff_t(b.natoms));
            for (auto& x : bodies) x = x > 0 ? x : -1;
            b.bodies = mdz::share(std::move(bodies));
        }
        b.int_columns.clear();
        const std::size_t natoms = b.natoms, nframes = b.nframes, n3 = natoms * 3;
        const std::vector<float>& pos = *b.positions();
        std::fprintf(stderr, "%s: %zu atoms x %zu frames\n", file.c_str(), natoms, nframes);

        std::mt19937 rng(42);
        std::vector<std::size_t> probes(std::min<std::size_t>(300, nframes));
        for (auto& p : probes) p = rng() % nframes;
        const std::string scratch = tmp + "/bench_files.mdz";
        std::vector<float> back(pos.size());

        for (const auto& m : methods()) {
            for (unsigned threads : thread_counts) {
                // Codec choice does not depend on the thread count, so only
                // the default and lossless methods are run multi-threaded.
                if (threads != thread_counts.front() && std::strcmp(m.name, "mdz") != 0 &&
                    std::strcmp(m.name, "mdz-lossless") != 0)
                    continue;
                mdz::set_num_threads(threads);
                std::size_t size = 0;
                const double enc = best_of(repeat, [&] {
                    auto bytes = mdz::to_bytes(mdz::compress(b, m.opts));
                    size = bytes.size();
                    std::ofstream(scratch, std::ios::binary)
                        .write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
                });
                double dec, random_us, seq_us;
                {
                    auto r = mdz::Reader::open(scratch);
                    dec = best_of(repeat, [&] { r.read_positions(0, nframes, back.data()); });
                }
                const double err = max_err_angstrom(pos, back);
                std::vector<float> frame(n3);
                {
                    // Random access to single frames, using a fresh reader so
                    // that no decoded group is cached.
                    auto r = mdz::Reader::open(scratch);
                    auto t = Clock::now();
                    for (auto p : probes) r.read_positions(p, 1, frame.data());
                    random_us = secs(t) * 1e6 / double(probes.size());
                }
                {
                    // Frame-by-frame iteration, as a typical analysis loop reads.
                    auto r = mdz::Reader::open(scratch);
                    auto t = Clock::now();
                    for (std::size_t f = 0; f < nframes; ++f) r.read_positions(f, 1, frame.data());
                    seq_us = secs(t) * 1e6 / double(nframes);
                }
                emit(file, m.name, threads, natoms, nframes, size, enc, dec, random_us, seq_us, err);
            }
        }
        mdz::set_num_threads(0);
        std::filesystem::remove(scratch);

        // XTC through mdz's own XTC writer and reader, which implement the
        // GROMACS algorithm at its standard 0.001 nm precision. The format is
        // inherently single-threaded.
        {
            std::vector<std::uint8_t> xtc;
            const double enc = best_of(repeat, [&] {
                xtc.clear();
                auto w = mdz::find_format("xtc")->open_writer(mdz::memory_output(&xtc), natoms, {});
                w->write_batch(b);
                w->finish();
            });
            const double dec = best_of(repeat, [&] {
                auto rd = mdz::find_format("xtc")->open_reader(mdz::memory_input(xtc), {});
                std::size_t at = 0;
                while (auto chunk = rd->read_batch(100)) {
                    const auto& p = *chunk->positions();
                    std::memcpy(back.data() + at, p.data(), p.size() * sizeof(float));
                    at += p.size();
                }
            });
            emit(file, "xtc", 1, natoms, nframes, xtc.size(), enc, dec, -1, -1, max_err_angstrom(pos, back));
        }
        // zstd level 3 over the raw float32 positions, as a baseline for
        // lossless general-purpose compression.
        {
            mdz::ByteSpan raw(reinterpret_cast<const std::uint8_t*>(pos.data()), pos.size() * 4);
            std::vector<float>().swap(back);  // free the decode buffer
            std::vector<std::uint8_t> z, out;
            const double enc = best_of(repeat, [&] { mdz::detail::zstd_compress(raw, 3, z); });
            const double dec = best_of(repeat, [&] { mdz::detail::zstd_decompress_bounded(z, raw.size(), out); });
            emit(file, "zstd-raw", 1, natoms, nframes, z.size(), enc, dec, -1, -1, 0.0);
        }
    }
    return 0;
}
