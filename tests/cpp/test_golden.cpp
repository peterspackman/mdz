// Golden-file tests. Each case in tests/cpp/golden holds an input batch, the
// options used to compress it, and the expected .mdz bytes. Compressing the
// input must reproduce those bytes exactly, and decoding them must match the
// recorded hash.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

#include "mdz/mdz.hpp"
#include "test.hpp"
#include "zstd.hpp"

namespace fs = std::filesystem;

namespace {

std::vector<std::uint8_t> read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

template <class T>
std::vector<T> read_array(const fs::path& p) {
    auto b = read_file(p);
    std::vector<T> v(b.size() / sizeof(T));
    if (!v.empty()) std::memcpy(v.data(), b.data(), v.size() * sizeof(T));
    return v;
}

std::map<std::string, std::string> read_meta(const fs::path& p) {
    std::map<std::string, std::string> m;
    std::ifstream f(p);
    std::string line;
    while (std::getline(f, line)) {
        auto eq = line.find('=');
        if (eq != std::string::npos) m[line.substr(0, eq)] = line.substr(eq + 1);
    }
    return m;
}

std::uint64_t fnv1a(const std::vector<float>& v) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    const auto* p = reinterpret_cast<const std::uint8_t*>(v.data());
    for (std::size_t i = 0; i < v.size() * 4; ++i) h = (h ^ p[i]) * 0x100000001b3ull;
    return h;
}

struct Case {
    mdz::FrameBatch batch;
    mdz::CompressOptions opts;
    std::map<std::string, std::string> meta;
};

Case load_case(const fs::path& dir) {
    Case c;
    c.meta = read_meta(dir / "meta.txt");
    auto& m = c.meta;
    c.batch.natoms = std::stoul(m["natoms"]);
    c.batch.nframes = std::stoul(m["nframes"]);
    auto& o = c.opts;
    o.gof_size = static_cast<std::uint32_t>(std::stoul(m["gof_size"]));
    o.precision = std::strtof(m["precision"].c_str(), nullptr);
    o.predictor_order = static_cast<unsigned>(std::stoul(m["predictor_order"]));
    o.zstd_level = std::stoi(m["zstd_level"]);
    o.adaptive = m["adaptive"] == "1";
    o.anchor_step = static_cast<unsigned>(std::stoul(m["anchor_step"]));
    o.codec = mdz::parse_codec_mode(m["codec"]);
    o.spatial_delta = m["spatial_delta"] == "1";
    o.lossless = m["lossless"] == "1";
    if (m.count("source_format")) o.source_format = m["source_format"];
    for (const auto& [k, v] : m) {
        if (k.rfind("column_precision.", 0) == 0) o.column_precision[k.substr(17)] = std::strtof(v.c_str(), nullptr);
        if (k.rfind("attr.", 0) == 0) c.batch.attributes[k.substr(5)] = v;
    }
    if (m.count("units")) c.batch.units = mdz::LammpsUnits::from_style(m["units"]);

    for (const auto& e : fs::directory_iterator(dir)) {
        const std::string f = e.path().filename().string();
        auto stem = [&](std::size_t pre, std::size_t suf) { return f.substr(pre, f.size() - pre - suf); };
        if (f.rfind("col.", 0) == 0) c.batch.columns[stem(4, 4)] = read_array<float>(e.path());
        else if (f.rfind("icol.", 0) == 0) c.batch.int_columns[stem(5, 4)] = read_array<std::int32_t>(e.path());
        else if (f.rfind("prop.", 0) == 0) c.batch.frame_props[stem(5, 4)] = read_array<double>(e.path());
        else if (f.rfind("iprop.", 0) == 0) c.batch.int_frame_props[stem(6, 4)] = read_array<std::int64_t>(e.path());
        else if (f == "box.f64") {
            auto v = read_array<double>(e.path());
            c.batch.box.emplace(v.size() / 9);
            std::memcpy(c.batch.box->data(), v.data(), v.size() * 8);
        } else if (f == "origin.f64") {
            auto v = read_array<double>(e.path());
            c.batch.origin.emplace(v.size() / 3);
            std::memcpy(c.batch.origin->data(), v.data(), v.size() * 8);
        } else if (f == "atom_types.i32") c.batch.atom_types = mdz::share(read_array<std::int32_t>(e.path()));
        else if (f == "atom_ids.i32") c.batch.atom_ids = mdz::share(read_array<std::int32_t>(e.path()));
        else if (f == "atom_names.txt") {
            std::vector<std::string> names;
            std::ifstream in(e.path());
            std::string line;
            while (std::getline(in, line)) names.push_back(line);
            c.batch.atom_names = mdz::share(std::move(names));
        }
    }
    if (!c.batch.columns.count("positions")) c.batch.columns["positions"] = {};
    return c;
}

std::vector<fs::path> cases() {
    std::vector<fs::path> v;
    for (const auto& e : fs::directory_iterator(MDZ_GOLDEN_DIR))
        if (e.is_directory()) v.push_back(e.path());
    std::sort(v.begin(), v.end());
    return v;
}

}  // namespace

// The fixtures were compressed with zstd 1.5.7. Other zstd versions may produce
// different but equally valid bytes, so byte comparison is skipped for them.
// Decoding is checked with any version.
bool same_zstd() {
    if (mdz::detail::zstd_version() == 10507) return true;
    std::fprintf(stderr, "  skipped: built against zstd %u, fixtures need 1.5.7\n", mdz::detail::zstd_version());
    return false;
}

// With MDZ_REGEN_GOLDEN=1, every out.mdz and its decoded hash are rewritten
// by this build instead of being checked. Use this only after a deliberate
// change to the encoder.
void regenerate(const fs::path& dir, const Case& c) {
    auto bytes = mdz::to_bytes(mdz::compress(c.batch, c.opts));
    std::ofstream(dir / "out.mdz", std::ios::binary)
        .write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    char hash[32];
    std::snprintf(hash, sizeof hash, "%016llx",
                  static_cast<unsigned long long>(fnv1a(mdz::decompress_positions(mdz::load_bytes(bytes)))));
    std::ifstream in(dir / "meta.txt");
    std::string meta, line;
    while (std::getline(in, line))
        if (line.rfind("decoded_fnv1a=", 0) != 0) meta += line + "\n";
    std::ofstream(dir / "meta.txt") << meta << "decoded_fnv1a=" << hash << "\n";
    std::fprintf(stderr, "  regenerated %s (%zu bytes)\n", dir.filename().string().c_str(), bytes.size());
}

TEST(golden_compress_is_byte_identical) {
    if (!same_zstd()) return;
    auto all = cases();
    REQUIRE(all.size() >= 10);
    const char* regen = std::getenv("MDZ_REGEN_GOLDEN");
    for (const auto& dir : all) {
        Case c = load_case(dir);
        if (regen && *regen == '1') {
            regenerate(dir, c);
            continue;
        }
        auto want = read_file(dir / "out.mdz");
        auto got = mdz::to_bytes(mdz::compress(c.batch, c.opts));
        if (got != want) {
            std::size_t at = 0;
            while (at < got.size() && at < want.size() && got[at] == want[at]) ++at;
            std::fprintf(stderr, "  %s: %zu bytes vs %zu, first difference at %zu\n",
                         dir.filename().string().c_str(), got.size(), want.size(), at);
            CHECK(got == want);
        }
    }
}

TEST(golden_compress_is_byte_identical_single_threaded) {
    if (!same_zstd()) return;
    mdz::set_num_threads(1);
    for (const auto& dir : cases()) {
        Case c = load_case(dir);
        CHECK(mdz::to_bytes(mdz::compress(c.batch, c.opts)) == read_file(dir / "out.mdz"));
    }
    mdz::set_num_threads(0);
}

TEST(golden_decode_matches_reference) {
    for (const auto& dir : cases()) {
        Case c = load_case(dir);
        auto ct = mdz::load((dir / "out.mdz").string());
        char want[32], got[32];
        std::snprintf(got, sizeof got, "%016llx",
                      static_cast<unsigned long long>(fnv1a(mdz::decompress_positions(ct))));
        std::snprintf(want, sizeof want, "%s", c.meta["decoded_fnv1a"].c_str());
        if (std::strcmp(want, got) != 0) {
            std::fprintf(stderr, "  %s: decoded hash %s, reference %s\n", dir.filename().string().c_str(), got, want);
            CHECK(false);
        }
    }
}

TEST(golden_full_decode_restores_exact_columns) {
    // Everything the multicolumn case stores exactly must decode exactly.
    Case c = load_case(fs::path(MDZ_GOLDEN_DIR) / "multicolumn_real_units");
    auto ct = mdz::load((fs::path(MDZ_GOLDEN_DIR) / "multicolumn_real_units" / "out.mdz").string());
    mdz::FrameBatch b = mdz::decompress(ct);
    CHECK_EQ(b.nframes, c.batch.nframes);
    CHECK(b.int_columns == c.batch.int_columns);
    CHECK(b.int_frame_props == c.batch.int_frame_props);
    CHECK(b.frame_props == c.batch.frame_props);
    CHECK(*b.box == *c.batch.box);
    CHECK(*b.origin == *c.batch.origin);
    CHECK(b.columns["dipole"] == c.batch.columns["dipole"]);
    CHECK(*b.atom_names == *c.batch.atom_names);
    CHECK(*b.atom_ids == *c.batch.atom_ids);
    CHECK(b.units == c.batch.units);
    CHECK(b.attributes == c.batch.attributes);
    // Lossy columns must be within their error bound, which is 4 quantization
    // steps with adaptive precision.
    for (const char* name : {"positions", "velocities", "charge"}) {
        const auto& a = c.batch.columns[name];
        const auto& r = b.columns[name];
        REQUIRE(a.size() == r.size());
        double worst = 0;
        for (std::size_t i = 0; i < a.size(); ++i) worst = std::max(worst, double(std::fabs(a[i] - r[i])));
        // Positions and velocities are in real units (A, A/fs). The step is
        // 0.000125 nm = 0.00125 A, and 0.000125 nm/ps = 1.25e-6 A/fs.
        CHECK(worst <= 4 * 0.00125 + 1e-6);
    }
}
