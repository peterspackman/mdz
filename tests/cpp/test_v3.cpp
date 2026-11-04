// Version 3 files, written by the Rust mdz 0.3, are read so they can be
// converted. fixtures/v3/nacl.mdz is examples/lammps/nacl.lammpstrj repeated
// seven times, compressed by that version.

#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>

#include "mdz/mdz.hpp"
#include "test.hpp"

namespace {

const std::string kFile = std::string(MDZ_FIXTURES_DIR) + "/v3/nacl.mdz";

// FNV-1a of the positions the Rust reader decodes from the fixture.
constexpr std::uint64_t kPositionsHash = 0x264512ca6176ebd0ull;

std::uint64_t fnv1a(const std::vector<float>& v) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    const auto* p = reinterpret_cast<const std::uint8_t*>(v.data());
    for (std::size_t i = 0; i < v.size() * 4; ++i) h = (h ^ p[i]) * 0x100000001b3ull;
    return h;
}

}  // namespace

TEST(v3_positions_match_the_rust_reader) {
    mdz::CompressedTrajectory ct = mdz::load(kFile);
    CHECK_EQ(ct.natoms, std::size_t(512));
    CHECK_EQ(ct.nframes, std::size_t(42));
    CHECK_EQ(ct.gofs.size(), std::size_t(3));
    CHECK_EQ(fnv1a(mdz::decompress_positions(ct)), kPositionsHash);
}

TEST(v3_cell_becomes_a_box_and_per_frame_columns_split_by_group) {
    mdz::CompressedTrajectory ct = mdz::load(kFile);
    CHECK(ct.has_column("box"));
    CHECK(!ct.has_column("cell"));
    mdz::FrameBatch b = mdz::decompress(ct, 18, 24);
    REQUIRE(b.box.has_value());
    REQUIRE(b.box->size() == 6);
    const double side = 2.2560801;  // nm
    for (const mdz::Box& m : *b.box)
        for (std::size_t i = 0; i < 9; ++i) CHECK(std::abs(m[i] - (i % 4 == 0 ? side : 0.0)) < 1e-12);
    const auto& step = b.int_frame_props["step"];
    REQUIRE(step.size() == 6);
    for (std::size_t f = 0; f < 6; ++f) CHECK_EQ(step[f], std::int64_t(((18 + f) % 6) * 100));
    CHECK(b.columns.count("velocities") == 1);
}

TEST(v3_converts_to_v4_without_reencoding) {
    mdz::CompressedTrajectory ct = mdz::load(kFile);
    std::vector<std::uint8_t> bytes = mdz::to_bytes(ct);
    CHECK(bytes[3] == 4);
    mdz::LoadStatus st;
    mdz::CompressedTrajectory v4 = mdz::load_bytes(bytes, &st);
    CHECK(st.complete);
    CHECK_EQ(fnv1a(mdz::decompress_positions(v4)), kPositionsHash);
    CHECK(mdz::decompress(v4).int_frame_props["step"] == mdz::decompress(ct).int_frame_props["step"]);
    for (std::size_t gi = 0; gi < ct.gofs.size(); ++gi) {
        const mdz::ByteSpan a = ct.column("positions")->blob(gi), b = v4.column("positions")->blob(gi);
        CHECK(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size()) == 0);
    }
}

TEST(v3_damaged_files_are_errors) {
    std::ifstream f(kFile, std::ios::binary);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    REQUIRE(bytes.size() > 200);
    for (std::size_t cut : {std::size_t(10), std::size_t(40), std::size_t(200), bytes.size() - 1}) {
        std::vector<std::uint8_t> b(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(cut));
        CHECK_THROWS(mdz::load_bytes(b));
    }
}
