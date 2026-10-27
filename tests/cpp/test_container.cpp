#include <cstring>
#include <random>

#include "mdz/mdz.hpp"
#include "test.hpp"

namespace {

mdz::FrameBatch sample(std::size_t natoms, std::size_t nframes) {
    mdz::FrameBatch b(natoms, nframes);
    auto& p = b.columns["positions"];
    p.resize(natoms * nframes * 3);
    for (std::size_t i = 0; i < p.size(); ++i) p[i] = static_cast<float>(i % 1000) * 0.01f;
    std::vector<std::string> names;
    for (std::size_t i = 0; i < natoms; ++i) names.push_back("A" + std::to_string(i));
    b.atom_names = mdz::share(std::move(names));
    b.int_frame_props["step"].resize(nframes);
    for (std::size_t f = 0; f < nframes; ++f) b.int_frame_props["step"][f] = static_cast<std::int64_t>(f * 10);
    return b;
}

std::vector<std::uint8_t> sample_bytes() {
    mdz::CompressOptions o;
    o.gof_size = 10;
    return mdz::to_bytes(mdz::compress(sample(8, 26), o));
}

}  // namespace

TEST(save_load_roundtrip) {
    auto bytes = sample_bytes();
    mdz::LoadStatus st;
    auto ct = mdz::load_bytes(bytes, &st);
    CHECK(st.complete);
    CHECK_EQ(ct.nframes, std::size_t(26));
    CHECK_EQ(ct.gofs.size(), std::size_t(3));
    CHECK(mdz::to_bytes(ct) == bytes);
    auto b = mdz::decompress(ct, 9, 21);
    CHECK_EQ(b.nframes, std::size_t(12));
    CHECK_EQ(b.int_frame_props["step"].front(), std::int64_t(90));
    CHECK_EQ(b.int_frame_props["step"].back(), std::int64_t(200));
    CHECK_EQ((*b.atom_names)[3], std::string("A3"));
}

TEST(truncated_files_keep_every_complete_group) {
    auto bytes = sample_bytes();
    // The footer offset is the first 8 bytes of the 16-byte trailer.
    std::uint64_t footer;
    std::memcpy(&footer, bytes.data() + bytes.size() - 16, 8);
    std::size_t last_whole = 0;
    for (std::size_t cut = 0; cut < bytes.size(); ++cut) {
        std::vector<std::uint8_t> head(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(cut));
        try {
            mdz::LoadStatus st;
            auto ct = mdz::load_bytes(head, &st);
            CHECK(!st.complete);
            CHECK(ct.nframes == 0 || ct.nframes == 10 || ct.nframes == 20 || ct.nframes == 26);
            CHECK(ct.nframes >= last_whole);
            last_whole = ct.nframes;
            if (cut >= footer) CHECK_EQ(ct.nframes, std::size_t(26));
            // Anything that loads must also decode.
            mdz::decompress(ct);
        } catch (const mdz::Error& e) {
            // Only a cut inside the header or schema may fail to load.
            CHECK(e.kind() == mdz::ErrorKind::Corrupt);
            CHECK(last_whole == 0);
        }
    }
    CHECK_EQ(last_whole, std::size_t(26));
}

TEST(corrupt_record_in_a_closed_file_is_an_error) {
    auto bytes = sample_bytes();
    // Flip a byte in the middle of the file, which is well past the header
    // and schema and inside the data section.
    auto ct0 = mdz::load_bytes(bytes);
    std::size_t at = bytes.size() / 2;
    bytes[at] ^= 0x40;
    try {
        auto ct = mdz::load_bytes(bytes);
        // Loading may succeed because CRCs are checked lazily, but decoding
        // must fail.
        CHECK_THROWS(mdz::decompress(ct));
    } catch (const mdz::Error& e) {
        CHECK(e.kind() == mdz::ErrorKind::Corrupt);
    }
}

TEST(garbage_and_unknown_versions_are_refused) {
    CHECK_THROWS(mdz::load_bytes({}));
    CHECK_THROWS(mdz::load_bytes(std::vector<std::uint8_t>(8, 0)));
    auto bytes = sample_bytes();
    auto v = bytes;
    v[3] = 3;  // MDZ\x03
    CHECK_THROWS(mdz::load_bytes(v));
    v = bytes;
    v[6] = 1;  // a header flag
    CHECK_THROWS(mdz::load_bytes(v));
    v = bytes;
    v[30] ^= 1;  // a schema byte, which then fails its CRC
    CHECK_THROWS(mdz::load_bytes(v));
}

TEST(random_corruption_never_crashes) {
    auto bytes = sample_bytes();
    std::mt19937 rng(7);
    for (int trial = 0; trial < 3000; ++trial) {
        auto v = bytes;
        int flips = 1 + static_cast<int>(rng() % 4);
        for (int k = 0; k < flips; ++k) v[rng() % v.size()] = static_cast<std::uint8_t>(rng());
        if (rng() % 4 == 0) v.resize(rng() % v.size());
        try {
            auto ct = mdz::load_bytes(v);
            mdz::decompress(ct);
        } catch (const mdz::Error&) {
        }
    }
    CHECK(true);
}

TEST(reader_random_access_matches_full_decode) {
    std::vector<float> pos(30 * 3 * 57);
    std::mt19937 rng(3);
    for (auto& x : pos) x = static_cast<float>(rng() % 10000) * 0.001f;
    mdz::FrameBatch b(30, 57);
    b.columns["positions"] = pos;
    b.box.emplace(57, mdz::Box{3, 0, 0, 0, 3, 0, 0, 0, 3});
    mdz::CompressOptions o;
    o.gof_size = 8;
    auto r = mdz::Reader::from_bytes(mdz::to_bytes(mdz::compress(b, o)));
    auto all = r.read_positions(0, r.nframes());
    CHECK_EQ(r.nframes(), std::size_t(57));
    REQUIRE(r.boxes().has_value());
    CHECK_EQ((*r.boxes())[5][4], 3.0f);
    for (std::size_t first : {0, 3, 7, 8, 30, 56})
        for (std::size_t count : {1, 2, 9, 20}) {
            if (first + count > 57) continue;
            auto part = r.read_positions(first, count);
            CHECK(std::equal(part.begin(), part.end(), all.begin() + static_cast<std::ptrdiff_t>(first * 90)));
        }
    CHECK_THROWS(r.read_positions(50, 10, all.data()));
    CHECK(r.read_positions(57, 5).empty());
}

TEST(partial_group_reads_match_full_decode_for_every_codec) {
    // Single-frame and partial reads go through the partial-decode and
    // group-cache paths. They must agree exactly with a full decode.
    std::vector<float> pos(25 * 3 * 41);
    std::mt19937 rng(11);
    for (std::size_t i = 0; i < pos.size(); ++i) pos[i] = static_cast<float>(i % 75) * 0.05f + (rng() % 100) * 1e-4f;
    for (auto mode : {mdz::CodecMode::Temporal, mdz::CodecMode::Intra, mdz::CodecMode::Lossless}) {
        for (unsigned anchor : {1u, 4u}) {
            if (mode == mdz::CodecMode::Lossless && anchor > 1) continue;
            mdz::FrameBatch b(25, 41);
            b.columns["positions"] = pos;
            mdz::CompressOptions o;
            o.gof_size = 10;
            o.codec = mode;
            o.lossless = mode == mdz::CodecMode::Lossless;
            o.anchor_step = anchor;
            auto r = mdz::Reader::from_bytes(mdz::to_bytes(mdz::compress(b, o)));
            const auto all = mdz::decompress_positions(r.trajectory());
            // Read frame by frame forwards, then backwards, then in odd ranges.
            for (std::size_t f = 0; f < 41; ++f) {
                auto one = r.read_frame(f);
                CHECK(std::equal(one.begin(), one.end(), all.begin() + static_cast<std::ptrdiff_t>(f * 75)));
            }
            for (std::size_t f = 41; f-- > 0;) {
                auto one = r.read_frame(f);
                CHECK(std::equal(one.begin(), one.end(), all.begin() + static_cast<std::ptrdiff_t>(f * 75)));
            }
            for (auto [first, count] : {std::pair<std::size_t, std::size_t>{3, 4}, {8, 5}, {19, 2}, {0, 41}, {35, 6}}) {
                auto part = r.read_positions(first, count);
                CHECK(std::equal(part.begin(), part.end(), all.begin() + static_cast<std::ptrdiff_t>(first * 75)));
            }
        }
    }
}
