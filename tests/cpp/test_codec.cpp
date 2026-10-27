#include <cmath>
#include <cstring>
#include <random>

#include "codec.hpp"
#include "kernels.hpp"
#include "mdz/mdz.hpp"
#include "test.hpp"
#include "zstd.hpp"

namespace {

std::vector<float> traj(std::size_t natoms, std::size_t nframes, std::uint32_t seed = 1) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-0.5f, 0.5f);
    std::vector<float> base(natoms * 3), out(natoms * nframes * 3);
    for (auto& b : base) b = u(rng) * 5.0f;
    for (std::size_t f = 0; f < nframes; ++f)
        for (std::size_t i = 0; i < natoms * 3; ++i) out[f * natoms * 3 + i] = base[i] + 0.01f * f + u(rng) * 0.001f;
    return out;
}

mdz::FrameBatch batch_of(std::vector<float> pos, std::size_t natoms) {
    mdz::FrameBatch b(natoms, pos.size() / (natoms * 3));
    b.columns["positions"] = std::move(pos);
    return b;
}

double max_err(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, double(std::fabs(a[i] - b[i])));
    return m;
}

}  // namespace

TEST(every_codec_roundtrips_within_bound) {
    auto pos = traj(50, 37);
    for (auto mode : {mdz::CodecMode::Temporal, mdz::CodecMode::Intra, mdz::CodecMode::Auto}) {
        for (unsigned anchor : {1u, 5u}) {
            mdz::CompressOptions o;
            o.precision = 0.001f;
            o.codec = mode;
            o.anchor_step = anchor;
            o.gof_size = 15;
            auto ct = mdz::compress(batch_of(pos, 50), o);
            auto back = mdz::decompress_positions(mdz::load_bytes(mdz::to_bytes(ct)));
            REQUIRE(back.size() == pos.size());
            CHECK(max_err(pos, back) <= 4 * 0.001 + 1e-6);
        }
    }
}

TEST(lossless_is_bit_exact_on_awkward_values) {
    const float awkward[] = {0.0f, -0.0f, 1.0f, -1.0f, 1.17549435e-38f, -1.17549435e-38f, 1e-40f, -1e-40f,
                             3.40282347e38f, -3.40282347e38f, 3.14159265f, -2.71828183f, 1.0f / 3.0f, 123456.79f,
                             -0.00012345679f};
    const std::size_t natoms = 5, n3 = 15, nframes = 8;
    std::vector<float> data(nframes * n3);
    for (std::size_t i = 0; i < n3; ++i) data[i] = awkward[i];
    for (std::size_t f = 1; f < nframes; ++f)
        for (std::size_t i = 0; i < n3; ++i) data[f * n3 + i] = data[(f - 1) * n3 + i] * 1.000001f + 0.001f;
    // Values that overflowed to inf are replaced, because lossless mode
    // rejects non-finite input.
    for (auto& v : data)
        if (!std::isfinite(v)) v = 1.0f;
    for (unsigned order = 1; order <= 4; ++order) {
        mdz::CompressOptions o;
        o.lossless = true;
        o.predictor_order = order;
        auto back = mdz::decompress_positions(mdz::compress(batch_of(data, natoms), o));
        REQUIRE(back.size() == data.size());
        CHECK(std::memcmp(back.data(), data.data(), data.size() * 4) == 0);
    }
}

TEST(lossless_overrides_rounding_settings) {
    mdz::CompressOptions o;
    o.lossless = true;
    o.precision = 0.001f;
    o.anchor_step = 5;
    o.codec = mdz::CodecMode::Intra;
    auto r = o.resolved();
    CHECK_EQ(r.precision, 0.0f);
    CHECK(!r.adaptive);
    CHECK_EQ(r.anchor_step, 1u);
    CHECK(!r.spatial_delta);
    CHECK(r.codec == mdz::CodecMode::Lossless);
    auto ct = mdz::compress(batch_of(traj(10, 5), 10), o);
    CHECK(!ct.column("positions")->desc.lossy);
}

TEST(non_finite_and_out_of_range_values_are_refused) {
    auto pos = traj(10, 3);
    pos[7] = NAN;
    CHECK_THROWS(mdz::compress(batch_of(pos, 10)));
    mdz::CompressOptions lossless;
    lossless.lossless = true;
    CHECK_THROWS(mdz::compress(batch_of(pos, 10), lossless));
    pos[7] = 1e9f;
    mdz::CompressOptions o;
    o.precision = 0.0001f;
    CHECK_THROWS(mdz::compress(batch_of(pos, 10), o));
}

TEST(bad_parameters_are_refused) {
    using mdz::detail::validate_params;
    validate_params(0.001f, 10, 5);
    CHECK_THROWS(validate_params(0.0f, 10, 5));
    CHECK_THROWS(validate_params(-1.0f, 10, 5));
    CHECK_THROWS(validate_params(NAN, 10, 5));
    CHECK_THROWS(validate_params(0.001f, 0, 5));
    CHECK_THROWS(validate_params(0.001f, 65536, 5));
    CHECK_THROWS(validate_params(0.001f, 10, 0));
    CHECK_THROWS(validate_params(0.001f, 10, 256));
    CHECK_THROWS(validate_params(0.001f, 60000, 2));
}

TEST(unknown_codec_mode_is_refused) {
    const std::uint8_t payload[] = {3, 1, 99, 0, 0, 0, 0};
    float out[3];
    try {
        mdz::detail::decode_f32x3_payload(mdz::ByteSpan(payload, sizeof payload), 1, 0.001f, 1, out);
        CHECK(false);
    } catch (const mdz::Error& e) {
        CHECK(std::strstr(e.what(), "99") != nullptr);
    }
}

TEST(auto_never_picks_lossless) {
    auto pos = traj(30, 6);
    auto m = mdz::detail::pick_codec_mode(pos.data(), 6, 30, 0.001f, true, 1, true, 3);
    CHECK(m != mdz::CodecMode::Lossless);
}

TEST(huge_residuals_fall_back_to_intra) {
    std::vector<float> pos(40 * 3 * 4);
    for (std::size_t i = 0; i < pos.size(); ++i) pos[i] = (i % 2 ? 1.0f : -1.0f) * 200000.0f;
    std::vector<std::uint8_t> blob;
    mdz::detail::GofParams p;
    p.precision = 0.001f;
    p.codec = mdz::CodecMode::Temporal;
    auto r = mdz::detail::encode_gof_blob(pos.data(), 4, 40, p, blob);
    CHECK(r.mode == mdz::CodecMode::Intra);
}

TEST(decompression_bombs_are_refused) {
    std::vector<std::uint8_t> payload(1 << 20, 0), out;
    auto blob = mdz::detail::zstd_compress(payload, 3);
    mdz::detail::zstd_decompress_bounded(blob, 1 << 20, out);
    CHECK_EQ(out.size(), std::size_t(1) << 20);
    CHECK_THROWS(mdz::detail::zstd_decompress_bounded(blob, (1 << 20) - 1, out));
    const std::uint8_t garbage[] = {0x28, 0xb5, 0x2f, 0xfd, 0, 0};
    CHECK_THROWS(mdz::detail::zstd_decompress_bounded(mdz::ByteSpan(garbage, 6), 100, out));
    CHECK_THROWS(mdz::detail::zstd_decompress_bounded(mdz::ByteSpan(), 100, out));
}

TEST(gof_table_locates_frames) {
    auto t = mdz::GofTable::uniform(25, 10);
    CHECK_EQ(t.size(), std::size_t(3));
    CHECK_EQ(t.span(2).start, std::size_t(20));
    CHECK_EQ(t.span(2).end, std::size_t(25));
    CHECK_EQ(t.overlapping(0, 1).start, std::size_t(0));
    CHECK_EQ(t.overlapping(0, 1).end, std::size_t(1));
    CHECK_EQ(t.overlapping(9, 11).end, std::size_t(2));
    CHECK_EQ(t.overlapping(20, 100).start, std::size_t(2));
    CHECK_EQ(t.overlapping(25, 30).size(), std::size_t(0));
    mdz::GofTable v;
    v.push(3);
    v.push(1);
    v.push(5);
    CHECK_EQ(v.overlapping(3, 4).start, std::size_t(1));
    CHECK_EQ(v.overlapping(2, 5).end, std::size_t(3));
    CHECK_EQ(v.max_len(), std::size_t(5));
    CHECK(!v.push(0));
}

TEST(cell_params_and_boxes_convert_exactly_at_right_angles) {
    mdz::CellParams p = {2.0, 90.0, 3.0, 90.0, 90.0, 4.0};
    mdz::Box b = mdz::cell_params_to_box(p);
    mdz::Box want = {2, 0, 0, 0, 3, 0, 0, 0, 4};
    CHECK(b == want);
    CHECK(mdz::box_to_cell_params(b) == p);
    mdz::CellParams tri = {2.0, 70.0, 3.0, 110.0, 95.0, 4.0};
    auto back = mdz::box_to_cell_params(mdz::cell_params_to_box(tri));
    for (int i = 0; i < 6; ++i) CHECK(std::fabs(back[i] - tri[i]) < 1e-9);
}
