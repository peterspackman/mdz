#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "kernels.hpp"
#include "test.hpp"

using namespace mdz::detail;

namespace {

// Reference rounding: round half away from zero, saturate to the int32 range,
// and map NaN to 0.
std::int32_t reference_round(float y) {
    if (std::isnan(y)) return 0;
    float r = std::round(y);
    if (r >= 2147483648.0f) return INT32_MAX;
    if (r < -2147483648.0f) return INT32_MIN;
    return static_cast<std::int32_t>(r);
}

}  // namespace

TEST(round_matches_reference_on_hard_values) {
    const float cases[] = {0.0f,         -0.0f,         0.5f,         -0.5f,           1.5f,        -1.5f,
                           2.5f,         0.49999997f,   -0.49999997f, 8388607.5f,      8388609.0f,  16777217.0f,
                           2147483520.f, -2147483648.f, 2147483648.f, -2147483904.f,   1e30f,       -1e30f,
                           INFINITY,     -INFINITY,     NAN,          1.17549435e-38f, 0.50000006f, -2.5f};
    for (float y : cases) CHECK_EQ(round_to_i32(y), reference_round(y));
}

TEST(round_matches_reference_on_every_float_sampled) {
    // Every 97th bit pattern, about 44 million values covering every exponent.
    for (std::uint64_t u = 0; u < (1ull << 32); u += 97) {
        float y = bits_float(static_cast<std::uint32_t>(u));
        if (round_to_i32(y) != reference_round(y)) {
            CHECK_EQ(round_to_i32(y), reference_round(y));
            return;
        }
    }
}

TEST(zigzag_roundtrip) {
    const std::int32_t v[] = {0, -1, 1, -2, 2, INT32_MIN, INT32_MAX, -100000, 100000};
    for (auto x : v) CHECK_EQ(unzigzag(zigzag(x)), x);
    CHECK_EQ(zigzag(-1), 1u);
    CHECK_EQ(zigzag(1), 2u);
}

TEST(residual_frame_roundtrip_and_truncation) {
    std::mt19937 rng(1);
    for (bool spatial : {false, true}) {
        const std::size_t natoms = 257;
        std::vector<std::int32_t> q(natoms * 3), back(natoms * 3);
        for (auto& x : q) x = static_cast<std::int32_t>(rng());
        q[0] = INT32_MIN;
        q[1] = INT32_MAX;
        std::vector<std::uint8_t> buf(natoms * 15);
        std::size_t n = pack_residual_frame(q.data(), natoms, spatial, buf.data());
        CHECK_EQ(read_residual_frame(buf.data(), n, natoms, spatial, back.data()), n);
        CHECK(q == back);
        CHECK_EQ(read_residual_frame(buf.data(), n - 1, natoms, spatial, back.data()), std::size_t(0));
    }
    // A varint carrying bits beyond 32 must be rejected.
    const std::uint8_t wide[] = {0xFF, 0xFF, 0xFF, 0xFF, 0x1F, 0, 0};
    std::int32_t out[3];
    CHECK_EQ(read_residual_frame(wide, sizeof wide, 1, false, out), std::size_t(0));
}
