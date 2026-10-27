// Rigid bodies as a compression model.

#include <cmath>
#include <cstring>
#include <random>

#include "mdz/mdz.hpp"
#include "rigid.hpp"
#include "test.hpp"

namespace {

using Vec = std::array<double, 3>;

// Rotate v by the unit quaternion q (w, x, y, z).
Vec rotate(const std::array<double, 4>& q, const Vec& v) {
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    const double m[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                         2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                         2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)};
    return {m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
            m[6] * v[0] + m[7] * v[1] + m[8] * v[2]};
}

std::array<double, 4> normalize(std::array<double, 4> q) {
    const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    for (auto& c : q) c /= n;
    return q;
}

// Builds `nmol` rigid molecules of `shape` (offsets from the centre, in nm)
// that tumble and drift, plus `nfree` free atoms. `step` scales the motion per
// frame; large values make successive frames effectively uncorrelated. `flex`
// adds random displacement within each molecule, so the bodies are not quite
// rigid.
mdz::FrameBatch molecules(const std::vector<Vec>& shape, std::size_t nmol, std::size_t nfree, std::size_t nframes,
                          double step, double flex = 0.0, std::uint32_t seed = 7) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> g(0.0, 1.0);
    std::uniform_real_distribution<double> u(0.0, 4.0);
    const std::size_t k = shape.size(), natoms = nmol * k + nfree;
    std::vector<Vec> centre(nmol), freepos(nfree);
    std::vector<std::array<double, 4>> q(nmol);
    // Molecule centres sit on a lattice 0.6 nm apart so they do not overlap,
    // as in a real liquid. Free atoms are placed anywhere.
    const std::size_t side = std::size_t(std::ceil(std::cbrt(double(nmol))));
    for (std::size_t m = 0; m < nmol; ++m) {
        centre[m] = {0.6 * double(m % side), 0.6 * double(m / side % side), 0.6 * double(m / side / side)};
        q[m] = normalize({g(rng), g(rng), g(rng), g(rng)});
    }
    for (auto& p : freepos) p = {u(rng), u(rng), u(rng)};
    mdz::FrameBatch b(natoms, nframes);
    auto& pos = b.columns["positions"];
    pos.reserve(natoms * nframes * 3);
    for (std::size_t f = 0; f < nframes; ++f) {
        for (std::size_t m = 0; m < nmol; ++m) {
            for (int d = 0; d < 3; ++d) centre[m][d] += step * 0.05 * g(rng);
            q[m] = normalize({q[m][0] + step * g(rng), q[m][1] + step * g(rng), q[m][2] + step * g(rng),
                              q[m][3] + step * g(rng)});
            for (const Vec& s : shape) {
                Vec r = rotate(q[m], s);
                for (int d = 0; d < 3; ++d) pos.push_back(float(centre[m][d] + r[d] + flex * g(rng)));
            }
        }
        for (auto& p : freepos)
            for (int d = 0; d < 3; ++d) pos.push_back(float(p[d] += step * 0.05 * g(rng)));
    }
    return b;
}

const std::vector<Vec> kWater = {{0, 0, 0.0065}, {0.0757, 0, -0.052}, {-0.0757, 0, -0.052}};

std::vector<Vec> ring(int n, double r) {
    const double pi = 3.14159265358979323846;
    std::vector<Vec> s;
    for (int i = 0; i < n; ++i) s.push_back({r * std::cos(2 * pi * i / n), r * std::sin(2 * pi * i / n), 0});
    return s;
}

// Largest absolute difference. Checks against the error bound allow 1e-6 nm on
// top of it, since adding the decoded residual to the prediction rounds to the
// nearest float.
double max_err(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, double(std::fabs(a[i] - b[i])));
    return m;
}

}  // namespace

TEST(detects_rigid_molecules) {
    auto b = molecules(kWater, 50, 7, 12, 0.05);
    auto bodies = mdz::detail::detect_rigid_bodies(b);
    REQUIRE(bodies.size() == 157);
    // Each water is one body of three atoms. Free atoms belong to no body.
    for (std::size_t m = 0; m < 50; ++m) {
        CHECK(bodies[3 * m] >= 0);
        CHECK_EQ(bodies[3 * m], bodies[3 * m + 1]);
        CHECK_EQ(bodies[3 * m], bodies[3 * m + 2]);
        if (m) CHECK(bodies[3 * m] != bodies[3 * m - 1]);
    }
    for (std::size_t i = 150; i < 157; ++i) CHECK_EQ(bodies[i], -1);
    // Molecules that flex beyond the tolerance are not detected as rigid.
    auto floppy = molecules(kWater, 20, 0, 12, 0.05, 0.01);
    for (auto x : mdz::detail::detect_rigid_bodies(floppy)) CHECK_EQ(x, -1);
}

TEST(rigid_coding_keeps_the_error_bound) {
    for (double step : {0.01, 0.3}) {
        for (double flex : {0.0, 0.002}) {
            auto b = molecules(kWater, 120, 10, 30, step, flex);
            for (auto codec : {mdz::CodecMode::Auto, mdz::CodecMode::Temporal, mdz::CodecMode::Intra}) {
                mdz::CompressOptions o;
                o.rigid = true;
                o.codec = codec;
                o.gof_size = 10;
                // Declare the bodies explicitly, because flexible molecules
                // would not be detected.
                std::vector<std::int32_t> bodies(b.natoms, -1);
                for (std::size_t i = 0; i < 360; ++i) bodies[i] = std::int32_t(i / 3);
                b.bodies = mdz::share(bodies);
                auto ct = mdz::load_bytes(mdz::to_bytes(mdz::compress(b, o)));
                auto back = mdz::decompress_positions(ct);
                CHECK(max_err(*b.positions(), back) <= 4 * o.precision + 1e-6);
            }
        }
    }
}

TEST(rigid_coding_shrinks_large_bodies) {
    // Tumbling twelve-atom rings. Rigid coding stores six numbers per body per
    // frame instead of 36 coordinates.
    auto b = molecules(ring(12, 0.14), 200, 0, 20, 0.3);
    mdz::CompressOptions plain, rigid;
    rigid.rigid = true;
    const auto a = mdz::to_bytes(mdz::compress(b, plain)).size();
    const auto r = mdz::to_bytes(mdz::compress(b, rigid)).size();
    std::fprintf(stderr, "  12-atom rings: %zu bytes plain, %zu rigid (%.2fx)\n", a, r, double(a) / double(r));
    CHECK(r * 2 < a);
    // Water is reported for information. Rigid coding may or may not be
    // smaller for it.
    auto w = molecules(kWater, 500, 0, 20, 0.3);
    const auto wa = mdz::to_bytes(mdz::compress(w, plain)).size();
    const auto wr = mdz::to_bytes(mdz::compress(w, rigid)).size();
    std::fprintf(stderr, "  water: %zu bytes plain, %zu rigid (%.2fx)\n", wa, wr, double(wa) / double(wr));
    CHECK(wr <= wa);  // each group keeps whichever encoding is smaller
}

TEST(rigid_files_read_back_everywhere) {
    auto b = molecules(ring(6, 0.14), 40, 5, 27, 0.2);
    mdz::CompressOptions o;
    o.rigid = true;
    o.gof_size = 8;
    auto ct = mdz::compress(b, o);
    REQUIRE(ct.rigid);
    auto bytes = mdz::to_bytes(ct);
    auto r = mdz::Reader::from_bytes(bytes);
    auto all = r.read_positions(0, r.nframes());
    CHECK(max_err(*b.positions(), all) <= 4 * o.precision + 1e-6);
    for (std::size_t f : {0, 5, 9, 26}) {
        auto one = r.read_frame(f);
        CHECK(std::equal(one.begin(), one.end(), all.begin() + std::ptrdiff_t(f * 245 * 3)));
    }
    // The bodies are stored with the topology and round-trip.
    auto back = mdz::decompress(mdz::load_bytes(bytes));
    REQUIRE(back.bodies);
    CHECK_EQ(back.bodies->size(), std::size_t(245));
    // Rigid groups cannot be decoded without the bodies from the metadata.
    auto stripped = mdz::load_bytes(bytes);
    stripped.rigid.reset();
    CHECK_THROWS(mdz::decompress_positions(stripped));
}

TEST(streaming_with_rigid_bodies) {
    auto b = molecules(kWater, 60, 4, 25, 0.2);
    mdz::CompressOptions o;
    o.rigid = true;
    o.gof_size = 5;
    const std::string path = mdz_test::scratch_dir() + "/rigid.mdz";
    {
        auto w = mdz::Writer::create(path, o);
        for (std::size_t f = 0; f < b.nframes; ++f) w.write(b.slice_frames(f, f + 1));
        w.close();
    }
    auto back = mdz::decompress_positions(mdz::load(path));
    CHECK(max_err(*b.positions(), back) <= 4 * o.precision + 1e-6);
    // Appending reuses the bodies already stored in the file.
    {
        auto w = mdz::Writer::append(path, o);
        w.write(b.slice_frames(0, 5));
        w.close();
    }
    CHECK_EQ(mdz::load(path).nframes, std::size_t(30));
}

TEST(streaming_keeps_the_first_groups_choice_of_bodies) {
    // The first group is a gas with no bodies, and molecules appear later.
    // The file's metadata is fixed when the first group is written, so later
    // groups must not detect bodies of their own.
    auto gas = molecules({{0, 0, 0}}, 0, 60, 5, 1.0);
    auto mol = molecules(kWater, 20, 0, 10, 0.05);
    mdz::CompressOptions o;
    o.rigid = true;
    o.gof_size = 5;
    const std::string path = mdz_test::scratch_dir() + "/rigid_late.mdz";
    {
        auto w = mdz::Writer::create(path, o);
        w.write(gas);
        w.write(mol);
        w.close();
    }
    auto back = mdz::decompress_positions(mdz::load(path));
    std::vector<float> want(*gas.positions());
    want.insert(want.end(), mol.positions()->begin(), mol.positions()->end());
    CHECK(max_err(want, back) <= 4 * o.precision + 1e-6);
    // The same applies when appending with rigid enabled to a file that has
    // no bodies.
    {
        auto w = mdz::Writer::append(path, o);
        w.write(mol);
        w.close();
    }
    CHECK_EQ(mdz::load(path).nframes, std::size_t(25));
}

TEST(rigid_is_not_used_for_lossless_or_varying_counts) {
    auto b = molecules(kWater, 10, 0, 6, 0.1);
    mdz::CompressOptions o;
    o.rigid = true;
    o.lossless = true;
    auto back = mdz::decompress_positions(mdz::compress(b, o));
    CHECK(std::memcmp(back.data(), b.positions()->data(), back.size() * 4) == 0);
    o.lossless = false;
    o.variable_atoms = true;
    b.bodies = mdz::share(std::vector<std::int32_t>(b.natoms, 0));
    CHECK_THROWS(mdz::compress(b, o));
}
