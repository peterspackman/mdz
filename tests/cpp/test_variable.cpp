// Trajectories whose atom count varies from frame to frame.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>

#include "mdz/mdz.hpp"
#include "test.hpp"

namespace {

// A grand-canonical-style run in which atoms are inserted and deleted over
// time, each moving smoothly while it exists. Each frame is built as a
// fixed-count batch, as a format reader would produce it, and then appended.
mdz::FrameBatch gcmc(std::size_t nframes, std::uint32_t seed = 3) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.0f, 3.0f);
    std::map<std::int32_t, std::array<float, 3>> alive;
    std::int32_t next_id = 1;
    for (int i = 0; i < 40; ++i) alive[next_id++] = {u(rng), u(rng), u(rng)};
    mdz::FrameBatch all;
    for (std::size_t f = 0; f < nframes; ++f) {
        for (int k = 0; k < 2 && !alive.empty(); ++k) {
            auto it = alive.begin();
            std::advance(it, rng() % alive.size());
            if (rng() % 2) alive.erase(it);
        }
        for (int k = 0; k < 2; ++k)
            if (rng() % 2) alive[next_id++] = {u(rng), u(rng), u(rng)};
        mdz::FrameBatch b(alive.size(), 1);
        std::vector<std::int32_t> ids, types;
        std::vector<std::string> names;
        auto& p = b.columns["positions"];
        auto& v = b.columns["velocities"];
        for (auto& [id, x] : alive) {
            for (auto& c : x) c += 0.001f * float(int(rng() % 21) - 10);
            p.insert(p.end(), x.begin(), x.end());
            v.insert(v.end(), {0.1f * float(id % 7), 0.0f, -0.2f});
            ids.push_back(id);
            types.push_back(1 + id % 3);
            names.push_back(id % 3 == 0 ? "Ow" : "Na");
        }
        b.atom_ids = mdz::share(std::move(ids));
        b.atom_types = mdz::share(std::move(types));
        b.atom_names = mdz::share(std::move(names));
        b.int_frame_props["step"] = {std::int64_t(f) * 10};
        b.box = std::vector<mdz::Box>{{3, 0, 0, 0, 3, 0, 0, 0, 3}};
        all.append(b);
    }
    return all;
}

void check_same(const mdz::FrameBatch& a, const mdz::FrameBatch& b, float tol) {
    REQUIRE(a.nframes == b.nframes);
    for (std::size_t f = 0; f < a.nframes; ++f) {
        mdz::FrameBatch x = a.frame(f), y = b.frame(f);
        REQUIRE(x.natoms == y.natoms);
        CHECK(*x.atom_ids == *y.atom_ids);
        CHECK(*x.atom_types == *y.atom_types);
        CHECK(*x.atom_names == *y.atom_names);
        for (const char* col : {"positions", "velocities"}) {
            const auto &p = *x.column(col), &q = *y.column(col);
            REQUIRE(p.size() == q.size());
            for (std::size_t i = 0; i < p.size(); ++i)
                if (std::fabs(p[i] - q[i]) > tol) {
                    CHECK(std::fabs(p[i] - q[i]) <= tol);
                    return;
                }
        }
        CHECK(x.int_frame_props.at("step") == y.int_frame_props.at("step"));
    }
}

std::string path(const char* name) { return mdz_test::scratch_dir() + "/" + name; }

}  // namespace

TEST(appending_different_atom_counts_gives_a_varying_batch) {
    auto b = gcmc(30);
    REQUIRE(b.variable());
    CHECK_EQ(b.natoms, std::size_t(0));
    CHECK(b.positions()->size() == b.total_atoms() * 3);
    CHECK(b.int_columns.count(mdz::columns::kId) == 1);
    CHECK(b.strings && b.strings->size() == 2);
    // A single frame carries its own slice of the per-atom metadata.
    auto f = b.frame(7);
    CHECK(!f.variable());
    CHECK_EQ(f.natoms, (*b.frame_natoms)[7]);
    CHECK(f.atom_names && f.atom_names->size() == f.natoms);
    auto data = b.available_data();
    CHECK(std::find(data.begin(), data.end(), "atom_names") != data.end());
}

TEST(varying_atoms_roundtrip_lossy_and_lossless) {
    auto b = gcmc(53);
    for (bool lossless : {false, true}) {
        mdz::CompressOptions o;
        o.gof_size = 10;
        o.lossless = lossless;
        auto bytes = mdz::to_bytes(mdz::compress(b, o));
        auto ct = mdz::load_bytes(bytes);
        CHECK(ct.variable_atoms);
        CHECK(ct.frame_natoms() == *b.frame_natoms);
        check_same(b, mdz::decompress(ct), lossless ? 0.0f : 0.0005f);
        // A partial range that crosses group boundaries.
        auto mid = mdz::decompress(ct, 8, 23);
        check_same(b.slice_frames(8, 23), mid, lossless ? 0.0f : 0.0005f);
    }
}

TEST(reader_handles_varying_atoms) {
    auto b = gcmc(41);
    mdz::CompressOptions o;
    o.gof_size = 8;
    auto r = mdz::Reader::from_bytes(mdz::to_bytes(mdz::compress(b, o)));
    CHECK(r.variable_atoms());
    CHECK_EQ(r.natoms(), std::size_t(0));
    CHECK(r.frame_natoms() == *b.frame_natoms);
    const auto off = b.atom_offsets();
    for (std::size_t f : {0, 7, 8, 40}) {
        auto x = r.read_frame(f);
        REQUIRE(x.size() == b.atoms_in(f) * 3);
        for (std::size_t i = 0; i < x.size(); ++i)
            CHECK(std::fabs(x[i] - (*b.positions())[off[f] * 3 + i]) <= 0.0005f);
        CHECK(r.atom_ids(f) == *b.frame(f).atom_ids);
    }
    auto many = r.read_positions(5, 20);
    CHECK_EQ(many.size(), (off[25] - off[5]) * 3);
    auto names = r.read_column("mdz.name");
    CHECK_EQ(names.i32.size(), b.total_atoms());
}

TEST(streaming_varying_atoms_and_recovery) {
    auto b = gcmc(37);
    mdz::CompressOptions o;
    o.gof_size = 6;
    o.variable_atoms = true;
    {
        auto w = mdz::Writer::create(path("var.mdz"), o);
        for (std::size_t f = 0; f < 30; ++f) w.write(b.frame(f));
        w.flush();
        std::filesystem::copy_file(path("var.mdz"), path("var_crash.mdz"),
                                   std::filesystem::copy_options::overwrite_existing);
        for (std::size_t f = 30; f < b.nframes; ++f) w.write(b.frame(f));
        w.close();
    }
    auto full = mdz::load(path("var.mdz"));
    check_same(b, mdz::decompress(full), 0.0005f);
    mdz::LoadStatus st;
    auto crashed = mdz::load(path("var_crash.mdz"), &st);
    CHECK(!st.complete);
    CHECK_EQ(crashed.nframes, std::size_t(30));
    {
        auto w = mdz::Writer::append(path("var_crash.mdz"), o);
        for (std::size_t f = 30; f < b.nframes; ++f) w.write(b.frame(f));
        w.close();
    }
    check_same(b, mdz::decompress(mdz::load(path("var_crash.mdz"))), 0.0005f);
}

TEST(fixed_files_refuse_changing_atom_counts) {
    auto b = gcmc(12);
    mdz::CompressOptions o;
    o.gof_size = 4;
    auto w = mdz::Writer::create(path("fixed.mdz"), o);
    // Write frames while the atom count is unchanged, then check that the
    // first frame with a different count is rejected.
    std::size_t f = 0;
    while (f + 1 < b.nframes && b.atoms_in(f + 1) == b.atoms_in(0)) ++f;
    for (std::size_t i = 0; i <= f; ++i) w.write(b.frame(i));
    REQUIRE(f + 1 < b.nframes);
    try {
        w.write(b.frame(f + 1));
        CHECK(false);
    } catch (const mdz::Error& e) {
        CHECK(std::strstr(e.what(), "variable_atoms") != nullptr);
    }
}

TEST(duplicate_ids_in_a_frame_are_refused) {
    mdz::FrameBatch b(3, 1);
    b.columns["positions"] = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    b.atom_ids = mdz::share(std::vector<std::int32_t>{5, 6, 5});
    mdz::CompressOptions o;
    o.variable_atoms = true;
    CHECK_THROWS(mdz::compress(b, o));
}

TEST(fixed_data_in_variable_mode_costs_little) {
    // A fixed-count trajectory stored in variable-atom mode. The membership
    // blobs should add almost nothing to the size.
    mdz::FrameBatch b(500, 60);
    auto& p = b.columns["positions"];
    std::mt19937 rng(1);
    for (std::size_t i = 0; i < 500 * 60 * 3; ++i) p.push_back(float(i % 1500) * 0.002f + float(rng() % 100) * 1e-5f);
    mdz::CompressOptions fixed, var;
    var.variable_atoms = true;
    const auto a = mdz::to_bytes(mdz::compress(b, fixed)).size();
    const auto v = mdz::to_bytes(mdz::compress(b, var)).size();
    CHECK(double(v) < double(a) * 1.02 + 200);
    auto back = mdz::decompress(mdz::load_bytes(mdz::to_bytes(mdz::compress(b, var))));
    CHECK(back.variable());
    CHECK_EQ(back.total_atoms(), std::size_t(500 * 60));
}

TEST(unknown_header_flags_are_rejected) {
    // Variable-atom files set a header flag. Readers must reject flags they do
    // not recognise.
    auto bytes = mdz::to_bytes(mdz::compress(gcmc(5), {}));
    CHECK_EQ(int(bytes[6]), 1);
    bytes[6] = 2;  // an unknown flag
    CHECK_THROWS(mdz::load_bytes(bytes));
}
