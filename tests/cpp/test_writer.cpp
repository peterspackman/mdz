#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>

#include "mdz/mdz.hpp"
#include "test.hpp"

namespace {

std::vector<std::uint8_t> slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

mdz::FrameBatch frames(std::size_t natoms, std::size_t first, std::size_t n) {
    mdz::FrameBatch b(natoms, n);
    auto& p = b.columns["positions"];
    for (std::size_t f = first; f < first + n; ++f)
        for (std::size_t i = 0; i < natoms * 3; ++i) p.push_back(static_cast<float>(i) * 0.1f + f * 0.001f);
    for (std::size_t f = first; f < first + n; ++f) b.int_frame_props["step"].push_back(static_cast<std::int64_t>(f));
    b.box.emplace(n, mdz::Box{2, 0, 0, 0, 2, 0, 0, 0, 2});
    return b;
}

std::string path(const char* name) { return mdz_test::scratch_dir() + "/" + name; }

}  // namespace

TEST(streaming_writes_the_same_bytes_as_one_shot) {
    mdz::CompressOptions o;
    o.gof_size = 5;
    const std::string p = path("stream.mdz");
    {
        auto w = mdz::Writer::create(p, o);
        for (std::size_t f = 0; f < 20; ++f) w.write(frames(12, f, 1));
        w.close();
    }
    CHECK(slurp(p) == mdz::to_bytes(mdz::compress(frames(12, 0, 20), o)));
}

TEST(writer_survives_being_dropped_and_appends) {
    mdz::CompressOptions o;
    o.gof_size = 4;
    const std::string p = path("append.mdz");
    {
        auto w = mdz::Writer::create(p, o);
        w.write(frames(6, 0, 10));  // two full groups committed, two frames buffered
        CHECK_EQ(w.frames_committed(), std::uint64_t(8));
        CHECK_EQ(w.frames_written(), std::uint64_t(10));
        // The destructor finishes the file on a best-effort basis.
    }
    {
        auto r = mdz::Reader::open(p);
        CHECK(r.complete());
        CHECK_EQ(r.nframes(), std::size_t(10));
    }
    {
        auto w = mdz::Writer::append(p, o);
        CHECK_EQ(w.frames_written(), std::uint64_t(10));
        w.write(frames(6, 10, 7));
        w.close();
    }
    auto r = mdz::Reader::open(p);
    CHECK(r.complete());
    CHECK_EQ(r.nframes(), std::size_t(17));
    auto steps = r.read_column("step");
    REQUIRE(steps.i64.size() == 17);
    for (std::size_t f = 0; f < 17; ++f) CHECK_EQ(steps.i64[f], static_cast<std::int64_t>(f));
    auto all = r.read_positions(0, 17);
    auto want = frames(6, 0, 17).columns["positions"];
    for (std::size_t i = 0; i < all.size(); ++i) CHECK(std::fabs(all[i] - want[i]) < 0.001f);
}

TEST(a_crashed_writer_leaves_every_complete_group) {
    mdz::CompressOptions o;
    o.gof_size = 3;
    const std::string p = path("crash.mdz");
    {
        auto w = mdz::Writer::create(p, o);
        w.write(frames(5, 0, 10));
        w.flush();
        // Simulate a crash by copying the file before close().
        std::filesystem::copy_file(p, path("crash_copy.mdz"), std::filesystem::copy_options::overwrite_existing);
        w.close();
    }
    // Also make a copy with the last record partly cut off.
    auto bytes = slurp(path("crash_copy.mdz"));
    bytes.resize(bytes.size() - 7);
    {
        std::ofstream f(path("torn.mdz"), std::ios::binary);
        f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    {
        auto r = mdz::Reader::open(path("crash_copy.mdz"));
        CHECK(!r.complete());
        CHECK_EQ(r.nframes(), std::size_t(10));
        // Closed before the append below: Windows cannot truncate a file that
        // is memory-mapped.
        auto t = mdz::Reader::open(path("torn.mdz"));
        CHECK(!t.complete());
        CHECK_EQ(t.nframes(), std::size_t(9));
        CHECK(t.status().trailing_bytes > 0);
    }
    // Appending to the torn file resumes after the last complete group.
    {
        auto w = mdz::Writer::append(path("torn.mdz"), o);
        w.write(frames(5, 9, 4));
        w.close();
    }
    auto a = mdz::Reader::open(path("torn.mdz"));
    CHECK(a.complete());
    CHECK_EQ(a.nframes(), std::size_t(13));
}

TEST(writer_refuses_mismatched_frames) {
    mdz::CompressOptions o;
    o.gof_size = 2;
    auto w = mdz::Writer::create(path("mismatch.mdz"), o);
    w.write(frames(4, 0, 2));
    CHECK_THROWS(w.write(frames(5, 2, 2)));
    auto no_step = frames(4, 2, 2);
    no_step.int_frame_props.clear();
    CHECK_THROWS(w.write(no_step));
    auto short_box = frames(4, 2, 2);
    short_box.box->pop_back();
    CHECK_THROWS(w.write(short_box));
    w.close();
    CHECK_THROWS(w.write(frames(4, 2, 2)));
    mdz::CompressOptions other = o;
    other.precision = 0.01f;
    auto a = mdz::Writer::append(path("mismatch.mdz"), other);
    CHECK_THROWS(a.write(frames(4, 2, 2)));
}
