// DCD, XTC and npy readers and writers.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>

#include "mdz/mdz.hpp"
#include "test.hpp"

namespace {

using Bytes = std::vector<std::uint8_t>;

// An output stream that cannot seek, like a pipe.
class PipeOutput final : public mdz::OutputStream {
public:
    explicit PipeOutput(Bytes* out) : out_(out) {}
    void write(const void* d, std::size_t n) override {
        const auto* c = static_cast<const std::uint8_t*>(d);
        out_->insert(out_->end(), c, c + n);
    }
    void flush() override {}
    void close() override {}
    const std::string& name() const override { return name_; }

private:
    Bytes* out_;
    std::string name_ = "<pipe>";
};

const mdz::FormatInfo& fmt(std::string_view name) {
    const mdz::FormatInfo* f = mdz::find_format(name);
    REQUIRE(f != nullptr);
    return *f;
}

std::unique_ptr<mdz::TrajectoryReader> reader(std::string_view format, Bytes bytes) {
    return fmt(format).open_reader(mdz::memory_input(std::move(bytes)), {});
}

// Writes batches to memory through either a seekable or a pipe-like stream.
Bytes write(std::string_view format, const std::vector<mdz::FrameBatch>& batches, bool seekable = true,
            std::size_t natoms = 0) {
    Bytes out;
    std::unique_ptr<mdz::OutputStream> s;
    if (seekable)
        s = mdz::memory_output(&out);
    else
        s = std::make_unique<PipeOutput>(&out);
    if (natoms == 0 && !batches.empty()) natoms = batches[0].natoms;
    auto w = fmt(format).open_writer(std::move(s), natoms, {});
    for (const auto& b : batches) w->write_batch(b);
    w->finish();
    return out;
}

Bytes write(std::string_view format, const mdz::FrameBatch& b, bool seekable = true) {
    return write(format, std::vector<mdz::FrameBatch>{b}, seekable);
}

mdz::FrameBatch read_all(std::string_view format, Bytes bytes) {
    auto r = reader(format, std::move(bytes));
    auto b = r->read_all();
    REQUIRE(b.has_value());
    return *b;
}

// Atom a in frame f is at (0.1 a + 0.01 f, 0.2 a, 0.3 a) nm.
mdz::FrameBatch sample(std::size_t natoms, std::size_t nframes) {
    mdz::FrameBatch b(natoms, nframes);
    auto& p = b.columns["positions"];
    p.resize(natoms * nframes * 3);
    for (std::size_t f = 0; f < nframes; ++f)
        for (std::size_t a = 0; a < natoms; ++a) {
            float* q = &p[(f * natoms + a) * 3];
            q[0] = static_cast<float>(a) * 0.1f + static_cast<float>(f) * 0.01f;
            q[1] = static_cast<float>(a) * 0.2f;
            q[2] = static_cast<float>(a) * 0.3f;
        }
    return b;
}

std::vector<mdz::Box> boxes(std::size_t n, const mdz::CellParams& cell) {
    return std::vector<mdz::Box>(n, mdz::cell_params_to_box(cell));
}

bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

template <class F>
mdz::ErrorKind error_kind(F&& f) {
    try {
        f();
    } catch (const mdz::Error& e) {
        return e.kind();
    }
    return mdz::ErrorKind::OutOfRange;  // stands for "did not throw"; no test expects this kind
}

Bytes file_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return Bytes(std::istreambuf_iterator<char>(f), {});
}

}  // namespace

// --- DCD ---
TEST(dcd_roundtrip) {
    auto b = sample(10, 3);
    b.box = boxes(3, {3.0, 90.0, 3.0, 90.0, 90.0, 3.0});
    auto r = reader("dcd", write("dcd", b));
    CHECK_EQ(r->natoms(), std::size_t(10));
    REQUIRE(r->nframes().has_value());
    CHECK_EQ(r->nframes().value_or(999), std::size_t(3));
    auto got = r->read_all();
    REQUIRE(got.has_value());
    CHECK_EQ(got->nframes, std::size_t(3));
    const auto& p = *b.positions();
    const auto& q = *got->positions();
    REQUIRE(q.size() == p.size());
    bool ok = true;
    for (std::size_t i = 0; i < p.size(); ++i) ok = ok && near(p[i], q[i], 1e-5);
    CHECK(ok);
    REQUIRE(got->box.has_value());
    REQUIRE(got->box->size() == 3);
    for (const auto& m : *got->box)
        for (std::size_t k = 0; k < 9; ++k) CHECK(near(m[k], k % 4 == 0 ? 3.0 : 0.0, 1e-5));
}

TEST(dcd_triclinic_cell_roundtrip) {
    // Non-orthogonal angles round-trip. The writer stores cosines and the
    // reader converts them back to degrees.
    auto b = sample(4, 1);
    const mdz::CellParams cell = {2.0, 70.0, 3.0, 110.0, 95.0, 4.0};
    b.box = boxes(1, cell);
    auto got = read_all("dcd", write("dcd", b));
    REQUIRE(got.box.has_value());
    const auto back = mdz::box_to_cell_params((*got.box)[0]);
    for (std::size_t k = 0; k < 6; ++k) CHECK(near(back[k], cell[k], 1e-4));
}

TEST(dcd_without_cell_has_no_box) {
    auto got = read_all("dcd", write("dcd", sample(5, 2)));
    CHECK(!got.box.has_value());
    CHECK_EQ(got.nframes, std::size_t(2));
}

TEST(dcd_rejects_truncated_and_garbage) {
    CHECK(error_kind([] { reader("dcd", Bytes(64, 0)); }) == mdz::ErrorKind::Corrupt);
    CHECK(error_kind([] { reader("dcd", Bytes()); }) == mdz::ErrorKind::Corrupt);
    auto b = sample(8, 2);
    b.box = boxes(2, {3.0, 90.0, 3.0, 90.0, 90.0, 3.0});
    const Bytes full = write("dcd", b);
    // Every truncation of the file fails, either at open or at read, and
    // always with Corrupt.
    for (std::size_t cut = 0; cut < full.size(); ++cut) {
        Bytes head(full.begin(), full.begin() + static_cast<std::ptrdiff_t>(cut));
        CHECK(error_kind([&] { reader("dcd", head)->read_all(); }) == mdz::ErrorKind::Corrupt);
    }
    // A corrupt coordinate record marker.
    Bytes bad = full;
    bad[bad.size() - 40] ^= 0x01;
    CHECK(error_kind([&] { reader("dcd", bad)->read_all(); }) == mdz::ErrorKind::Corrupt);
    // A header claiming a huge atom count must fail on the missing data
    // without allocating for the claimed size.
    bad = full;
    const std::int32_t huge = 0x7ffffff0;
    std::memcpy(bad.data() + 92 + 8 + 80 + 4 + 4, &huge, 4);
    CHECK(error_kind([&] { reader("dcd", bad)->read_all(); }) == mdz::ErrorKind::Corrupt);
    // Fixed atoms are not supported.
    bad = full;
    bad[40] = 1;
    CHECK(error_kind([&] { reader("dcd", bad); }) == mdz::ErrorKind::Corrupt);
}

TEST(dcd_reads_big_endian) {
    auto b = sample(6, 2);
    b.box = boxes(2, {2.0, 70.0, 3.0, 110.0, 95.0, 4.0});
    const Bytes le = write("dcd", b);
    // Byte-swap every field. Most fields are 4-byte words. The exceptions are
    // "CORD" and the title, which are left alone, and the cell values, which
    // are 8 bytes.
    Bytes be = le;
    auto swap = [&](std::size_t at, std::size_t w) {
        std::reverse(be.begin() + static_cast<std::ptrdiff_t>(at), be.begin() + static_cast<std::ptrdiff_t>(at + w));
    };
    std::size_t at = 0;
    swap(0, 4);
    for (at = 8; at < 92; at += 4) swap(at, 4);
    swap(92, 4);
    swap(96, 4);
    swap(96 + 4 + 80, 4);
    for (at = 184; at < 196; at += 4) swap(at, 4);
    while (at < be.size()) {
        swap(at, 4);
        for (std::size_t k = 0; k < 6; ++k) swap(at + 4 + 8 * k, 8);
        swap(at + 52, 4);
        at += 56;
        for (std::size_t c = 0; c < 3; ++c)
            for (std::size_t k = 0; k < 6 + 2; ++k, at += 4) swap(at, 4);
    }
    CHECK_EQ(at, be.size());
    auto x = read_all("dcd", le), y = read_all("dcd", be);
    CHECK(*x.positions() == *y.positions());
    CHECK(*x.box == *y.box);
}

TEST(dcd_header_timing_comes_from_step_and_time) {
    // Steps 100, 110, ... and 2 fs per step. DCD's timestep is in AKMA units.
    auto all = sample(3, 4);
    all.int_frame_props["step"] = {100, 110, 120, 130};
    all.frame_props["time"] = {0.2, 0.22, 0.24, 0.26};
    // The second frame arrives in a later batch.
    std::vector<mdz::FrameBatch> parts = {all.slice_frames(0, 1), all.slice_frames(1, 4)};
    for (bool seekable : {true, false}) {
        const Bytes h = write("dcd", parts, seekable);
        std::int32_t istart, nsavc, nstep;
        float delta;
        std::memcpy(&istart, h.data() + 12, 4);
        std::memcpy(&nsavc, h.data() + 16, 4);
        std::memcpy(&nstep, h.data() + 20, 4);
        std::memcpy(&delta, h.data() + 44, 4);
        CHECK_EQ(istart, 100);
        CHECK_EQ(nsavc, 10);
        CHECK_EQ(nstep, 40);
        CHECK(std::fabs(delta * 0.0488882129 - 0.002) < 1e-8);
    }
    // Without them the header says nothing about time.
    const Bytes h = write("dcd", sample(3, 4));
    float delta;
    std::memcpy(&delta, h.data() + 44, 4);
    CHECK(delta == 0.0f);
}

TEST(dcd_header_frame_count_is_patched) {
    // On a seekable stream the header is written first and the frame count is
    // patched at finish(). On a pipe the writer buffers and writes the header
    // once. Both must produce the same bytes.
    auto all = sample(7, 10);
    all.box = boxes(10, {2.5, 90.0, 2.5, 90.0, 90.0, 2.5});
    std::vector<mdz::FrameBatch> parts = {all.slice_frames(0, 3), all.slice_frames(3, 4), all.slice_frames(4, 10)};
    const Bytes one = write("dcd", all);
    CHECK(write("dcd", parts) == one);
    CHECK(write("dcd", parts, false) == one);
    std::int32_t n;
    std::memcpy(&n, one.data() + 8, 4);
    CHECK_EQ(n, 10);
    std::memcpy(&n, one.data() + 20, 4);
    CHECK_EQ(n, 10);
    auto r = reader("dcd", one);
    CHECK_EQ(r->nframes().value_or(999), std::size_t(10));
    // Reading in batches of any size gives the same frames.
    auto b1 = r->read_batch(4);
    REQUIRE(b1.has_value());
    CHECK_EQ(b1->nframes, std::size_t(4));
    auto rest = r->read_all();
    REQUIRE(rest.has_value());
    CHECK_EQ(rest->nframes, std::size_t(6));
    CHECK(!r->read_batch(5).has_value());
}

TEST(dcd_writer_edge_cases) {
    // A writer with no frames produces a valid, empty trajectory.
    Bytes out;
    {
        auto w = fmt("dcd").open_writer(mdz::memory_output(&out), 5, {});
        w->finish();
    }
    auto r = reader("dcd", out);
    CHECK_EQ(r->natoms(), std::size_t(5));
    CHECK_EQ(r->nframes().value_or(999), std::size_t(0));
    CHECK(!r->read_batch(10).has_value());
    // A writer destroyed without finish() still finishes.
    out.clear();
    {
        auto w = fmt("dcd").open_writer(mdz::memory_output(&out), 3, {});
        w->write_batch(sample(3, 2));
    }
    CHECK_EQ(read_all("dcd", out).nframes, std::size_t(2));
    // Missing positions or a wrong atom count are rejected.
    mdz::FrameBatch empty(3, 1);
    CHECK(error_kind([&] { write("dcd", empty); }) == mdz::ErrorKind::InvalidInput);
    CHECK(error_kind([&] { write("dcd", {sample(3, 1), sample(4, 1)}); }) == mdz::ErrorKind::InvalidInput);
    // The header declares unit cells, so a later frame without one is
    // rejected.
    auto withbox = sample(3, 1);
    withbox.box = boxes(1, {2.0, 90.0, 2.0, 90.0, 90.0, 2.0});
    CHECK(error_kind([&] { write("dcd", {withbox, sample(3, 1)}); }) == mdz::ErrorKind::InvalidInput);
}

TEST(dcd_writer_converts_lammps_units) {
    const mdz::LammpsUnits* real = mdz::LammpsUnits::from_style("real");
    REQUIRE(real != nullptr);
    auto b = sample(4, 1);
    b.units = real;
    auto& p = *b.column("positions");
    for (float& v : p) v *= 10.0f;  // angstroms
    auto got = read_all("dcd", write("dcd", b));
    const auto want = sample(4, 1);
    bool ok = true;
    for (std::size_t i = 0; i < p.size(); ++i) ok = ok && near((*got.positions())[i], (*want.positions())[i], 1e-5);
    CHECK(ok);
}

// Reference values for the bundled example files.
TEST(dcd_reads_examples) {
    {
        auto r = mdz::open_reader(std::string(MDZ_EXAMPLES_DIR) + "/traj.dcd");
        CHECK_EQ(r->natoms(), std::size_t(4096));
        CHECK_EQ(r->nframes().value_or(999), std::size_t(25));
        auto b = r->read_all();
        REQUIRE(b.has_value());
        CHECK_EQ(b->nframes, std::size_t(25));
        const auto& p = *b->positions();
        REQUIRE(p.size() == std::size_t(25 * 4096 * 3));
        CHECK_EQ(p[0], 3.1510798931121826f);
        CHECK_EQ(p[1], 3.0497043132781982f);
        CHECK_EQ(p[2], 0.43148526549339294f);
        CHECK_EQ(p[3], 3.235734224319458f);
        CHECK_EQ(p[p.size() - 3], 0.5611537098884583f);
        CHECK_EQ(p[p.size() - 1], -0.009358466602861881f);
        REQUIRE(b->box.has_value());
        CHECK_EQ(b->box->size(), std::size_t(25));
        const auto& m = (*b->box)[0];
        CHECK(near(m[0], 3.3772107, 1e-6) && near(m[4], 3.3772107, 1e-6) && near(m[8], 3.3772107, 1e-6));
        CHECK(m[1] == 0.0 && m[2] == 0.0 && m[5] == 0.0);
    }
    {
        auto r = mdz::open_reader(std::string(MDZ_EXAMPLES_DIR) + "/every-frame.dcd");
        CHECK_EQ(r->natoms(), std::size_t(769));
        CHECK_EQ(r->nframes().value_or(999), std::size_t(120));
        auto b = r->read_all();
        REQUIRE(b.has_value());
        const auto& p = *b->positions();
        REQUIRE(p.size() == std::size_t(120 * 769 * 3));
        CHECK_EQ(p[0], 0.843000054359436f);
        CHECK_EQ(p[1], 1.0379999876022339f);
        CHECK_EQ(p[2], 0.9151000380516052f);
        CHECK_EQ(p[p.size() - 3], 0.24719659984111786f);
        CHECK_EQ(p[p.size() - 1], 0.2579815685749054f);
        REQUIRE(b->box.has_value());
        CHECK(near((*b->box)[0][0], 1.9716, 1e-6));
    }
}

// --- XTC ---
TEST(xtc_roundtrip) {
    auto b = sample(20, 4);
    b.box = boxes(4, {2.0, 70.0, 3.0, 110.0, 95.0, 4.0});
    b.frame_props["time"] = {0.5, 1.0, 1.5, 2.0};
    b.int_frame_props["step"] = {100, 200, 300, 400};
    auto r = reader("xtc", write("xtc", b));
    CHECK_EQ(r->natoms(), std::size_t(20));
    CHECK(!r->nframes().has_value());
    auto got = r->read_all();
    REQUIRE(got.has_value());
    CHECK_EQ(got->nframes, std::size_t(4));
    const auto& p = *b.positions();
    const auto& q = *got->positions();
    REQUIRE(q.size() == p.size());
    bool ok = true;
    for (std::size_t i = 0; i < p.size(); ++i) ok = ok && near(p[i], q[i], 0.0005 + 1e-6);
    CHECK(ok);
    CHECK(got->frame_props["time"] == std::vector<double>({0.5, 1.0, 1.5, 2.0}));
    CHECK(got->int_frame_props["step"] == std::vector<std::int64_t>({100, 200, 300, 400}));
    REQUIRE(got->box.has_value());
    for (std::size_t k = 0; k < 9; ++k)
        CHECK_EQ((*got->box)[2][k], static_cast<double>(static_cast<float>((*b.box)[2][k])));
}

TEST(xtc_small_frames_are_exact) {
    // Up to 9 atoms are stored as plain floats.
    auto b = sample(9, 3);
    auto got = read_all("xtc", write("xtc", b));
    CHECK(*got.positions() == *b.positions());
    // A frame without a box reads back with a zero box.
    REQUIRE(got.box.has_value());
    CHECK((*got.box)[0] == mdz::Box{});
}

TEST(xtc_step_and_time_fall_back_to_the_global_frame_index) {
    auto all = sample(12, 6);
    std::vector<mdz::FrameBatch> parts = {all.slice_frames(0, 2), all.slice_frames(2, 6)};
    const Bytes one = write("xtc", all);
    CHECK(write("xtc", parts) == one);
    CHECK(write("xtc", parts, false) == one);
    auto got = read_all("xtc", one);
    CHECK(got.int_frame_props["step"] == std::vector<std::int64_t>({0, 1, 2, 3, 4, 5}));
    CHECK_EQ(got.frame_props["time"][5], static_cast<double>(5.0f * 0.001f));
    // When a per-frame property is shorter than the batch, frames past its end
    // fall back to the frame index.
    auto b = sample(12, 3);
    b.int_frame_props["step"] = {7};
    got = read_all("xtc", write("xtc", b));
    CHECK(got.int_frame_props["step"] == std::vector<std::int64_t>({7, 1, 2}));
}

TEST(xtc_rejects_step_not_fitting_i32) {
    auto b = sample(12, 1);
    b.int_frame_props["step"] = {std::int64_t(1) << 31};
    CHECK(error_kind([&] { write("xtc", b); }) == mdz::ErrorKind::InvalidInput);
    b.int_frame_props["step"] = {INT32_MIN};
    auto got = read_all("xtc", write("xtc", b));
    CHECK_EQ(got.int_frame_props["step"][0], std::int64_t(INT32_MIN));
}

TEST(xtc_rejects_overflow_and_nonfinite) {
    // Frames with more than 9 atoms use the quantized path, which is where the
    // float-to-int32 conversion happens.
    for (float v : {1.0e9f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
        auto bad = sample(10, 1);
        (*bad.column("positions"))[0] = v;
        Bytes out;
        auto w = fmt("xtc").open_writer(mdz::memory_output(&out), 10, {});
        w->write_batch(sample(10, 1));
        CHECK(error_kind([&] { w->write_batch(bad); }) == mdz::ErrorKind::InvalidInput);
        w->finish();
        // The good frame is kept and no partial frame follows it.
        CHECK_EQ(read_all("xtc", out).nframes, std::size_t(1));
    }
    auto ok = sample(10, 1);
    (*ok.column("positions"))[0] = 1.234f;
    CHECK_EQ(read_all("xtc", write("xtc", ok)).nframes, std::size_t(1));
}

TEST(xtc_rejects_truncated_and_garbage) {
    CHECK(error_kind([] { reader("xtc", Bytes()); }) == mdz::ErrorKind::Corrupt);
    CHECK(error_kind([] { reader("xtc", Bytes(64, 0)); }) == mdz::ErrorKind::Corrupt);
    const auto three = sample(30, 3);
    const Bytes full = write("xtc", three);
    // XTC frames are self-contained, so the file is its frames concatenated.
    std::vector<std::size_t> ends;
    for (std::size_t f = 0; f < 3; ++f)
        ends.push_back((ends.empty() ? 0 : ends.back()) + write("xtc", three.slice_frames(f, f + 1)).size());
    CHECK_EQ(ends.back(), full.size());
    for (std::size_t cut = 0; cut < full.size(); ++cut) {
        Bytes head(full.begin(), full.begin() + static_cast<std::ptrdiff_t>(cut));
        std::size_t frames = 0;
        mdz::ErrorKind k = error_kind([&] {
            auto b = reader("xtc", head)->read_all();
            frames = b ? b->nframes : 0;
        });
        // A cut at a frame boundary, or up to 3 bytes past one, reads cleanly
        // because a partial magic number is treated as end of file. Any other
        // cut is Corrupt.
        std::size_t whole = 0;
        bool clean = false;
        for (std::size_t e : ends)
            if (cut >= e) {
                whole += 1;
                clean = cut - e < 4;
            }
        if (clean) {
            CHECK(k == mdz::ErrorKind::OutOfRange);
            CHECK_EQ(frames, whole);
        } else {
            CHECK(k == mdz::ErrorKind::Corrupt);
        }
    }
    // A later frame with a different atom count makes the batch variable.
    Bytes mixed = full;
    Bytes other = write("xtc", sample(31, 1));
    mixed.insert(mixed.end(), other.begin(), other.end());
    mdz::FrameBatch m = read_all("xtc", mixed);
    CHECK(m.variable());
    CHECK_EQ(m.frame_natoms->back(), std::size_t(31));
    // A payload length far beyond the end of the data.
    Bytes bad = full;
    bad[56 + 32] = 0x7f;
    CHECK(error_kind([&] { reader("xtc", bad)->read_all(); }) == mdz::ErrorKind::Corrupt);
    // A huge atom count in both header fields.
    bad = full;
    bad[4] = bad[52] = 0x7f;
    CHECK(error_kind([&] { reader("xtc", bad)->read_all(); }) == mdz::ErrorKind::Corrupt);
}

TEST(dcd_to_xtc_and_back) {
    auto dcd = mdz::open_reader(std::string(MDZ_EXAMPLES_DIR) + "/traj.dcd");
    Bytes xtc;
    {
        auto w = fmt("xtc").open_writer(mdz::memory_output(&xtc), dcd->natoms(), {});
        while (auto b = dcd->read_batch(7)) w->write_batch(*b);
        w->finish();
    }
    auto src = mdz::open_reader(std::string(MDZ_EXAMPLES_DIR) + "/traj.dcd")->read_all();
    auto got = read_all("xtc", xtc);
    REQUIRE(src.has_value());
    CHECK_EQ(got.nframes, std::size_t(25));
    CHECK_EQ(got.natoms, std::size_t(4096));
    const auto& p = *src->positions();
    const auto& q = *got.positions();
    REQUIRE(p.size() == q.size());
    double worst = 0;
    for (std::size_t i = 0; i < p.size(); ++i) worst = std::max(worst, std::fabs(double(p[i]) - q[i]));
    CHECK(worst <= 0.0005 + 1e-6);
    for (std::size_t f = 0; f < 25; ++f) {
        CHECK_EQ(got.int_frame_props["step"][f], std::int64_t(f));
        CHECK_EQ(got.frame_props["time"][f], double(static_cast<float>(f) * 0.001f));
        for (std::size_t k = 0; k < 9; ++k) CHECK_EQ((*got.box)[f][k], double(static_cast<float>((*src->box)[f][k])));
    }
}

// --- npy ---
TEST(npy_roundtrip) {
    auto b = sample(4, 5);
    const Bytes out = write("npy", b);
    // The header matches NumPy's, padded so the data starts on a 64-byte
    // boundary.
    std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (5, 4, 3), }";
    dict.append(128 - 10 - 1 - dict.size(), ' ');
    dict += '\n';
    REQUIRE(out.size() == 128 + 60 * 4);
    CHECK(std::string(out.begin() + 10, out.begin() + 128) == dict);
    CHECK_EQ(out[8] | (out[9] << 8), 118);
    auto r = reader("npy", out);
    CHECK_EQ(r->natoms(), std::size_t(4));
    CHECK_EQ(r->nframes().value_or(999), std::size_t(5));
    auto got = r->read_all();
    REQUIRE(got.has_value());
    CHECK(*got->positions() == *b.positions());
}

TEST(npy_patched_and_buffered_headers_match) {
    auto all = sample(3, 1000);
    std::vector<mdz::FrameBatch> parts = {all.slice_frames(0, 1), all.slice_frames(1, 999),
                                          all.slice_frames(999, 1000)};
    const Bytes one = write("npy", all, false);
    CHECK(write("npy", parts) == one);
    CHECK(write("npy", parts, false) == one);
    CHECK(*read_all("npy", one).positions() == *all.positions());
    // A writer with no frames.
    Bytes out;
    fmt("npy").open_writer(mdz::memory_output(&out), 3, {})->finish();
    auto r = reader("npy", out);
    CHECK_EQ(r->nframes().value_or(999), std::size_t(0));
    CHECK(!r->read_batch(1).has_value());
}

namespace {

Bytes npy_bytes(const std::string& dict, std::size_t nvalues, int major = 1) {
    Bytes h = {0x93, 'N', 'U', 'M', 'P', 'Y', static_cast<std::uint8_t>(major), 0};
    std::string d = dict + "\n";
    if (major == 1) {
        h.push_back(static_cast<std::uint8_t>(d.size()));
        h.push_back(static_cast<std::uint8_t>(d.size() >> 8));
    } else {
        for (int k = 0; k < 4; ++k) h.push_back(static_cast<std::uint8_t>(d.size() >> (8 * k)));
    }
    h.insert(h.end(), d.begin(), d.end());
    for (std::size_t i = 0; i < nvalues; ++i) {
        float v = static_cast<float>(i);
        std::uint8_t q[4];
        std::memcpy(q, &v, 4);
        h.insert(h.end(), q, q + 4);
    }
    return h;
}

}  // namespace

TEST(npy_reads_shapes_and_versions) {
    // A 2-D shape (natoms, 3) is read as a single frame.
    auto b = read_all("npy", npy_bytes("{'descr': '<f4', 'fortran_order': False, 'shape': (4, 3), }", 12));
    CHECK_EQ(b.nframes, std::size_t(1));
    CHECK_EQ(b.natoms, std::size_t(4));
    CHECK_EQ((*b.positions())[11], 11.0f);
    // Version 2 headers, and the native byte-order spellings of the dtype.
    b = read_all("npy", npy_bytes("{'descr': 'f4', 'fortran_order': False, 'shape': (2, 2, 3), }", 12, 2));
    CHECK_EQ(b.nframes, std::size_t(2));
    b = read_all("npy", npy_bytes("{'descr': '=f4', 'shape': (1,2,3)}", 6));
    CHECK_EQ(b.natoms, std::size_t(2));
}

TEST(npy_rejects_bad_files) {
    auto corrupt = [](const Bytes& bytes) {
        return error_kind([&] { reader("npy", bytes)->read_all(); }) == mdz::ErrorKind::Corrupt;
    };
    const std::string ok = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2, 3), }";
    CHECK(!corrupt(npy_bytes(ok, 12)));
    CHECK(corrupt(Bytes(5, 0)));
    CHECK(corrupt(Bytes(64, 0)));
    CHECK(corrupt(npy_bytes(ok, 12, 3)));  // version
    CHECK(corrupt(npy_bytes(ok, 11)));     // short
    CHECK(corrupt(npy_bytes(ok, 13)));     // long
    Bytes odd = npy_bytes(ok, 12);
    odd.pop_back();
    CHECK(corrupt(odd));  // not a whole number of floats
    odd = npy_bytes(ok, 12);
    odd.push_back(0);
    CHECK(corrupt(odd));
    CHECK(corrupt(npy_bytes("{'descr': '<f8', 'fortran_order': False, 'shape': (2, 2, 3), }", 24)));
    CHECK(corrupt(npy_bytes("{'descr': '>f4', 'fortran_order': False, 'shape': (2, 2, 3), }", 12)));
    CHECK(corrupt(npy_bytes("{'descr': '<f4', 'fortran_order': True, 'shape': (2, 2, 3), }", 12)));
    CHECK(corrupt(npy_bytes("{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2, 4), }", 16)));
    CHECK(corrupt(npy_bytes("{'descr': '<f4', 'fortran_order': False, 'shape': (12,), }", 12)));
    CHECK(corrupt(npy_bytes("{'descr': '<f4', 'fortran_order': False, 'shape': (2, x, 3), }", 12)));
    CHECK(corrupt(npy_bytes("{'descr': '<f4', 'fortran_order': False, }", 12)));
    CHECK(corrupt(npy_bytes("{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2, 3", 12)));
    // A shape whose element count overflows, and one far larger than the
    // data.
    CHECK(corrupt(npy_bytes("{'descr': '<f4', 'shape': (18446744073709551615, 18446744073709551615, 3), }", 12)));
    CHECK(corrupt(npy_bytes("{'descr': '<f4', 'shape': (1000000, 1000000, 3), }", 12)));
    // A header length past the end of the file.
    Bytes cut = npy_bytes(ok, 0);
    cut.resize(40);
    CHECK(corrupt(cut));
    // The writer requires positions.
    CHECK(error_kind([] { write("npy", mdz::FrameBatch(2, 1)); }) == mdz::ErrorKind::InvalidInput);
}

TEST(binary_formats_on_disk) {
    // open_writer and open_reader on real files, choosing the format by
    // extension.
    const std::string dir = mdz_test::scratch_dir();
    auto b = sample(11, 3);
    b.box = boxes(3, {2.0, 90.0, 2.0, 90.0, 90.0, 2.0});
    for (const char* ext : {"dcd", "xtc", "npy"}) {
        const std::string path = dir + "/binfmt." + ext;
        {
            auto w = mdz::open_writer(path, 11);
            w->write_batch(b.slice_frames(0, 1));
            w->write_batch(b.slice_frames(1, 3));
            w->finish();
        }
        CHECK(file_bytes(path) == write(ext, b));
        auto r = mdz::open_reader(path);
        auto got = r->read_all();
        REQUIRE(got.has_value());
        CHECK_EQ(got->nframes, std::size_t(3));
        CHECK_EQ(got->natoms, std::size_t(11));
    }
}
