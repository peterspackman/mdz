// Trajectory formats with a varying atom count. Readers that see atoms come
// and go return variable batches. Writers for formats that record a count per
// frame write them, and formats with a single count per file reject them.

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <optional>
#include <sstream>
#include <string>

#include "mdz/error.hpp"
#include "mdz/mdz.hpp"
#include "test.hpp"

namespace {

using mdz::FrameBatch;

std::vector<std::uint8_t> bytes_of(std::string_view s) { return std::vector<std::uint8_t>(s.begin(), s.end()); }

const mdz::FormatInfo& fmt(std::string_view name) {
    const mdz::FormatInfo* f = mdz::find_format(name);
    REQUIRE(f != nullptr);
    return *f;
}

std::unique_ptr<mdz::TrajectoryReader> reader(std::string_view format, const std::vector<std::uint8_t>& bytes) {
    return fmt(format).open_reader(mdz::memory_input(bytes, "test." + std::string(format)), {});
}

FrameBatch read_all(std::string_view format, const std::vector<std::uint8_t>& bytes) {
    auto b = reader(format, bytes)->read_all();
    REQUIRE(b.has_value());
    return std::move(*b);
}

FrameBatch read_all(std::string_view format, std::string_view text) { return read_all(format, bytes_of(text)); }

std::vector<std::uint8_t> write(std::string_view format, const std::vector<const FrameBatch*>& batches) {
    std::vector<std::uint8_t> out;
    auto w = fmt(format).open_writer(mdz::memory_output(&out), batches.empty() ? 0 : batches[0]->natoms, {});
    for (const FrameBatch* b : batches) w->write_batch(*b);
    w->finish();
    return out;
}

std::vector<std::uint8_t> write(std::string_view format, const FrameBatch& b) { return write(format, {&b}); }

std::string text_of(const std::vector<std::uint8_t>& b) { return std::string(b.begin(), b.end()); }

// Returns the kind of mdz::Error that f throws, or nullopt if it does not throw.
std::optional<mdz::ErrorKind> error_kind(const std::function<void()>& f) {
    try {
        f();
    } catch (const mdz::Error& e) {
        return e.kind();
    }
    return std::nullopt;
}

// A deposition-style run. Each frame holds the atoms alive at that point,
// sorted by id, and each atom moves smoothly. One atom is added every frame
// and one is removed every third frame. The batch is built a frame at a time,
// as readers do. Positions are in nm and exact to 3 decimals in Angstrom, so
// every text format round-trips them.
FrameBatch deposition(std::size_t nframes, bool with_types = true) {
    std::map<std::int32_t, int> alive;  // id -> birth frame
    std::int32_t next = 1;
    for (int i = 0; i < 5; ++i) alive[next++] = 0;
    FrameBatch all;
    for (std::size_t f = 0; f < nframes; ++f) {
        if (f > 0) alive[next++] = static_cast<int>(f);
        if (f % 3 == 2) alive.erase(alive.begin());
        FrameBatch b(alive.size(), 1);
        auto& p = b.columns["positions"];
        std::vector<std::int32_t> ids, types;
        std::vector<std::string> names;
        for (const auto& [id, born] : alive) {
            const float t = static_cast<float>(f) - static_cast<float>(born);
            p.insert(p.end(), {0.1f * static_cast<float>(id), 0.05f + 0.001f * t, 0.2f});
            ids.push_back(id);
            types.push_back(1 + id % 2);
            names.push_back(id % 2 ? "Ar" : "Ne");
        }
        b.atom_ids = mdz::share(std::move(ids));
        if (with_types) b.atom_types = mdz::share(std::move(types));
        b.atom_names = mdz::share(std::move(names));
        b.box = std::vector<mdz::Box>{{2, 0, 0, 0, 2, 0, 0, 0, 2}};
        b.int_frame_props["step"] = {static_cast<std::int64_t>(f) * 10};
        all.append(b);
    }
    return all;
}

// Checks frame by frame that atom counts, ids (if `ids`) and positions (in nm,
// within `tol`) match.
void check_frames(const FrameBatch& want, const FrameBatch& got, float tol, bool ids = true) {
    REQUIRE(got.nframes == want.nframes);
    for (std::size_t f = 0; f < want.nframes; ++f) {
        FrameBatch a = want.frame(f).in_internal_units(), b = got.frame(f).in_internal_units();
        REQUIRE(a.natoms == b.natoms);
        if (ids) {
            REQUIRE(b.atom_ids != nullptr);
            CHECK(*a.atom_ids == *b.atom_ids);
        }
        const auto &p = *a.positions(), &q = *b.positions();
        REQUIRE(p.size() == q.size());
        for (std::size_t i = 0; i < p.size(); ++i)
            if (std::fabs(p[i] - q[i]) > tol) {
                CHECK(std::fabs(p[i] - q[i]) <= tol);
                return;
            }
    }
}

std::string lammps_frame(std::int64_t step, const std::vector<int>& ids, const char* extra = "") {
    std::ostringstream o;
    o << "ITEM: TIMESTEP\n"
      << step << "\nITEM: NUMBER OF ATOMS\n"
      << ids.size() << "\nITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\nITEM: ATOMS id type x y z" << extra << "\n";
    for (int id : ids) {
        o << id << " " << 1 + id % 3 << " " << id << ".5 " << step << " 1";
        if (*extra) o << " " << id * 2;
        o << "\n";
    }
    return o.str();
}

}  // namespace

// --- LAMMPS ---
TEST(lammps_reader_follows_atoms_that_come_and_go) {
    // Atom 4 is deposited, then atom 2 is deleted. The ids are out of order in
    // one frame.
    const std::string dump = lammps_frame(0, {1, 2, 3}) + lammps_frame(10, {1, 2, 3}) + lammps_frame(20, {3, 1, 4, 2}) +
                             lammps_frame(30, {1, 3, 4});
    auto r = reader("lammps", bytes_of(dump));
    // The first two frames have the same atoms, so they form a fixed batch.
    auto b1 = r->read_batch(2);
    REQUIRE(b1.has_value());
    CHECK(!b1->variable());
    CHECK_EQ(b1->natoms, std::size_t(3));
    // From the first change onwards, batches are variable.
    auto b2 = r->read_batch(5);
    REQUIRE(b2.has_value());
    REQUIRE(b2->variable());
    CHECK(*b2->frame_natoms == (std::vector<std::size_t>{4, 3}));
    CHECK_EQ(r->natoms(), std::size_t(3));  // count from the first frame
    CHECK(!r->read_batch(5));
    // Atoms are sorted by id, with ids and types taken from the dump.
    FrameBatch f = b2->frame(0);
    CHECK(*f.atom_ids == (std::vector<std::int32_t>{1, 2, 3, 4}));
    CHECK(*f.atom_types == (std::vector<std::int32_t>{2, 3, 1, 2}));
    CHECK(f.positions()->at(9) == 4.5f);  // x of atom 4, in the dump's units
    CHECK(*b2->frame(1).atom_ids == (std::vector<std::int32_t>{1, 3, 4}));
    CHECK(b2->int_frame_props.at("step") == (std::vector<std::int64_t>{20, 30}));
    CHECK_EQ(b2->attributes.at("lammps.units"), std::string("real"));

    // Reading the whole file gives one variable batch holding every frame.
    FrameBatch all = read_all("lammps", dump);
    REQUIRE(all.variable());
    CHECK(*all.frame_natoms == (std::vector<std::size_t>{3, 3, 4, 3}));
    // Extra per-atom columns are reordered along with their atoms.
    FrameBatch extra = read_all("lammps", lammps_frame(0, {2, 1}, " c_q") + lammps_frame(1, {1, 3, 2}, " c_q"));
    REQUIRE(extra.variable());
    CHECK(extra.columns.at("c_q") == (std::vector<float>{2, 4, 2, 4, 6}));
}

TEST(lammps_writer_writes_each_frame_with_its_atoms) {
    FrameBatch b = deposition(9);
    REQUIRE(b.variable());
    const std::string text = text_of(write("lammps", b));
    // Each frame records its own atom count and the ids and types of its atoms.
    CHECK(text.find("ITEM: NUMBER OF ATOMS\n5\n") != std::string::npos);
    CHECK(text.find("ITEM: NUMBER OF ATOMS\n7\n") != std::string::npos);
    FrameBatch back = read_all("lammps", text);
    REQUIRE(back.variable());
    CHECK(*back.frame_natoms == *b.frame_natoms);
    check_frames(b, back, 1e-6f);
    for (std::size_t f = 0; f < b.nframes; ++f) CHECK(*back.frame(f).atom_types == *b.frame(f).atom_types);
    CHECK(back.int_frame_props.at("step") == b.int_frame_props.at("step"));
    // Writing the frames as a sequence of fixed batches with different counts
    // gives the same dump as writing the equivalent variable batch.
    std::vector<FrameBatch> frames;
    std::vector<const FrameBatch*> ptrs;
    for (std::size_t f = 0; f < b.nframes; ++f) frames.push_back(b.frame(f));
    for (const auto& f : frames) ptrs.push_back(&f);
    CHECK(text_of(write("lammps", ptrs)) == text);
}

TEST(lammps_varying_dump_round_trips_through_mdz) {
    FrameBatch b = deposition(23);
    const std::string text = text_of(write("lammps", b));
    FrameBatch in = read_all("lammps", text);
    mdz::CompressOptions o;
    o.gof_size = 5;
    auto r = mdz::Reader::from_bytes(mdz::to_bytes(mdz::compress(in, o)));
    CHECK(r.variable_atoms());
    CHECK(r.frame_natoms() == *b.frame_natoms);
    FrameBatch back = r.read_batch(0, r.nframes());
    // Writing back to a dump preserves the atoms in each frame.
    FrameBatch again = read_all("lammps", write("lammps", back));
    CHECK(*again.frame_natoms == *b.frame_natoms);
    check_frames(in, again, 0.001f);
    for (std::size_t f = 0; f < b.nframes; ++f) CHECK(*again.frame(f).atom_types == *in.frame(f).atom_types);
}

// --- extxyz / xyz ---
TEST(extxyz_reader_takes_ids_from_the_id_property) {
    const std::string two =
        "2\nProperties=species:S:1:pos:R:3:id:I:1\nO 0 0 0 7\nH 1 0 0 3\n"
        "3\nProperties=species:S:1:pos:R:3:id:I:1\nO 0 0 0 7\nH 1 0 0 3\nC 2 0 0 9\n";
    FrameBatch b = read_all("extxyz", two);
    REQUIRE(b.variable());
    CHECK(*b.frame_natoms == (std::vector<std::size_t>{2, 3}));
    CHECK(*b.frame(1).atom_ids == (std::vector<std::int32_t>{7, 3, 9}));
    CHECK(*b.frame(1).atom_names == (std::vector<std::string>{"O", "H", "C"}));
    // A frame with the same count but different ids is also variable.
    FrameBatch c = read_all("extxyz",
                            "1\nProperties=species:S:1:pos:R:3:id:I:1\nO 0 0 0 1\n"
                            "1\nProperties=species:S:1:pos:R:3:id:I:1\nO 0 0 0 2\n");
    CHECK(c.variable());
    // Without ids, atoms are identified by their position in the frame.
    FrameBatch d = read_all("extxyz",
                            "1\nProperties=species:S:1:pos:R:3\nO 0 0 0\n"
                            "2\nProperties=species:S:1:pos:R:3\nO 0 0 0\nH 1 1 1\n");
    REQUIRE(d.variable());
    CHECK(d.int_columns.count(mdz::columns::kId) == 0);
    // A fixed-count file with ids reads as a fixed batch and writes back
    // byte-identical.
    const std::string fixed =
        "2\nProperties=species:S:1:pos:R:3:id:I:1 pbc=\"T T T\"\n"
        "O 0.000000 0.000000 0.000000 7\nH 1.000000 0.000000 0.000000 3\n";
    FrameBatch e = read_all("extxyz", fixed + fixed);
    CHECK(!e.variable());
    CHECK(*e.atom_ids == (std::vector<std::int32_t>{7, 3}));
    CHECK(text_of(write("extxyz", e)) == fixed + fixed);
}

TEST(extxyz_writer_writes_each_frame_with_its_atoms_and_ids) {
    FrameBatch b = deposition(10);
    const std::string text = text_of(write("extxyz", b));
    CHECK(text.find("Properties=species:S:1:pos:R:3:id:I:1 ") != std::string::npos);
    FrameBatch back = read_all("extxyz", text);
    REQUIRE(back.variable());
    CHECK(*back.frame_natoms == *b.frame_natoms);
    check_frames(b, back, 1e-6f);
    for (std::size_t f = 0; f < b.nframes; ++f) CHECK(*back.frame(f).atom_names == *b.frame(f).atom_names);
    // Successive batches with different atom counts.
    FrameBatch f0 = b.frame(0), f3 = b.frame(3);
    FrameBatch two = read_all("extxyz", write("extxyz", {&f0, &f3}));
    CHECK(*two.frame_natoms == (std::vector<std::size_t>{5, 7}));
}

TEST(xyz_reads_and_writes_frames_of_different_sizes) {
    FrameBatch b = read_all("xyz", "1\nc\nC 0 0 0\n2\nc\nC 0 0 0\nO 1 2 3\n2\nc\nC 0 0 1\nO 1 2 3\n");
    REQUIRE(b.variable());
    CHECK(*b.frame_natoms == (std::vector<std::size_t>{1, 2, 2}));
    CHECK(*b.frame(2).atom_names == (std::vector<std::string>{"C", "O"}));
    CHECK(b.positions()->at(11) == 0.1f);
    // An empty first frame has no names to compare against.
    FrameBatch e = read_all("xyz", "0\nc\n1\nc\nC 0 0 0\n");
    REQUIRE(e.variable());
    CHECK(*e.frame_natoms == (std::vector<std::size_t>{0, 1}));
    CHECK(*e.frame(1).atom_names == (std::vector<std::string>{"C"}));
    // Reading one frame at a time, every batch after the change is variable.
    auto r = reader("xyz", bytes_of("2\nc\nC 0 0 0\nO 1 2 3\n2\nc\nN 0 0 0\nO 1 2 3\n2\nc\nN 0 0 0\nO 1 2 3\n"));
    CHECK(!r->read_batch(1)->variable());
    auto second = r->read_batch(1);
    CHECK(second->variable());
    CHECK(*second->frame(0).atom_names == (std::vector<std::string>{"N", "O"}));
    CHECK(r->read_batch(1)->variable());

    FrameBatch d = deposition(8);
    FrameBatch back = read_all("xyz", write("xyz", d));
    REQUIRE(back.variable());
    CHECK(*back.frame_natoms == *d.frame_natoms);
    check_frames(d, back, 1e-6f, false);
    for (std::size_t f = 0; f < d.nframes; ++f) CHECK(*back.frame(f).atom_names == *d.frame(f).atom_names);
}

// --- PDB ---
TEST(pdb_models_of_different_sizes_vary_by_serial) {
    const char* models =
        "MODEL        1\n"
        "ATOM      5  O   WAT A   1       1.000   2.000   3.000\n"
        "ATOM      9  H   WAT A   1       1.500   2.000   3.000\nENDMDL\n"
        "MODEL        2\n"
        "ATOM      9  H   WAT A   1       1.500   2.500   3.000\nENDMDL\nEND\n";
    FrameBatch b = read_all("pdb", models);
    REQUIRE(b.variable());
    CHECK(*b.frame_natoms == (std::vector<std::size_t>{2, 1}));
    FrameBatch f1 = b.frame(1);
    CHECK(*f1.atom_ids == (std::vector<std::int32_t>{9}));
    CHECK(*f1.atom_names == (std::vector<std::string>{"H"}));
    CHECK(*f1.residue_names == (std::vector<std::string>{"WAT"}));

    FrameBatch d = deposition(7);
    const std::string text = text_of(write("pdb", d));
    // Every frame, including the first, is written as a MODEL.
    CHECK(text.find("MODEL        1\n") != std::string::npos);
    FrameBatch back = read_all("pdb", text);
    REQUIRE(back.variable());
    CHECK(*back.frame_natoms == *d.frame_natoms);
    check_frames(d, back, 1e-6f);
    for (std::size_t f = 0; f < d.nframes; ++f) CHECK(*back.frame(f).atom_names == *d.frame(f).atom_names);
}

// --- XTC ---
TEST(xtc_reads_and_writes_frames_of_different_sizes) {
    FrameBatch d = deposition(12);
    // XTC has no ids, so atoms are matched by position in the frame.
    FrameBatch back = read_all("xtc", write("xtc", d));
    REQUIRE(back.variable());
    CHECK(*back.frame_natoms == *d.frame_natoms);
    check_frames(d, back, 0.0006f, false);
    CHECK(back.int_frame_props.at("step") == d.int_frame_props.at("step"));
    // Write fixed batches with different counts in sequence. When read back in
    // small batches, the first is fixed and every batch from the change
    // onwards is variable.
    FrameBatch a = d.frame(0), c = d.frame(5);
    auto r = reader("xtc", write("xtc", {&a, &a, &c}));
    auto first = r->read_batch(2);
    CHECK(!first->variable());
    CHECK_EQ(first->natoms, a.natoms);
    auto rest = r->read_batch(2);
    REQUIRE(rest->variable());
    CHECK(*rest->frame_natoms == (std::vector<std::size_t>{c.natoms}));
    CHECK_EQ(r->natoms(), a.natoms);
}

// --- formats with one atom count per file ---
TEST(dcd_and_npy_refuse_a_varying_atom_count) {
    FrameBatch d = deposition(4);
    FrameBatch f0 = d.frame(0), f3 = d.frame(3);
    for (const char* name : {"dcd", "npy"}) {
        try {
            write(name, d);
            CHECK(false);
        } catch (const mdz::Error& e) {
            CHECK(e.kind() == mdz::ErrorKind::InvalidInput);
            CHECK(std::string(e.what()).find(name == std::string("dcd") ? "DCD" : "npy") != std::string::npos);
            CHECK(std::string(e.what()).find("varies") != std::string::npos);
        }
        CHECK(error_kind([&] { write(name, {&f0, &f3}); }) == mdz::ErrorKind::InvalidInput);
        // A constant atom count is accepted.
        CHECK(!write(name, {&f0, &f0}).empty());
    }
}

// --- data loss reporting ---
TEST(data_loss_of_a_varying_batch_names_the_metadata) {
    FrameBatch d = deposition(4);
    REQUIRE(d.variable());
    auto has = [](const std::vector<std::string>& v, const char* s) {
        return std::find(v.begin(), v.end(), s) != v.end();
    };
    // In this batch the atom types are per-atom columns, and LAMMPS keeps
    // them.
    auto lammps = mdz::check_data_loss(d, "lammps");
    CHECK(!has(lammps, "atom_types"));
    CHECK(!has(lammps, "atom_ids"));
    CHECK(has(lammps, "atom_names"));
    for (const auto& s : lammps) CHECK(s.rfind("mdz.", 0) != 0);
    // extxyz keeps atom names, and keeps ids for a variable batch.
    auto ext = mdz::check_data_loss(d, "extxyz");
    CHECK(!has(ext, "atom_names"));
    CHECK(!has(ext, "atom_ids"));
    CHECK(has(ext, "atom_types"));
    // For a fixed batch the extxyz writer does not write ids.
    CHECK(has(mdz::check_data_loss(d.frame(0), "extxyz"), "atom_ids"));
    auto xyz = mdz::check_data_loss(d, "xyz");
    CHECK(has(xyz, "atom_ids"));
    CHECK(has(xyz, "atom_types"));
    CHECK(!has(xyz, "atom_names"));
    auto pdb = mdz::check_data_loss(d, "pdb");
    CHECK(!has(pdb, "atom_ids"));
    CHECK(!has(pdb, "atom_names"));
}
