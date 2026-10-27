// Text formats: xyz, extxyz, PDB and LAMMPS dumps. Writer output is compared
// byte for byte against reference files in tests/cpp/fixtures/text.

#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

#include "mdz/error.hpp"
#include "mdz/mdz.hpp"
#include "formats/text.hpp"
#include "test.hpp"

namespace {

using mdz::FrameBatch;

std::vector<std::uint8_t> bytes_of(std::string_view s) { return std::vector<std::uint8_t>(s.begin(), s.end()); }

FrameBatch read_text(std::string_view format, std::string_view text, const mdz::FormatOptions& opts = {}) {
    auto r = mdz::find_format(format)->open_reader(mdz::memory_input(bytes_of(text), "test." + std::string(format)), opts);
    auto b = r->read_all();
    REQUIRE(b.has_value());
    return std::move(*b);
}

std::string write_text(std::string_view format, const std::vector<const FrameBatch*>& batches,
                       const mdz::FormatOptions& opts = {}) {
    std::vector<std::uint8_t> out;
    auto w = mdz::find_format(format)->open_writer(mdz::memory_output(&out), batches.empty() ? 0 : batches[0]->natoms,
                                                   opts);
    for (const FrameBatch* b : batches) w->write_batch(*b);
    w->finish();
    return std::string(out.begin(), out.end());
}

std::string write_text(std::string_view format, const FrameBatch& b, const mdz::FormatOptions& opts = {}) {
    return write_text(format, std::vector<const FrameBatch*>{&b}, opts);
}

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.good());
    std::ostringstream os;
    os << f.rdbuf();
    return os.str();
}

std::string fixture(const std::string& name) { return slurp(std::string(MDZ_FIXTURES_DIR) + "/text/" + name); }

// Compares byte for byte and reports the first line that differs.
void check_same_text(const std::string& got, const std::string& want, const char* what) {
    if (got == want) return;
    std::istringstream g(got), w(want);
    std::string lg, lw;
    int line = 1;
    while (true) {
        const bool a = static_cast<bool>(std::getline(g, lg)), b = static_cast<bool>(std::getline(w, lw));
        if (!a && !b) break;
        if (!a || !b || lg != lw) break;
        ++line;
    }
    mdz_test::fail(__FILE__, __LINE__,
                   std::string(what) + ": differs at line " + std::to_string(line) + ":\n  got:  " + lg +
                       "\n  want: " + lw,
                   false);
}

mdz::Shared<std::string> strs(std::vector<std::string> v) { return mdz::share(std::move(v)); }

std::vector<std::string> lines_of(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream is(s);
    for (std::string l; std::getline(is, l);) out.push_back(l);
    return out;
}

bool near(double a, double b, double tol) { return std::fabs(a - b) < tol; }

const mdz::LammpsUnits* style(const char* name) { return mdz::LammpsUnits::from_style(name); }

mdz::FormatOptions lammps_opts(const char* units) {
    mdz::FormatOptions o;
    o.lammps_units = style(units);
    return o;
}

}  // namespace

// --- xyz ---
TEST(xyz_roundtrip) {
    FrameBatch b(2, 2);
    b.atom_names = strs({"C", "O"});
    b.columns["positions"] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.11f, 0.21f, 0.31f, 0.41f, 0.51f, 0.61f};
    const std::string dir = mdz_test::scratch_dir();
    {
        auto w = mdz::open_writer(dir + "/rt.xyz", 2);
        w->write_batch(b);
        w->finish();
    }
    auto r = mdz::open_reader(dir + "/rt.xyz");
    auto back = r->read_batch(100);
    REQUIRE(back.has_value());
    CHECK_EQ(back->natoms, std::size_t(2));
    CHECK_EQ(back->nframes, std::size_t(2));
    const auto& p0 = b.columns["positions"];
    const auto& p1 = back->columns["positions"];
    REQUIRE(p1.size() == p0.size());
    for (std::size_t i = 0; i < p0.size(); ++i) CHECK(std::fabs(p0[i] - p1[i]) < 1e-4f);
    CHECK_EQ((*back->atom_names)[1], std::string("O"));
}

TEST(xyz_errors) {
    CHECK_THROWS(read_text("xyz", "2\nc\nC 0 0 0\nO 1 2\n"));
    CHECK_THROWS(read_text("xyz", "2\nc\nC 0 0 0\n"));
    CHECK_THROWS(read_text("xyz", "x\nc\n"));
    CHECK_THROWS(read_text("xyz", "1\nc\nC 0 0 zz\n"));
    try {
        read_text("xyz", "1\nc\nC 0 0 zz\n");
    } catch (const mdz::Error& e) {
        CHECK(e.kind() == mdz::ErrorKind::Corrupt);
        CHECK(std::string(e.what()).find("test.xyz:3") != std::string::npos);
    }
    FrameBatch none(1, 1);
    CHECK_THROWS(write_text("xyz", none));
}

// --- extxyz ---
TEST(extxyz_parse_properties) {
    // species:S:1:pos:R:3:vel:R:3 maps to names, positions and velocities.
    auto b = read_text("extxyz", "1\nProperties=species:S:1:pos:R:3:vel:R:3\nO 1 2 3 4 5 6\n");
    CHECK_EQ((*b.atom_names)[0], std::string("O"));
    CHECK(near(b.columns["positions"][2], 0.3, 1e-7));
    CHECK(near(b.columns["velocities"][0], 0.4, 1e-7));
    CHECK_EQ(b.columns.size(), std::size_t(2));
    // Column aliases, skipped string columns, a default type and a trailing
    // partial triple.
    b = read_text("extxyz", "1\nProperties=species:S:1:pos:R:3:name:S:1:velo:R:3:forces::1:x:R\nO 1 2 3 n 4 5 6 7\n");
    CHECK(near(b.columns["velocities"][2], 0.6, 1e-7));
    CHECK(near(b.columns["forces"][0], 7 * 96.4853 * 10, 1e-2));
    CHECK_EQ(b.columns.size(), std::size_t(3));
}

TEST(extxyz_parse_comment) {
    auto b = read_text("extxyz",
                       "1\nLattice=\"1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0\" Properties=species:S:1:pos:R:3 "
                       "pbc=\"T T T\" Time=2.5 bare energy=1 step=7\nO 1 2 3\n");
    REQUIRE(b.box.has_value());
    CHECK(near((*b.box)[0][4], 0.1, 1e-12));
    CHECK_EQ(b.frame_props["time"][0], 2.5);
    CHECK(near(b.frame_props["energy"][0], 96.4853, 1e-12));
    CHECK_EQ(b.int_frame_props["step"][0], std::int64_t(7));
    // Without a Properties key the columns default to species and positions.
    b = read_text("extxyz", "1\nTime=x\nO 1 2 3\n");
    CHECK(b.frame_props.empty());
    CHECK_EQ((*b.atom_names)[0], std::string("O"));
}

TEST(extxyz_parse_lattice) {
    auto b = read_text("extxyz", "1\nLattice=\"10.0 0.0 0.0 0.0 20.0 0.0 0.0 0.0 30.0\"\nO 1 2 3\n");
    auto cell = b.cell_params();
    REQUIRE(cell.has_value());
    const auto& uc = (*cell)[0];
    CHECK(near(uc[0], 1.0, 1e-6));
    CHECK(near(uc[1], 90.0, 1e-4));
    CHECK(near(uc[2], 2.0, 1e-6));
    CHECK(near(uc[5], 3.0, 1e-6));
    // The full lattice matrix is kept, including tilt.
    b = read_text("extxyz", "1\nLattice=\"10 0 0 2 9 0 -1 0.5 8\"\nO 1 2 3\n");
    CHECK(near((*b.box)[0][3], 0.2, 1e-12));
    CHECK(near((*b.box)[0][6], -0.1, 1e-12));
    // A lattice without exactly nine numbers gives no box.
    b = read_text("extxyz", "1\nLattice=\"10 0 0 2 9 0 -1 0.5\"\nO 1 2 3\n");
    CHECK(!b.box.has_value());
}

TEST(extxyz_rejects_short_atom_line) {
    CHECK_THROWS(read_text("extxyz", "2\nProperties=species:S:1:pos:R:3\nO 1.0 2.0 3.0\nH 4.0 5.0\n"));
    CHECK_THROWS(read_text("extxyz", "2\nProperties=species:S:1:pos:R:3\nO 1.0 2.0 3.0\n"));
    CHECK_THROWS(read_text("extxyz", "1\nProperties=species:S:1:pos:R:3:n:I:1\nO 1.0 2.0 3.0 1.5\n"));
}

TEST(extxyz_roundtrip) {
    FrameBatch b(3, 2);
    b.atom_names = strs({"O", "H", "H"});
    b.columns["positions"] = {0.1f,  0.2f,  0.3f,  0.4f,  0.5f,  0.6f,  0.7f,  0.8f,  0.9f,
                              0.11f, 0.21f, 0.31f, 0.41f, 0.51f, 0.61f, 0.71f, 0.81f, 0.91f};
    const mdz::Box cube = mdz::cell_params_to_box({3.0, 90.0, 3.0, 90.0, 90.0, 3.0});
    b.box = std::vector<mdz::Box>{cube, cube};
    const std::string text = write_text("extxyz", b);
    auto back = read_text("extxyz", text);
    CHECK_EQ(back.natoms, std::size_t(3));
    CHECK_EQ(back.nframes, std::size_t(2));
    for (std::size_t i = 0; i < 18; ++i) CHECK(std::fabs(back.columns["positions"][i] - b.columns["positions"][i]) < 1e-4f);
    REQUIRE(back.box.has_value());
    CHECK_EQ(back.box->size(), std::size_t(2));
    CHECK(near((*back.cell_params())[0][0], 3.0, 0.01));
}

TEST(extxyz_integer_columns_roundtrip_and_order) {
    auto b = read_text("extxyz",
                       "2\nProperties=species:S:1:pos:R:3:tag:I:1:mol:I:1 pbc=\"T T T\"\n"
                       "O 1.0 2.0 3.0 100 7\nH 4.0 5.0 6.0 200 8\n");
    CHECK(b.int_columns["tag"] == (std::vector<std::int32_t>{100, 200}));
    CHECK(b.int_columns["mol"] == (std::vector<std::int32_t>{7, 8}));
    CHECK(!b.columns.count("tag"));
    const std::string text = write_text("extxyz", b);
    CHECK(text.find("Properties=species:S:1:pos:R:3:mol:I:1:tag:I:1") != std::string::npos);
    const std::string atom0 = lines_of(text)[2];
    CHECK(atom0.size() > 6 && atom0.substr(atom0.size() - 6) == " 7 100");
    auto b2 = read_text("extxyz", text);
    CHECK(b2.int_columns["mol"] == (std::vector<std::int32_t>{7, 8}));
    CHECK(b2.int_columns["tag"] == (std::vector<std::int32_t>{100, 200}));
}

// --- pdb ---
TEST(pdb_roundtrip) {
    FrameBatch b(3, 2);
    b.atom_names = strs({"O", "H", "H"});
    b.columns["positions"] = {0.1f,  0.2f,  0.3f,  0.4f,  0.5f,  0.6f,  0.7f,  0.8f,  0.9f,
                              0.11f, 0.21f, 0.31f, 0.41f, 0.51f, 0.61f, 0.71f, 0.81f, 0.91f};
    const mdz::Box cube = mdz::cell_params_to_box({3.0, 90.0, 3.0, 90.0, 90.0, 3.0});
    b.box = std::vector<mdz::Box>{cube, cube};
    auto back = read_text("pdb", write_text("pdb", b));
    CHECK_EQ(back.natoms, std::size_t(3));
    CHECK_EQ(back.nframes, std::size_t(2));
    for (std::size_t i = 0; i < 18; ++i) CHECK(std::fabs(back.columns["positions"][i] - b.columns["positions"][i]) < 1e-3f);
    REQUIRE(back.atom_names);
    CHECK(*back.atom_names == (std::vector<std::string>{"O", "H", "H"}));
    REQUIRE(back.box.has_value());
    CHECK((*back.box)[0] == cube);
}

TEST(pdb_topology_roundtrip) {
    FrameBatch b(3, 1);
    b.columns["positions"] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
    b.atom_names = strs({"N", "CA", "O"});
    b.elements = strs({"N", "C", "O"});
    b.atom_ids = mdz::share(std::vector<std::int32_t>{10, 11, 12});
    b.residue_names = strs({"ALA", "ALA", "HOH"});
    b.residue_ids = mdz::share(std::vector<std::int32_t>{1, 1, 2});
    b.chain_ids = strs({"A", "A", "B"});
    b.hetatm = mdz::share(std::vector<std::uint8_t>{0, 0, 1});
    auto back = read_text("pdb", write_text("pdb", b));
    CHECK(*back.atom_names == (std::vector<std::string>{"N", "CA", "O"}));
    CHECK(*back.elements == (std::vector<std::string>{"N", "C", "O"}));
    CHECK(*back.residue_names == (std::vector<std::string>{"ALA", "ALA", "HOH"}));
    CHECK(*back.residue_ids == (std::vector<std::int32_t>{1, 1, 2}));
    CHECK(*back.chain_ids == (std::vector<std::string>{"A", "A", "B"}));
    CHECK(*back.hetatm == (std::vector<std::uint8_t>{0, 0, 1}));
    CHECK(*back.atom_ids == (std::vector<std::int32_t>{10, 11, 12}));
    CHECK(back.columns.count("occupancy") && back.columns.count("bfactor"));
}

TEST(pdb_writer_nonascii_name_and_atom_ids) {
    FrameBatch b(2, 1);
    b.atom_names = strs({"\xC3\x84\xC3\x96", "C"});
    b.atom_ids = mdz::share(std::vector<std::int32_t>{41, 42});
    b.columns["positions"] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    std::vector<std::string> atoms;
    for (const auto& l : lines_of(write_text("pdb", b)))
        if (l.rfind("ATOM", 0) == 0) atoms.push_back(l);
    REQUIRE(atoms.size() == 2);
    CHECK(atoms[0].find("   41") != std::string::npos);
    CHECK(atoms[1].find("   42") != std::string::npos);
    // A two-character name starts in column 14 however many bytes it takes.
    CHECK(atoms[0].compare(12, 6, " \xC3\x84\xC3\x96 ") == 0);
    CHECK(atoms[1].compare(12, 4, " C  ") == 0);
}

TEST(pdb_single_frame_no_model) {
    FrameBatch b(2, 1);
    b.atom_names = strs({"C", "N"});
    b.columns["positions"] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    const std::string text = write_text("pdb", b);
    CHECK(text.find("MODEL") == std::string::npos);
    CHECK(text.find("ATOM") != std::string::npos);
    CHECK(text.find("END") != std::string::npos);
    auto back = read_text("pdb", text);
    CHECK_EQ(back.natoms, std::size_t(2));
    CHECK_EQ(back.nframes, std::size_t(1));
}

TEST(pdb_cryst1_parsing) {
    auto b = read_text("pdb",
                       "CRYST1   30.000   40.000   50.000  90.00  90.00  90.00 P 1           1\n"
                       "ATOM      1  O   WAT A   1       1.000   2.000   3.000  1.00  0.00           O\n"
                       "ATOM      2  H1  WAT A   1       4.000   5.000   6.000  1.00  0.00           H\n"
                       "END\n");
    CHECK_EQ(b.natoms, std::size_t(2));
    auto cell = b.cell_params();
    REQUIRE(cell.has_value());
    CHECK(near((*cell)[0][0], 3.0, 0.01));
    CHECK(near((*cell)[0][2], 4.0, 0.01));
    CHECK(near((*cell)[0][5], 5.0, 0.01));
    // The angles are exactly 90 degrees, so the box is diagonal.
    CHECK_EQ((*b.box)[0][3], 0.0);
    CHECK_EQ((*b.box)[0][4], 4.0);
}

TEST(pdb_reader_details) {
    // Short records get default occupancy and B-factor, and an element
    // guessed from the atom name.
    auto b = read_text("pdb",
                       "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n"
                       "HETATM    7 CL   ION X  12      -1.000   2.500   3.000\n"
                       "ATOM      8  CA  GLY Y  13       1.000   2.000   3.000  0.50 12.00\n");
    CHECK(!b.box.has_value());  // a 1x1x1 CRYST1 cell is a placeholder
    CHECK(*b.atom_names == (std::vector<std::string>{"CL", "CA"}));
    CHECK(*b.elements == (std::vector<std::string>{"C", "C"}));
    CHECK(*b.hetatm == (std::vector<std::uint8_t>{1, 0}));
    CHECK(*b.chain_ids == (std::vector<std::string>{"X", "Y"}));
    CHECK(*b.residue_ids == (std::vector<std::int32_t>{12, 13}));
    CHECK(b.columns["occupancy"] == (std::vector<float>{1.0f, 0.5f}));
    CHECK(b.columns["bfactor"] == (std::vector<float>{0.0f, 12.0f}));
    CHECK(b.columns["positions"][0] == -1.0f * 0.1f);
    // A record without coordinates is malformed.
    CHECK_THROWS(read_text("pdb", "ATOM      1  O   WAT A   1       1.000   2.000\n"));
    // MODEL/ENDMDL frames. A change in atom count makes the batch variable.
    const char* two = "MODEL        1\nATOM      1  O   WAT A   1       1.000   2.000   3.000\nENDMDL\n"
                      "MODEL        2\nATOM      1  O   WAT A   1       1.500   2.000   3.000\nENDMDL\nEND\n";
    b = read_text("pdb", two);
    CHECK_EQ(b.nframes, std::size_t(2));
    CHECK(b.columns["positions"][3] == 1.5f * 0.1f);
    b = read_text("pdb", "MODEL        1\nATOM      1  O   WAT A   1       1.000   2.000   3.000\nENDMDL\n"
                         "MODEL        2\nATOM      1  O   WAT A   1       1.500   2.000   3.000\n"
                         "ATOM      2  O   WAT A   1       1.500   2.000   3.000\nENDMDL\n");
    CHECK(b.variable());
    CHECK(*b.frame_natoms == (std::vector<std::size_t>{1, 2}));
}

// --- lammps ---
namespace {

const char* kTriclinic = R"(ITEM: TIMESTEP
100
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS xy xz yz pp pp ff
-1.5 12.5 2.0
5.0 15.0 -1.0
-3.0 7.0 0.5
ITEM: ATOMS id type xu yu zu vx vy vz fx fy fz ix iy iz q mol c_pe
3 2 1.0 6.0 0.0 0.01 0.02 0.03 1.0 2.0 3.0 0 1 -1 -0.5 7 -12.25
1 1 3.0 7.0 1.0 -0.01 0.0 0.0 0.5 0.0 0.0 0 0 0 0.5 7 -11.5
2 1 4.0 8.0 2.0 0.0 0.0 0.0 0.0 0.0 0.0 1 0 0 0.0 8 -10.0
)";

const char* kGeneral = R"(ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS abc origin pp pp pp
10 0 0 1
1 10 0 2
0 1 10 3
ITEM: ATOMS id type xs ys zs
2 1 0.25 0.75 0.125
1 1 0.5 0.5 0.5
)";

const char* kOrtho = R"(ITEM: TIMESTEP
5
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
-1.41 21.15
-1.41 21.15
0 10
ITEM: ATOMS id type x y z
1 1 0 0 0
)";

const char* kTimed =
    "ITEM: TIME\n1.5\nITEM: TIMESTEP\n750\nITEM: NUMBER OF ATOMS\n1\n"
    "ITEM: BOX BOUNDS pp pp pp\n0 1\n0 1\n0 1\nITEM: ATOMS id type x y z\n"
    "1 1 0.5 0.5 0.5\n"
    "ITEM: TIME\n3\nITEM: TIMESTEP\n1500\nITEM: NUMBER OF ATOMS\n1\n"
    "ITEM: BOX BOUNDS pp pp pp\n0 1\n0 1\n0 1\nITEM: ATOMS id type x y z\n"
    "1 1 0.5 0.5 0.6\n";

FrameBatch read_lammps(std::string_view text, const char* units) {
    return read_text("lammps", text, lammps_opts(units));
}

}  // namespace

TEST(lammps_triclinic_box_origin_and_units_are_exact) {
    FrameBatch native = read_lammps(kTriclinic, "real");
    REQUIRE(native.units);
    CHECK(native.units->style == "real");
    CHECK(((*native.origin)[0] == mdz::Vec3{-0.5, 5.0, -3.0}));
    CHECK_EQ(native.columns["velocities"][0], -0.01f);
    FrameBatch b = native.in_internal_units();
    CHECK(b.units == nullptr);
    const mdz::Vec3 o = (*b.origin)[0];
    const mdz::Box m = (*b.box)[0];
    CHECK(near(o[0], -0.05, 1e-12) && near(o[1], 0.5, 1e-12) && near(o[2], -0.3, 1e-12));
    CHECK(near(m[0], 1.1, 1e-12) && near(m[3], 0.2, 1e-12) && near(m[6], -0.1, 1e-12) && near(m[7], 0.05, 1e-12));
    CHECK(near(m[4], 0.95, 1e-12) && near(m[8], 1.0, 1e-12));
    CHECK(*b.atom_ids == (std::vector<std::int32_t>{1, 2, 3}));
    CHECK(*b.atom_types == (std::vector<std::int32_t>{1, 1, 2}));
    const auto& p = b.columns["positions"];
    CHECK(std::fabs(p[0] - 0.3f) < 1e-6f && std::fabs(p[6] - 0.1f) < 1e-6f);
    CHECK(std::fabs(b.columns["velocities"][0] + 1.0f) < 1e-5f);
    CHECK(std::fabs(b.columns["forces"][0] - 0.5f * 41.84f) < 1e-4f);
    CHECK(b.int_columns["image"] == (std::vector<std::int32_t>{0, 0, 0, 1, 0, 0, 0, 1, -1}));
    CHECK(b.int_columns["mol"] == (std::vector<std::int32_t>{7, 8, 7}));
    CHECK(b.columns["q"] == (std::vector<float>{0.5f, 0.0f, -0.5f}));
    CHECK(b.columns["c_pe"] == (std::vector<float>{-11.5f, -10.0f, -12.25f}));
    CHECK_EQ(b.attributes["lammps.boundary"], std::string("pp pp ff"));
    CHECK_EQ(b.attributes["lammps.units"], std::string("real"));
    CHECK_EQ(b.attributes["lammps.coords"], std::string("unwrapped"));
    CHECK_EQ(b.attributes["lammps.box"], std::string("triclinic"));
    CHECK_EQ(b.attributes["lammps.columns"], std::string("id type xu yu zu vx vy vz fx fy fz ix iy iz q mol c_pe"));
    CHECK(b.int_frame_props["step"] == (std::vector<std::int64_t>{100}));
}

TEST(lammps_dump_round_trips_through_reader_and_writer) {
    FrameBatch b = read_lammps(kTriclinic, "real");
    const std::string text = write_text("lammps", b);
    const auto lines = lines_of(text);
    CHECK_EQ(lines[4], std::string("ITEM: BOX BOUNDS xy xz yz pp pp ff"));
    CHECK_EQ(lines[8], std::string("ITEM: ATOMS id type xu yu zu vx vy vz fx fy fz ix iy iz q mol c_pe"));
    FrameBatch again = read_lammps(text, "real");
    CHECK(again.origin == b.origin);
    for (int i = 0; i < 9; ++i) CHECK(near((*again.box)[0][i], (*b.box)[0][i], 1e-12));
    for (const char* key : {"positions", "velocities", "forces", "q", "c_pe"}) {
        const auto& x = again.columns[key];
        const auto& y = b.columns[key];
        REQUIRE(x.size() == y.size());
        for (std::size_t i = 0; i < x.size(); ++i) CHECK(std::fabs(x[i] - y[i]) <= 1e-6f * std::max(std::fabs(y[i]), 1.0f));
    }
    CHECK(again.int_columns == b.int_columns);
    CHECK(again.attributes == b.attributes);
}

TEST(lammps_scaled_coordinates_and_general_boxes) {
    const char* text = "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\nITEM: BOX BOUNDS abc origin pp pp pp\n"
                       "10 0 0 1\n1 10 0 2\n0 1 10 3\nITEM: ATOMS id type xs ys zs\n1 1 0.5 0.5 0.5\n";
    FrameBatch b = read_lammps(text, "metal");
    FrameBatch internal = b.in_internal_units();
    const auto& p = internal.columns["positions"];
    const float want[3] = {0.65f, 0.75f, 0.8f};
    for (int k = 0; k < 3; ++k) CHECK(std::fabs(p[static_cast<std::size_t>(k)] - want[k]) < 1e-6f);
    CHECK_EQ(b.attributes["lammps.box"], std::string("general"));
    const std::string out = write_text("lammps", b);
    CHECK(out.find("ITEM: BOX BOUNDS abc origin pp pp pp") != std::string::npos);
    CHECK(out.find("ITEM: ATOMS id type xs ys zs\n1 1 0.5 0.5 0.5\n") != std::string::npos);
}

TEST(lammps_orthogonal_boxes_keep_their_origin) {
    const std::string out = write_text("lammps", read_lammps(kOrtho, "metal"));
    CHECK(out.find("ITEM: BOX BOUNDS pp pp pp\n-1.41 21.15") != std::string::npos);
}

TEST(lammps_changing_atoms_make_the_batch_vary) {
    auto frame = [](const char* ids, const char* ty) {
        return std::string("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\nITEM: BOX BOUNDS pp pp pp\n0 1\n0 1\n0 1\n"
                           "ITEM: ATOMS id type x y z\n") +
               ids[0] + " " + ty + " 0 0 0\n" + ids[1] + " 1 0 0 0\n";
    };
    // Different ids or changed types mean different atoms, even at the same
    // count.
    CHECK(read_text("lammps", frame("12", "1") + frame("13", "1")).variable());
    CHECK(read_text("lammps", frame("12", "1") + frame("12", "2")).variable());
    // The same atoms in a different order are still the same atoms.
    FrameBatch b = read_text("lammps", frame("12", "1") + frame("21", "1"));
    CHECK_EQ(b.nframes, std::size_t(2));
    CHECK(!b.variable());
}

TEST(lammps_writer_converts_to_a_requested_unit_style) {
    FrameBatch b = read_lammps(kTriclinic, "real");
    const std::string text = write_text("lammps", b, lammps_opts("metal"));
    std::istringstream atom1(lines_of(text)[9]);
    std::string tok;
    for (int i = 0; i < 6; ++i) atom1 >> tok;
    CHECK(near(std::stod(tok), -10.0, 1e-4));
}

TEST(lammps_simulation_time_round_trips) {
    FrameBatch b = read_lammps(kTimed, "metal");
    CHECK(b.frame_props["time"] == (std::vector<double>{1.5, 3.0}));
    CHECK(b.int_frame_props["step"] == (std::vector<std::int64_t>{750, 1500}));
    const std::string out = write_text("lammps", b);
    CHECK(out.rfind("ITEM: TIME\n1.5\nITEM: TIMESTEP\n750\n", 0) == 0);
    CHECK(out.find("ITEM: TIME\n3\nITEM: TIMESTEP\n1500\n") != std::string::npos);
    CHECK(mdz::check_data_loss(b, "lammps").empty());
    FrameBatch real = read_lammps(kTimed, "real");
    CHECK(real.frame_props["time"] == (std::vector<double>{1.5e-3, 3e-3}));
}

TEST(lammps_reader_details) {
    // The default unit style is real. Blank lines between frames are skipped.
    FrameBatch b = read_text("lammps", std::string(kOrtho) + "\n\n" + kOrtho);
    CHECK(b.units && b.units->style == "real");
    CHECK_EQ(b.nframes, std::size_t(2));
    // A dump without boundary flags and without an id column. Atoms are
    // numbered in file order.
    b = read_text("lammps",
                  "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\nITEM: BOX BOUNDS\n0 1\n0 2\n0 3\n"
                  "ITEM: ATOMS x y z element i_flag\n1 2 3 Na 4\n4 5 6 Cl -5\n");
    CHECK_EQ(b.attributes["lammps.boundary"], std::string("pp pp pp"));
    CHECK_EQ(b.attributes["lammps.coords"], std::string("wrapped"));
    CHECK(*b.atom_ids == (std::vector<std::int32_t>{1, 2}));
    CHECK(!b.atom_types);
    CHECK(*b.elements == (std::vector<std::string>{"Na", "Cl"}));
    CHECK(b.int_columns["i_flag"] == (std::vector<std::int32_t>{4, -5}));
    // Malformed dumps.
    const std::string head = "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\nITEM: BOX BOUNDS pp pp pp\n0 1\n0 1\n0 1\n";
    CHECK_THROWS(read_text("lammps", head + "ITEM: ATOMS id type q\n1 1 0\n"));            // no coordinates
    CHECK_THROWS(read_text("lammps", head + "ITEM: ATOMS id type x y z\n1 1 0 0\n"));      // short line
    CHECK_THROWS(read_text("lammps", head + "ITEM: ATOMS id type x y z\n1 1 0 0 0 0\n"));  // long line
    CHECK_THROWS(read_text("lammps", head + "ITEM: ATOMS id type x y z\n1 1 0 0 q\n"));    // bad number
    CHECK_THROWS(read_text("lammps", head + "ITEM: ATOMS id type x y z\n"));               // truncated
    CHECK_THROWS(read_text("lammps", head + "ITEM: ATOMS id type x y z\n3000000000 1 0 0 0\n"));
    CHECK_THROWS(read_text("lammps", "ITEM: TIMESTEP\nx\n"));
    CHECK_THROWS(read_text("lammps", "ITEM: STEP\n0\n"));
    CHECK_THROWS(read_text("lammps", "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\nITEM: BOX BOUNDS pp pp pp\n0\n"));
    try {
        read_text("lammps", head + "ITEM: ATOMS id type x y z\n1 1 0 0 q\n");
    } catch (const mdz::Error& e) {
        CHECK(e.kind() == mdz::ErrorKind::Corrupt);
        CHECK(std::string(e.what()).find("test.lammps:10") != std::string::npos);
    }
    // The writer rejects an unknown unit style.
    FrameBatch u(1, 1);
    u.columns["positions"] = {0, 0, 0};
    u.attributes["lammps.units"] = "furlongs";
    CHECK_THROWS(write_text("lammps", u));
}

TEST(lammps_examples_match_reference) {
    // Reference counts and values for the bundled example dumps.
    auto nacl = mdz::open_reader(std::string(MDZ_EXAMPLES_DIR) + "/lammps/nacl.lammpstrj")->read_all();
    REQUIRE(nacl.has_value());
    CHECK_EQ(nacl->natoms, std::size_t(512));
    CHECK_EQ(nacl->nframes, std::size_t(6));
    const auto& p = nacl->columns["positions"];
    CHECK_EQ(p[0], 6.25e-8f);
    CHECK_EQ(p[p.size() - 3], 19.891f);
    CHECK_EQ(p[p.size() - 2], 19.62395f);
    CHECK_EQ(p[p.size() - 1], 19.672478f);
    CHECK_EQ(nacl->int_frame_props["step"].back(), std::int64_t(500));
    CHECK_EQ((*nacl->box)[0][0], 22.560800999999998);
    CHECK_EQ((*nacl->origin)[0][0], -1.41005);
    CHECK_EQ(nacl->attributes["lammps.coords"], std::string("scaled"));
    CHECK_EQ(nacl->attributes["lammps.columns"], std::string("xs ys zs vx vy vz"));
    CHECK_EQ(nacl->columns["velocities"].back(), 0.000626494f);
    CHECK_EQ(nacl->columns.size(), std::size_t(2));
    double sum = 0;
    for (float x : p) sum += x;
    CHECK(near(sum, 90964.040633, 1e-6));

    auto ch = mdz::open_reader(std::string(MDZ_EXAMPLES_DIR) + "/lammps/ch3cn_water.lammpstrj")->read_all();
    REQUIRE(ch.has_value());
    CHECK_EQ(ch->natoms, std::size_t(375));
    CHECK_EQ(ch->nframes, std::size_t(10));
    const auto& q = ch->columns["positions"];
    CHECK_EQ(q[0], 10.4408f);
    CHECK_EQ(q[2], 2.07727f);
    CHECK_EQ(q.back(), 14.6957f);
    CHECK_EQ(ch->int_frame_props["step"].front(), std::int64_t(4662600));
    CHECK_EQ(ch->int_frame_props["step"].back(), std::int64_t(4665300));
    CHECK_EQ(ch->columns["velocities"][0], 43.0091f);
    CHECK_EQ(ch->columns["velocities"].back(), 5.3734f);
    CHECK((std::vector<std::int32_t>(ch->atom_types->begin(), ch->atom_types->begin() + 5) ==
           std::vector<std::int32_t>{2, 2, 1, 1, 1}));
    CHECK((std::vector<std::int32_t>(ch->int_columns["image"].begin(), ch->int_columns["image"].begin() + 6) ==
           std::vector<std::int32_t>{1, -1, 1, 1, -1, 1}));
    CHECK_EQ(ch->atom_ids->back(), 375);
    double msum = 0, psum = 0;
    for (float x : ch->columns["mass"]) msum += x;
    for (float x : q) psum += x;
    CHECK_EQ(msum, 22500.0);
    CHECK(near(psum, 88173.004720, 1e-6));
    CHECK_EQ(ch->attributes["lammps.columns"], std::string("id type x y z ix iy iz vx vy vz fx fy fz mass"));
}

// --- writer output against reference fixtures ---
namespace {

std::vector<float> ramp(int n, float scale, float offset, bool divide = false) {
    std::vector<float> v(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) v[static_cast<std::size_t>(i)] = divide ? static_cast<float>(i) / scale : static_cast<float>(i) * scale + offset;
    return v;
}

const std::vector<float> kPos18 = {0.1f,  -0.2f, 0.3f,  1.2345678f, 0.0f, -0.0000004f, 12.5f, -7.25f, 0.333333f,
                                   0.11f, 0.21f, 0.31f, 0.41f,      0.51f, 0.61f,      0.71f, 0.81f,  0.91f};

mdz::Box diag(double a, double b, double c) { return {a, 0, 0, 0, b, 0, 0, 0, c}; }

FrameBatch batch_ext() {
    FrameBatch b(3, 2);
    b.atom_names = strs({"O", "H", "H"});
    b.elements = strs({"O", "H", "Hx"});
    b.columns["positions"] = kPos18;
    b.columns["velocities"] = ramp(18, 0.37f, -2.0f);
    b.columns["forces"] = ramp(18, -13.1f, 50.0f);
    b.columns["charge"] = {-0.8f, 0.4f, 0.4f, -0.81f, 0.405f, 0.405f};
    b.columns["dipole"] = ramp(18, 7.0f, 0.0f, true);
    b.int_columns["tag"] = {100, 200, -300, 101, 201, -301};
    b.int_columns["pair"] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    b.frame_props["time"] = {0.5, 1.0};
    b.frame_props["energy"] = {-100.25, -99.5};
    b.int_frame_props["step"] = {10, 20};
    b.box = std::vector<mdz::Box>{diag(3.0, 3.5, 4.0), diag(3.01, 3.49, 4.2)};
    return b;
}

FrameBatch batch_internal() {
    FrameBatch b(3, 2);
    b.columns["positions"] = {0.1f,  -0.2f, 0.3f,  1.2345678f, 0.0f, -0.0000004f, 2.5f,  -1.25f, 0.333333f,
                              0.11f, 0.21f, 0.31f, 0.41f,      0.51f, 0.61f,      0.71f, 0.81f,  0.91f};
    b.columns["velocities"] = ramp(18, 0.37f, -2.0f);
    b.columns["forces"] = ramp(18, -13.1f, 50.0f);
    b.columns["charge"] = {-0.8f, 0.4f, 0.4f, -0.81f, 0.405f, 0.405f};
    b.int_columns["mol"] = {1, 1, 2, 1, 1, 2};
    b.atom_types = mdz::share(std::vector<std::int32_t>{1, 2, 2});
    b.atom_ids = mdz::share(std::vector<std::int32_t>{5, 6, 9});
    b.frame_props["time"] = {0.002, 0.004};
    b.int_frame_props["step"] = {1000, 2000};
    b.box = std::vector<mdz::Box>{diag(3.0, 3.5, 4.0), mdz::Box{3.0, 0, 0, 0.4, 3.5, 0, -0.2, 0.1, 4.0}};
    return b;
}

// Returns the first frame of a dump, written as a dump of its own.
std::string first_frame(const std::string& path) {
    auto r = mdz::open_reader(path);
    auto b = r->read_batch(1);
    REQUIRE(b.has_value());
    return write_text("lammps", *b);
}

}  // namespace

TEST(xyz_writer_matches_reference) {
    FrameBatch b(3, 2);
    b.atom_names = strs({"O", "H1", "Na+"});
    b.elements = strs({"O", "H"});
    b.columns["positions"] = kPos18;
    check_same_text(write_text("xyz", b), fixture("xyz_basic.xyz"), "xyz_basic.xyz");
}

TEST(extxyz_writer_matches_reference) {
    check_same_text(write_text("extxyz", batch_ext()), fixture("extxyz_full.extxyz"), "extxyz_full.extxyz");
    FrameBatch m(2, 1);
    m.columns["positions"] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    check_same_text(write_text("extxyz", m), fixture("extxyz_min.extxyz"), "extxyz_min.extxyz");
    // The reader accepts what the writer produced.
    FrameBatch back = read_text("extxyz", fixture("extxyz_full.extxyz"));
    CHECK_EQ(back.nframes, std::size_t(2));
    CHECK(back.int_columns["pair"] == batch_ext().int_columns["pair"]);
    CHECK(near(back.frame_props["energy"][1], -99.5, 1e-4));
    CHECK(near((*back.box)[1][8], 4.2, 1e-12));
}

TEST(pdb_writer_matches_reference) {
    FrameBatch b(4, 2);
    b.columns["positions"] = {0.1f,  0.2f,  0.3f,  -0.4f, 0.5f,  0.6f,  12.34567f, -99.9999f, 0.0f, 0.7f, 0.8f, 0.9f,
                              0.11f, 0.21f, 0.31f, -0.41f, 0.51f, 0.61f, 1.0f,      2.0f,      3.0f, 0.71f, 0.81f, 0.91f};
    b.columns["occupancy"] = {1.0f, 0.5f, 0.25f, 0.125f, 1.0f, 0.5f, 0.25f};
    b.columns["bfactor"] = {10.0f, 20.5f, 30.25f, 99.99f, 1.0f, 2.0f, 3.0f, 4.0f};
    b.atom_names = strs({"N", "CA", "HD21", "\xC3\x84"});
    b.elements = strs({"N", "C", "", ""});
    b.atom_ids = mdz::share(std::vector<std::int32_t>{10, 11, 123456, -3});
    b.residue_names = strs({"ALA", "ALA", "ASN", "HOH"});
    b.residue_ids = mdz::share(std::vector<std::int32_t>{1, 1, 10234, -5});
    b.chain_ids = strs({"A", "A", "", "B"});
    b.hetatm = mdz::share(std::vector<std::uint8_t>{0, 0, 0, 1});
    b.box = std::vector<mdz::Box>{diag(3.0, 3.5, 4.0)};
    FrameBatch c(4, 1);
    c.columns["positions"] = std::vector<float>(12, 0.5f);
    c.atom_names = strs({"O", "OW", "C1", "Cl"});
    check_same_text(write_text("pdb", {&b, &c}), fixture("pdb_topology.pdb"), "pdb_topology.pdb");
    check_same_text(write_text("pdb", c), fixture("pdb_single.pdb"), "pdb_single.pdb");
    // PDB columns are byte offsets. The two-byte UTF-8 name shifts the
    // coordinates out of their columns, so the record is rejected.
    CHECK_THROWS(read_text("pdb", fixture("pdb_topology.pdb")));
    // With an ASCII name the file reads back with its topology. Each model
    // keeps its own, because the third model has different atoms (serials
    // and names) from the first two.
    b.atom_names = strs({"N", "CA", "HD21", "OW"});
    FrameBatch all = read_text("pdb", write_text("pdb", {&b, &c}));
    CHECK_EQ(all.nframes, std::size_t(3));
    REQUIRE(all.variable());
    FrameBatch back = all.frame(0);
    CHECK(*back.atom_names == (std::vector<std::string>{"N", "CA", "HD21", "OW"}));
    CHECK(*back.elements == (std::vector<std::string>{"N", "C", "H", "OW"}));
    CHECK(*back.atom_ids == (std::vector<std::int32_t>{10, 11, 23456, 99997}));
    CHECK(*back.chain_ids == (std::vector<std::string>{"A", "A", "A", "B"}));
    CHECK(*back.residue_ids == (std::vector<std::int32_t>{1, 1, 234, 9995}));
    CHECK(*all.frame(1).atom_names == *back.atom_names);
    CHECK(*all.frame(2).atom_names == (std::vector<std::string>{"O", "OW", "C1", "Cl"}));
    CHECK(*all.frame(2).atom_ids == (std::vector<std::int32_t>{1, 2, 3, 4}));
}

TEST(lammps_writer_matches_reference) {
    const FrameBatch tri = read_lammps(kTriclinic, "real");
    check_same_text(write_text("lammps", tri), fixture("lammps_triclinic.lammpstrj"), "lammps_triclinic");
    check_same_text(write_text("lammps", tri, lammps_opts("metal")), fixture("lammps_triclinic_metal.lammpstrj"),
                    "lammps_triclinic_metal");
    check_same_text(write_text("lammps", read_lammps(kGeneral, "metal")), fixture("lammps_general.lammpstrj"),
                    "lammps_general");
    check_same_text(write_text("lammps", read_lammps(kOrtho, "metal")), fixture("lammps_ortho.lammpstrj"),
                    "lammps_ortho");
    check_same_text(write_text("lammps", read_lammps(kTimed, "metal")), fixture("lammps_time.lammpstrj"),
                    "lammps_time");
    check_same_text(write_text("lammps", batch_internal()), fixture("lammps_internal.lammpstrj"), "lammps_internal");
    check_same_text(write_text("lammps", batch_internal(), lammps_opts("lj")), fixture("lammps_internal_lj.lammpstrj"),
                    "lammps_internal_lj");
    const std::string ex = std::string(MDZ_EXAMPLES_DIR) + "/lammps/";
    check_same_text(first_frame(ex + "nacl.lammpstrj"), fixture("lammps_nacl1.lammpstrj"), "lammps_nacl1");
    check_same_text(first_frame(ex + "ch3cn_water.lammpstrj"), fixture("lammps_ch3cn1.lammpstrj"), "lammps_ch3cn1");
    // A dump in LAMMPS units written through the fixed-unit writers.
    check_same_text(write_text("xyz", tri), fixture("triclinic.xyz"), "triclinic.xyz");
    check_same_text(write_text("extxyz", tri), fixture("triclinic.extxyz"), "triclinic.extxyz");
    check_same_text(write_text("pdb", tri), fixture("triclinic.pdb"), "triclinic.pdb");
}

TEST(lammps_sorting_and_numbers) {
    const std::string head = "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n4\nITEM: BOX BOUNDS pp pp pp\n0 1\n0 1\n0 1\n"
                             "ITEM: ATOMS id x y z q mol\n";
    // Sparse ids with a repeat. Atoms are sorted by id, equal ids stay in
    // input order, and the other columns move with their atoms.
    FrameBatch b = read_text("lammps", head + "5 1 0 0 0.5 1\n3 2 0 0 0.25 2\n9 3 0 0 0.75 3\n3 4 0 0 0.125 4\n");
    CHECK(*b.atom_ids == (std::vector<std::int32_t>{3, 3, 5, 9}));
    CHECK(b.columns["q"] == (std::vector<float>{0.25f, 0.125f, 0.5f, 0.75f}));
    CHECK(b.int_columns["mol"] == (std::vector<std::int32_t>{2, 4, 1, 3}));
    CHECK(b.columns["positions"][3] == 4.0f);
    // Dense ids out of order.
    b = read_text("lammps", head + "2 2 0 0 0 0\n4 4 0 0 0 0\n1 1 0 0 0 0\n3 3 0 0 0 0\n");
    CHECK(*b.atom_ids == (std::vector<std::int32_t>{1, 2, 3, 4}));
    for (int i = 0; i < 4; ++i) CHECK(b.columns["positions"][static_cast<std::size_t>(3 * i)] == static_cast<float>(i + 1));
    // The number parser accepts a leading '+', inf and nan, and out-of-range
    // values (which overflow to inf or underflow to zero).
    b = read_text("lammps", head + "+1 1e400 -1e-400 +2 nan +3\n2 inf -inf 0 1e50 -0\n3 0 0 0 0 0\n4 0 0 0 0 0\n");
    const auto& p = b.columns["positions"];
    CHECK(std::isinf(p[0]) && p[0] > 0);
    CHECK(p[1] == 0.0f && std::signbit(p[1]));
    CHECK_EQ(p[2], 2.0f);
    CHECK(std::isnan(b.columns["q"][0]));
    CHECK(std::isinf(b.columns["q"][1]));
    CHECK(b.int_columns["mol"][0] == 3);
    CHECK_THROWS(read_text("lammps", head + "+-1 0 0 0 0 0\n2 0 0 0 0 0\n3 0 0 0 0 0\n4 0 0 0 0 0\n"));
    CHECK_THROWS(read_text("lammps", head + "1 0 0 0 0 1.5\n2 0 0 0 0 0\n3 0 0 0 0 0\n4 0 0 0 0 0\n"));
}

TEST(text_numbers_parse_like_the_reference) {
    using mdz::text::parse;
    float f;
    double d;
    std::int64_t i;
    CHECK(parse("+1.5", d) && d == 1.5);
    CHECK(!parse("+-1", d));
    CHECK(!parse("++1", d));
    CHECK(!parse("1.5x", d));
    CHECK(!parse("", d));
    CHECK(parse("1e400", d) && std::isinf(d));
    CHECK(parse("-1e400", d) && std::isinf(d) && d < 0);
    CHECK(parse("1e-400", d) && d == 0.0);
    CHECK(parse("1e50", f) && std::isinf(f));
    CHECK(parse("inf", d) && std::isinf(d));
    CHECK(parse("NaN", f) && std::isnan(f));
    CHECK(parse("+42", i) && i == 42);
    CHECK(!parse("+-42", i));
    CHECK(!parse("4.2", i));
}
