// LazyReader: reading a file in byte ranges must give what Reader gives.

#include <cmath>
#include <cstring>

#include "mdz/lazy_reader.hpp"
#include "mdz/mdz.hpp"
#include "test.hpp"

namespace {

using Bytes = std::vector<std::uint8_t>;

// A file served in byte ranges, counting what is read.
struct Source {
    const Bytes& file;
    std::size_t bytes_read = 0;
    mdz::ByteSpan read(std::uint64_t lo, std::uint64_t hi) {
        if (hi > file.size()) hi = file.size();
        if (lo > hi) lo = hi;
        bytes_read += hi - lo;
        return {file.data() + lo, static_cast<std::size_t>(hi - lo)};
    }
};

// Open as a browser would: head, trailer and footer, or a scan of the
// records if the file was not closed.
mdz::LazyReader open_lazy(Source& src) {
    const std::uint64_t size = src.file.size();
    mdz::ByteSpan head = src.read(0, mdz::LazyReader::kHeaderSize);
    head = src.read(0, mdz::LazyReader::head_size(head));
    mdz::LazyReader r = mdz::LazyReader::open(head, size);
    bool closed = false;
    if (size >= mdz::LazyReader::kTrailerSize)
        if (auto f = r.footer_range(src.read(size - mdz::LazyReader::kTrailerSize, size)))
            closed = r.read_footer(f->first, src.read(f->first, f->second));
    if (!closed)
        while (std::uint64_t n = r.record_size(src.read(r.scan_offset(), r.scan_offset() + 8)))
            if (!r.add_record(src.read(r.scan_offset(), r.scan_offset() + n))) break;
    return r;
}

mdz::FrameBatch decode(const mdz::LazyReader& r, Source& src, std::size_t gi,
                       const std::vector<std::string>& columns = {}) {
    const auto& g = r.groups()[gi];
    return r.decode_group(gi, src.read(g.offset, g.offset + g.size), columns);
}

mdz::FrameBatch smooth(std::size_t natoms, std::size_t nframes) {
    mdz::FrameBatch b(natoms, nframes);
    auto& p = b.columns["positions"];
    auto& v = b.columns["velocities"];
    for (std::size_t f = 0; f < nframes; ++f)
        for (std::size_t i = 0; i < natoms; ++i)
            for (int k = 0; k < 3; ++k) {
                p.push_back(static_cast<float>(i % 7) * 0.3f +
                            0.01f * static_cast<float>(f) * static_cast<float>(k + 1) +
                            0.002f * std::sin(static_cast<float>(i + f)));
                v.push_back(0.1f * static_cast<float>(k) - 0.01f * static_cast<float>(i % 5));
            }
    std::vector<std::string> elements;
    for (std::size_t i = 0; i < natoms; ++i) elements.push_back(i % 2 ? "H" : "O");
    b.elements = mdz::share(std::move(elements));
    b.box = std::vector<mdz::Box>(nframes, mdz::Box{3, 0, 0, 0.5, 3, 0, 0, 0, 3});
    for (std::size_t f = 0; f < nframes; ++f) {
        b.frame_props["time"].push_back(0.5 * static_cast<double>(f));
        b.int_frame_props["step"].push_back(static_cast<std::int64_t>(f) * 100);
    }
    return b;
}

Bytes file_of(const mdz::FrameBatch& b, mdz::CompressOptions o = {}) {
    if (o.gof_size == 20) o.gof_size = 4;
    return mdz::to_bytes(mdz::compress(b, o));
}

void check_same(const mdz::FrameBatch& a, const mdz::FrameBatch& b) {
    CHECK_EQ(a.nframes, b.nframes);
    CHECK_EQ(a.natoms, b.natoms);
    CHECK(a.columns == b.columns);
    CHECK(a.int_columns == b.int_columns);
    CHECK(a.frame_props == b.frame_props);
    CHECK(a.int_frame_props == b.int_frame_props);
    CHECK(a.box == b.box);
    CHECK(a.frame_natoms == b.frame_natoms);
}

// Every group decoded lazily matches Reader over the same frames.
void check_matches_reader(const Bytes& file, bool complete) {
    Source src{file};
    mdz::LazyReader r = open_lazy(src);
    mdz::Reader full = mdz::Reader::from_bytes(file);
    CHECK_EQ(r.complete(), complete);
    CHECK_EQ(r.nframes(), full.nframes());
    CHECK_EQ(r.natoms(), full.natoms());
    CHECK_EQ(r.variable_atoms(), full.variable_atoms());
    CHECK_EQ(r.precision(), full.precision());
    CHECK_EQ(r.groups().size(), full.n_gofs());
    for (std::size_t gi = 0; gi < r.groups().size(); ++gi) {
        const auto& g = r.groups()[gi];
        check_same(decode(r, src, gi), full.read_batch(g.first_frame, g.nframes));
        CHECK_EQ(r.group_of(g.first_frame), gi);
        CHECK_EQ(r.group_of(g.first_frame + g.nframes - 1), gi);
    }
}

}  // namespace

TEST(lazy_closed_file_matches_reader) {
    const Bytes file = file_of(smooth(30, 18));
    check_matches_reader(file, true);
}

TEST(lazy_open_reads_only_head_trailer_and_footer) {
    const Bytes file = file_of(smooth(200, 40));
    Source src{file};
    mdz::LazyReader r = open_lazy(src);
    CHECK_EQ(r.groups().size(), std::size_t(10));
    CHECK(src.bytes_read < 1024);
    const std::size_t before = src.bytes_read;
    decode(r, src, 7);
    CHECK_EQ(src.bytes_read - before, std::size_t(r.groups()[7].size));
}

TEST(lazy_metadata_and_units) {
    const Bytes file = file_of(smooth(10, 4));
    Source src{file};
    mdz::LazyReader r = open_lazy(src);
    CHECK(r.metadata().elements == mdz::Reader::from_bytes(file).metadata().elements);
    CHECK_EQ(r.length_to_nm(), 1.0);
    CHECK_EQ(r.columns().front().name, std::string("positions"));
}

TEST(lazy_unclosed_file_is_scanned) {
    Bytes file = file_of(smooth(30, 18));
    std::uint64_t cut = 0;
    {
        Source whole{file};
        const auto last = open_lazy(whole).groups().back();
        cut = last.offset + last.size / 2;
    }
    // The footer and half the last group are lost; 4 groups of 4 frames remain.
    file.resize(cut);
    check_matches_reader(file, false);
    Source src{file};
    CHECK_EQ(open_lazy(src).nframes(), std::size_t(16));
}

TEST(lazy_bad_footer_falls_back_to_a_scan) {
    Bytes file = file_of(smooth(30, 18));
    const std::uint64_t size = file.size();
    std::uint64_t footer_at = 0;
    std::memcpy(&footer_at, file.data() + size - 16, 8);
    file[footer_at + 14] ^= 0xFF;  // inside the footer body: its CRC fails
    Source src{file};
    mdz::LazyReader r = open_lazy(src);
    CHECK(!r.complete());
    CHECK_EQ(r.nframes(), std::size_t(18));
}

TEST(lazy_short_head_asks_for_more) {
    const Bytes file = file_of(smooth(10, 4));
    const mdz::ByteSpan all(file.data(), file.size());
    const std::uint64_t need = mdz::LazyReader::head_size(all.first(24));
    CHECK(need > 24);
    CHECK_THROWS(mdz::LazyReader::open(all.first(static_cast<std::size_t>(need) - 1), file.size()));
    CHECK_THROWS(mdz::LazyReader::head_size(all.first(10)));
    Bytes garbage(64, 0x55);
    CHECK_THROWS(mdz::LazyReader::head_size(mdz::ByteSpan(garbage.data(), garbage.size())));
}

TEST(lazy_wrong_or_damaged_record_is_corrupt) {
    Bytes file = file_of(smooth(30, 18));
    Source src{file};
    mdz::LazyReader r = open_lazy(src);
    const auto& g1 = r.groups()[1];
    const auto& g2 = r.groups()[2];
    bool corrupt = false;
    try {
        r.decode_group(1, src.read(g2.offset, g2.offset + g2.size));
    } catch (const mdz::Error& e) {
        corrupt = e.kind() == mdz::ErrorKind::Corrupt;
    }
    CHECK(corrupt);
    file[g1.offset + 30] ^= 0x01;
    CHECK_THROWS(r.decode_group(1, src.read(g1.offset, g1.offset + g1.size)));
    CHECK_THROWS(r.decode_group(99, src.read(g1.offset, g1.offset + g1.size)));
    CHECK_THROWS(r.group_of(r.nframes()));
}

TEST(lazy_decodes_only_the_columns_asked_for) {
    const Bytes file = file_of(smooth(30, 8));
    Source src{file};
    mdz::LazyReader r = open_lazy(src);
    mdz::FrameBatch all = decode(r, src, 0);
    mdz::FrameBatch some = decode(r, src, 0, {"box", "time"});
    CHECK(some.columns.count("velocities") == 0);
    CHECK(some.int_frame_props.count("step") == 0);
    CHECK(some.columns.at("positions") == all.columns.at("positions"));
    CHECK(some.box == all.box);
    CHECK(some.frame_props.at("time") == all.frame_props.at("time"));
}

TEST(lazy_variable_atom_counts) {
    mdz::FrameBatch all;
    for (std::size_t f = 0; f < 11; ++f) {
        const std::size_t n = 5 + (f * 3) % 7;
        mdz::FrameBatch b(n, 1);
        std::vector<std::int32_t> ids;
        std::vector<std::string> names;
        for (std::size_t i = 0; i < n; ++i) {
            const auto id = static_cast<std::int32_t>(i * 2 + f % 3);
            ids.push_back(id);
            names.push_back(id % 2 ? "Na" : "Cl");
            for (int k = 0; k < 3; ++k)
                b.columns["positions"].push_back(0.1f * static_cast<float>(id) + 0.01f * static_cast<float>(f + k));
        }
        b.atom_ids = mdz::share(std::move(ids));
        b.atom_names = mdz::share(std::move(names));
        all.append(b);
    }
    mdz::CompressOptions o;
    o.gof_size = 3;
    const Bytes file = mdz::to_bytes(mdz::compress(all, o));
    check_matches_reader(file, true);
    Source src{file};
    mdz::LazyReader r = open_lazy(src);
    CHECK(r.variable_atoms());
    mdz::FrameBatch g = decode(r, src, 1);
    REQUIRE(g.frame_natoms.has_value());
    CHECK_EQ((*g.frame_natoms)[0], std::size_t(5 + (3 * 3) % 7));
}

TEST(lazy_rigid_bodies) {
    // Dimers that translate and rotate rigidly.
    const std::size_t nmol = 40, nframes = 12;
    mdz::FrameBatch b(nmol * 2, nframes);
    std::vector<std::int32_t> bodies;
    for (std::size_t m = 0; m < nmol; ++m) bodies.insert(bodies.end(), {std::int32_t(m), std::int32_t(m)});
    b.bodies = mdz::share(std::move(bodies));
    auto& p = b.columns["positions"];
    for (std::size_t f = 0; f < nframes; ++f)
        for (std::size_t m = 0; m < nmol; ++m) {
            const float a = 0.1f * static_cast<float>(f + m);
            const float cx = 0.5f * static_cast<float>(m % 8) + 0.01f * static_cast<float>(f);
            const float cy = 0.5f * static_cast<float>(m / 8);
            p.insert(p.end(), {cx + 0.05f * std::cos(a), cy + 0.05f * std::sin(a), 1.0f});
            p.insert(p.end(), {cx - 0.05f * std::cos(a), cy - 0.05f * std::sin(a), 1.0f});
        }
    mdz::CompressOptions o;
    o.rigid = true;
    o.gof_size = 4;
    check_matches_reader(mdz::to_bytes(mdz::compress(b, o)), true);
}
