// Trajectories whose atom count varies from frame to frame, such as
// grand-canonical Monte Carlo, deposition, evaporation, or a dump of a region.
//
// Each group of frames is compressed over the union of the atoms (identified
// by id) present in any of its frames, as if every frame held all of them.
// An atom missing from a frame takes its values from the nearest frame that
// has it. The codecs therefore see a smooth, fixed set of atoms, and an absent
// atom costs almost nothing. Each group also has a membership blob, stored in
// column kMembershipColumn, that records the union's ids and the frames each
// atom is present in. Decoding uses it to drop the absent atoms again.
//
// Membership blob layout, zstd compressed, all integers little-endian:
//   u8   tag 0x25
//   u32  count
//   count ids, ascending, each as a zigzag LEB128 varint of the difference
//        from the previous id (the first from 0)
//   u32  nframes
//   nframes rows of ceil(count / 8) bytes; bit a of row f is set if atom a
//        is present in frame f
//   u32  nstrings
//   nstrings strings, each a u16 length followed by UTF-8 bytes
// The string table holds the strings that the group's per-atom string columns
// (atom names, elements, ...) index into, so each group can be decoded on its
// own.

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>

#include "bytes.hpp"
#include "mdz/error.hpp"
#include "mdz/trajectory.hpp"
#include "parallel.hpp"
#include "variable.hpp"
#include "zstd.hpp"

namespace mdz {
namespace detail {

namespace {

constexpr std::uint8_t kTagMembership = 0x25;

[[noreturn]] void invalid(const std::string& m) { throw Error(ErrorKind::InvalidInput, m); }

struct Membership {
    std::vector<std::int64_t> ids;  // ascending
    std::vector<std::string> strings;
    std::size_t nframes = 0;
    std::size_t row_bytes = 0;
    std::vector<std::uint8_t> bits;

    void reset(std::size_t n_ids, std::size_t frames) {
        nframes = frames;
        row_bytes = (n_ids + 7) / 8;
        bits.assign(frames * row_bytes, 0);
    }
    bool present(std::size_t f, std::size_t a) const noexcept { return (bits[f * row_bytes + a / 8] >> (a % 8)) & 1u; }
    void set(std::size_t f, std::size_t a) noexcept { bits[f * row_bytes + a / 8] |= std::uint8_t(1u << (a % 8)); }
    std::size_t count(std::size_t f) const noexcept {
        std::size_t c = 0;
        for (std::size_t a = 0; a < ids.size(); ++a) c += present(f, a);
        return c;
    }
};

void put_varint(Bytes& out, std::uint64_t v) {
    while (v >= 0x80) {
        out.push_back(static_cast<std::uint8_t>(v | 0x80));
        v >>= 7;
    }
    out.push_back(static_cast<std::uint8_t>(v));
}

std::uint64_t get_varint(ByteCursor& c) {
    std::uint64_t v = 0;
    for (unsigned shift = 0;; shift += 7) {
        if (shift > 63) corrupt("membership varint too long");
        const std::uint8_t b = c.u8();
        v |= std::uint64_t(b & 0x7F) << shift;
        if (!(b & 0x80)) return v;
    }
}

std::uint64_t zz64(std::int64_t v) {
    return (static_cast<std::uint64_t>(v) << 1) ^ static_cast<std::uint64_t>(v >> 63);
}
std::int64_t unzz64(std::uint64_t v) { return static_cast<std::int64_t>((v >> 1) ^ (0 - (v & 1))); }

Bytes encode_membership(const Membership& m, int level) {
    Bytes p;
    p.push_back(kTagMembership);
    put_le<std::uint32_t>(p, static_cast<std::uint32_t>(m.ids.size()));
    std::int64_t prev = 0;
    for (std::int64_t id : m.ids) {
        put_varint(p,
                   zz64(static_cast<std::int64_t>(static_cast<std::uint64_t>(id) - static_cast<std::uint64_t>(prev))));
        prev = id;
    }
    put_le<std::uint32_t>(p, static_cast<std::uint32_t>(m.nframes));
    put_bytes(p, m.bits.data(), m.bits.size());
    put_le<std::uint32_t>(p, static_cast<std::uint32_t>(m.strings.size()));
    for (const auto& s : m.strings) {
        if (s.size() > 65535) invalid("atom string '" + s.substr(0, 40) + "...' is longer than 65535 bytes");
        put_le<std::uint16_t>(p, static_cast<std::uint16_t>(s.size()));
        put_str(p, s);
    }
    return zstd_compress(p, level);
}

Membership decode_membership(ByteSpan blob, std::size_t expected_frames) {
    Bytes raw;
    zstd_decompress_bounded(blob, SIZE_MAX, raw);
    ByteCursor c(raw);
    if (c.u8() != kTagMembership) corrupt("not a membership blob");
    Membership m;
    const std::size_t n = c.u32();
    // Every id costs at least a byte.
    if (n > c.remaining()) corrupt("membership blob holds more ids than bytes");
    m.ids.resize(n);
    std::int64_t prev = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const std::int64_t id = static_cast<std::int64_t>(static_cast<std::uint64_t>(prev) +
                                                          static_cast<std::uint64_t>(unzz64(get_varint(c))));
        if (i > 0 && id <= prev) corrupt("membership ids are not ascending");
        m.ids[i] = prev = id;
    }
    const std::size_t frames = c.u32();
    if (frames != expected_frames)
        corrupt("membership covers " + std::to_string(frames) + " frames, expected " + std::to_string(expected_frames));
    m.nframes = frames;
    m.row_bytes = (n + 7) / 8;
    ByteSpan bits = c.take(sat_mul(frames, m.row_bytes));
    m.bits.assign(bits.begin(), bits.end());
    const std::size_t ns = c.u32();
    if (ns > c.remaining() / 2) corrupt("membership string table holds more strings than bytes");
    m.strings.reserve(ns);
    for (std::size_t i = 0; i < ns; ++i) m.strings.emplace_back(c.str(c.u16()));
    if (c.remaining() != 0) corrupt("membership blob has trailing bytes");
    return m;
}

// Returns frames [s, e) of a varying-count batch as a fixed-count batch over
// the union of their atoms, with absent atoms filled from the nearest frame
// that has them. Fills `m` with the membership recording which were absent.
FrameBatch densify(const FrameBatch& b, const std::vector<std::size_t>& off, std::size_t s, std::size_t e,
                   Membership& m) {
    const std::size_t n = e - s;
    const std::vector<std::int32_t>* ids = nullptr;
    if (auto it = b.int_columns.find(columns::kId); it != b.int_columns.end()) ids = &it->second;
    auto id_of = [&](std::size_t f, std::size_t a) -> std::int64_t {
        return ids ? (*ids)[off[f] + a] : static_cast<std::int64_t>(a);
    };

    std::vector<std::int64_t> all;
    all.reserve(off[e] - off[s]);
    for (std::size_t f = s; f < e; ++f)
        for (std::size_t a = 0; a < b.atoms_in(f); ++a) all.push_back(id_of(f, a));
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());
    m.ids = all;
    m.reset(all.size(), n);
    const std::size_t N = all.size();

    // The union slot of every atom in every frame.
    std::vector<std::uint32_t> slot(off[e] - off[s]);
    for (std::size_t f = s; f < e; ++f)
        for (std::size_t a = 0; a < b.atoms_in(f); ++a) {
            const auto k =
                static_cast<std::size_t>(std::lower_bound(all.begin(), all.end(), id_of(f, a)) - all.begin());
            if (m.present(f - s, k))
                invalid("atom id " + std::to_string(id_of(f, a)) + " appears twice in frame " + std::to_string(f));
            m.set(f - s, k);
            slot[off[f] - off[s] + a] = static_cast<std::uint32_t>(k);
        }

    FrameBatch d(N, n);
    auto spread = [&](const auto& src, auto& dst, std::size_t total) {
        const std::size_t w = total == 0 ? 0 : src.size() / total;
        dst.assign(N * n * w, {});
        for (std::size_t f = s; f < e; ++f)
            for (std::size_t a = 0; a < b.atoms_in(f); ++a) {
                const std::size_t from = (off[f] + a) * w, to = ((f - s) * N + slot[off[f] - off[s] + a]) * w;
                std::copy(src.begin() + static_cast<std::ptrdiff_t>(from),
                          src.begin() + static_cast<std::ptrdiff_t>(from + w),
                          dst.begin() + static_cast<std::ptrdiff_t>(to));
            }
        // Fill absent slots from the nearest frame that has the atom. That is
        // the closest earlier frame if there is one, otherwise the first later
        // one.
        for (std::size_t k = 0; k < N; ++k) {
            std::size_t first = 0;
            while (!m.present(first, k)) ++first;
            for (std::size_t f = 0; f < n; ++f) {
                if (m.present(f, k)) continue;
                const std::size_t from = f < first ? first : f - 1;
                std::copy(dst.begin() + static_cast<std::ptrdiff_t>((from * N + k) * w),
                          dst.begin() + static_cast<std::ptrdiff_t>((from * N + k + 1) * w),
                          dst.begin() + static_cast<std::ptrdiff_t>((f * N + k) * w));
            }
        }
    };
    const std::size_t total = off[b.nframes];
    for (const auto& [name, v] : b.columns) spread(v, d.columns[name], total);
    for (const auto& [name, v] : b.int_columns)
        if (name != columns::kId) spread(v, d.int_columns[name], total);
    // String columns index the batch's string table. Re-index them into a
    // per-group table holding only the strings the group uses.
    std::vector<std::int32_t> local;
    for (const char* name : {columns::kName, columns::kElement, columns::kResname, columns::kChain}) {
        auto it = d.int_columns.find(name);
        if (it == d.int_columns.end()) continue;
        if (!b.strings) invalid(std::string("column '") + name + "' needs FrameBatch::strings");
        local.resize(b.strings->size(), -1);
        for (auto& x : it->second) {
            const auto k = static_cast<std::size_t>(x);
            if (x < 0 || k >= b.strings->size())
                invalid(std::string("column '") + name + "' has string index " + std::to_string(x) + " out of range");
            if (local[k] < 0) {
                local[k] = static_cast<std::int32_t>(m.strings.size());
                m.strings.push_back((*b.strings)[k]);
            }
            x = local[k];
        }
    }
    FrameBatch part = b.slice_frames(s, e);
    d.frame_props = std::move(part.frame_props);
    d.int_frame_props = std::move(part.int_frame_props);
    d.box = std::move(part.box);
    d.origin = std::move(part.origin);
    d.attributes = b.attributes;
    d.units = b.units;
    return d;
}

// Views group gi of a varying-count trajectory as a one-group, fixed-count
// trajectory over the group's union of atoms. The blobs are not copied and
// remain owned by ct, which must outlive the view.
CompressedTrajectory group_view(const CompressedTrajectory& ct, std::size_t gi, std::size_t natoms) {
    CompressedTrajectory g;
    g.natoms = natoms;
    g.nframes = ct.gofs.span(gi).size();
    g.gof_size = ct.gof_size;
    g.gofs.push(g.nframes);
    g.metadata = ct.metadata;
    for (const auto& c : ct.columns) {
        if (c.desc.name == kMembershipColumn) continue;
        g.columns.emplace_back(c.desc, nullptr, std::vector<ByteSpan>{c.blob(gi)}, nullptr);
    }
    return g;
}

const CompressedColumn& membership_column(const CompressedTrajectory& ct) {
    const CompressedColumn* m = ct.column(kMembershipColumn);
    if (!m) corrupt("a file with a varying atom count has no membership column");
    return *m;
}

}  // namespace

struct AtomCounts {
    std::mutex mutex;
    bool done = false;
    std::vector<std::size_t> counts;
};

CompressedTrajectory compress_variable(const FrameBatch& in, const CompressOptions& opts) {
    FrameBatch converted;
    const FrameBatch* bp = &in;
    if (!in.variable()) {
        converted = in;
        converted.make_variable();
        bp = &converted;
    }
    const FrameBatch& b = *bp;
    if (!b.positions()) invalid("FrameBatch has no positions column");
    if (b.frame_natoms->size() != b.nframes) invalid("frame_natoms must have one entry per frame");
    const std::vector<std::size_t> off = b.atom_offsets();
    const std::size_t total = off[b.nframes];
    for (const auto& [name, v] : b.columns)
        if ((total == 0 && !v.empty()) || (total && v.size() % total))
            invalid("column '" + name + "' has " + std::to_string(v.size()) + " values for " + std::to_string(total) +
                    " atoms over all frames");
    if (b.positions()->size() != total * 3) invalid("positions must have 3 values per atom");
    for (const auto& [name, v] : b.int_columns)
        if ((total == 0 && !v.empty()) || (total && v.size() % total))
            invalid("column '" + name + "' has " + std::to_string(v.size()) + " values for " + std::to_string(total) +
                    " atoms over all frames");
    opts.resolved().validate();

    CompressOptions inner = opts;
    inner.variable_atoms = false;
    inner.source_format.reset();
    const GofTable gofs = GofTable::uniform(b.nframes, opts.gof_size);
    const std::size_t ngroups = gofs.size();
    std::vector<Membership> members(ngroups);
    std::vector<CompressedTrajectory> parts(ngroups);
    parallel_for(ngroups, [&](std::size_t g) {
        const auto s = gofs.span(g);
        parts[g] = compress(densify(b, off, s.start, s.end, members[g]), inner);
    });
    if (ngroups == 0) {
        // With no frames, compress an empty batch anyway so the file still
        // describes its columns.
        Membership none;
        CompressedTrajectory empty = compress(densify(b, off, 0, 0, none), inner);
        parts.push_back(std::move(empty));
    }

    CompressedTrajectory ct;
    ct.natoms = 0;
    ct.nframes = b.nframes;
    ct.gof_size = opts.gof_size;
    ct.gofs = gofs;
    ct.variable_atoms = true;
    const CompressedTrajectory& first = parts[0];
    for (std::size_t k = 0; k < first.columns.size(); ++k) {
        std::vector<Bytes> blobs;
        blobs.reserve(ngroups);
        for (std::size_t g = 0; g < ngroups; ++g) {
            const CompressedColumn& c = parts[g].columns[k];
            if (c.desc != first.columns[k].desc) invalid("column '" + c.desc.name + "' changes shape between groups");
            ByteSpan blob = c.blob(0);
            blobs.emplace_back(blob.begin(), blob.end());
        }
        ct.columns.emplace_back(first.columns[k].desc, std::move(blobs));
    }
    {
        ColumnDesc d;
        d.name = kMembershipColumn;
        d.dtype = Dtype::I64;
        d.width = 1;
        std::vector<Bytes> blobs(ngroups);
        parallel_for(ngroups, [&](std::size_t g) { blobs[g] = encode_membership(members[g], opts.zstd_level); });
        ct.columns.emplace_back(std::move(d), std::move(blobs));
    }

    Metadata meta;
    meta.source_format = opts.source_format;
    meta.attributes = b.attributes;
    if (b.units) meta.attributes[kUnitsAttr] = "lammps:" + std::string(b.units->style);
    if (!meta.empty()) ct.metadata = meta.encode();

    ct.atom_counts = std::make_shared<AtomCounts>();
    ct.atom_counts->done = true;
    ct.atom_counts->counts = *b.frame_natoms;
    return ct;
}

std::shared_ptr<AtomCounts> new_atom_counts() { return std::make_shared<AtomCounts>(); }

std::vector<std::size_t> variable_frame_natoms(const CompressedTrajectory& ct) {
    std::shared_ptr<AtomCounts> counts = ct.atom_counts;
    if (!counts) corrupt("missing atom counts");
    std::lock_guard<std::mutex> lock(counts->mutex);
    if (!counts->done) {
        const CompressedColumn& mem = membership_column(ct);
        std::vector<std::size_t> c(ct.nframes);
        for (std::size_t gi = 0; gi < ct.gofs.size(); ++gi) {
            const auto s = ct.gofs.span(gi);
            const Membership m = decode_membership(mem.blob(gi), s.size());
            for (std::size_t f = 0; f < s.size(); ++f) c[s.start + f] = m.count(f);
        }
        counts->counts = std::move(c);
        counts->done = true;
    }
    return counts->counts;
}

// Decodes frames [start, end) of a varying-count trajectory, either every
// column or only `only`, and compacts each frame back to the atoms it holds.
FrameBatch decompress_variable(const CompressedTrajectory& ct, std::size_t start, std::size_t end,
                               const std::string* only) {
    const CompressedColumn& mem = membership_column(ct);
    FrameBatch out(0, 0);
    out.frame_natoms.emplace();
    const auto groups = ct.gofs.overlapping(start, end);
    // Decode groups in parallel, then join them in order.
    std::vector<FrameBatch> parts(groups.size());
    parallel_for(groups.size(), [&](std::size_t k) {
        const std::size_t gi = groups.start + k;
        const auto s = ct.gofs.span(gi);
        const std::size_t lo = std::max(start, s.start) - s.start, hi = std::min(end, s.end) - s.start;
        const Membership m = decode_membership(mem.blob(gi), s.size());
        CompressedTrajectory g = group_view(ct, gi, m.ids.size());
        if (only) {
            auto keep = std::remove_if(g.columns.begin(), g.columns.end(),
                                       [&](const CompressedColumn& c) { return c.desc.name != *only; });
            g.columns.erase(keep, g.columns.end());
        }
        FrameBatch dense = decompress(g, lo, hi);
        const std::size_t N = m.ids.size();
        FrameBatch& p = parts[k];
        p.nframes = hi - lo;
        p.frame_natoms.emplace();
        std::vector<std::size_t> keep_slots;  // row of each present atom in `dense`
        for (std::size_t f = lo; f < hi; ++f) {
            std::size_t c = 0;
            for (std::size_t a = 0; a < N; ++a)
                if (m.present(f, a)) {
                    keep_slots.push_back((f - lo) * N + a);
                    ++c;
                }
            p.frame_natoms->push_back(c);
        }
        auto compact = [&](const auto& src, auto& dst) {
            const std::size_t rows = N * (hi - lo), w = rows == 0 ? 0 : src.size() / rows;
            dst.resize(keep_slots.size() * w);
            for (std::size_t i = 0; i < keep_slots.size(); ++i)
                std::copy(src.begin() + static_cast<std::ptrdiff_t>(keep_slots[i] * w),
                          src.begin() + static_cast<std::ptrdiff_t>((keep_slots[i] + 1) * w),
                          dst.begin() + static_cast<std::ptrdiff_t>(i * w));
        };
        for (const auto& [name, v] : dense.columns) compact(v, p.columns[name]);
        for (const auto& [name, v] : dense.int_columns) compact(v, p.int_columns[name]);
        if (!only) {
            auto& ids = p.int_columns[columns::kId];
            ids.reserve(keep_slots.size());
            for (std::size_t r : keep_slots) {
                const std::int64_t id = m.ids[r % N];
                if (id < INT32_MIN || id > INT32_MAX)
                    corrupt("atom id " + std::to_string(id) + " does not fit 32 bits");
                ids.push_back(static_cast<std::int32_t>(id));
            }
        }
        p.frame_props = std::move(dense.frame_props);
        p.int_frame_props = std::move(dense.int_frame_props);
        p.box = std::move(dense.box);
        p.origin = std::move(dense.origin);
        p.attributes = std::move(dense.attributes);
        p.units = dense.units;
        if (!m.strings.empty()) p.strings = share(std::vector<std::string>(m.strings));
    });
    for (auto& p : parts) out.append(p);
    out.natoms = 0;
    return out;
}

}  // namespace detail
}  // namespace mdz
