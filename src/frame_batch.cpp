#include "mdz/frame_batch.hpp"

#include <algorithm>
#include <cmath>

#include "mdz/error.hpp"

namespace mdz {
namespace {

constexpr double kPi = 3.14159265358979323846;

// cos and sin of an angle in degrees, exact at 90 degrees.
void cos_sin_deg(double deg, double& c, double& s) {
    if (deg == 90.0) {
        c = 0.0;
        s = 1.0;
        return;
    }
    const double r = deg * (kPi / 180.0);
    c = std::cos(r);
    s = std::sin(r);
}

template <class T>
std::vector<T> cut(const std::vector<T>& v, std::size_t w, std::size_t start, std::size_t end) {
    std::size_t lo = std::min(start * w, v.size()), hi = std::min(end * w, v.size());
    return std::vector<T>(v.begin() + static_cast<std::ptrdiff_t>(lo), v.begin() + static_cast<std::ptrdiff_t>(hi));
}

template <class V>
void extend(std::vector<V>& a, const std::vector<V>& b) {
    a.insert(a.end(), b.begin(), b.end());
}

}  // namespace

Box cell_params_to_box(const CellParams& p) noexcept {
    const double a = p[0], b = p[2], c = p[5];
    double ca, sa, cb, sb, cg, sg;
    cos_sin_deg(p[4], ca, sa);
    cos_sin_deg(p[3], cb, sb);
    cos_sin_deg(p[1], cg, sg);
    (void)sa;
    (void)sb;
    const double bx = b * cg, by = b * sg;
    const double cx = c * cb;
    const double cy = std::fabs(sg) > 1e-10 ? c * (ca - cb * cg) / sg : 0.0;
    const double cz2 = c * c - cx * cx - cy * cy;
    const double cz = cz2 > 0.0 ? std::sqrt(cz2) : 0.0;
    return {a, 0.0, 0.0, bx, by, 0.0, cx, cy, cz};
}

CellParams box_to_cell_params(const Box& m) noexcept {
    auto dot = [&](int r, int s) {
        return m[3 * r] * m[3 * s] + m[3 * r + 1] * m[3 * s + 1] + m[3 * r + 2] * m[3 * s + 2];
    };
    const double la = std::sqrt(dot(0, 0)), lb = std::sqrt(dot(1, 1)), lc = std::sqrt(dot(2, 2));
    auto angle = [&](int r, int s, double lr, double ls) {
        const double d = lr * ls;
        if (d == 0.0) return 90.0;
        const double c = dot(r, s) / d;
        if (c == 0.0) return 90.0;
        return std::acos(std::clamp(c, -1.0, 1.0)) * (180.0 / kPi);
    };
    return {la, angle(0, 1, la, lb), lb, angle(0, 2, la, lc), angle(1, 2, lb, lc), lc};
}

const std::vector<float>* FrameBatch::column(std::string_view name) const {
    auto it = columns.find(name);
    return it == columns.end() ? nullptr : &it->second;
}

std::vector<float>* FrameBatch::column(std::string_view name) {
    auto it = columns.find(name);
    return it == columns.end() ? nullptr : &it->second;
}

std::optional<std::vector<CellParams>> FrameBatch::cell_params() const {
    if (!box) return std::nullopt;
    std::vector<CellParams> out;
    out.reserve(box->size());
    for (const Box& m : *box) out.push_back(box_to_cell_params(m));
    return out;
}

std::size_t FrameBatch::total_atoms() const noexcept {
    if (!variable()) return natoms * nframes;
    std::size_t n = 0;
    for (std::size_t c : *frame_natoms) n += c;
    return n;
}

std::vector<std::size_t> FrameBatch::atom_offsets() const {
    std::vector<std::size_t> o(nframes + 1, 0);
    for (std::size_t f = 0; f < nframes; ++f) o[f + 1] = o[f] + atoms_in(f);
    return o;
}

FrameBatch FrameBatch::slice_frames(std::size_t start, std::size_t end) const {
    start = std::min(start, nframes);
    end = std::clamp(end, start, nframes);
    FrameBatch out(natoms, end - start);
    // Per-atom columns are sliced by atom index, using the per-frame offsets
    // when the count varies.
    std::size_t lo = start * natoms, hi = end * natoms, total = natoms * nframes;
    if (variable()) {
        const auto o = atom_offsets();
        lo = o[start];
        hi = o[end];
        total = o[nframes];
        out.frame_natoms.emplace(frame_natoms->begin() + static_cast<std::ptrdiff_t>(start),
                                 frame_natoms->begin() + static_cast<std::ptrdiff_t>(end));
    }
    auto atoms = [&](const auto& v) {
        using V = std::decay_t<decltype(v)>;
        const std::size_t w = total == 0 ? 0 : v.size() / total;
        return V(v.begin() + static_cast<std::ptrdiff_t>(std::min(lo * w, v.size())),
                 v.begin() + static_cast<std::ptrdiff_t>(std::min(hi * w, v.size())));
    };
    for (const auto& [k, v] : columns) out.columns.emplace(k, atoms(v));
    for (const auto& [k, v] : int_columns) out.int_columns.emplace(k, atoms(v));
    for (const auto& [k, v] : frame_props) out.frame_props.emplace(k, cut(v, 1, start, end));
    for (const auto& [k, v] : int_frame_props) out.int_frame_props.emplace(k, cut(v, 1, start, end));
    if (box) out.box = cut(*box, 1, start, end);
    if (origin) out.origin = cut(*origin, 1, start, end);
    out.attributes = attributes;
    out.units = units;
    out.strings = strings;
    out.atom_names = atom_names;
    out.elements = elements;
    out.atom_types = atom_types;
    out.atom_ids = atom_ids;
    out.residue_names = residue_names;
    out.residue_ids = residue_ids;
    out.chain_ids = chain_ids;
    out.hetatm = hetatm;
    out.bodies = bodies;
    return out;
}

namespace {

// Pair each static metadata field with the column that carries it when the
// atom count varies.
template <class F>
void for_each_int_meta(FrameBatch& b, F&& f) {
    f(columns::kId, b.atom_ids);
    f(columns::kType, b.atom_types);
    f(columns::kResid, b.residue_ids);
}

template <class F>
void for_each_string_meta(FrameBatch& b, F&& f) {
    f(columns::kName, b.atom_names);
    f(columns::kElement, b.elements);
    f(columns::kResname, b.residue_names);
    f(columns::kChain, b.chain_ids);
}

const char* const kStringColumns[] = {columns::kName, columns::kElement, columns::kResname, columns::kChain};

// The index of s in table, adding it if it is new.
std::int32_t intern(std::vector<std::string>& table, std::map<std::string, std::int32_t, std::less<>>& index,
                    const std::string& s) {
    auto it = index.find(s);
    if (it != index.end()) return it->second;
    const auto i = static_cast<std::int32_t>(table.size());
    table.push_back(s);
    index.emplace(s, i);
    return i;
}

}  // namespace

void FrameBatch::make_variable() {
    if (variable()) return;
    if (bodies) throw Error(ErrorKind::InvalidInput, "rigid bodies need a fixed atom count");
    const std::size_t n = natoms, nf = nframes;
    frame_natoms.emplace(nf, n);
    auto replicate = [&](const char* name, const std::vector<std::int32_t>& per_atom) {
        auto& c = int_columns[name];
        c.clear();
        c.reserve(n * nf);
        for (std::size_t f = 0; f < nf; ++f) c.insert(c.end(), per_atom.begin(), per_atom.end());
    };
    for_each_int_meta(*this, [&](const char* name, Shared<std::int32_t>& m) {
        if (m) replicate(name, *m);
        m.reset();
    });
    if (hetatm) replicate(columns::kHetatm, std::vector<std::int32_t>(hetatm->begin(), hetatm->end()));
    hetatm.reset();
    std::vector<std::string> table = strings ? *strings : std::vector<std::string>{};
    std::map<std::string, std::int32_t, std::less<>> index;
    for (std::size_t i = 0; i < table.size(); ++i) index.emplace(table[i], static_cast<std::int32_t>(i));
    for_each_string_meta(*this, [&](const char* name, Shared<std::string>& m) {
        if (!m) return;
        std::vector<std::int32_t> ix(m->size());
        for (std::size_t i = 0; i < m->size(); ++i) ix[i] = intern(table, index, (*m)[i]);
        replicate(name, ix);
        m.reset();
    });
    strings = table.empty() ? nullptr : share(std::move(table));
    natoms = 0;
}

FrameBatch FrameBatch::frame(std::size_t f) const {
    FrameBatch out = slice_frames(f, f + 1);
    if (!variable()) return out;
    out.natoms = (*frame_natoms)[f];
    out.frame_natoms.reset();
    for_each_int_meta(out, [&](const char* name, Shared<std::int32_t>& m) {
        auto it = out.int_columns.find(name);
        if (it == out.int_columns.end()) return;
        m = share(std::move(it->second));
        out.int_columns.erase(it);
    });
    if (auto it = out.int_columns.find(columns::kHetatm); it != out.int_columns.end()) {
        out.hetatm = share(std::vector<std::uint8_t>(it->second.begin(), it->second.end()));
        out.int_columns.erase(it);
    }
    for_each_string_meta(out, [&](const char* name, Shared<std::string>& m) {
        auto it = out.int_columns.find(name);
        if (it == out.int_columns.end()) return;
        std::vector<std::string> v(it->second.size());
        for (std::size_t i = 0; i < v.size(); ++i) {
            const auto k = static_cast<std::size_t>(it->second[i]);
            if (!strings || k >= strings->size())
                throw Error(ErrorKind::Corrupt, std::string(name) + " index " + std::to_string(k) + " is out of range");
            v[i] = (*strings)[k];
        }
        m = share(std::move(v));
        out.int_columns.erase(it);
    });
    out.strings.reset();
    return out;
}

void FrameBatch::append_fixed(const FrameBatch& b) {
    natoms = b.natoms;
    nframes += b.nframes;
    for (const auto& [k, v] : b.columns) extend(columns[k], v);
    for (const auto& [k, v] : b.int_columns) extend(int_columns[k], v);
    for (const auto& [k, v] : b.frame_props) extend(frame_props[k], v);
    for (const auto& [k, v] : b.int_frame_props) extend(int_frame_props[k], v);
    if (b.box) {
        if (!box) box.emplace();
        extend(*box, *b.box);
    }
    if (b.origin) {
        if (!origin) origin.emplace();
        extend(*origin, *b.origin);
    }
    if (!atom_names) atom_names = b.atom_names;
    if (!elements) elements = b.elements;
    if (!atom_types) atom_types = b.atom_types;
    if (!atom_ids) atom_ids = b.atom_ids;
    if (!residue_names) residue_names = b.residue_names;
    if (!residue_ids) residue_ids = b.residue_ids;
    if (!chain_ids) chain_ids = b.chain_ids;
    if (!hetatm) hetatm = b.hetatm;
    if (!bodies) bodies = b.bodies;
    if (attributes.empty()) attributes = b.attributes;
    if (!units) units = b.units;
}

namespace {

// Whether two batches' static per-atom metadata can describe the same atoms.
// A field missing from either batch is not a conflict, because the first
// batch that has it wins.
template <class T>
bool same(const Shared<T>& a, const Shared<T>& b) {
    return !a || !b || a == b || *a == *b;
}

bool same_atoms(const FrameBatch& a, const FrameBatch& b) {
    return same(a.atom_names, b.atom_names) && same(a.elements, b.elements) && same(a.atom_types, b.atom_types) &&
           same(a.atom_ids, b.atom_ids) && same(a.residue_names, b.residue_names) &&
           same(a.residue_ids, b.residue_ids) && same(a.chain_ids, b.chain_ids) && same(a.hetatm, b.hetatm) &&
           same(a.bodies, b.bodies);
}

}  // namespace

void FrameBatch::append(const FrameBatch& b) {
    // If both batches hold the same atoms, b's frames are appended directly.
    // Otherwise (a different count, or the same count of different atoms) the
    // result has a varying atom count.
    if (!variable() && !b.variable() && (nframes == 0 || (b.natoms == natoms && same_atoms(*this, b)))) {
        append_fixed(b);
        return;
    }
    make_variable();
    FrameBatch v = b;
    v.make_variable();
    // A side with no atoms has no values to conflict with, so frames of no
    // atoms may lack columns that the other frames carry.
    if (total_atoms() > 0 && v.total_atoms() > 0) {
        auto keys = [](const auto& m) {
            std::vector<std::string> k;
            for (const auto& kv : m) k.push_back(kv.first);
            return k;
        };
        if (keys(columns) != keys(v.columns) || keys(int_columns) != keys(v.int_columns))
            throw Error(ErrorKind::InvalidInput,
                        "frames with different atom counts must carry the same per-atom columns");
    }
    // Re-index the strings of v into this batch's table.
    if (v.strings) {
        std::vector<std::string> table = strings ? *strings : std::vector<std::string>{};
        std::map<std::string, std::int32_t, std::less<>> index;
        for (std::size_t i = 0; i < table.size(); ++i) index.emplace(table[i], static_cast<std::int32_t>(i));
        std::vector<std::int32_t> remap(v.strings->size());
        for (std::size_t i = 0; i < remap.size(); ++i) remap[i] = intern(table, index, (*v.strings)[i]);
        for (const char* name : kStringColumns)
            if (auto it = v.int_columns.find(name); it != v.int_columns.end())
                for (auto& x : it->second) x = remap[static_cast<std::size_t>(x)];
        strings = share(std::move(table));
    }
    for (const auto& [k, x] : v.columns) extend(columns[k], x);
    for (const auto& [k, x] : v.int_columns) extend(int_columns[k], x);
    for (const auto& [k, x] : v.frame_props) extend(frame_props[k], x);
    for (const auto& [k, x] : v.int_frame_props) extend(int_frame_props[k], x);
    if (v.box) {
        if (!box) box.emplace();
        extend(*box, *v.box);
    }
    if (v.origin) {
        if (!origin) origin.emplace();
        extend(*origin, *v.origin);
    }
    extend(*frame_natoms, *v.frame_natoms);
    nframes += v.nframes;
    if (attributes.empty()) attributes = v.attributes;
    if (!units) units = v.units;
}

void FrameBatch::convert_units(const LammpsUnits* to) {
    auto factor = [&](double LammpsUnits::* f) {
        const double from = units ? units->*f : 1.0;
        const double t = to ? to->*f : 1.0;
        return from / t;
    };
    const double fl = factor(&LammpsUnits::length), fv = factor(&LammpsUnits::velocity),
                 ff = factor(&LammpsUnits::force);
    auto scale = [](std::vector<float>& v, double f) {
        if (f == 1.0) return;
        for (float& x : v) x = static_cast<float>(static_cast<double>(x) * f);
    };
    if (auto* v = column("positions")) scale(*v, fl);
    if (auto* v = column("velocities")) scale(*v, fv);
    if (auto* v = column("forces")) scale(*v, ff);
    if (fl != 1.0) {
        if (box)
            for (Box& m : *box)
                for (double& x : m) x *= fl;
        if (origin)
            for (Vec3& o : *origin)
                for (double& x : o) x *= fl;
    }
    units = to;
}

FrameBatch FrameBatch::in_internal_units() const {
    FrameBatch b = *this;
    if (units) b.convert_units(nullptr);
    return b;
}

std::vector<std::string> FrameBatch::available_data() const {
    static const std::pair<const char*, const char*> meta[] = {
        {columns::kId, "atom_ids"},           {columns::kType, "atom_types"}, {columns::kResid, "residue_ids"},
        {columns::kHetatm, "hetatm"},         {columns::kName, "atom_names"}, {columns::kElement, "elements"},
        {columns::kResname, "residue_names"}, {columns::kChain, "chain_ids"}};
    std::vector<std::string> d;
    for (const auto& kv : columns) d.push_back(kv.first);
    for (const auto& kv : int_columns) {
        std::string name = kv.first;
        for (const auto& [col, logical] : meta)
            if (name == col) name = logical;
        d.push_back(name);
    }
    for (const auto& kv : frame_props) d.push_back(kv.first);
    for (const auto& kv : int_frame_props) d.push_back(kv.first);
    if (box) d.emplace_back("box");
    if (origin) d.emplace_back("origin");
    if (atom_names) d.emplace_back("atom_names");
    if (elements) d.emplace_back("elements");
    if (atom_types) d.emplace_back("atom_types");
    if (atom_ids) d.emplace_back("atom_ids");
    if (residue_names) d.emplace_back("residue_names");
    if (residue_ids) d.emplace_back("residue_ids");
    if (chain_ids) d.emplace_back("chain_ids");
    if (hetatm) d.emplace_back("hetatm");
    std::sort(d.begin(), d.end());
    return d;
}

}  // namespace mdz
