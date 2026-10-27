// Converts frames into columns and columns back into frames. This is the
// layer above the group-of-frames codecs.
//
// Blob encodings, each zstd-compressed, with little-endian integers:
//   per-atom F32 x 3 (positions, velocities, forces): [3][order][mode][payload],
//     see codec.cpp
//   0x20 quantized F32: [tag][precision f32][count u32][count x i32]
//   0x24 exact F32:     [tag][count u32][4 byte planes of count u32 words]
//   0x21 I32:           [tag][count u32][count x i32]
//   0x22 F64:           [tag][count u32][count x f64]
//   0x23 I64:           [tag][count u32][count x i64]

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

#include "bytes.hpp"
#include "codec.hpp"
#include "container.hpp"
#include "kernels.hpp"
#include "mdz/error.hpp"
#include "mdz/mdz.hpp"
#include "mdz/trajectory.hpp"
#include "parallel.hpp"
#include "rigid.hpp"
#include "variable.hpp"
#include "zstd.hpp"

namespace mdz {

using detail::Bytes;
using detail::ByteCursor;
using detail::corrupt;
using detail::put_le;
using detail::put_u8;
using detail::sat_add;
using detail::sat_mul;

namespace {

constexpr std::uint8_t kTagQuantF32 = 0x20;
constexpr std::uint8_t kTagInt32 = 0x21;
constexpr std::uint8_t kTagRawF64 = 0x22;
constexpr std::uint8_t kTagInt64 = 0x23;
constexpr std::uint8_t kTagRawF32 = 0x24;

[[noreturn]] void invalid(const std::string& m) { throw Error(ErrorKind::InvalidInput, m); }

// --- simple tagged blobs ---

template <class T>
Bytes encode_plain(std::uint8_t tag, const T* v, std::size_t n, int level) {
    thread_local Bytes payload;
    payload.clear();
    payload.reserve(5 + n * sizeof(T));
    put_u8(payload, tag);
    put_le<std::uint32_t>(payload, static_cast<std::uint32_t>(n));
    detail::put_bytes(payload, v, n * sizeof(T));
    return detail::zstd_compress(payload, level);
}

Bytes encode_quant_f32(const float* v, std::size_t n, float precision, int level) {
    thread_local Bytes payload;
    payload.clear();
    payload.resize(9 + n * 4);
    payload[0] = kTagQuantF32;
    detail::store_le(payload.data() + 1, precision);
    detail::store_le(payload.data() + 5, static_cast<std::uint32_t>(n));
    const float inv = precision > 0.0f ? 1.0f / precision : 1.0f;
    std::uint8_t* dst = payload.data() + 9;
    for (std::size_t i = 0; i < n; ++i) detail::store_le(dst + 4 * i, detail::round_to_i32(v[i] * inv));
    return detail::zstd_compress(payload, level);
}

Bytes encode_raw_f32(const float* v, std::size_t n, int level) {
    thread_local Bytes payload;
    thread_local std::vector<std::uint32_t> words;
    words.resize(n);
    for (std::size_t i = 0; i < n; ++i) words[i] = detail::float_bits(v[i]);
    payload.clear();
    payload.resize(5 + n * 4);
    payload[0] = kTagRawF32;
    detail::store_le(payload.data() + 1, static_cast<std::uint32_t>(n));
    detail::byte_planes(words.data(), n, payload.data() + 5);
    return detail::zstd_compress(payload, level);
}

// How many values a tagged blob must hold. Per-atom data must hold exactly n,
// and a short group is corrupt. Per-frame data may hold fewer than n, because
// a per-frame column need not cover every frame.
struct Count {
    std::size_t n;
    bool exact;
};

std::string describe(Count c) { return (c.exact ? "exactly " : "at most ") + std::to_string(c.n); }

// Validate a tagged payload's framing and return its value bytes.
ByteSpan tagged_body(ByteSpan raw, std::uint8_t tag, std::size_t extra, std::size_t elem, Count count,
                     const char* what) {
    ByteCursor c(raw);
    if (c.u8() != tag) corrupt(std::string("not a ") + what + " blob");
    c.skip(extra);
    const std::size_t stored = c.u32();
    if (count.exact ? stored != count.n : stored > count.n)
        corrupt(std::string(what) + " blob holds " + std::to_string(stored) + " values, expected " + describe(count));
    ByteSpan body = c.rest();
    if (body.size() != sat_mul(stored, elem)) corrupt(std::string(what) + " blob length mismatch");
    return body;
}

void decompress_tagged(ByteSpan blob, std::size_t extra, std::size_t elem, Count count, Bytes& raw) {
    detail::zstd_decompress_bounded(blob, sat_add(sat_mul(count.n, elem), 5 + extra), raw);
}

template <class T>
void read_values(ByteSpan body, std::vector<T>& out) {
    const std::size_t n = body.size() / sizeof(T), at = out.size();
    if (n == 0) return;
    out.resize(at + n);
    std::memcpy(out.data() + at, body.data(), n * sizeof(T));
}

// Decode a scalar F32 payload, quantized or exact, appending to out.
void decode_scalar_f32_payload(ByteSpan raw, Count count, std::vector<float>& out) {
    if (!raw.empty() && raw[0] == kTagRawF32) {
        ByteSpan body = tagged_body(raw, kTagRawF32, 0, 4, count, "raw_f32");
        const std::size_t n = body.size() / 4, at = out.size();
        out.resize(at + n);
        const std::uint8_t *p0 = body.data(), *p1 = p0 + n, *p2 = p1 + n, *p3 = p2 + n;
        for (std::size_t i = 0; i < n; ++i)
            out[at + i] = detail::bits_float(p0[i] | (std::uint32_t(p1[i]) << 8) | (std::uint32_t(p2[i]) << 16) |
                                             (std::uint32_t(p3[i]) << 24));
        return;
    }
    ByteSpan body = tagged_body(raw, kTagQuantF32, 4, 4, count, "quant_f32");
    const float precision = detail::load_le<float>(raw.data() + 1);
    const std::size_t n = body.size() / 4, at = out.size();
    out.resize(at + n);
    for (std::size_t i = 0; i < n; ++i)
        out[at + i] = static_cast<float>(detail::load_le<std::int32_t>(body.data() + 4 * i)) * precision;
}

// --- compression helpers ---

bool is_xyz(std::string_view name) { return name == "positions" || name == "velocities" || name == "forces"; }

// The quantization step for a per-atom column other than positions,
// velocities and forces. A step of 0 stores the column exactly. The standard
// scalars (charge, bfactor, occupancy) default to the global precision.
float extra_precision(std::string_view name, const CompressOptions& o) {
    if (o.lossless) return 0.0f;
    auto it = o.column_precision.find(name);
    if (it != o.column_precision.end()) return it->second;
    if (name == "charge" || name == "bfactor" || name == "occupancy") return o.precision;
    return 0.0f;
}

// Precision options are given in internal units. A batch stored in a LAMMPS
// unit style needs them converted to its own units.
CompressOptions in_batch_units(const CompressOptions& opts, const FrameBatch& b) {
    if (!b.units) return opts;
    CompressOptions o = opts;
    const std::pair<const char*, double> f[] = {
        {"positions", b.units->length}, {"velocities", b.units->velocity}, {"forces", b.units->force}};
    for (const auto& [name, factor] : f) {
        const float p = opts.precision_for(name);
        if (p > 0.0f) o.column_precision[name] = static_cast<float>(static_cast<double>(p) / factor);
    }
    return o;
}

std::size_t width_of(std::size_t len, std::size_t natoms, std::size_t nframes) {
    const std::size_t n = natoms * nframes;
    return n == 0 ? 1 : len / n;
}

void check_shape(const FrameBatch& b) {
    const std::size_t natoms = b.natoms, nframes = b.nframes, per = natoms * nframes;
    auto any_width = [&](const std::string& name, std::size_t len) {
        if (per == 0) {
            if (len != 0) invalid("column '" + name + "' has values but the batch has no atoms or frames");
            return;
        }
        if (len == 0 || len % per != 0)
            invalid("column '" + name + "' has " + std::to_string(len) + " values, not a multiple of " +
                    std::to_string(natoms) + " atoms \xC3\x97 " + std::to_string(nframes) + " frames");
        if (len / per > 255) invalid("column '" + name + "' is too wide");
    };
    for (const auto& [name, v] : b.columns) {
        if (is_xyz(name)) {
            if (v.size() != per * 3)
                invalid("column '" + name + "' has " + std::to_string(v.size()) + " values; expected " +
                        std::to_string(natoms) + " atoms \xC3\x97 " + std::to_string(nframes) + " frames \xC3\x97 3");
        } else {
            any_width(name, v.size());
        }
    }
    for (const auto& [name, v] : b.int_columns) any_width(name, v.size());
    for (const auto& [name, v] : b.frame_props)
        if (v.size() > nframes)
            invalid("frame property '" + name + "' has " + std::to_string(v.size()) + " values for " +
                    std::to_string(nframes) + " frames");
    if (b.box && b.box->size() > nframes) invalid("more box matrices than the " + std::to_string(nframes) + " frames");
    if (b.origin && b.origin->size() > nframes)
        invalid("more box origins than the " + std::to_string(nframes) + " frames");
    for (const auto& [name, v] : b.int_frame_props) {
        if (v.size() > nframes)
            invalid("frame property '" + name + "' has " + std::to_string(v.size()) + " values for " +
                    std::to_string(nframes) + " frames");
        if (b.frame_props.count(name)) invalid("frame property '" + name + "' is both a float and an integer");
    }
    if (b.attributes.size() > 200)
        invalid(std::to_string(b.attributes.size()) + " attributes; at most 200 are stored");
    for (const auto& kv : b.attributes)
        if (kv.first.size() > 250) invalid("attribute name '" + kv.first + "' is longer than 250 bytes");
    auto meta = [&](const char* name, std::size_t len, bool present) {
        if (present && len != natoms)
            invalid(std::string(name) + " has " + std::to_string(len) + " entries for " + std::to_string(natoms) +
                    " atoms");
    };
    meta("atom_names", b.atom_names ? b.atom_names->size() : 0, bool(b.atom_names));
    meta("elements", b.elements ? b.elements->size() : 0, bool(b.elements));
    meta("atom_types", b.atom_types ? b.atom_types->size() : 0, bool(b.atom_types));
    meta("atom_ids", b.atom_ids ? b.atom_ids->size() : 0, bool(b.atom_ids));
    meta("residue_names", b.residue_names ? b.residue_names->size() : 0, bool(b.residue_names));
    meta("residue_ids", b.residue_ids ? b.residue_ids->size() : 0, bool(b.residue_ids));
    meta("chain_ids", b.chain_ids ? b.chain_ids->size() : 0, bool(b.chain_ids));
    meta("hetatm", b.hetatm ? b.hetatm->size() : 0, bool(b.hetatm));
}

void validate_values(const std::string& name, const std::vector<float>& col, const CompressOptions& o) {
    try {
        if (o.lossless) {
            for (std::size_t i = 0; i < col.size(); ++i)
                if (!std::isfinite(col[i]))
                    invalid("value at index " + std::to_string(i) + " is " + (std::isnan(col[i]) ? "NaN" : "inf") +
                            "; mdz cannot store non-finite (NaN/Inf) values");
            return;
        }
        detail::validate_coords(col.data(), col.size(), o.precision_for(name));
    } catch (const Error& e) {
        invalid("column '" + name + "': " + e.what());
    }
}

ColumnDesc describe_xyz(ColumnDesc d, float precision, const CompressOptions& o) {
    d.precision = precision;
    d.lossy = precision > 0.0f;
    d.pred_order = static_cast<std::uint8_t>(o.predictor_order);
    d.anchor_step = static_cast<std::uint8_t>(o.anchor_step);
    d.codec_flags = static_cast<std::uint8_t>((o.adaptive ? kFlagAdaptive : 0) | (o.anchor_step > 1 ? kFlagBFrames : 0) |
                                              (o.spatial_delta ? kFlagSpatialDelta : 0));
    return d;
}

std::vector<Bytes> encode_xyz_groups(const float* data, std::size_t natoms, std::size_t nframes, float precision,
                                     const CompressOptions& o, const detail::RigidModel* rigid = nullptr) {
    const std::size_t g = o.gof_size, ngroups = (nframes + g - 1) / g, n3 = natoms * 3;
    std::vector<Bytes> blobs(ngroups);
    detail::GofParams p;
    p.precision = precision;
    p.predictor_order = o.predictor_order;
    p.adaptive = o.adaptive;
    p.anchor_step = o.anchor_step;
    p.spatial_delta = o.spatial_delta;
    p.codec = o.codec;
    p.zstd_level = o.zstd_level;
    p.rigid = rigid;
    detail::parallel_for(ngroups, [&](std::size_t i) {
        const std::size_t start = i * g, end = std::min(start + g, nframes);
        detail::encode_gof_blob(data + start * n3, end - start, natoms, p, blobs[i]);
    });
    return blobs;
}

std::vector<Bytes> encode_scalar_groups(const float* data, std::size_t per_frame, std::size_t nframes,
                                        std::size_t g, float precision, int level) {
    const std::size_t ngroups = (nframes + g - 1) / g;
    std::vector<Bytes> blobs(ngroups);
    detail::parallel_for(ngroups, [&](std::size_t i) {
        const std::size_t start = i * g, end = std::min(start + g, nframes);
        const float* s = data + start * per_frame;
        const std::size_t n = (end - start) * per_frame;
        blobs[i] = precision > 0.0f ? encode_quant_f32(s, n, precision, level) : encode_raw_f32(s, n, level);
    });
    return blobs;
}

template <class T>
std::vector<Bytes> encode_plain_groups(std::uint8_t tag, const T* data, std::size_t per_frame,
                                       std::size_t nframes, std::size_t g, int level) {
    const std::size_t ngroups = (nframes + g - 1) / g;
    std::vector<Bytes> blobs(ngroups);
    detail::parallel_for(ngroups, [&](std::size_t i) {
        const std::size_t start = i * g, end = std::min(start + g, nframes);
        blobs[i] = encode_plain(tag, data + start * per_frame, (end - start) * per_frame, level);
    });
    return blobs;
}

// One blob per group from a per-frame column with `width` values per frame.
// The column may cover fewer frames than the trajectory, and groups past its
// end get empty blobs.
template <class T>
std::vector<Bytes> encode_frame_groups(std::uint8_t tag, const std::vector<T>& flat, std::size_t width,
                                       const GofTable& gofs, int level) {
    std::vector<Bytes> blobs(gofs.size());
    for (std::size_t i = 0; i < gofs.size(); ++i) {
        const auto s = gofs.span(i);
        const std::size_t lo = std::min(s.start * width, flat.size()), hi = std::min(s.end * width, flat.size());
        blobs[i] = encode_plain(tag, flat.data() + lo, hi - lo, level);
    }
    return blobs;
}

template <std::size_t N>
std::vector<double> flatten(const std::vector<std::array<double, N>>& v) {
    std::vector<double> out;
    out.reserve(v.size() * N);
    for (const auto& a : v) out.insert(out.end(), a.begin(), a.end());
    return out;
}

std::string_view unit_label(const FrameBatch& b, std::string_view name) {
    if (!b.units) return {};
    if (name == "positions" || name == "box" || name == "origin") return b.units->length_unit;
    if (name == "velocities") return b.units->velocity_unit;
    if (name == "forces") return b.units->force_unit;
    return {};
}

Metadata metadata_of(const FrameBatch& b, const CompressOptions& o) {
    Metadata m;
    if (b.bodies) m.bodies = *b.bodies;
    auto own = [](const auto& s) {
        using V = std::decay_t<decltype(*s)>;
        return s ? std::optional<V>(*s) : std::nullopt;
    };
    m.atom_names = own(b.atom_names);
    m.elements = own(b.elements);
    m.atom_types = own(b.atom_types);
    m.atom_ids = own(b.atom_ids);
    m.residues = own(b.residue_names);
    m.residue_ids = own(b.residue_ids);
    m.chain_ids = own(b.chain_ids);
    if (b.hetatm) m.hetatm = std::vector<std::int32_t>(b.hetatm->begin(), b.hetatm->end());
    m.source_format = o.source_format;
    m.attributes = b.attributes;
    if (b.units) m.attributes[kUnitsAttr] = "lammps:" + std::string(b.units->style);
    return m;
}

const LammpsUnits* units_of(std::string_view sys) {
    std::string_view style = sys.substr(0, 7) == "lammps:" ? sys.substr(7) : sys;
    const LammpsUnits* u = LammpsUnits::from_style(style);
    if (!u) corrupt("unknown unit system '" + std::string(sys) + "'");
    return u;
}

void to_nm(std::vector<float>& v, double f) {
    if (f == 1.0) return;
    for (float& x : v) x = static_cast<float>(static_cast<double>(x) * f);
}

}  // namespace

// --- CompressedColumn / CompressedTrajectory ---

CompressedColumn::CompressedColumn(ColumnDesc d, std::vector<std::vector<std::uint8_t>> blobs) : desc(std::move(d)) {
    auto owned = std::make_shared<std::vector<std::vector<std::uint8_t>>>(std::move(blobs));
    blobs_.reserve(owned->size());
    for (const auto& b : *owned) blobs_.emplace_back(b.data(), b.size());
    backing_ = std::move(owned);
}

CompressedColumn::CompressedColumn(ColumnDesc d, std::shared_ptr<const void> backing, std::vector<ByteSpan> blobs,
                                   std::shared_ptr<detail::GroupChecks> checks)
    : desc(std::move(d)), backing_(std::move(backing)), blobs_(std::move(blobs)), checks_(std::move(checks)) {}

ByteSpan CompressedColumn::blob(std::size_t gi) const {
    if (gi >= blobs_.size()) corrupt("column '" + desc.name + "' missing GoF " + std::to_string(gi));
    if (checks_) checks_->verify(gi);
    return blobs_[gi];
}

std::size_t CompressedColumn::compressed_size() const noexcept {
    std::size_t n = 0;
    for (const auto& b : blobs_) n += b.size();
    return n;
}

const CompressedColumn* CompressedTrajectory::column(std::string_view name) const noexcept {
    for (const auto& c : columns)
        if (c.desc.name == name) return &c;
    return nullptr;
}

std::size_t CompressedTrajectory::compressed_size() const noexcept {
    std::size_t n = metadata ? metadata->size() : 0;
    for (const auto& c : columns) n += c.compressed_size();
    return n;
}

std::size_t CompressedTrajectory::raw_size() const {
    std::size_t n = 0;
    if (!variable_atoms) {
        for (const auto& c : columns) n += c.desc.raw_size(natoms, nframes);
        return n;
    }
    std::size_t atoms = 0;
    for (std::size_t c : frame_natoms()) atoms += c;
    for (const auto& c : columns)
        if (c.desc.name != kMembershipColumn)
            n += c.desc.per_frame ? c.desc.raw_size(0, nframes) : atoms * c.desc.width * dtype_size(c.desc.dtype);
    return n;
}

Metadata CompressedTrajectory::decoded_metadata() const {
    if (!metadata || metadata->empty()) return {};
    return Metadata::decode(*metadata);
}

double CompressedTrajectory::length_to_nm() const {
    if (!metadata || metadata->empty()) return 1.0;
    Metadata m = decoded_metadata();
    auto it = m.attributes.find(kUnitsAttr);
    if (it == m.attributes.end()) return 1.0;
    return units_of(it->second)->length;
}

std::vector<std::size_t> CompressedTrajectory::frame_natoms(std::size_t start, std::size_t end) const {
    start = std::min(start, nframes);
    end = std::max(start, std::min(end, nframes));
    if (!variable_atoms) return std::vector<std::size_t>(end - start, natoms);
    std::vector<std::size_t> all = detail::variable_frame_natoms(*this);
    return std::vector<std::size_t>(all.begin() + static_cast<std::ptrdiff_t>(start),
                                    all.begin() + static_cast<std::ptrdiff_t>(end));
}

std::size_t DecodedColumn::size() const noexcept {
    switch (dtype) {
        case Dtype::F64: return f64.size();
        case Dtype::I32: return i32.size();
        case Dtype::I64: return i64.size();
        default: return f32.size();
    }
}

// --- compress ---

CompressedTrajectory compress(const FrameBatch& batch, const CompressOptions& user_opts) {
    if (batch.variable() || user_opts.variable_atoms) return detail::compress_variable(batch, user_opts);
    const CompressOptions opts = in_batch_units(user_opts.resolved(), batch);
    const std::vector<float>* positions = batch.positions();
    if (!positions) invalid("FrameBatch has no positions column");
    const std::size_t natoms = batch.natoms, nframes = batch.nframes;
    const int level = opts.zstd_level;

    opts.validate();
    check_shape(batch);
    for (const auto& [name, col] : batch.columns)
        if (is_xyz(name) || extra_precision(name, opts) > 0.0f) validate_values(name, col, opts);

    CompressedTrajectory ct;
    ct.natoms = natoms;
    ct.nframes = nframes;
    ct.gof_size = opts.gof_size;
    ct.gofs = GofTable::uniform(nframes, opts.gof_size);

    // Rigid bodies, if enabled, model the positions. Use the bodies given in
    // the batch, or detect them.
    FrameBatch with_bodies;
    const FrameBatch* src = &batch;
    if (opts.rigid && !opts.lossless) {
        auto bodies = batch.bodies ? *batch.bodies : detail::detect_rigid_bodies(batch);
        auto model = detail::RigidModel::build(bodies, natoms);
        if (model->nbodies() > 0) {
            ct.rigid = model;
            if (!batch.bodies) {
                // Record detected bodies in the metadata so that readers can
                // rebuild the model.
                with_bodies.bodies = share(std::move(bodies));
                src = &with_bodies;
            }
        }
    }
    auto xyz = [&](const char* name, const std::vector<float>& data, ColumnDesc d,
                   const detail::RigidModel* rigid = nullptr) {
        const float prec = opts.precision_for(name);
        ct.columns.emplace_back(describe_xyz(std::move(d), prec, opts),
                                encode_xyz_groups(data.data(), natoms, nframes, prec, opts, rigid));
    };
    xyz("positions", *positions, ColumnDesc::positions(0.0f), ct.rigid.get());
    if (auto* v = batch.column("velocities")) xyz("velocities", *v, ColumnDesc::velocities(0.0f));
    if (auto* f = batch.column("forces")) xyz("forces", *f, ColumnDesc::forces(0.0f));

    // The remaining per-atom columns, by name. The standard scalars are
    // quantized at the default precision. Anything else is stored exactly
    // unless it has its own precision.
    for (const auto& [name, data] : batch.columns) {
        if (is_xyz(name)) continue;
        const std::size_t width = width_of(data.size(), natoms, nframes);
        const float prec = extra_precision(name, opts);
        ColumnDesc d = ColumnDesc::scalar(name, "", prec);
        d.width = static_cast<std::uint32_t>(width);
        ct.columns.emplace_back(std::move(d),
                                encode_scalar_groups(data.data(), natoms * width, nframes, opts.gof_size, prec, level));
    }
    for (const auto& [name, data] : batch.int_columns) {
        const std::size_t width = width_of(data.size(), natoms, nframes);
        ColumnDesc d = ColumnDesc::int_column(name);
        d.width = static_cast<std::uint32_t>(width);
        ct.columns.emplace_back(std::move(d), encode_plain_groups(kTagInt32, data.data(), natoms * width, nframes,
                                                                  opts.gof_size, level));
    }

    // Per-frame columns get one blob per group, like the per-atom ones, so
    // that every group is self-contained.
    if (batch.box)
        ct.columns.emplace_back(ColumnDesc::frame_vector("box", 9, "nm"),
                                encode_frame_groups(kTagRawF64, flatten(*batch.box), 9, ct.gofs, level));
    if (batch.origin)
        ct.columns.emplace_back(ColumnDesc::frame_vector("origin", 3, "nm"),
                                encode_frame_groups(kTagRawF64, flatten(*batch.origin), 3, ct.gofs, level));
    for (const auto& [name, v] : batch.frame_props) {
        const char* unit = name == "time" ? "ps" : name == "energy" ? "kJ/mol" : "";
        ct.columns.emplace_back(ColumnDesc::frame_scalar(name, unit),
                                encode_frame_groups(kTagRawF64, v, 1, ct.gofs, level));
    }
    for (const auto& [name, v] : batch.int_frame_props) {
        ColumnDesc d = ColumnDesc::frame_scalar(name, "");
        d.dtype = Dtype::I64;
        ct.columns.emplace_back(std::move(d), encode_frame_groups(kTagInt64, v, 1, ct.gofs, level));
    }

    Metadata meta = metadata_of(batch, opts);
    if (src->bodies) meta.bodies = *src->bodies;
    if (!meta.empty()) ct.metadata = meta.encode();
    for (auto& c : ct.columns) {
        std::string_view label = unit_label(batch, c.desc.name);
        if (!label.empty()) c.desc.unit = std::string(label);
    }
    return ct;
}

CompressedTrajectory compress_positions(const float* positions, std::size_t natoms, std::size_t nframes,
                                        const std::vector<Box>* boxes, const CompressOptions& user_opts) {
    const CompressOptions opts = user_opts.resolved();
    opts.validate();
    if (boxes && boxes->size() > nframes) invalid("more box matrices than frames");
    const std::size_t n = natoms * nframes * 3;
    if (opts.lossless) {
        for (std::size_t i = 0; i < n; ++i)
            if (!std::isfinite(positions[i]))
                invalid("column 'positions': value at index " + std::to_string(i) +
                        " is non-finite; mdz cannot store non-finite (NaN/Inf) values");
    } else {
        try {
            detail::validate_coords(positions, n, opts.precision_for("positions"));
        } catch (const Error& e) {
            invalid(std::string("column 'positions': ") + e.what());
        }
    }
    CompressedTrajectory ct;
    ct.natoms = natoms;
    ct.nframes = nframes;
    ct.gof_size = opts.gof_size;
    ct.gofs = GofTable::uniform(nframes, opts.gof_size);
    const float prec = opts.precision_for("positions");
    ct.columns.emplace_back(describe_xyz(ColumnDesc::positions(0.0f), prec, opts),
                            encode_xyz_groups(positions, natoms, nframes, prec, opts));
    if (boxes && !boxes->empty())
        ct.columns.emplace_back(ColumnDesc::frame_vector("box", 9, "nm"),
                                encode_frame_groups(kTagRawF64, flatten(*boxes), 9, ct.gofs, opts.zstd_level));
    if (opts.source_format) {
        Metadata m;
        m.source_format = opts.source_format;
        ct.metadata = m.encode();
    }
    return ct;
}

// --- decode ---

namespace detail {

// Decode group gi of a per-atom F32 column into out (frames * natoms * width
// floats). Only frames [lo, hi) are certain to be decoded, and the returned
// span gives the frames that are valid. Scalar columns always decode the
// whole group. raw and tmp are scratch buffers.
DecodedSpan decode_f32_group(const CompressedColumn& col, std::size_t gi, std::size_t frames, std::size_t natoms,
                             Bytes& raw, std::vector<float>& tmp, float* out, std::size_t lo, std::size_t hi,
                             const RigidModel* rigid) {
    const std::size_t width = col.desc.width;
    const std::size_t values = sat_mul(sat_mul(frames, natoms), width);
    std::size_t cap = sat_add(9, sat_mul(values, 4));
    if (width == 3) cap = std::max(cap, f32x3_gof_cap(frames, natoms));
    zstd_decompress_bounded(col.blob(gi), cap, raw);
    if (width == 3 && is_f32x3_payload(raw))
        return decode_f32x3_payload(raw, natoms, col.desc.precision, frames, out, lo, hi, rigid);
    tmp.clear();
    decode_scalar_f32_payload(raw, {values, true}, tmp);
    if (values) std::memcpy(out, tmp.data(), values * sizeof(float));
    return {0, frames};
}

// Upper bound on the number of values a decompressed payload of `bytes`
// bytes can decode to. The intra codec spends at least one bit per atom
// (three values), and every other encoding at least one byte per value.
inline std::size_t max_values_for(std::size_t bytes) noexcept { return sat_mul(bytes, 24); }

}  // namespace detail

namespace {

DecodedColumn empty_of(Dtype dt, bool per_frame) {
    DecodedColumn d;
    if (per_frame)
        d.dtype = dt == Dtype::U32 ? Dtype::I32 : (dt == Dtype::F32 || dt == Dtype::F64 || dt == Dtype::I32 || dt == Dtype::I64 ? dt : Dtype::F32);
    else
        d.dtype = (dt == Dtype::I32 || dt == Dtype::U32) ? Dtype::I32 : Dtype::F32;
    return d;
}

DecodedColumn decode_per_frame(const CompressedColumn& col, std::size_t start, std::size_t end, const GofTable& gofs) {
    const ColumnDesc& d = col.desc;
    const std::size_t width = d.width;
    DecodedColumn out = empty_of(d.dtype, true);
    if (gofs.empty() && col.n_gofs() == 0) return out;
    if (col.n_gofs() != gofs.size())
        corrupt("per_frame column '" + d.name + "' has " + std::to_string(col.n_gofs()) + " blobs for " +
                std::to_string(gofs.size()) + " groups");
    const auto groups = gofs.overlapping(start, end);
    if (groups.size() == 0) return out;
    const std::size_t first = gofs.span(groups.start).start;
    Bytes raw;
    for (std::size_t gi = groups.start; gi < groups.end; ++gi) {
        const Count count{sat_mul(gofs.span(gi).size(), width), false};
        ByteSpan blob = col.blob(gi);
        switch (d.dtype) {
            case Dtype::F64:
                decompress_tagged(blob, 0, 8, count, raw);
                read_values(tagged_body(raw, kTagRawF64, 0, 8, count, "raw_f64"), out.f64);
                break;
            case Dtype::I32:
            case Dtype::U32:
                decompress_tagged(blob, 0, 4, count, raw);
                read_values(tagged_body(raw, kTagInt32, 0, 4, count, "int32"), out.i32);
                break;
            case Dtype::I64:
                decompress_tagged(blob, 0, 8, count, raw);
                read_values(tagged_body(raw, kTagInt64, 0, 8, count, "int64"), out.i64);
                break;
            case Dtype::F32:
                decompress_tagged(blob, 4, 4, count, raw);
                decode_scalar_f32_payload(raw, count, out.f32);
                break;
            default:
                corrupt("unsupported per_frame dtype " + std::string(dtype_name(d.dtype)));
        }
    }
    // Some writers emit short groups. Clamp to the frames actually present.
    auto clamp = [&](auto& v) {
        const std::size_t hi = std::min(v.size(), (end - first) * width);
        const std::size_t lo = std::min(hi, (start - first) * width);
        v.erase(v.begin() + static_cast<std::ptrdiff_t>(hi), v.end());
        v.erase(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(lo));
    };
    clamp(out.f32);
    clamp(out.f64);
    clamp(out.i32);
    clamp(out.i64);
    return out;
}

}  // namespace

DecodedColumn decode_column(const CompressedTrajectory& ct, const CompressedColumn& col, std::size_t start,
                            std::size_t end) {
    const ColumnDesc& d = col.desc;
    const GofTable& gofs = ct.gofs;
    start = std::min(start, gofs.nframes());
    end = std::min(end, gofs.nframes());
    if (d.per_frame) return decode_per_frame(col, start, std::max(start, end), gofs);
    if (ct.variable_atoms) {
        // Values for the atoms present in each frame, concatenated frame by
        // frame.
        if (d.name == kMembershipColumn) throw Error(ErrorKind::InvalidInput, "the membership column is not data");
        FrameBatch b = detail::decompress_variable(ct, start, std::max(start, end), &d.name);
        DecodedColumn out = empty_of(d.dtype, false);
        if (auto it = b.columns.find(d.name); it != b.columns.end()) out.f32 = std::move(it->second);
        if (auto it = b.int_columns.find(d.name); it != b.int_columns.end()) out.i32 = std::move(it->second);
        return out;
    }

    const std::size_t natoms = ct.natoms;
    const std::size_t per_frame = detail::checked_mul(natoms, d.width, "column too large");
    DecodedColumn out = empty_of(d.dtype, false);
    const auto groups = gofs.overlapping(start, end);
    if (groups.size() == 0) return out;
    if (col.n_gofs() != gofs.size())
        corrupt("column '" + d.name + "' has " + std::to_string(col.n_gofs()) + " groups, expected " +
                std::to_string(gofs.size()));

    if (d.dtype == Dtype::I32 || d.dtype == Dtype::U32) {
        Bytes raw;
        std::vector<std::int32_t> vals;
        for (std::size_t gi = groups.start; gi < groups.end; ++gi) {
            const auto s = gofs.span(gi);
            const Count count{sat_mul(s.size(), per_frame), true};
            decompress_tagged(col.blob(gi), 0, 4, count, raw);
            vals.clear();
            read_values(tagged_body(raw, kTagInt32, 0, 4, count, "int32"), vals);
            const std::size_t lo = (std::max(start, s.start) - s.start) * per_frame;
            const std::size_t hi = (std::min(end, s.end) - s.start) * per_frame;
            out.i32.insert(out.i32.end(), vals.begin() + static_cast<std::ptrdiff_t>(lo),
                           vals.begin() + static_cast<std::ptrdiff_t>(hi));
        }
        return out;
    }
    if (d.dtype != Dtype::F32)
        corrupt("unsupported column dtype/width " + std::string(dtype_name(d.dtype)) + " / " + std::to_string(d.width));

    // Groups decode independently, so work through them in batches. Size the
    // output for each batch, checking the size against the compressed bytes
    // present rather than trusting header counts alone, then decode the batch
    // in parallel directly into place.
    const std::size_t batch = std::max<std::size_t>(4 * num_threads(), 1);
    for (std::size_t g0 = groups.start; g0 < groups.end; g0 += batch) {
        const std::size_t g1 = std::min(g0 + batch, groups.end), nb = g1 - g0;
        std::vector<std::size_t> offs(nb);
        std::size_t total = out.f32.size();
        for (std::size_t k = 0; k < nb; ++k) {
            const auto s = gofs.span(g0 + k);
            offs[k] = total;
            total = detail::checked_add(
                total, detail::checked_mul(std::min(end, s.end) - std::max(start, s.start), per_frame, "column too large"),
                "column too large");
        }
        // Bound the allocation by the compressed bytes present. A blob can
        // expand at most 32768x, and each value costs at least 1/24 byte.
        std::size_t compressed = 0;
        for (std::size_t k = 0; k < nb; ++k) compressed = sat_add(compressed, col.blob(g0 + k).size());
        if (total - out.f32.size() > detail::max_values_for(sat_mul(compressed, 32 * 1024)))
            corrupt("column '" + d.name + "' claims more frames than its data holds");
        out.f32.resize(total);
        float* base = out.f32.data();
        detail::parallel_for(nb, [&](std::size_t k) {
            const std::size_t gi = g0 + k;
            const auto s = gofs.span(gi);
            thread_local Bytes raw;
            thread_local std::vector<float> tmp, whole;
            const std::size_t lo = std::max(start, s.start), hi = std::min(end, s.end);
            if (lo == s.start && hi == s.end) {
                detail::decode_f32_group(col, gi, s.size(), natoms, raw, tmp, base + offs[k], 0, s.size(),
                                         ct.rigid.get());
            } else {
                whole.resize(s.size() * per_frame);
                detail::decode_f32_group(col, gi, s.size(), natoms, raw, tmp, whole.data(), lo - s.start,
                                         hi - s.start, ct.rigid.get());
                if (per_frame)
                    std::memcpy(base + offs[k], whole.data() + (lo - s.start) * per_frame,
                                (hi - lo) * per_frame * sizeof(float));
            }
        });
    }
    return out;
}

FrameBatch decompress(const CompressedTrajectory& ct, std::size_t start, std::size_t end) {
    start = std::min(start, ct.nframes);
    end = std::max(start, std::min(end, ct.nframes));
    if (ct.variable_atoms) return detail::decompress_variable(ct, start, end, nullptr);
    FrameBatch b(ct.natoms, end - start);
    for (const auto& col : ct.columns) {
        DecodedColumn v = decode_column(ct, col, start, end);
        const ColumnDesc& d = col.desc;
        if (d.per_frame) {
            if (d.name == "box" && d.width == 9 && v.dtype == Dtype::F64) {
                b.box.emplace(v.f64.size() / 9);
                if (!b.box->empty()) std::memcpy(b.box->data(), v.f64.data(), b.box->size() * 9 * sizeof(double));
            } else if (d.name == "origin" && d.width == 3 && v.dtype == Dtype::F64) {
                b.origin.emplace(v.f64.size() / 3);
                if (!b.origin->empty()) std::memcpy(b.origin->data(), v.f64.data(), b.origin->size() * 3 * sizeof(double));
            } else if (v.dtype == Dtype::F64) {
                b.frame_props[d.name] = std::move(v.f64);
            } else if (v.dtype == Dtype::I32) {
                b.int_frame_props[d.name] = std::vector<std::int64_t>(v.i32.begin(), v.i32.end());
            } else if (v.dtype == Dtype::I64) {
                b.int_frame_props[d.name] = std::move(v.i64);
            }
        } else if (v.dtype == Dtype::F32) {
            b.columns[d.name] = std::move(v.f32);
        } else if (v.dtype == Dtype::I32) {
            b.int_columns[d.name] = std::move(v.i32);
        }
    }
    if (ct.metadata && !ct.metadata->empty()) {
        Metadata m = Metadata::decode(*ct.metadata);
        auto sh = [](auto& opt) {
            using V = std::decay_t<decltype(*opt)>;
            return opt ? std::make_shared<const V>(std::move(*opt)) : nullptr;
        };
        b.atom_names = sh(m.atom_names);
        b.elements = sh(m.elements);
        b.atom_types = sh(m.atom_types);
        b.atom_ids = sh(m.atom_ids);
        b.residue_names = sh(m.residues);
        b.residue_ids = sh(m.residue_ids);
        b.chain_ids = sh(m.chain_ids);
        b.bodies = sh(m.bodies);
        if (m.hetatm) {
            std::vector<std::uint8_t> h(m.hetatm->size());
            for (std::size_t i = 0; i < h.size(); ++i) h[i] = (*m.hetatm)[i] != 0;
            b.hetatm = share(std::move(h));
        }
        b.attributes = std::move(m.attributes);
        auto it = b.attributes.find(kUnitsAttr);
        if (it != b.attributes.end()) {
            b.units = units_of(it->second);
            b.attributes.erase(it);
        }
    }
    return b;
}

std::vector<float> decompress_positions(const CompressedTrajectory& ct) {
    const CompressedColumn* col = ct.column("positions");
    if (!col) throw Error(ErrorKind::Corrupt, "trajectory has no positions column");
    DecodedColumn v = decode_column(ct, *col);
    if (v.dtype != Dtype::F32) throw Error(ErrorKind::Corrupt, "positions column did not decode as f32");
    to_nm(v.f32, ct.length_to_nm());
    return std::move(v.f32);
}

}  // namespace mdz
