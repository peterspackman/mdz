// Rigid bodies as a compression model for coordinates.
//
// The atoms of a rigid body keep their positions relative to each other, so
// each frame of a body can be described by its centre and orientation. That
// is six numbers per body, rather than three per atom. A group coded this way
// holds:
//
//   - A reference geometry per body: its atoms relative to its centre in the
//     group's first frame.
//   - The free atoms, the body centres, and the body orientations, each coded
//     as a separate stream with the ordinary group codecs. An orientation is
//     stored as the vector part of a unit quaternion, scaled by the largest
//     reference radius so that it is a length. Centres and orientations are
//     kept apart because the intra codec sizes a frame's integers by the
//     frame's overall range. Centres span the box, but orientations only span
//     a body's radius.
//   - The residual of every body atom against the prediction rebuilt from the
//     decoded centres and orientations (not the original ones).
//
// The decoder rebuilds the prediction from the same decoded values with the
// same arithmetic. That arithmetic uses only correctly rounded IEEE
// operations and sqrt, so it is bit-reproducible on any target. A body atom's
// error is therefore exactly the coding error of its residual. The column's
// error bound holds whatever the bodies do, and a body that is not quite
// rigid costs extra bits but never accuracy.
//
// Let E be the column's error bound: 4x its step with adaptive precision, or
// half the step without. Centres are coded at a fixed step of E/2,
// orientations at E/4, and the reference geometry at E/8. The prediction is
// then within 0.8 E, and almost every residual rounds to zero. The residual
// corrects whatever error remains, so the bound always holds.
//
// Payload body, after the header bytes [3][0][3], little-endian:
//   u16  nframes
//   f32  rscale, ref_step, centre_step, orient_step
//   then five parts, each prefixed by its u32 byte length:
//     1. the M x 3 quantized reference coordinates, packed as one
//        temporal-codec residual frame without spatial delta
//     2. the free atoms' payload (F atoms)
//     3. the centres' payload (B "atoms")
//     4. the orientations' payload (B "atoms")
//     5. the residuals' payload (M atoms)
// Parts 2 to 5 are complete coordinate-codec payloads, header bytes included.
// F is the number of free atoms, B the number of bodies and M the number of
// body atoms.

#include "rigid.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <string>

#include "kernels.hpp"
#include "mdz/error.hpp"

namespace mdz::detail {
namespace {

// The row-major rotation matrix of the unit quaternion whose vector part is
// (x, y, z) and whose scalar part is the non-negative root. The encoder and
// decoder both use this, so its arithmetic must not change.
inline void rotation(float x, float y, float z, float m[9]) noexcept {
    const float s = 1.0f - x * x - y * y - z * z;
    const float w = s > 0.0f ? std::sqrt(s) : 0.0f;
    m[0] = 1.0f - 2.0f * (y * y + z * z);
    m[1] = 2.0f * (x * y - w * z);
    m[2] = 2.0f * (x * z + w * y);
    m[3] = 2.0f * (x * y + w * z);
    m[4] = 1.0f - 2.0f * (x * x + z * z);
    m[5] = 2.0f * (y * z - w * x);
    m[6] = 2.0f * (x * z - w * y);
    m[7] = 2.0f * (y * z + w * x);
    m[8] = 1.0f - 2.0f * (x * x + y * y);
}

inline void apply(const float m[9], const float c[3], const float r[3], float out[3]) noexcept {
    out[0] = c[0] + (m[0] * r[0] + m[1] * r[1] + m[2] * r[2]);
    out[1] = c[1] + (m[3] * r[0] + m[4] * r[1] + m[5] * r[2]);
    out[2] = c[2] + (m[6] * r[0] + m[7] * r[1] + m[8] * r[2]);
}

// Three atoms that span a body well, as indices into its members. They are
// the first atom, the atom farthest from it, and the atom farthest from the
// line through those two. Chosen once per group from the reference geometry.
struct Triad {
    std::size_t a = 0, b = 0, c = 0;
};

Triad pick_triad(const float* ref, std::size_t cnt) {
    Triad t;
    auto d2 = [&](std::size_t i, std::size_t j) {
        double s = 0;
        for (int d = 0; d < 3; ++d)
            s += double(ref[3 * i + d] - ref[3 * j + d]) * double(ref[3 * i + d] - ref[3 * j + d]);
        return s;
    };
    for (std::size_t i = 1; i < cnt; ++i)
        if (d2(0, i) > d2(0, t.b)) t.b = i;
    double best = -1;
    for (std::size_t i = 1; i < cnt; ++i) {
        // Squared distance from the line through a and b.
        double u[3], v[3];
        for (int d = 0; d < 3; ++d) {
            u[d] = double(ref[3 * t.b + d]) - ref[d];
            v[d] = double(ref[3 * i + d]) - ref[d];
        }
        const double cx = u[1] * v[2] - u[2] * v[1], cy = u[2] * v[0] - u[0] * v[2], cz = u[0] * v[1] - u[1] * v[0];
        const double h = cx * cx + cy * cy + cz * cz;
        if (h > best) {
            best = h;
            t.c = i;
        }
    }
    return t;
}

// An orthonormal basis e[0..2] from three points, with e0 along p1 - p0 and e1
// in the plane of all three. If the third point lies on that line, as in a
// two-atom or linear body, e1 is an arbitrary but fixed perpendicular. That
// is safe because rotation about the axis does not move such a body's atoms.
void basis(const double p0[3], const double p1[3], const double p2[3], double e[3][3]) {
    double u[3], v[3];
    for (int d = 0; d < 3; ++d) {
        u[d] = p1[d] - p0[d];
        v[d] = p2[d] - p0[d];
    }
    double n = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    if (!(n > 0)) {
        u[0] = 1;
        u[1] = u[2] = 0;
        n = 1;
    }
    for (int d = 0; d < 3; ++d) e[0][d] = u[d] / n;
    double dot = e[0][0] * v[0] + e[0][1] * v[1] + e[0][2] * v[2];
    for (int d = 0; d < 3; ++d) v[d] -= dot * e[0][d];
    double m = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (!(m > 1e-9 * n)) {
        // Take the coordinate axis least aligned with e0 and orthogonalize it.
        int k = 0;
        for (int d = 1; d < 3; ++d)
            if (std::fabs(e[0][d]) < std::fabs(e[0][k])) k = d;
        v[0] = v[1] = v[2] = 0;
        v[k] = 1;
        dot = e[0][k];
        for (int d = 0; d < 3; ++d) v[d] -= dot * e[0][d];
        m = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }
    for (int d = 0; d < 3; ++d) e[1][d] = v[d] / m;
    e[2][0] = e[0][1] * e[1][2] - e[0][2] * e[1][1];
    e[2][1] = e[0][2] * e[1][0] - e[0][0] * e[1][2];
    e[2][2] = e[0][0] * e[1][1] - e[0][1] * e[1][0];
}

// The unit quaternion (with w >= 0) of the rotation taking basis r onto basis
// f. The rotation is R = sum_k f_k r_k^T, converted with Shepperd's method.
void quaternion_between(const double r[3][3], const double f[3][3], double q[4]) {
    double m[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) m[i][j] = f[0][i] * r[0][j] + f[1][i] * r[1][j] + f[2][i] * r[2][j];
    const double tr = m[0][0] + m[1][1] + m[2][2];
    if (tr > 0) {
        const double s = std::sqrt(tr + 1.0) * 2;
        q[0] = 0.25 * s;
        q[1] = (m[2][1] - m[1][2]) / s;
        q[2] = (m[0][2] - m[2][0]) / s;
        q[3] = (m[1][0] - m[0][1]) / s;
    } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        const double s = std::sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2;
        q[0] = (m[2][1] - m[1][2]) / s;
        q[1] = 0.25 * s;
        q[2] = (m[0][1] + m[1][0]) / s;
        q[3] = (m[0][2] + m[2][0]) / s;
    } else if (m[1][1] > m[2][2]) {
        const double s = std::sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2;
        q[0] = (m[0][2] - m[2][0]) / s;
        q[1] = (m[0][1] + m[1][0]) / s;
        q[2] = 0.25 * s;
        q[3] = (m[1][2] + m[2][1]) / s;
    } else {
        const double s = std::sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2;
        q[0] = (m[1][0] - m[0][1]) / s;
        q[1] = (m[0][2] + m[2][0]) / s;
        q[2] = (m[1][2] + m[2][1]) / s;
        q[3] = 0.25 * s;
    }
    const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    for (int k = 0; k < 4; ++k) q[k] /= n;
    if (q[0] < 0)
        for (int k = 0; k < 4; ++k) q[k] = -q[k];
}

}  // namespace

std::shared_ptr<const RigidModel> RigidModel::build(const std::vector<std::int32_t>& body_of_atom, std::size_t natoms) {
    if (body_of_atom.size() != natoms)
        throw Error(ErrorKind::InvalidInput, "rigid body index has " + std::to_string(body_of_atom.size()) +
                                                 " entries for " + std::to_string(natoms) + " atoms");
    auto m = std::make_shared<RigidModel>();
    m->natoms = natoms;
    // Bodies are stored in index order, with atoms ascending within each.
    // Single-atom bodies become free atoms.
    std::vector<std::uint32_t> order(natoms);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(),
                     [&](std::uint32_t a, std::uint32_t b) { return body_of_atom[a] < body_of_atom[b]; });
    std::vector<bool> is_free(natoms, false);
    for (std::size_t i = 0; i < natoms;) {
        const std::int32_t b = body_of_atom[order[i]];
        std::size_t j = i;
        while (j < natoms && body_of_atom[order[j]] == b) ++j;
        if (b < 0 || j - i < 2) {
            for (std::size_t k = i; k < j; ++k) is_free[order[k]] = true;
        } else {
            m->body_start.push_back(static_cast<std::uint32_t>(m->members.size()));
            m->members.insert(m->members.end(), order.begin() + static_cast<std::ptrdiff_t>(i),
                              order.begin() + static_cast<std::ptrdiff_t>(j));
        }
        i = j;
    }
    if (!m->body_start.empty()) m->body_start.push_back(static_cast<std::uint32_t>(m->members.size()));
    for (std::size_t a = 0; a < natoms; ++a)
        if (is_free[a]) m->free_atoms.push_back(static_cast<std::uint32_t>(a));
    return m;
}

void encode_rigid_payload(const float* frames, std::size_t n, std::size_t natoms, const RigidModel& model,
                          const GofParams& p, Bytes& payload) {
    const std::size_t F = model.free_atoms.size(), B = model.nbodies(), M = model.members.size();
    const std::size_t n3 = natoms * 3;
    const float prec = p.precision;
    const float bound = p.adaptive ? 4.0f * prec : 0.5f * prec;
    const float ref_step = bound / 8.0f, centre_step = bound / 2.0f, orient_step = bound / 4.0f;
    auto atom = [&](std::size_t f, std::size_t a) { return frames + f * n3 + a * 3; };

    // Each body's centre in frame f, rounded to float to match the coded value.
    std::vector<std::array<double, 3>> centre(B);
    auto centres = [&](std::size_t f) {
        for (std::size_t b = 0; b < B; ++b) {
            double c[3] = {0, 0, 0};
            for (std::size_t k = model.body_start[b]; k < model.body_start[b + 1]; ++k)
                for (int d = 0; d < 3; ++d) c[d] += atom(f, model.members[k])[d];
            const double cnt = double(model.body_start[b + 1] - model.body_start[b]);
            for (int d = 0; d < 3; ++d) centre[b][d] = double(float(c[d] / cnt));
        }
    };

    // The reference geometry is the first frame relative to each body's
    // centre. It is quantized so that the decoder holds exactly the same
    // numbers.
    std::vector<std::int32_t> refq(M * 3);
    std::vector<float> ref(M * 3);
    float rscale = 0.0f;
    centres(0);
    const float inv_ref = 1.0f / ref_step;
    for (std::size_t b = 0; b < B; ++b)
        for (std::size_t k = model.body_start[b]; k < model.body_start[b + 1]; ++k) {
            float r2 = 0;
            for (int d = 0; d < 3; ++d) {
                const float rel = atom(0, model.members[k])[d] - float(centre[b][d]);
                refq[3 * k + d] = round_to_i32(rel * inv_ref);
                ref[3 * k + d] = static_cast<float>(refq[3 * k + d]) * ref_step;
                r2 += ref[3 * k + d] * ref[3 * k + d];
            }
            rscale = std::max(rscale, std::sqrt(r2));
        }
    if (!(rscale > 0.0f)) rscale = 1.0f;

    // Orientations come from a basis built from three atoms per body, the same
    // way in the reference and in each frame. This is exact for a rigid body
    // and much cheaper than a least-squares fit. The difference would only
    // show up in the residuals.
    std::vector<Triad> triad(B);
    std::vector<std::array<std::array<double, 3>, 3>> ref_basis(B);
    for (std::size_t b = 0; b < B; ++b) {
        const std::size_t s = model.body_start[b], cnt = model.body_start[b + 1] - s;
        triad[b] = pick_triad(ref.data() + 3 * s, cnt);
        double p0[3], p1[3], p2[3];
        for (int d = 0; d < 3; ++d) {
            p0[d] = ref[3 * (s + triad[b].a) + d];
            p1[d] = ref[3 * (s + triad[b].b) + d];
            p2[d] = ref[3 * (s + triad[b].c) + d];
        }
        double e[3][3];
        basis(p0, p1, p2, e);
        for (int k = 0; k < 3; ++k)
            for (int d = 0; d < 3; ++d) ref_basis[b][k][d] = e[k][d];
    }

    // Free atoms, centres and orientations.
    std::vector<float> Fr(n * F * 3), C(n * B * 3), O(n * B * 3);
    std::vector<std::array<double, 4>> q(B, {1, 0, 0, 0});
    for (std::size_t f = 0; f < n; ++f) {
        for (std::size_t i = 0; i < F; ++i)
            std::copy(atom(f, model.free_atoms[i]), atom(f, model.free_atoms[i]) + 3, Fr.data() + (f * F + i) * 3);
        centres(f);
        for (std::size_t b = 0; b < B; ++b) {
            const std::size_t s = model.body_start[b];
            double p0[3], p1[3], p2[3], e[3][3], r[3][3];
            for (int d = 0; d < 3; ++d) {
                p0[d] = atom(f, model.members[s + triad[b].a])[d];
                p1[d] = atom(f, model.members[s + triad[b].b])[d];
                p2[d] = atom(f, model.members[s + triad[b].c])[d];
            }
            basis(p0, p1, p2, e);
            for (int k = 0; k < 3; ++k)
                for (int d = 0; d < 3; ++d) r[k][d] = ref_basis[b][k][d];
            quaternion_between(r, e, q[b].data());
            for (int d = 0; d < 3; ++d) {
                C[(f * B + b) * 3 + d] = float(centre[b][d]);
                O[(f * B + b) * 3 + d] = float(q[b][d + 1]) * rscale;
            }
        }
    }
    GofParams free_p = p, centre_p = p, orient_p = p;
    free_p.rigid = centre_p.rigid = orient_p.rigid = nullptr;
    centre_p.adaptive = orient_p.adaptive = false;
    centre_p.precision = centre_step;
    orient_p.precision = orient_step;
    Bytes fpay, cpay, opay;
    encode_gof_payload(Fr.data(), n, F, free_p, fpay);
    encode_gof_payload(C.data(), n, B, centre_p, cpay);
    encode_gof_payload(O.data(), n, B, orient_p, opay);

    // Residuals against what the decoder will predict from the decoded
    // centres and orientations.
    std::vector<float> Cd(C.size()), Od(O.size());
    decode_f32x3_payload(cpay, B, centre_step, n, Cd.data());
    decode_f32x3_payload(opay, B, orient_step, n, Od.data());
    std::vector<float> R(n * M * 3);
    for (std::size_t f = 0; f < n; ++f)
        for (std::size_t b = 0; b < B; ++b) {
            float m[9];
            const float* qv = Od.data() + (f * B + b) * 3;
            rotation(qv[0] / rscale, qv[1] / rscale, qv[2] / rscale, m);
            for (std::size_t k = model.body_start[b]; k < model.body_start[b + 1]; ++k) {
                float pred[3];
                apply(m, Cd.data() + (f * B + b) * 3, ref.data() + 3 * k, pred);
                const float* x = atom(f, model.members[k]);
                for (int d = 0; d < 3; ++d) R[(f * M + k) * 3 + d] = x[d] - pred[d];
            }
        }
    // Residuals are mostly zero and otherwise noise, so there is nothing to
    // predict from frame to frame. The temporal codec with a zero-order
    // predictor stores them as plain quantized values, which is compact and
    // the fastest option to decode.
    Bytes rpay(3);
    encode_gof_temporal(R.data(), n, M, prec, 0, p.adaptive, 1, p.spatial_delta, rpay);
    rpay[0] = 3;
    rpay[1] = 0;
    rpay[2] = static_cast<std::uint8_t>(CodecMode::Temporal);

    Bytes refb(M * 15);
    refb.resize(pack_residual_frame(refq.data(), M, false, refb.data()));
    payload.clear();
    payload.push_back(3);
    payload.push_back(0);
    payload.push_back(static_cast<std::uint8_t>(kRigidMode));
    put_le<std::uint16_t>(payload, static_cast<std::uint16_t>(n));
    for (float v : {rscale, ref_step, centre_step, orient_step}) put_le<float>(payload, v);
    for (const Bytes* part : {&refb, &fpay, &cpay, &opay, &rpay}) {
        put_le<std::uint32_t>(payload, static_cast<std::uint32_t>(part->size()));
        put_bytes(payload, part->data(), part->size());
    }
}

DecodedSpan decode_rigid_payload(ByteSpan body, std::size_t natoms, float precision, std::size_t expected,
                                 const RigidModel* model, float* out, std::size_t lo, std::size_t hi) {
    if (!model) corrupt("a rigid-body group in a file that defines no rigid bodies");
    if (model->natoms != natoms) corrupt("rigid bodies are defined for a different atom count");
    const std::size_t F = model->free_atoms.size(), B = model->nbodies(), M = model->members.size();
    const std::size_t n3 = natoms * 3;
    ByteCursor c(body);
    const std::size_t n = c.u16();
    if (n != expected) corrupt("GoF holds " + std::to_string(n) + " frames, expected " + std::to_string(expected));
    float v[4];
    for (float& x : v) {
        x = c.f32();
        if (!(std::isfinite(x) && x > 0.0f)) corrupt("bad rigid-body scale or step");
    }
    const float rscale = v[0], ref_step = v[1], centre_step = v[2], orient_step = v[3];
    ByteSpan refb = c.take(c.u32());
    ByteSpan fpay = c.take(c.u32()), cpay = c.take(c.u32()), opay = c.take(c.u32()), rpay = c.take(c.u32());
    if (c.remaining() != 0) corrupt("rigid-body group has trailing bytes");
    std::vector<std::int32_t> refq(M * 3);
    if (M && read_residual_frame(refb.data(), refb.size(), M, false, refq.data()) != refb.size())
        corrupt("rigid-body reference geometry is malformed");
    std::vector<float> ref(M * 3);
    for (std::size_t i = 0; i < ref.size(); ++i) ref[i] = static_cast<float>(refq[i]) * ref_step;
    // Each inner payload decodes into a buffer no larger than `out`, since
    // every body has at least two atoms and so F, B and M are all <= natoms.
    // Only the requested frames are reconstructed. Each part decodes at least
    // those frames, as described for decode_f32x3_payload.
    hi = std::min(hi, n);
    lo = std::min(lo, hi);
    std::vector<float> Fr(n * F * 3), C(n * B * 3), O(n * B * 3), R(n * M * 3);
    decode_f32x3_payload(fpay, F, precision, n, Fr.data(), lo, hi);
    decode_f32x3_payload(cpay, B, centre_step, n, C.data(), lo, hi);
    decode_f32x3_payload(opay, B, orient_step, n, O.data(), lo, hi);
    decode_f32x3_payload(rpay, M, precision, n, R.data(), lo, hi);
    for (std::size_t f = lo; f < hi; ++f) {
        float* o = out + f * n3;
        for (std::size_t i = 0; i < F; ++i)
            std::copy(Fr.data() + (f * F + i) * 3, Fr.data() + (f * F + i) * 3 + 3, o + 3 * model->free_atoms[i]);
        for (std::size_t b = 0; b < B; ++b) {
            float m[9];
            const float* qv = O.data() + (f * B + b) * 3;
            rotation(qv[0] / rscale, qv[1] / rscale, qv[2] / rscale, m);
            for (std::size_t k = model->body_start[b]; k < model->body_start[b + 1]; ++k) {
                float pred[3];
                apply(m, C.data() + (f * B + b) * 3, ref.data() + 3 * k, pred);
                const float* r = R.data() + (f * M + k) * 3;
                float* x = o + 3 * model->members[k];
                for (int d = 0; d < 3; ++d) x[d] = pred[d] + r[d];
            }
        }
    }
    return {lo, hi};
}

std::vector<std::int32_t> detect_rigid_bodies(const FrameBatch& b, double bond, double tolerance,
                                              std::size_t max_body) {
    const std::size_t natoms = b.natoms, nframes = b.nframes;
    std::vector<std::int32_t> body(natoms, -1);
    const std::vector<float>* pos = b.positions();
    if (!pos || nframes == 0 || natoms < 2 || b.variable()) return body;
    // bond and tolerance are in nm, but positions may be in a LAMMPS unit
    // style.
    const double to_nm = b.units ? b.units->length : 1.0;
    bond /= to_nm;
    tolerance /= to_nm;

    // Minimum image for an orthorhombic box, if there is one.
    double L[3] = {0, 0, 0};
    if (b.box && !b.box->empty()) {
        const Box& m = (*b.box)[0];
        if (m[1] == 0 && m[2] == 0 && m[3] == 0 && m[5] == 0 && m[6] == 0 && m[7] == 0) {
            L[0] = m[0];
            L[1] = m[4];
            L[2] = m[8];
        }
    }
    auto dist2 = [&](std::size_t f, std::size_t i, std::size_t j) {
        const float* x = pos->data() + f * natoms * 3;
        double s = 0;
        for (int d = 0; d < 3; ++d) {
            double dx = double(x[3 * i + d]) - double(x[3 * j + d]);
            if (L[d] > 0) dx -= L[d] * std::nearbyint(dx / L[d]);
            s += dx * dx;
        }
        return s;
    };

    // Join atoms closer than `bond` in frame 0, using a cell list and
    // union-find.
    std::vector<std::uint32_t> parent(natoms);
    std::iota(parent.begin(), parent.end(), 0u);
    auto find = [&](std::uint32_t a) {
        while (parent[a] != a) a = parent[a] = parent[parent[a]];
        return a;
    };
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    for (std::size_t i = 0; i < natoms; ++i)
        for (int d = 0; d < 3; ++d) {
            lo[d] = std::min(lo[d], double((*pos)[3 * i + d]));
            hi[d] = std::max(hi[d], double((*pos)[3 * i + d]));
        }
    long nc[3];
    for (int d = 0; d < 3; ++d) nc[d] = std::max(1L, std::min(1L << 10, long((hi[d] - lo[d]) / bond)));
    auto cell_of = [&](std::size_t i, int d) {
        long c = long((double((*pos)[3 * i + d]) - lo[d]) / bond);
        return std::clamp(c, 0L, nc[d] - 1);
    };
    std::vector<std::vector<std::uint32_t>> cells(std::size_t(nc[0] * nc[1] * nc[2]));
    for (std::size_t i = 0; i < natoms; ++i)
        cells[std::size_t((cell_of(i, 0) * nc[1] + cell_of(i, 1)) * nc[2] + cell_of(i, 2))].push_back(std::uint32_t(i));
    const double bond2 = bond * bond;
    for (std::size_t i = 0; i < natoms; ++i) {
        const long ci[3] = {cell_of(i, 0), cell_of(i, 1), cell_of(i, 2)};
        for (long dx = -1; dx <= 1; ++dx)
            for (long dy = -1; dy <= 1; ++dy)
                for (long dz = -1; dz <= 1; ++dz) {
                    long c[3] = {ci[0] + dx, ci[1] + dy, ci[2] + dz};
                    bool ok = true;
                    for (int d = 0; d < 3; ++d) {
                        // Wrap across a periodic box. Without one, stop at the edge.
                        if (c[d] < 0 || c[d] >= nc[d]) {
                            if (L[d] > 0)
                                c[d] = (c[d] + nc[d]) % nc[d];
                            else
                                ok = false;
                        }
                    }
                    if (!ok) continue;
                    for (std::uint32_t j : cells[std::size_t((c[0] * nc[1] + c[1]) * nc[2] + c[2])])
                        if (j > i && dist2(0, i, j) < bond2) parent[find(std::uint32_t(i))] = find(j);
                }
    }
    std::vector<std::vector<std::uint32_t>> clusters(natoms);
    for (std::size_t i = 0; i < natoms; ++i) clusters[find(std::uint32_t(i))].push_back(std::uint32_t(i));

    // Keep the clusters that stay rigid in up to eight sampled frames.
    std::vector<std::size_t> sample;
    for (std::size_t k = 1; k <= 8 && k < nframes; ++k)
        sample.push_back(k * (nframes - 1) / std::min<std::size_t>(8, nframes - 1));
    std::int32_t next = 0;
    for (const auto& cl : clusters) {
        if (cl.size() < 2 || cl.size() > max_body) continue;
        bool rigid = true;
        for (std::size_t f : sample) {
            for (std::size_t a = 0; a < cl.size() && rigid; ++a)
                for (std::size_t c2 = a + 1; c2 < cl.size() && rigid; ++c2)
                    if (std::fabs(std::sqrt(dist2(f, cl[a], cl[c2])) - std::sqrt(dist2(0, cl[a], cl[c2]))) > tolerance)
                        rigid = false;
            if (!rigid) break;
        }
        if (!rigid) continue;
        for (std::uint32_t a : cl) body[a] = next;
        ++next;
    }
    return body;
}

}  // namespace mdz::detail
