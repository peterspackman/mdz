#include "codec.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "bytes.hpp"
#include "kernels.hpp"
#include "predict.hpp"
#include "rigid.hpp"
#include "xdr3d.hpp"
#include "zstd.hpp"

namespace mdz::detail {
namespace {

constexpr std::uint8_t kGofVersion = 3;
constexpr unsigned kNumLevels = 4;
constexpr float kMultipliers[kNumLevels] = {1.0f, 2.0f, 4.0f, 8.0f};

std::string fmt(float v) {
    if (std::isnan(v)) return "NaN";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.9g", static_cast<double>(v));
    return buf;
}

// Frame layout of a group. Anchors are every `anchor_step` frames, plus the
// last frame. Each remaining (B) frame is interpolated from the anchors on
// either side.
struct GofLayout {
    std::vector<std::size_t> anchors;
    std::vector<std::size_t> bframes;
    std::vector<std::pair<std::size_t, std::size_t>> brefs;
};

GofLayout build_gof_layout(std::size_t n, std::size_t anchor_step) {
    GofLayout L;
    if (anchor_step <= 1 || n <= 2) {
        L.anchors.resize(n);
        for (std::size_t i = 0; i < n; ++i) L.anchors[i] = i;
        return L;
    }
    for (std::size_t i = 0; i < n; i += anchor_step) L.anchors.push_back(i);
    if (L.anchors.back() != n - 1) L.anchors.push_back(n - 1);
    std::size_t ai = 0;
    for (std::size_t i = 0; i < n; ++i) {
        while (ai < L.anchors.size() && L.anchors[ai] < i) ++ai;
        if (ai < L.anchors.size() && L.anchors[ai] == i) continue;
        L.bframes.push_back(i);
        L.brefs.emplace_back(L.anchors[ai - 1], L.anchors[ai]);
    }
    return L;
}

// The last order+1 anchors, oldest first.
struct History {
    std::vector<std::size_t> idx;
    std::size_t order;
    explicit History(std::size_t o) : order(o) {}
    void push(std::size_t i) {
        idx.push_back(i);
        if (idx.size() > order + 1) idx.erase(idx.begin());
    }
};

void predict_from(const History& h, const float* base, std::size_t n3, std::vector<const float*>& ptrs, float* out) {
    ptrs.resize(h.idx.size());
    for (std::size_t k = 0; k < h.idx.size(); ++k) ptrs[k] = base + h.idx[k] * n3;
    predict_forward(ptrs.data(), ptrs.size(), h.order, n3, out);
}

void compute_precision_classes(const std::vector<float>& max_abs, float base, std::vector<std::uint8_t>& classes) {
    float inv[kNumLevels];
    for (unsigned k = 0; k < kNumLevels; ++k) inv[k] = 1.0f / (base * kMultipliers[k]);
    classes.resize(max_abs.size());
    for (std::size_t i = 0; i < max_abs.size(); ++i) {
        const float m = max_abs[i];
        std::uint8_t c = 0;
        // The coarsest level whose quantized magnitude fits 64 steps.
        for (unsigned k = kNumLevels - 1; k >= 1; --k) {
            if (m * inv[k] < 64.0f) {
                c = static_cast<std::uint8_t>(k);
                break;
            }
        }
        classes[i] = c;
    }
}

void build_precision_array(const std::uint8_t* classes, std::size_t n, float base, std::vector<float>& out) {
    out.resize(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = base * kMultipliers[classes[i]];
}

// Returns the largest magnitude in the group, and throws on NaN or infinity.
// With the sign bit cleared, the integer order of the bit patterns is the
// magnitude order, and every non-finite value sorts above every finite one.
// A single integer max, which vectorizes, therefore answers both questions.
float max_abs_finite(const float* v, std::size_t n) {
    std::uint32_t m = 0;
    for (std::size_t i = 0; i < n; ++i) m = std::max(m, float_bits(v[i]) & 0x7FFFFFFFu);
    if (m >= 0x7F800000u) throw Error(ErrorKind::InvalidInput, "mdz cannot store non-finite (NaN/Inf) values");
    return bits_float(m);
}

bool temporal_residuals_fit(float max_abs, float precision) {
    // The coefficients of the widest (cubic) predictor sum to 15 in magnitude,
    // so a prediction is at most 15x the largest coordinate and a residual at
    // most 16x. Adaptive steps are never finer than the base step.
    return static_cast<double>(max_abs) * 16.0 / static_cast<double>(precision) < static_cast<double>(INT32_MAX - 1);
}

[[noreturn]] void frame_count_mismatch(std::size_t found, std::size_t expected) {
    corrupt("GoF holds " + std::to_string(found) + " frames, expected " + std::to_string(expected));
}

}  // namespace

// --- Temporal ---

void encode_gof_temporal(const float* frames, std::size_t n, std::size_t natoms, float precision, unsigned order,
                         bool adaptive, unsigned anchor_step, bool spatial, std::vector<std::uint8_t>& out) {
    const std::size_t n3 = natoms * 3;
    const bool use_b = anchor_step > 1 && n > 2;
    const GofLayout L = build_gof_layout(n, use_b ? anchor_step : 1);
    auto frame = [&](std::size_t i) { return frames + i * n3; };

    std::vector<float> pred(n3), residual(n3), deq(n3), recon(n * n3);
    std::vector<std::int32_t> q(n3);
    std::vector<const float*> ptrs;
    History hist(order);

    // Pass 1, for adaptive precision only. Each atom's largest residual at the
    // uniform step selects its precision class.
    std::vector<std::uint8_t> classes;
    std::vector<float> prec;
    const bool adapt = adaptive && n > 1;
    if (adapt) {
        std::vector<float> max_abs(natoms, 0.0f);
        auto update_max = [&] {
            for (std::size_t a = 0; a < natoms; ++a) {
                float mx = std::max(std::max(std::fabs(residual[3 * a]), std::fabs(residual[3 * a + 1])),
                                    std::fabs(residual[3 * a + 2]));
                if (mx > max_abs[a]) max_abs[a] = mx;
            }
        };
        for (std::size_t ai = 0; ai < L.anchors.size(); ++ai) {
            const std::size_t idx = L.anchors[ai];
            float* r = recon.data() + idx * n3;
            if (ai == 0) {
                quantize_uniform(frame(idx), n3, precision, q.data());
                dequantize_uniform(q.data(), n3, precision, r);
            } else {
                predict_from(hist, recon.data(), n3, ptrs, pred.data());
                for (std::size_t i = 0; i < n3; ++i) residual[i] = frame(idx)[i] - pred[i];
                update_max();
                quantize_uniform(residual.data(), n3, precision, q.data());
                dequantize_uniform(q.data(), n3, precision, deq.data());
                for (std::size_t i = 0; i < n3; ++i) r[i] = pred[i] + deq[i];
            }
            hist.push(idx);
        }
        for (std::size_t b = 0; b < L.bframes.size(); ++b) {
            auto [left, right] = L.brefs[b];
            const std::size_t bi = L.bframes[b];
            predict_interpolate(recon.data() + left * n3, recon.data() + right * n3, left, right, bi, n3, pred.data());
            for (std::size_t i = 0; i < n3; ++i) residual[i] = frame(bi)[i] - pred[i];
            update_max();
        }
        compute_precision_classes(max_abs, precision, classes);
        build_precision_array(classes.data(), natoms, precision, prec);
    }

    // Pass 2: encode.
    const std::uint8_t flags = static_cast<std::uint8_t>((adapt ? 1 : 0) | (use_b ? 2 : 0) | (spatial ? 4 : 0));
    out.reserve(out.size() + n * n3 * 2 + 16);
    put_u8(out, flags);
    put_le<std::uint16_t>(out, static_cast<std::uint16_t>(n));
    if (use_b) {
        put_u8(out, static_cast<std::uint8_t>(anchor_step));
        put_u8(out, static_cast<std::uint8_t>(L.anchors.size()));
    }
    if (adapt) {
        std::size_t start = out.size();
        out.resize(start + (natoms + 3) / 4, 0);
        for (std::size_t i = 0; i < natoms; ++i)
            out[start + i / 4] |= static_cast<std::uint8_t>((classes[i] & 3) << ((i % 4) * 2));
    }

    auto pack = [&] {
        std::size_t at = out.size();
        out.resize(at + n3 * 5);
        out.resize(at + pack_residual_frame(q.data(), natoms, spatial, out.data() + at));
    };
    auto quantize = [&] {
        if (adapt) {
            quantize_adaptive(residual.data(), prec.data(), natoms, q.data());
            dequantize_adaptive(q.data(), prec.data(), natoms, deq.data());
        } else {
            quantize_uniform(residual.data(), n3, precision, q.data());
            dequantize_uniform(q.data(), n3, precision, deq.data());
        }
    };

    // Anchors. The decoder rebuilds each as prediction + dequantized residual;
    // computing exactly that here keeps the two in lockstep.
    hist.idx.clear();
    for (std::size_t ai = 0; ai < L.anchors.size(); ++ai) {
        const std::size_t idx = L.anchors[ai];
        float* r = recon.data() + idx * n3;
        if (ai == 0) {
            quantize_uniform(frame(idx), n3, precision, q.data());
            pack();
            dequantize_uniform(q.data(), n3, precision, r);
        } else {
            predict_from(hist, recon.data(), n3, ptrs, pred.data());
            for (std::size_t i = 0; i < n3; ++i) residual[i] = frame(idx)[i] - pred[i];
            quantize();
            pack();
            for (std::size_t i = 0; i < n3; ++i) r[i] = pred[i] + deq[i];
        }
        hist.push(idx);
    }
    for (std::size_t b = 0; b < L.bframes.size(); ++b) {
        auto [left, right] = L.brefs[b];
        const std::size_t bi = L.bframes[b];
        predict_interpolate(recon.data() + left * n3, recon.data() + right * n3, left, right, bi, n3, pred.data());
        for (std::size_t i = 0; i < n3; ++i) residual[i] = frame(bi)[i] - pred[i];
        quantize();
        pack();
    }
}

static DecodedSpan decode_gof_temporal(ByteSpan data, std::size_t natoms, float base, std::size_t order,
                                       std::size_t expected, float* out, std::size_t hi) {
    if (!(std::isfinite(base) && base > 0.0f)) corrupt("temporal GoF in a column with precision " + fmt(base));
    const std::size_t n3 = checked_mul(natoms, 3, "atom count overflows");
    ByteCursor cur(data);
    const std::uint8_t flags = cur.u8();
    const bool adaptive = flags & 1, has_b = flags & 2, spatial = flags & 4;
    const std::size_t nframes = cur.u16();
    if (nframes != expected) frame_count_mismatch(nframes, expected);
    std::size_t anchor_step = 1;
    if (has_b) {
        anchor_step = cur.u8();
        cur.u8();  // anchor count, which the layout implies
    }
    const GofLayout L = build_gof_layout(nframes, anchor_step);

    std::vector<float> prec;
    if (adaptive) {
        ByteSpan packed = cur.take((natoms + 3) / 4);
        std::vector<std::uint8_t> classes(natoms);
        for (std::size_t i = 0; i < natoms; ++i) classes[i] = (packed[i / 4] >> ((i % 4) * 2)) & 3;
        build_precision_array(classes.data(), natoms, base, prec);
    }
    // Each value takes at least one varint byte. Reject a payload too short
    // for the group before writing anything to `out`.
    const std::size_t total = sat_mul(n3, nframes);
    if (total > cur.remaining()) corrupt("temporal GoF too short for its atom count");

    std::vector<std::int32_t> q(n3);
    std::vector<float> pred(n3);
    std::vector<const float*> ptrs;
    auto next = [&] {
        ByteSpan rest = cur.rest();
        std::size_t used = read_residual_frame(rest.data(), rest.size(), natoms, spatial, q.data());
        if (used == 0 && n3 != 0) corrupt("temporal GoF residuals truncated");
        cur.skip(used);
    };
    auto reconstruct = [&](float* dst) {
        if (adaptive) {
            for (std::size_t a = 0; a < natoms; ++a) {
                const float p = prec[a];
                dst[3 * a] = pred[3 * a] + static_cast<float>(q[3 * a]) * p;
                dst[3 * a + 1] = pred[3 * a + 1] + static_cast<float>(q[3 * a + 1]) * p;
                dst[3 * a + 2] = pred[3 * a + 2] + static_cast<float>(q[3 * a + 2]) * p;
            }
        } else {
            for (std::size_t i = 0; i < n3; ++i) dst[i] = pred[i] + static_cast<float>(q[i]) * base;
        }
    };

    // Without B-frames the anchors are every frame in order, so decoding can
    // stop after the last frame requested.
    const bool prefix = L.bframes.empty() && hi < nframes;
    History hist(order);
    for (std::size_t ai = 0; ai < L.anchors.size(); ++ai) {
        const std::size_t idx = L.anchors[ai];
        if (prefix && idx >= hi) return {0, hi};
        next();
        float* dst = out + idx * n3;
        if (ai == 0) {
            for (std::size_t i = 0; i < n3; ++i) dst[i] = static_cast<float>(q[i]) * base;
        } else {
            predict_from(hist, out, n3, ptrs, pred.data());
            reconstruct(dst);
        }
        hist.push(idx);
    }
    for (std::size_t b = 0; b < L.bframes.size(); ++b) {
        next();
        auto [left, right] = L.brefs[b];
        const std::size_t bi = L.bframes[b];
        predict_interpolate(out + left * n3, out + right * n3, left, right, bi, n3, pred.data());
        reconstruct(out + bi * n3);
    }
    return {0, nframes};
}

// --- Intra-frame (XTC-style) ---

void encode_gof_intra(const float* frames, std::size_t n, std::size_t natoms, float precision, bool adaptive,
                      std::vector<std::uint8_t>& out) {
    const float xtc_precision = 1.0f / (adaptive ? precision * 8.0f : precision);
    put_le<std::uint16_t>(out, static_cast<std::uint16_t>(n));
    std::vector<std::uint8_t> buf;
    for (std::size_t f = 0; f < n; ++f) {
        buf.clear();
        xdr3d_encode(frames + f * natoms * 3, natoms, xtc_precision, buf);
        put_le<std::uint32_t>(out, static_cast<std::uint32_t>(buf.size()));
        put_bytes(out, buf.data(), buf.size());
    }
}

static DecodedSpan decode_gof_intra(ByteSpan data, std::size_t natoms, std::size_t expected, float* out, std::size_t lo,
                                    std::size_t hi) {
    ByteCursor cur(data);
    const std::size_t nframes = cur.u16();
    if (nframes != expected) frame_count_mismatch(nframes, expected);
    const std::size_t n3 = checked_mul(natoms, 3, "atom count overflows");
    // An XTC frame costs at least a bit per atom.
    if (sat_mul(natoms, nframes) > sat_mul(cur.remaining(), 8)) corrupt("intra-frame GoF too short for its atom count");
    thread_local std::vector<std::int32_t> lip;
    for (std::size_t f = 0; f < nframes; ++f) {
        const std::size_t len = cur.u32();
        ByteSpan blob = cur.take(len);
        // Frames are independent, so decode only the requested ones.
        if (n3 == 0 || f < lo || f >= hi) continue;
        try {
            xdr3d_decode(blob, natoms, lip, out + f * n3);
        } catch (const Error& e) {
            corrupt(std::string("intra-frame GoF: ") + e.what());
        }
    }
    return {std::min(lo, nframes), std::min(hi, nframes)};
}

// --- Lossless ---

void encode_gof_lossless(const float* frames, std::size_t n, std::size_t natoms, unsigned order,
                         std::vector<std::uint8_t>& out) {
    const std::size_t n3 = natoms * 3;
    std::vector<std::uint32_t> words(n * n3);
    std::vector<float> pred(n3);
    std::vector<const float*> ptrs;
    for (std::size_t i = 0; i < n; ++i) {
        const float* f = frames + i * n3;
        std::uint32_t* w = words.data() + i * n3;
        if (i == 0) {
            for (std::size_t j = 0; j < n3; ++j) w[j] = float_bits(f[j]);
            continue;
        }
        ptrs.resize(i);
        for (std::size_t k = 0; k < i; ++k) ptrs[k] = frames + k * n3;
        predict_forward(ptrs.data(), i, order, n3, pred.data());
        for (std::size_t j = 0; j < n3; ++j)
            w[j] = zigzag(static_cast<std::int32_t>(float_bits(f[j]) - float_bits(pred[j])));
    }
    put_le<std::uint16_t>(out, static_cast<std::uint16_t>(n));
    std::size_t at = out.size();
    out.resize(at + words.size() * 4);
    byte_planes(words.data(), words.size(), out.data() + at);
}

static DecodedSpan decode_gof_lossless(ByteSpan data, std::size_t natoms, std::size_t order, std::size_t expected,
                                       float* out, std::size_t hi) {
    ByteCursor cur(data);
    const std::size_t n = cur.u16();
    if (n != expected) frame_count_mismatch(n, expected);
    const std::size_t n3 = checked_mul(natoms, 3, "atom count overflows");
    const std::size_t count = sat_mul(n3, n);
    if (sat_mul(count, 4) != cur.remaining())
        corrupt("lossless GoF is " + std::to_string(data.size()) + " bytes, wrong for " + std::to_string(n) +
                " frames of " + std::to_string(natoms) + " atoms");
    if (n3 == 0) return {0, n};
    const std::uint8_t* p0 = cur.rest().data();
    const std::uint8_t *p1 = p0 + count, *p2 = p1 + count, *p3 = p2 + count;
    std::vector<float> pred(n3);
    std::vector<const float*> ptrs;
    const std::size_t last = std::min(n, hi);
    for (std::size_t i = 0; i < last; ++i) {
        float* dst = out + i * n3;
        const std::size_t base = i * n3;
        if (i == 0) {
            for (std::size_t j = 0; j < n3; ++j) {
                const std::size_t g = base + j;
                dst[j] = bits_float(p0[g] | (std::uint32_t(p1[g]) << 8) | (std::uint32_t(p2[g]) << 16) |
                                    (std::uint32_t(p3[g]) << 24));
            }
            continue;
        }
        ptrs.resize(i);
        for (std::size_t k = 0; k < i; ++k) ptrs[k] = out + k * n3;
        predict_forward(ptrs.data(), i, order, n3, pred.data());
        for (std::size_t j = 0; j < n3; ++j) {
            const std::size_t g = base + j;
            const std::uint32_t w =
                p0[g] | (std::uint32_t(p1[g]) << 8) | (std::uint32_t(p2[g]) << 16) | (std::uint32_t(p3[g]) << 24);
            dst[j] = bits_float(float_bits(pred[j]) + static_cast<std::uint32_t>(unzigzag(w)));
        }
    }
    return {0, last};
}

// --- Whole groups ---

unsigned auto_select_order(const float* frames, std::size_t n, std::size_t n3, unsigned max_order) {
    if (n < 2) return 1;
    const float* target = frames + (n - 1) * n3;
    std::vector<const float*> hist(n - 1);
    for (std::size_t k = 0; k + 1 < n; ++k) hist[k] = frames + k * n3;
    std::vector<float> pred(n3);
    unsigned best_order = 1;
    double best_cost = INFINITY;
    const std::size_t top = std::min<std::size_t>(max_order, n);
    for (std::size_t order = 1; order <= top && order <= 4; ++order) {
        predict_forward(hist.data(), hist.size(), order, n3, pred.data());
        double sum = 0.0;
        for (std::size_t i = 0; i < n3; ++i) sum += static_cast<double>(std::fabs(target[i] - pred[i]));
        const double cost = sum / static_cast<double>(n3);
        if (cost < best_cost) {
            best_cost = cost;
            best_order = static_cast<unsigned>(order);
        }
    }
    return best_order;
}

CodecMode pick_codec_mode(const float* frames, std::size_t n, std::size_t natoms, float precision, bool adaptive,
                          unsigned anchor_step, bool spatial, int zstd_level) {
    if (n < 2) return CodecMode::Intra;  // a single frame has nothing to predict from
    const std::size_t trial = std::min<std::size_t>(n, 4);
    const unsigned order = trial > 2 ? auto_select_order(frames, trial, natoms * 3, 3) : 1;
    std::vector<std::uint8_t> raw;
    encode_gof_temporal(frames, trial, natoms, precision, order, adaptive, anchor_step, spatial, raw);
    const std::size_t temporal = zstd_compressed_size(raw, zstd_level);
    raw.clear();
    encode_gof_intra(frames, trial, natoms, precision, adaptive, raw);
    const std::size_t intra = zstd_compressed_size(raw, zstd_level);
    return intra < temporal ? CodecMode::Intra : CodecMode::Temporal;
}

GofResult encode_gof_payload(const float* frames, std::size_t n, std::size_t natoms, const GofParams& p,
                             std::vector<std::uint8_t>& payload) {
    const std::size_t n3 = natoms * 3;
    const float max_abs = max_abs_finite(frames, n * n3);
    if (p.codec != CodecMode::Lossless) {
        validate_params(p.precision, std::max<std::size_t>(n, 1), std::max(p.anchor_step, 1u));
        if (static_cast<double>(max_abs) / static_cast<double>(p.precision) >= static_cast<double>(INT32_MAX - 1))
            throw Error(ErrorKind::InvalidInput,
                        "value " + fmt(max_abs) + " is out of range for precision " + fmt(p.precision) +
                            ": it would overflow the 32-bit quantizer. Use a coarser precision or rescale the data.");
    }
    CodecMode mode = p.codec == CodecMode::Auto ? pick_codec_mode(frames, n, natoms, p.precision, p.adaptive,
                                                                  p.anchor_step, p.spatial_delta, p.zstd_level)
                                                : p.codec;
    // Temporal residuals beyond the quantizer's range would saturate, so fall
    // back to intra. The intra codec quantizes the values themselves, and
    // validation has already bounded those.
    if (mode == CodecMode::Temporal && !temporal_residuals_fit(max_abs, p.precision)) mode = CodecMode::Intra;

    auto order = [&]() -> unsigned {
        if (p.predictor_order != 0) return p.predictor_order;
        return n > 2 ? auto_select_order(frames, n, n3, 3) : 1;
    };

    payload.clear();
    payload.resize(3);
    unsigned used = 0;
    switch (mode) {
        case CodecMode::Intra:
            encode_gof_intra(frames, n, natoms, p.precision, p.adaptive, payload);
            break;
        case CodecMode::Lossless:
            used = order();
            encode_gof_lossless(frames, n, natoms, used, payload);
            break;
        default:
            used = order();
            encode_gof_temporal(frames, n, natoms, p.precision, used, p.adaptive, p.anchor_step, p.spatial_delta,
                                payload);
            break;
    }
    payload[0] = kGofVersion;
    payload[1] = static_cast<std::uint8_t>(used);
    payload[2] = static_cast<std::uint8_t>(mode);
    return {mode, used};
}

GofResult encode_gof_blob(const float* frames, std::size_t n, std::size_t natoms, const GofParams& p,
                          std::vector<std::uint8_t>& blob) {
    thread_local std::vector<std::uint8_t> payload;
    GofResult r = encode_gof_payload(frames, n, natoms, p, payload);
    zstd_compress(payload, p.zstd_level, blob);
    // With rigid bodies, code the group both ways and keep the smaller.
    if (p.rigid && p.rigid->nbodies() > 0 && r.mode != CodecMode::Lossless) {
        thread_local std::vector<std::uint8_t> rigid_payload, rigid_blob;
        encode_rigid_payload(frames, n, natoms, *p.rigid, p, rigid_payload);
        zstd_compress(rigid_payload, p.zstd_level, rigid_blob);
        if (rigid_blob.size() < blob.size()) {
            blob.swap(rigid_blob);
            r = {CodecMode::Rigid, 0};
        }
    }
    return r;
}

std::size_t f32x3_gof_cap(std::size_t frames, std::size_t natoms) noexcept {
    // The temporal codec uses at most a 5-byte varint per value and the
    // lossless codec exactly 4 bytes. The intra codec stays under 8 bytes per
    // value plus a per-frame header. Allowing 8 bytes per value and 64 per
    // frame covers all three, and the natoms + 64 term covers the group
    // header and the adaptive precision classes.
    const std::size_t per_frame = sat_add(sat_mul(natoms, 3 * 8), 64);
    return sat_add(sat_add(sat_mul(frames, per_frame), natoms), 64);
}

DecodedSpan decode_f32x3_payload(ByteSpan raw, std::size_t natoms, float precision, std::size_t expected, float* out,
                                 std::size_t lo, std::size_t hi, const RigidModel* rigid) {
    if (raw.size() < 3) corrupt("GoF too small");
    if (raw[0] != kGofVersion) corrupt("unsupported f32x3 GoF version " + std::to_string(raw[0]));
    const std::size_t order = raw[1];
    ByteSpan body = raw.subspan(3, raw.size() - 3);
    switch (raw[2]) {
        case 0:
            return decode_gof_temporal(body, natoms, precision, order, expected, out, hi);
        case 1:
            return decode_gof_intra(body, natoms, expected, out, lo, hi);
        case 2:
            return decode_gof_lossless(body, natoms, order, expected, out, hi);
        case kRigidMode:
            return decode_rigid_payload(body, natoms, precision, expected, rigid, out, lo, hi);
        default:
            throw Error(ErrorKind::Corrupt, "GoF uses codec mode " + std::to_string(raw[2]) +
                                                ", which this build does not know how to decode; the file was "
                                                "written by a newer mdz");
    }
}

void validate_params(float precision, std::size_t gof_size, std::size_t anchor_step) {
    auto bad = [](const std::string& m) { throw Error(ErrorKind::InvalidInput, m); };
    if (!std::isfinite(precision) || precision <= 0.0f)
        bad("precision must be a positive, finite number (got " + fmt(precision) + ")");
    if (gof_size == 0) bad("gof_size must be >= 1");
    if (gof_size > 65535) bad("gof_size " + std::to_string(gof_size) + " exceeds the format limit of 65535");
    if (anchor_step == 0) bad("anchor_step must be >= 1");
    if (anchor_step > 255) bad("anchor_step " + std::to_string(anchor_step) + " exceeds the format limit of 255");
    if (anchor_step > 1) {
        const std::size_t n_anchors = (gof_size + anchor_step - 1) / anchor_step + 1;
        if (n_anchors > 255)
            bad("gof_size " + std::to_string(gof_size) + " with anchor_step " + std::to_string(anchor_step) +
                " yields up to " + std::to_string(n_anchors) +
                " anchors per group, exceeding the format limit of 255; reduce gof_size or increase anchor_step");
    }
}

void validate_coords(const float* v, std::size_t n, float precision) {
    // round(x) <= INT32_MIN + 1 or >= INT32_MAX - 1, with x = v / precision in
    // double, is exactly x <= -2147483646.5 or x >= 2147483645.5.
    const double inv = 1.0 / static_cast<double>(precision);
    std::size_t bad_at = n;
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(v[i]) * inv;
        // NaN fails both comparisons and is caught below.
        if (!(x > -2147483646.5 && x < 2147483645.5)) {
            bad_at = i;
            break;
        }
    }
    if (bad_at == n) return;
    const float val = v[bad_at];
    if (!std::isfinite(val))
        throw Error(ErrorKind::InvalidInput, "value at index " + std::to_string(bad_at) + " is " + fmt(val) +
                                                 "; mdz cannot bound non-finite (NaN/Inf) values");
    throw Error(ErrorKind::InvalidInput,
                "value at index " + std::to_string(bad_at) + " (" + fmt(val) + ") is out of range for precision " +
                    fmt(precision) +
                    ": it would overflow the 32-bit quantizer. Use a coarser precision or rescale the data.");
}

}  // namespace mdz::detail
