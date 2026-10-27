#pragma once
// Group-of-frames codecs for per-atom F32 x 3 columns (positions, velocities,
// forces): temporal prediction with adaptive quantization and B-frames, the
// XTC-style intra-frame codec, and the lossless codec.
//
// A group's blob is zstd([3][order][mode][payload]); see encode_gof_blob.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "mdz/options.hpp"
#include "mdz/span.hpp"

namespace mdz::detail {

struct RigidModel;

// The payload codec byte of a rigid-body group (see rigid.cpp).
inline constexpr std::uint8_t kRigidMode = 3;

struct GofParams {
    float precision = 0.000125f;
    // 0 picks an order per group.
    unsigned predictor_order = 0;
    bool adaptive = true;
    unsigned anchor_step = 1;
    bool spatial_delta = true;
    CodecMode codec = CodecMode::Auto;
    int zstd_level = 3;
    // Rigid bodies to try as a model for the group, if any.
    const RigidModel* rigid = nullptr;
};

struct GofResult {
    CodecMode mode;
    unsigned order;
};

// Encode nframes contiguous frames (nframes * natoms * 3 floats) into the
// blob that goes on disk, replacing `blob`. Throws Error(InvalidInput) for a
// non-finite value or (lossy codecs) one too large for the quantizer.
GofResult encode_gof_blob(const float* frames, std::size_t nframes, std::size_t natoms, const GofParams& p,
                          std::vector<std::uint8_t>& blob);
// The same group's payload before zstd ([3][order][mode][data]), without the
// rigid-body alternative.
GofResult encode_gof_payload(const float* frames, std::size_t nframes, std::size_t natoms, const GofParams& p,
                             std::vector<std::uint8_t>& payload);

// Raw payload encoders (appending to out), exposed for tests and trials.
void encode_gof_temporal(const float* frames, std::size_t n, std::size_t natoms, float precision, unsigned order,
                         bool adaptive, unsigned anchor_step, bool spatial, std::vector<std::uint8_t>& out);
// `precision` is the column's base step. With adaptive precision the coarsest
// of the four levels is 8x that step, which bounds the column's error at 4x
// the base step. Intra frames quantize every atom alike, so with adaptive
// precision they use the 8x step and meet the same bound.
void encode_gof_intra(const float* frames, std::size_t n, std::size_t natoms, float precision, bool adaptive,
                      std::vector<std::uint8_t>& out);
void encode_gof_lossless(const float* frames, std::size_t n, std::size_t natoms, unsigned order,
                         std::vector<std::uint8_t>& out);

// Trial-encode the first frames both ways and pick the smaller.
CodecMode pick_codec_mode(const float* frames, std::size_t n, std::size_t natoms, float precision, bool adaptive,
                          unsigned anchor_step, bool spatial, int zstd_level);

// The predictor order (1..max_order) that best predicts the last of n frames.
unsigned auto_select_order(const float* frames, std::size_t n, std::size_t n3, unsigned max_order);

// The largest decompressed payload a group of `frames` frames can have.
std::size_t f32x3_gof_cap(std::size_t frames, std::size_t natoms) noexcept;
// Whether a decompressed payload is a coordinate-codec group rather than a
// plain tagged scalar blob.
inline bool is_f32x3_payload(ByteSpan raw) noexcept { return !raw.empty() && raw[0] == 3; }
// Decode a decompressed coordinate-codec payload of expected_frames frames
// into out (expected_frames * natoms * 3). Only frames [lo, hi) are certain
// to be decoded. Decoders stop after the last requested frame, and the intra
// codec decodes only the requested frames, so reading one frame costs less
// than reading the whole group. The returned span gives the frames that are
// valid in `out`. It always covers [lo, hi), clamped to the group, and may
// extend beyond it. Throws Error(Corrupt).
struct DecodedSpan {
    std::size_t lo, hi;
};
DecodedSpan decode_f32x3_payload(ByteSpan raw, std::size_t natoms, float precision, std::size_t expected_frames,
                                 float* out, std::size_t lo = 0, std::size_t hi = SIZE_MAX,
                                 const RigidModel* rigid = nullptr);

// Throws Error(InvalidInput) if the settings exceed what the format holds.
void validate_params(float precision, std::size_t gof_size, std::size_t anchor_step);
// Throws Error(InvalidInput) if a value is non-finite or would overflow the
// 32-bit quantizer at `precision`.
void validate_coords(const float* v, std::size_t n, float precision);

}  // namespace mdz::detail
