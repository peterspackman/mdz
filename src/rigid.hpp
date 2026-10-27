#pragma once
// Rigid bodies as a compression model for coordinates: a body's atoms are
// predicted from its centre and orientation, so each frame costs six numbers
// per body plus small residuals. See rigid.cpp.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "bytes.hpp"
#include "codec.hpp"
#include "mdz/frame_batch.hpp"

namespace mdz::detail {

// Which atoms move together. Built from a per-atom body index, where -1 means
// no body. A body with a single atom is treated as a free atom.
struct RigidModel {
    std::vector<std::uint32_t> free_atoms;  // ascending
    std::vector<std::uint32_t> members;     // body atoms, body after body (bodies by index, atoms ascending)
    std::vector<std::uint32_t> body_start;  // nbodies + 1 offsets into members
    std::size_t natoms = 0;

    std::size_t nbodies() const noexcept { return body_start.empty() ? 0 : body_start.size() - 1; }
    // Throws Error(InvalidInput) unless body_of_atom has natoms entries.
    static std::shared_ptr<const RigidModel> build(const std::vector<std::int32_t>& body_of_atom, std::size_t natoms);
};

// Encode a group with the rigid model as a complete payload, header bytes
// included: [3][0][3][body]. Throws as encode_gof_blob does.
void encode_rigid_payload(const float* frames, std::size_t n, std::size_t natoms, const RigidModel& model,
                          const GofParams& p, Bytes& payload);

// Decode the body of a rigid payload (after the three header bytes). Frames
// [lo, hi) are guaranteed, as for decode_f32x3_payload.
DecodedSpan decode_rigid_payload(ByteSpan body, std::size_t natoms, float precision, std::size_t expected,
                                 const RigidModel* model, float* out, std::size_t lo = 0, std::size_t hi = SIZE_MAX);

// Finds groups of atoms whose pairwise distances stay within `tolerance` (nm)
// over the batch. Atoms closer than `bond` (nm) in the first frame are joined,
// using the minimum image when there is a box. A cluster is kept if every
// pairwise distance stays within `tolerance` in the sampled frames. Returns a
// body index per atom, with -1 for atoms in no rigid body.
std::vector<std::int32_t> detect_rigid_bodies(const FrameBatch& batch, double bond = 0.16, double tolerance = 0.002,
                                              std::size_t max_body = 64);

}  // namespace mdz::detail
