#pragma once
// Trajectories whose atom count varies from frame to frame; see variable.cpp.

#include <memory>
#include <string>
#include <vector>

#include "mdz/trajectory.hpp"

namespace mdz::detail {

CompressedTrajectory compress_variable(const FrameBatch& batch, const CompressOptions& opts);
// Decodes frames [start, end), either every column or only `only`. Each frame
// holds just the atoms present in it.
FrameBatch decompress_variable(const CompressedTrajectory& ct, std::size_t start, std::size_t end,
                               const std::string* only);
// The atom count of every frame. Decoded from the membership blobs on first
// use and cached in ct.atom_counts, which is guarded by a mutex.
std::vector<std::size_t> variable_frame_natoms(const CompressedTrajectory& ct);
std::shared_ptr<AtomCounts> new_atom_counts();

}  // namespace mdz::detail
