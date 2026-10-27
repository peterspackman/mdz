#pragma once
// GROMACS' reduced-precision coordinate codec (xdr3dfcoord), as used by XTC
// frames and by mdz's intra-frame codec. Follows xdrfile.c bit for bit.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "mdz/span.hpp"

namespace mdz::detail {

// Append the encoding of natoms coordinates (nm) at `precision` (1000 =
// 0.001 nm) to out. Throws Error(InvalidInput) for non-finite or
// out-of-range values.
void xdr3d_encode(const float* coords, std::size_t natoms, float precision, std::vector<std::uint8_t>& out);

// Decode one frame of natoms atoms from the front of `in` into out
// (natoms * 3), returning the bytes consumed. `lip` is scratch, reused across
// calls. Throws Error(Corrupt).
std::size_t xdr3d_decode(ByteSpan in, std::size_t natoms, std::vector<std::int32_t>& lip, float* out);

}  // namespace mdz::detail
