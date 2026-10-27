// Fuzzes the group-of-frames decoder directly with a decompressed payload.
// Mutating whole files mostly corrupts the zstd frames, so the decoders behind
// them are rarely reached that way. The first two bytes select the atom and
// frame counts.

#include <cstdlib>
#include <vector>

#include "codec.hpp"
#include "mdz/error.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (size < 2) return 0;
    // The atom count range covers both sides of the XTC codec's small-system
    // cutoff of 9 atoms.
    const std::size_t natoms = data[0] % 64, frames = data[1] % 24 + 1;
    std::vector<float> out(natoms * 3 * frames);
    try {
        mdz::detail::decode_f32x3_payload(mdz::ByteSpan(data + 2, size - 2), natoms, 0.001f, frames, out.data());
        // Partial decode of the same payload.
        mdz::detail::decode_f32x3_payload(mdz::ByteSpan(data + 2, size - 2), natoms, 0.001f, frames, out.data(),
                                          frames / 2, frames / 2 + 1);
    } catch (const mdz::Error&) {
    }
    return 0;
}
