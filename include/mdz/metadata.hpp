#pragma once
// The static metadata block stored in a file's schema: per-atom names, types
// and residues, plus free-form string attributes.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "mdz/frame_batch.hpp"
#include "mdz/span.hpp"

namespace mdz {

// Every field is optional. Per-atom arrays have one entry per atom.
struct Metadata {
    std::optional<std::vector<std::string>> atom_names;
    std::optional<std::vector<std::string>> elements;
    std::optional<std::vector<std::int32_t>> atom_types;
    std::optional<std::vector<std::int32_t>> atom_ids;
    std::optional<std::vector<std::string>> residues;
    std::optional<std::vector<std::int32_t>> residue_ids;
    std::optional<std::vector<std::string>> chain_ids;
    std::optional<std::vector<std::int32_t>> hetatm;
    // The rigid body index of each atom, or -1 for none.
    std::optional<std::vector<std::int32_t>> bodies;
    // The format the trajectory was compressed from.
    std::optional<std::string> source_format;
    // Free-form attributes, stored under the key `attr.<name>`.
    NamedMap<std::string> attributes;

    // True if no field is set.
    bool empty() const noexcept;
    // Serialize the block. Strings longer than 65535 bytes are truncated at a
    // UTF-8 character boundary. Throws Error(InvalidInput) if there are more
    // than 255 fields in total or a key is longer than 255 bytes.
    std::vector<std::uint8_t> encode() const;
    // Parse a block written by encode(). Unknown keys are ignored. Throws
    // Error(Corrupt) for malformed data or strings that are not valid UTF-8.
    static Metadata decode(ByteSpan data);
};

// The attribute that names the unit system of the coordinate columns when it
// is not the internal one. Its value is "lammps:<style>".
inline constexpr const char* kUnitsAttr = "mdz.units";

}  // namespace mdz
