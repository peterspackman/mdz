// Fuzzes the metadata block decoder (atom names, residues, ids, attributes).

#include "mdz/metadata.hpp"
#include "mdz/error.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    try {
        mdz::Metadata m = mdz::Metadata::decode(mdz::ByteSpan(data, size));
        // Anything that decodes must survive an encode and decode unchanged.
        mdz::Metadata again = mdz::Metadata::decode(m.encode());
        if (again.atom_names != m.atom_names || again.attributes != m.attributes) std::abort();
    } catch (const mdz::Error&) {
    }
    return 0;
}
