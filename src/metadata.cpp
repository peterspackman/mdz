#include "mdz/metadata.hpp"

#include <string>

#include "bytes.hpp"

namespace mdz {
namespace {

using detail::ByteCursor;
using detail::Bytes;
using detail::corrupt;
using detail::put_le;
using detail::put_str;
using detail::put_u8;

constexpr std::uint8_t kVersion = 1;
constexpr std::uint8_t kTagStringArray = 0;
constexpr std::uint8_t kTagI32Array = 1;
constexpr std::uint8_t kTagString = 2;

bool valid_utf8(std::string_view s) noexcept {
    const auto* p = reinterpret_cast<const unsigned char*>(s.data());
    std::size_t i = 0, n = s.size();
    while (i < n) {
        unsigned char c = p[i];
        std::size_t len;
        std::uint32_t cp;
        if (c < 0x80) {
            ++i;
            continue;
        } else if ((c & 0xE0) == 0xC0) {
            len = 2;
            cp = c & 0x1F;
        } else if ((c & 0xF0) == 0xE0) {
            len = 3;
            cp = c & 0x0F;
        } else if ((c & 0xF8) == 0xF0) {
            len = 4;
            cp = c & 0x07;
        } else {
            return false;
        }
        if (i + len > n) return false;
        for (std::size_t k = 1; k < len; ++k) {
            if ((p[i + k] & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (p[i + k] & 0x3F);
        }
        if ((len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) || (len == 4 && (cp < 0x10000 || cp > 0x10FFFF)) ||
            (cp >= 0xD800 && cp <= 0xDFFF))
            return false;
        i += len;
    }
    return true;
}

// Strings are truncated to 65535 bytes, on a character boundary.
void put_string(Bytes& out, std::string_view s) {
    std::size_t len = std::min<std::size_t>(s.size(), 65535);
    while (len > 0 && len < s.size() && (static_cast<unsigned char>(s[len]) & 0xC0) == 0x80) --len;
    put_le<std::uint16_t>(out, static_cast<std::uint16_t>(len));
    put_str(out, s.substr(0, len));
}

void put_string_array(Bytes& out, const std::vector<std::string>& v) {
    put_le<std::uint32_t>(out, static_cast<std::uint32_t>(v.size()));
    for (const auto& s : v) put_string(out, s);
}

void put_i32_array(Bytes& out, const std::vector<std::int32_t>& v) {
    put_le<std::uint32_t>(out, static_cast<std::uint32_t>(v.size()));
    detail::put_bytes(out, v.data(), v.size() * 4);
}

std::string get_string(ByteCursor& c) {
    std::size_t len = c.u16();
    std::string_view s = c.str(len);
    if (!valid_utf8(s)) corrupt("metadata string is not valid UTF-8");
    return std::string(s);
}

std::vector<std::string> get_string_array(ByteCursor& c) {
    std::size_t count = c.u32();
    // Each string costs at least its 2-byte length.
    if (count > c.remaining() / 2) corrupt("string array count exceeds its data");
    std::vector<std::string> out;
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i) out.push_back(get_string(c));
    return out;
}

std::vector<std::int32_t> get_i32_array(ByteCursor& c) {
    std::size_t count = c.u32();
    if (count > c.remaining() / 4) corrupt("truncated i32 array");
    ByteSpan b = c.take(count * 4);
    std::vector<std::int32_t> out(count);
    if (count) std::memcpy(out.data(), b.data(), count * 4);
    return out;
}

}  // namespace

bool Metadata::empty() const noexcept {
    return !atom_names && !elements && !atom_types && !atom_ids && !residues && !residue_ids && !chain_ids &&
           !hetatm && !bodies && !source_format && attributes.empty();
}

std::vector<std::uint8_t> Metadata::encode() const {
    struct Field {
        std::string key;
        std::uint8_t tag;
        Bytes payload;
    };
    std::vector<Field> fields;
    auto sa = [&](const char* key, const std::optional<std::vector<std::string>>& v) {
        if (!v) return;
        Field f{key, kTagStringArray, {}};
        put_string_array(f.payload, *v);
        fields.push_back(std::move(f));
    };
    auto ia = [&](const char* key, const std::optional<std::vector<std::int32_t>>& v) {
        if (!v) return;
        Field f{key, kTagI32Array, {}};
        put_i32_array(f.payload, *v);
        fields.push_back(std::move(f));
    };
    sa("atom_names", atom_names);
    sa("elements", elements);
    ia("atom_types", atom_types);
    ia("atom_ids", atom_ids);
    sa("residues", residues);
    ia("residue_ids", residue_ids);
    sa("chain_ids", chain_ids);
    ia("hetatm", hetatm);
    ia("bodies", bodies);
    if (source_format) {
        Field f{"source_format", kTagString, {}};
        put_string(f.payload, *source_format);
        fields.push_back(std::move(f));
    }
    for (const auto& [k, v] : attributes) {
        Field f{"attr." + k, kTagString, {}};
        put_string(f.payload, v);
        fields.push_back(std::move(f));
    }
    if (fields.size() > 255)
        throw Error(ErrorKind::InvalidInput, std::to_string(fields.size()) +
                                                 " metadata fields (atom data plus attributes); at most 255 fit");
    Bytes out;
    put_u8(out, kVersion);
    put_u8(out, static_cast<std::uint8_t>(fields.size()));
    for (const auto& f : fields) {
        if (f.key.size() > 255)
            throw Error(ErrorKind::InvalidInput, "metadata key '" + f.key + "' is longer than 255 bytes");
        put_u8(out, static_cast<std::uint8_t>(f.key.size()));
        put_str(out, f.key);
        put_u8(out, f.tag);
        detail::put_bytes(out, f.payload.data(), f.payload.size());
    }
    return out;
}

Metadata Metadata::decode(ByteSpan data) {
    if (data.size() < 2) corrupt("metadata too small");
    if (data[0] != kVersion) corrupt("unsupported metadata version " + std::to_string(data[0]));
    ByteCursor c(data, 2);
    Metadata m;
    for (std::size_t i = 0, n = data[1]; i < n; ++i) {
        std::size_t klen = c.u8();
        std::string_view key = c.str(klen);
        if (!valid_utf8(key)) corrupt("metadata key is not valid UTF-8");
        std::uint8_t tag = c.u8();
        if (tag == kTagStringArray) {
            auto v = get_string_array(c);
            if (key == "atom_names") m.atom_names = std::move(v);
            else if (key == "elements") m.elements = std::move(v);
            else if (key == "residues") m.residues = std::move(v);
            else if (key == "chain_ids") m.chain_ids = std::move(v);
        } else if (tag == kTagI32Array) {
            auto v = get_i32_array(c);
            if (key == "atom_types") m.atom_types = std::move(v);
            else if (key == "atom_ids") m.atom_ids = std::move(v);
            else if (key == "residue_ids") m.residue_ids = std::move(v);
            else if (key == "hetatm") m.hetatm = std::move(v);
            else if (key == "bodies") m.bodies = std::move(v);
        } else if (tag == kTagString) {
            std::string v = get_string(c);
            if (key == "source_format") m.source_format = std::move(v);
            else if (key.substr(0, 5) == "attr.") m.attributes[std::string(key.substr(5))] = std::move(v);
        } else {
            corrupt("unknown metadata tag " + std::to_string(tag));
        }
    }
    return m;
}

}  // namespace mdz
