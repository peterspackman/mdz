// The WebAssembly core of the JavaScript package (embind): a LazyReader that
// decodes the byte ranges lib/index.js reads for it. Arrays returned to JS
// are copies, never views of the wasm heap.

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "mdz/error.hpp"
#include "mdz/frame_batch.hpp"
#include "mdz/lazy_reader.hpp"
#include "mdz/mdz.hpp"
#include "mdz/metadata.hpp"
#include "mdz/trajectory.hpp"

using emscripten::val;

namespace {

const char* kind_name(mdz::ErrorKind k) {
    switch (k) {
        case mdz::ErrorKind::InvalidInput:
            return "InvalidInput";
        case mdz::ErrorKind::Corrupt:
            return "Corrupt";
        case mdz::ErrorKind::Io:
            return "Io";
        case mdz::ErrorKind::OutOfRange:
            return "OutOfRange";
    }
    return "Internal";
}

// Throw the MdzError class that lib/index.js passes in as a module property.
[[noreturn]] void throw_js(const std::string& kind, const std::string& message) {
    val::module_property("MdzError").new_(val(message), val(kind)).throw_();
    __builtin_unreachable();
}

// Run f, turning C++ exceptions into MdzError.
template <class F>
auto guard(F&& f) -> decltype(f()) {
    std::string kind, message;
    try {
        return f();
    } catch (const mdz::Error& e) {
        kind = kind_name(e.kind());
        message = e.what();
    } catch (const std::bad_alloc&) {
        kind = "OutOfMemory";
        message = "out of memory";
    } catch (const std::exception& e) {
        kind = "Internal";
        message = e.what();
    }
    throw_js(kind, message);
}

template <class T>
val typed(const char* type, const T* data, std::size_t n) {
    return val::global(type).new_(emscripten::typed_memory_view(n, data));
}
val f32(const std::vector<float>& v) { return typed("Float32Array", v.data(), v.size()); }
val f64(const std::vector<double>& v) { return typed("Float64Array", v.data(), v.size()); }
val i32(const std::vector<std::int32_t>& v) { return typed("Int32Array", v.data(), v.size()); }

val strings(const std::vector<std::string>& v) {
    val out = val::array();
    for (const auto& s : v) out.call<void>("push", val(s));
    return out;
}

std::vector<std::uint8_t> bytes_of(const val& u8) {
    std::vector<std::uint8_t> v(u8["length"].as<std::size_t>());
    val(emscripten::typed_memory_view(v.size(), v.data())).call<void>("set", u8);
    return v;
}

mdz::ByteSpan span_of(const std::vector<std::uint8_t>& v) { return {v.data(), v.size()}; }

// Stored lengths to float32 nm, rounded as Reader rounds them.
std::vector<float> to_nm(const std::vector<float>& v, double scale) {
    if (scale == 1.0) return v;
    std::vector<float> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = static_cast<float>(static_cast<double>(v[i]) * scale);
    return out;
}
template <std::size_t N>
std::vector<float> to_nm(const std::vector<std::array<double, N>>& v, double scale) {
    std::vector<float> out;
    out.reserve(v.size() * N);
    for (const auto& a : v)
        for (double x : a) out.push_back(static_cast<float>(x * scale));
    return out;
}

// A per-atom string attribute of a varying-count batch, from its string table.
val atom_strings(const mdz::FrameBatch& b, const char* column) {
    auto it = b.int_columns.find(column);
    if (it == b.int_columns.end() || !b.strings) return val::undefined();
    std::vector<std::string> out;
    out.reserve(it->second.size());
    for (std::int32_t k : it->second) {
        if (k < 0 || static_cast<std::size_t>(k) >= b.strings->size())
            throw mdz::Error(mdz::ErrorKind::Corrupt, std::string("string index out of range in ") + column);
        out.push_back((*b.strings)[static_cast<std::size_t>(k)]);
    }
    return strings(out);
}

class JsLazyReader {
public:
    explicit JsLazyReader(mdz::LazyReader r) : r_(std::move(r)) {}

    static double head_size(const val& header) {
        return guard([&] { return static_cast<double>(mdz::LazyReader::head_size(span_of(bytes_of(header)))); });
    }
    // Owned by JS: release it with delete().
    static JsLazyReader* open(const val& head, double file_size) {
        return guard([&] {
            return new JsLazyReader(
                mdz::LazyReader::open(span_of(bytes_of(head)), static_cast<std::uint64_t>(file_size)));
        });
    }

    // [begin, end) of the footer, or null.
    val footer_range(const val& trailer) const {
        return guard([&] {
            auto f = r_.footer_range(span_of(bytes_of(trailer)));
            if (!f) return val::null();
            val out = val::array();
            out.call<void>("push", static_cast<double>(f->first));
            out.call<void>("push", static_cast<double>(f->second));
            return out;
        });
    }
    bool read_footer(double offset, const val& footer) {
        return guard([&] { return r_.read_footer(static_cast<std::uint64_t>(offset), span_of(bytes_of(footer))); });
    }
    double scan_offset() const { return static_cast<double>(r_.scan_offset()); }
    double record_size(const val& head) const {
        return guard([&] { return static_cast<double>(r_.record_size(span_of(bytes_of(head)))); });
    }
    bool add_record(const val& record) {
        return guard([&] { return r_.add_record(span_of(bytes_of(record))); });
    }

    double natoms() const { return static_cast<double>(r_.natoms()); }
    double nframes() const { return static_cast<double>(r_.nframes()); }
    bool variable_atoms() const { return r_.variable_atoms(); }
    bool complete() const { return r_.complete(); }
    double gof_size() const { return r_.gof_size(); }
    double precision() const { return r_.precision(); }

    // [offset, size, first_frame, nframes] per group, flattened.
    val groups() const {
        std::vector<double> out;
        for (const auto& g : r_.groups())
            out.insert(out.end(), {static_cast<double>(g.offset), static_cast<double>(g.size),
                                   static_cast<double>(g.first_frame), static_cast<double>(g.nframes)});
        return f64(out);
    }

    val columns() const {
        val out = val::array();
        for (const auto& c : r_.columns()) {
            if (c.name == mdz::kMembershipColumn) continue;
            val d = val::object();
            d.set("name", c.name);
            d.set("dtype", std::string(mdz::dtype_name(c.dtype)));
            d.set("width", c.width);
            d.set("perFrame", c.per_frame);
            d.set("lossy", c.lossy);
            d.set("precision", c.precision);
            d.set("unit", c.unit);
            out.call<void>("push", d);
        }
        return out;
    }

    val metadata() const {
        return guard([&] {
            mdz::Metadata m = r_.metadata();
            val d = val::object();
            if (m.atom_names) d.set("atomNames", strings(*m.atom_names));
            if (m.elements) d.set("elements", strings(*m.elements));
            if (m.atom_types) d.set("atomTypes", i32(*m.atom_types));
            if (m.atom_ids) d.set("atomIds", i32(*m.atom_ids));
            if (m.residues) d.set("residues", strings(*m.residues));
            if (m.residue_ids) d.set("residueIds", i32(*m.residue_ids));
            if (m.chain_ids) d.set("chainIds", strings(*m.chain_ids));
            if (m.hetatm) {
                std::vector<std::uint8_t> h(m.hetatm->begin(), m.hetatm->end());
                d.set("hetatm", typed("Uint8Array", h.data(), h.size()));
            }
            if (m.bodies) d.set("bodies", i32(*m.bodies));
            if (m.source_format) d.set("sourceFormat", *m.source_format);
            val attrs = val::object();
            for (const auto& [k, v] : m.attributes) attrs.set(k, v);
            d.set("attributes", attrs);
            return d;
        });
    }

    // Decode a group, or only the named columns of it. Per-frame data has
    // nframes entries and per-atom data is packed frame after frame. Lengths
    // (positions, cells, origins) are in nm; other columns in stored units.
    val decode_group(double gi, const val& record, const val& names) const {
        return guard([&] {
            std::vector<std::string> cols;
            for (std::size_t i = 0, n = names["length"].as<std::size_t>(); i < n; ++i)
                cols.push_back(names[i].as<std::string>());
            const std::vector<std::uint8_t> bytes = bytes_of(record);
            mdz::FrameBatch b = r_.decode_group(static_cast<std::size_t>(gi), span_of(bytes), cols);
            const double scale = r_.length_to_nm();
            val out = val::object();
            out.set("nframes", static_cast<double>(b.nframes));
            std::vector<double> counts(b.nframes);
            for (std::size_t f = 0; f < b.nframes; ++f) counts[f] = static_cast<double>(b.atoms_in(f));
            out.set("natoms", f64(counts));
            if (const auto* p = b.positions()) out.set("positions", f32(to_nm(*p, scale)));
            if (b.box) out.set("cells", f32(to_nm(*b.box, scale)));
            if (b.origin) out.set("origins", f32(to_nm(*b.origin, scale)));
            val scalars = val::object();
            for (const auto& [k, v] : b.frame_props) scalars.set(k, f64(v));
            for (const auto& [k, v] : b.int_frame_props) scalars.set(k, f64(std::vector<double>(v.begin(), v.end())));
            out.set("scalars", scalars);
            val columns = val::object();
            for (const auto& [k, v] : b.columns)
                if (k != "positions") columns.set(k, f32(v));
            for (const auto& [k, v] : b.int_columns)
                if (k.rfind("mdz.", 0) != 0) columns.set(k, i32(v));
            out.set("columns", columns);
            if (b.variable()) {
                if (auto it = b.int_columns.find(mdz::columns::kId); it != b.int_columns.end())
                    out.set("ids", i32(it->second));
                if (auto it = b.int_columns.find(mdz::columns::kType); it != b.int_columns.end())
                    out.set("types", i32(it->second));
                out.set("elements", atom_strings(b, mdz::columns::kElement));
                out.set("names", atom_strings(b, mdz::columns::kName));
            }
            return out;
        });
    }

private:
    mdz::LazyReader r_;
};

// The wasm build has no threads: never let the library start any.
struct SingleThreaded {
    SingleThreaded() { mdz::set_num_threads(1); }
} single_threaded;

}  // namespace

EMSCRIPTEN_BINDINGS(mdz) {
    using namespace emscripten;
    function("version", optional_override([] { return std::string(mdz::version()); }));
    constant("headerSize", static_cast<double>(mdz::LazyReader::kHeaderSize));
    constant("trailerSize", static_cast<double>(mdz::LazyReader::kTrailerSize));
    constant("recordHeadSize", static_cast<double>(mdz::LazyReader::kRecordHeadSize));
    class_<JsLazyReader>("LazyReader")
        .class_function("headSize", &JsLazyReader::head_size)
        .class_function("open", &JsLazyReader::open, return_value_policy::take_ownership())
        .function("footerRange", &JsLazyReader::footer_range)
        .function("readFooter", &JsLazyReader::read_footer)
        .function("scanOffset", &JsLazyReader::scan_offset)
        .function("recordSize", &JsLazyReader::record_size)
        .function("addRecord", &JsLazyReader::add_record)
        .property("natoms", &JsLazyReader::natoms)
        .property("nframes", &JsLazyReader::nframes)
        .property("variableAtoms", &JsLazyReader::variable_atoms)
        .property("complete", &JsLazyReader::complete)
        .property("gofSize", &JsLazyReader::gof_size)
        .property("precision", &JsLazyReader::precision)
        .function("groups", &JsLazyReader::groups)
        .function("columns", &JsLazyReader::columns)
        .function("metadata", &JsLazyReader::metadata)
        .function("decodeGroup", &JsLazyReader::decode_group);
}
