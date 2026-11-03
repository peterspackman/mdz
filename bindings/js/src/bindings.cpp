// The WebAssembly core of the JavaScript package (embind). It decodes bytes it
// is given and never touches a filesystem. lib/index.js wraps it in the public
// API; this layer is not meant to be used directly.
//
// Every array returned to JS is a copy owned by JS, never a view of the wasm
// heap, which is invalidated whenever the heap grows.

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "mdz/error.hpp"
#include "mdz/frame_batch.hpp"
#include "mdz/mdz.hpp"
#include "mdz/metadata.hpp"
#include "mdz/reader.hpp"

using emscripten::val;

namespace {

// --- errors ---

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

// Throw an instance of the MdzError class that lib/index.js passes in as a
// module property.
[[noreturn]] void throw_js(const char* kind, const std::string& message) {
    val::module_property("MdzError").new_(val(message), val(kind)).throw_();
    __builtin_unreachable();
}

// Run f, turning C++ exceptions into MdzError. The message is copied out of
// the catch block first, so the C++ exception is released before JS unwinds.
template <class F>
auto guard(F&& f) -> decltype(f()) {
    std::string kind;
    std::string message;
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
    throw_js(kind.c_str(), message);
}

void check_frame(const mdz::Reader& r, double frame) {
    if (!(frame >= 0) || frame >= static_cast<double>(r.nframes()))
        throw mdz::Error(mdz::ErrorKind::OutOfRange, "frame " + std::to_string(static_cast<long long>(frame)) +
                                                         " out of range (nframes=" + std::to_string(r.nframes()) + ")");
}

// --- conversion to JS ---

template <class T>
val typed_copy(const char* type, const T* data, std::size_t n) {
    // Constructing a typed array from a memory view copies it.
    return val::global(type).new_(emscripten::typed_memory_view(n, data));
}
val f32(const std::vector<float>& v) { return typed_copy("Float32Array", v.data(), v.size()); }
val f64(const std::vector<double>& v) { return typed_copy("Float64Array", v.data(), v.size()); }
val i32(const std::vector<std::int32_t>& v) { return typed_copy("Int32Array", v.data(), v.size()); }

// int64 values (step, mdz.atoms) as a Float64Array. Exact up to 2^53, which
// covers any step count a simulation reaches.
val i64_as_f64(const std::vector<std::int64_t>& v) {
    std::vector<double> d(v.begin(), v.end());
    return f64(d);
}

val strings(const std::vector<std::string>& v) {
    val out = val::array();
    for (const auto& s : v) out.call<void>("push", val(s));
    return out;
}

val shape(std::initializer_list<std::size_t> dims) {
    val out = val::array();
    for (std::size_t d : dims) out.call<void>("push", val(static_cast<double>(d)));
    return out;
}

bool is_internal_column(const std::string& name) { return name == mdz::kMembershipColumn; }

// --- the reader ---

class JsReader {
public:
    explicit JsReader(mdz::Reader r) : r_(std::move(r)) {}

    static JsReader from_bytes(const val& bytes) {
        return guard([&] {
            // One copy from the JS array straight into the vector.
            std::vector<std::uint8_t> v(bytes["length"].as<std::size_t>());
            val(emscripten::typed_memory_view(v.size(), v.data())).call<void>("set", bytes);
            return JsReader(mdz::Reader::from_bytes(std::move(v)));
        });
    }

    double natoms() const { return static_cast<double>(r_.natoms()); }
    double nframes() const { return static_cast<double>(r_.nframes()); }
    bool variable_atoms() const { return r_.variable_atoms(); }
    bool complete() const { return r_.complete(); }
    double precision() const { return r_.precision(); }
    double gof_size() const { return r_.gof_size(); }
    double n_groups() const { return static_cast<double>(r_.n_gofs()); }

    val frame_natoms() const {
        return guard([&] {
            std::vector<std::size_t> n = r_.frame_natoms();
            std::vector<double> d(n.begin(), n.end());
            return f64(d);
        });
    }

    val atom_ids(double frame) const {
        return guard([&] {
            check_frame(r_, frame);
            return i32(r_.atom_ids(static_cast<std::size_t>(frame)));
        });
    }

    val column_names() const {
        std::vector<std::string> names = r_.column_names();
        std::vector<std::string> out;
        for (auto& n : names)
            if (!is_internal_column(n)) out.push_back(std::move(n));
        if (r_.trajectory().has_column("box")) out.emplace_back("cell");
        return strings(out);
    }

    val columns() const {
        val out = val::array();
        for (const auto& c : r_.trajectory().columns) {
            if (is_internal_column(c.desc.name)) continue;
            val d = val::object();
            d.set("name", c.desc.name);
            d.set("dtype", std::string(mdz::dtype_name(c.desc.dtype)));
            d.set("width", c.desc.width);
            d.set("perFrame", c.desc.per_frame);
            d.set("lossy", c.desc.lossy);
            d.set("precision", c.desc.precision);
            d.set("unit", c.desc.unit);
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
                d.set("hetatm", typed_copy("Uint8Array", h.data(), h.size()));
            }
            if (m.bodies) d.set("bodies", i32(*m.bodies));
            if (m.source_format) d.set("sourceFormat", *m.source_format);
            val attrs = val::object();
            for (const auto& [k, v] : m.attributes) attrs.set(k, v);
            d.set("attributes", attrs);
            return d;
        });
    }

    // Positions of frames [first, first + count) in nm, clamped to the end.
    val read_positions(double first, double count) const {
        return guard([&] {
            check_frame(r_, first);
            return f32(r_.read_positions(static_cast<std::size_t>(first), static_cast<std::size_t>(count)));
        });
    }

    // The 3x3 box of one frame in nm, rows a, b, c; null if there is none.
    val cell(double frame) const {
        return guard([&] {
            check_frame(r_, frame);
            const auto& boxes = r_.boxes();
            if (!boxes) return val::null();
            const auto& b = (*boxes)[static_cast<std::size_t>(frame)];
            return typed_copy("Float32Array", b.data(), b.size());
        });
    }

    // A whole column, or frames [first, first + count) of it, as
    // {name, dtype, shape, data}. Positions are in nm; other columns in their
    // stored units. "cell" is derived from "box" as (a, gamma, b, beta, alpha,
    // c) per frame.
    val read_column(const std::string& name, double first, double count) const {
        return guard([&] {
            const auto& ct = r_.trajectory();
            if (ct.nframes > 0 || first != 0) check_frame(r_, first);
            const std::size_t lo = static_cast<std::size_t>(first);
            const std::size_t n = std::min(static_cast<std::size_t>(count), ct.nframes - lo);
            const bool var = ct.variable_atoms;
            val out = val::object();
            out.set("name", name);
            if (name == "positions") {
                std::vector<float> p = r_.read_positions(lo, n);
                out.set("dtype", std::string("f32"));
                out.set("shape", var ? shape({p.size() / 3, 3}) : shape({n, ct.natoms, 3}));
                out.set("data", f32(p));
                return out;
            }
            const bool derived_cell = name == "cell" && !ct.column("cell");
            const mdz::CompressedColumn* col = ct.column(derived_cell ? "box" : name);
            if (!col || is_internal_column(name))
                throw mdz::Error(mdz::ErrorKind::InvalidInput, "no column named '" + name + "'");
            mdz::DecodedColumn d = r_.read_column(col->desc.name, lo, n);
            if (derived_cell) {
                std::vector<double> params(n * 6);
                for (std::size_t i = 0; i < n; ++i) {
                    mdz::Box m;
                    for (int k = 0; k < 9; ++k) m[k] = d.f64[i * 9 + k];
                    const mdz::CellParams p = mdz::box_to_cell_params(m);
                    for (int k = 0; k < 6; ++k) params[i * 6 + k] = p[k];
                }
                out.set("dtype", std::string("f64"));
                out.set("shape", shape({n, 6}));
                out.set("data", f64(params));
                return out;
            }
            const std::size_t w = col->desc.width;
            const std::size_t rows = col->desc.per_frame ? n : (var ? d.size() / w : n * ct.natoms);
            if (col->desc.per_frame)
                out.set("shape", w == 1 ? shape({n}) : (w == 9 && name == "box" ? shape({n, 3, 3}) : shape({n, w})));
            else if (var)
                out.set("shape", w == 1 ? shape({rows}) : shape({rows, w}));
            else
                out.set("shape", w == 1 ? shape({n, ct.natoms}) : shape({n, ct.natoms, w}));
            switch (d.dtype) {
                case mdz::Dtype::F64:
                    out.set("dtype", std::string("f64"));
                    out.set("data", f64(d.f64));
                    break;
                case mdz::Dtype::I32:
                    out.set("dtype", std::string("i32"));
                    out.set("data", i32(d.i32));
                    break;
                case mdz::Dtype::I64:
                    out.set("dtype", std::string("i64"));
                    out.set("data", i64_as_f64(d.i64));
                    break;
                default:
                    out.set("dtype", std::string("f32"));
                    out.set("data", f32(d.f32));
                    break;
            }
            return out;
        });
    }

    // One frame with every column: {natoms, positions (nm), cell (nm) or
    // null, scalars, columns}. Files with a varying atom count also give the
    // frame's ids and its per-atom elements, names and types.
    val frame(double frame) const {
        return guard([&] {
            check_frame(r_, frame);
            const std::size_t f = static_cast<std::size_t>(frame);
            mdz::FrameBatch b = r_.read_batch(f, 1);
            if (b.variable()) b = b.frame(0);
            val out = val::object();
            out.set("natoms", static_cast<double>(b.natoms));
            out.set("positions", f32(r_.read_positions(f, 1)));
            out.set("cell", cell(frame));
            val scalars = val::object();
            for (const auto& [k, v] : b.frame_props)
                if (!v.empty()) scalars.set(k, v[0]);
            for (const auto& [k, v] : b.int_frame_props)
                if (!v.empty()) scalars.set(k, static_cast<double>(v[0]));
            out.set("scalars", scalars);
            val cols = val::object();
            for (const auto& [k, v] : b.columns)
                if (k != "positions") cols.set(k, f32(v));
            for (const auto& [k, v] : b.int_columns)
                if (k.rfind("mdz.", 0) != 0) cols.set(k, i32(v));
            out.set("columns", cols);
            if (r_.variable_atoms()) {
                if (b.atom_ids) out.set("ids", i32(*b.atom_ids));
                if (b.elements) out.set("elements", strings(*b.elements));
                if (b.atom_names) out.set("names", strings(*b.atom_names));
                if (b.atom_types) out.set("types", i32(*b.atom_types));
            }
            return out;
        });
    }

private:
    mdz::Reader r_;
};

// The wasm build has no threads: never let the library start any.
struct SingleThreaded {
    SingleThreaded() { mdz::set_num_threads(1); }
} single_threaded;

}  // namespace

EMSCRIPTEN_BINDINGS(mdz) {
    using namespace emscripten;
    function("version", optional_override([] { return std::string(mdz::version()); }));
    class_<JsReader>("Reader")
        .class_function("fromBytes", &JsReader::from_bytes)
        .property("natoms", &JsReader::natoms)
        .property("nframes", &JsReader::nframes)
        .property("variableAtoms", &JsReader::variable_atoms)
        .property("complete", &JsReader::complete)
        .property("precision", &JsReader::precision)
        .property("gofSize", &JsReader::gof_size)
        .property("nGroups", &JsReader::n_groups)
        .function("frameNatoms", &JsReader::frame_natoms)
        .function("atomIds", &JsReader::atom_ids)
        .function("columnNames", &JsReader::column_names)
        .function("columns", &JsReader::columns)
        .function("metadata", &JsReader::metadata)
        .function("readPositions", &JsReader::read_positions)
        .function("cell", &JsReader::cell)
        .function("readColumn", &JsReader::read_column)
        .function("frame", &JsReader::frame);
}
