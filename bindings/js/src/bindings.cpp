// The WebAssembly core of the JavaScript package (embind): a LazyReader that
// decodes the byte ranges lib/index.js reads for it and, unless
// MDZ_JS_READ_ONLY is defined, a Writer and the other trajectory formats. Arrays returned to JS are copies, never views
// of the wasm heap.

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "mdz/error.hpp"
#include "mdz/formats.hpp"
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

// A batch as JS. Per-frame data has nframes entries and per-atom data is
// packed frame after frame. Lengths are multiplied by `scale` to give nm;
// other columns are as stored.
val batch_to_js(const mdz::FrameBatch& b, double scale) {
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
        if (auto it = b.int_columns.find(mdz::columns::kId); it != b.int_columns.end()) out.set("ids", i32(it->second));
        if (auto it = b.int_columns.find(mdz::columns::kType); it != b.int_columns.end())
            out.set("types", i32(it->second));
        out.set("elements", atom_strings(b, mdz::columns::kElement));
        out.set("names", atom_strings(b, mdz::columns::kName));
    }
    return out;
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

    // Decode a group, or only the named columns of it (see batch_to_js).
    val decode_group(double gi, const val& record, const val& names) const {
        return guard([&] {
            std::vector<std::string> cols;
            for (std::size_t i = 0, n = names["length"].as<std::size_t>(); i < n; ++i)
                cols.push_back(names[i].as<std::string>());
            const std::vector<std::uint8_t> bytes = bytes_of(record);
            mdz::FrameBatch b = r_.decode_group(static_cast<std::size_t>(gi), span_of(bytes), cols);
            return batch_to_js(b, r_.length_to_nm());
        });
    }

private:
    mdz::LazyReader r_;
};

#ifndef MDZ_JS_READ_ONLY

// The metadata fields of a fixed-count batch, keyed as Trajectory.metadata.
val batch_metadata(const mdz::FrameBatch& b) {
    val d = val::object();
    if (b.atom_names) d.set("atomNames", strings(*b.atom_names));
    if (b.elements) d.set("elements", strings(*b.elements));
    if (b.atom_types) d.set("atomTypes", i32(*b.atom_types));
    if (b.atom_ids) d.set("atomIds", i32(*b.atom_ids));
    if (b.residue_names) d.set("residues", strings(*b.residue_names));
    if (b.residue_ids) d.set("residueIds", i32(*b.residue_ids));
    if (b.chain_ids) d.set("chainIds", strings(*b.chain_ids));
    if (b.hetatm) d.set("hetatm", typed("Uint8Array", b.hetatm->data(), b.hetatm->size()));
    if (b.bodies) d.set("bodies", i32(*b.bodies));
    val attrs = val::object();
    for (const auto& [k, v] : b.attributes) attrs.set(k, v);
    d.set("attributes", attrs);
    return d;
}

// --- streams over JS memory ---

// Reads a Uint8Array held by JS, a slice at a time, so that the input is never
// copied whole into the wasm heap.
class JsInputStream final : public mdz::InputStream {
public:
    explicit JsInputStream(val bytes) : bytes_(std::move(bytes)), size_(bytes_["length"].as<std::size_t>()) {}
    std::size_t read(void* buf, std::size_t n) override {
        const std::size_t k = std::min(n, size_ - pos_);
        if (k == 0) return 0;
        val(emscripten::typed_memory_view(k, static_cast<std::uint8_t*>(buf)))
            .call<void>("set", bytes_.call<val>("subarray", static_cast<double>(pos_), static_cast<double>(pos_ + k)));
        pos_ += k;
        return k;
    }
    const std::string& name() const override { return name_; }

private:
    val bytes_;
    std::size_t size_;
    std::size_t pos_ = 0;
    std::string name_ = "<input>";
};

// Collects written bytes until JS takes them.
class DrainStream final : public mdz::OutputStream {
public:
    explicit DrainStream(std::vector<std::uint8_t>* buf) : buf_(buf) {}
    void write(const void* data, std::size_t n) override {
        const auto* p = static_cast<const std::uint8_t*>(data);
        buf_->insert(buf_->end(), p, p + n);
    }
    void flush() override {}
    void close() override {}
    const std::string& name() const override { return name_; }

private:
    std::vector<std::uint8_t>* buf_;
    std::string name_ = "<output>";
};

val take_bytes(std::vector<std::uint8_t>& buf) {
    val out = typed("Uint8Array", buf.data(), buf.size());
    buf.clear();
    buf.shrink_to_fit();
    return out;
}

// --- options from JS ---

bool has(const val& o, const char* key) {
    return o.typeOf().as<std::string>() == "object" && !o.isNull() && !o[key].isUndefined() && !o[key].isNull();
}

std::vector<std::string> string_list(const val& a) {
    std::vector<std::string> out;
    for (std::size_t i = 0, n = a["length"].as<std::size_t>(); i < n; ++i) out.push_back(a[i].as<std::string>());
    return out;
}

// Numbers from any array-like, in one copy: converted to `Type` (such as
// Float32Array) in JS first if needed.
template <class T>
std::vector<T> numbers(const val& a, const char* type) {
    const val t = a.instanceof(val::global(type)) ? a : val::global(type).new_(a);
    std::vector<T> out(t["length"].as<std::size_t>());
    val(emscripten::typed_memory_view(out.size(), out.data())).call<void>("set", t);
    return out;
}
std::vector<float> floats(const val& a) { return numbers<float>(a, "Float32Array"); }
std::vector<double> doubles(const val& a) { return numbers<double>(a, "Float64Array"); }
std::vector<std::int32_t> ints(const val& a) { return numbers<std::int32_t>(a, "Int32Array"); }

// Precisions are in nm (lib/index.js converts them).
mdz::CompressOptions compress_options(const val& o) {
    mdz::CompressOptions c;
    if (has(o, "precision")) c.precision = o["precision"].as<float>();
    if (has(o, "gofSize")) c.gof_size = o["gofSize"].as<std::uint32_t>();
    if (has(o, "predictorOrder")) c.predictor_order = o["predictorOrder"].as<std::uint32_t>();
    if (has(o, "zstdLevel")) c.zstd_level = o["zstdLevel"].as<int>();
    if (has(o, "adaptive")) c.adaptive = o["adaptive"].as<bool>();
    if (has(o, "anchorStep")) c.anchor_step = o["anchorStep"].as<std::uint32_t>();
    if (has(o, "codec")) c.codec = mdz::parse_codec_mode(o["codec"].as<std::string>());
    if (has(o, "lossless")) c.lossless = o["lossless"].as<bool>();
    if (has(o, "variableAtoms")) c.variable_atoms = o["variableAtoms"].as<bool>();
    if (has(o, "rigid")) c.rigid = o["rigid"].as<bool>();
    if (has(o, "columnPrecision")) {
        const val cp = o["columnPrecision"];
        for (const auto& k : string_list(val::global("Object").call<val>("keys", cp)))
            c.column_precision[k] = cp[k].as<float>();
    }
    c.validate();
    return c;
}

const mdz::LammpsUnits* lammps_units(const val& o) {
    if (!has(o, "lammpsUnits")) return nullptr;
    const std::string style = o["lammpsUnits"].as<std::string>();
    const mdz::LammpsUnits* u = mdz::LammpsUnits::from_style(style);
    if (!u) throw mdz::Error(mdz::ErrorKind::InvalidInput, "unknown LAMMPS unit style '" + style + "'");
    return u;
}

const mdz::FormatInfo& format_named(const std::string& name) {
    const mdz::FormatInfo* f = mdz::find_format(name);
    if (!f) throw mdz::Error(mdz::ErrorKind::InvalidInput, "unknown trajectory format '" + name + "'");
    return *f;
}

// --- formats ---

val formats() {
    val out = val::array();
    for (const auto& f : mdz::formats()) {
        val d = val::object();
        d.set("name", std::string(f.name));
        val ext = val::array();
        for (auto e : f.extensions) ext.call<void>("push", std::string(e));
        d.set("extensions", ext);
        d.set("readable", static_cast<bool>(f.open_reader));
        d.set("writable", static_cast<bool>(f.open_writer));
        out.call<void>("push", d);
    }
    return out;
}

std::string detect_format(const std::string& name) { return std::string(mdz::detect_format(name)); }

// Every frame of a file, in nm and the other internal units.
val parse(const val& bytes, const std::string& format, const val& opts) {
    return guard([&] {
        mdz::FormatOptions fo;
        fo.lammps_units = lammps_units(opts);
        auto reader = format_named(format).open_reader(std::make_unique<JsInputStream>(bytes), fo);
        std::optional<mdz::FrameBatch> b = reader->read_all();
        if (!b || b->nframes == 0) throw mdz::Error(mdz::ErrorKind::Corrupt, "no frames in the " + format + " input");
        if (b->units) b = b->in_internal_units();
        val out = batch_to_js(*b, 1.0);
        out.set("metadata", batch_metadata(*b));
        return out;
    });
}

// Convert between formats. A .mdz output starts with a fixed atom count and
// restarts with a varying one if the input's atoms change, as `mdz compress`
// does.
val convert(const val& bytes, const std::string& from, const std::string& to, const val& opts) {
    return guard([&] {
        const mdz::FormatInfo& in = format_named(from);
        const mdz::FormatInfo& out_fmt = format_named(to);
        if (!out_fmt.open_writer) throw mdz::Error(mdz::ErrorKind::InvalidInput, "cannot write " + to);
        mdz::FormatOptions fo;
        fo.lammps_units = lammps_units(opts);
        if (to == "mdz") {
            fo.compress = compress_options(opts);
            fo.compress.source_format = from;
        }
        std::vector<std::uint8_t> out;
        for (;;) {
            out.clear();
            try {
                auto reader = in.open_reader(std::make_unique<JsInputStream>(bytes), fo);
                auto writer = out_fmt.open_writer(mdz::memory_output(&out), reader->natoms(), fo);
                std::size_t frames = 0;
                while (auto b = reader->read_batch(100)) {
                    if (to == "npy" && b->units) b = b->in_internal_units();
                    writer->write_batch(*b);
                    frames += b->nframes;
                }
                if (frames == 0) throw mdz::Error(mdz::ErrorKind::Corrupt, "no frames in the " + from + " input");
                writer->finish();
                break;
            } catch (const mdz::Error& e) {
                // A fixed-count .mdz cannot take atoms that change; try again
                // as a varying-count file.
                if (to != "mdz" || fo.compress.variable_atoms || fo.compress.rigid ||
                    e.kind() != mdz::ErrorKind::InvalidInput)
                    throw;
                fo.compress.variable_atoms = true;
            }
        }
        return take_bytes(out);
    });
}

// --- writing .mdz ---

class JsWriter {
public:
    // Static per-atom metadata (elements, atomNames, atomTypes, atomIds,
    // bodies) is taken from `opts` and attached to every frame.
    explicit JsWriter(const val& opts) {
        guard([&] {
            const mdz::CompressOptions c = compress_options(opts);
            if (has(opts, "elements")) meta_.elements = mdz::share(string_list(opts["elements"]));
            if (has(opts, "atomNames")) meta_.atom_names = mdz::share(string_list(opts["atomNames"]));
            if (has(opts, "atomTypes")) meta_.atom_types = mdz::share(ints(opts["atomTypes"]));
            if (has(opts, "atomIds")) meta_.atom_ids = mdz::share(ints(opts["atomIds"]));
            if (has(opts, "bodies")) meta_.bodies = mdz::share(ints(opts["bodies"]));
            if (c.variable_atoms && (meta_.elements || meta_.atom_names || meta_.atom_types || meta_.atom_ids))
                throw mdz::Error(mdz::ErrorKind::InvalidInput,
                                 "with variableAtoms the atoms change between frames: pass ids, elements, names and "
                                 "types with each frame instead");
            variable_ = c.variable_atoms;
            writer_ = std::make_unique<mdz::Writer>(mdz::Writer::create(std::make_unique<DrainStream>(&buf_), c));
            return 0;
        });
    }

    // One frame, with lengths in nm. See lib/index.js for the fields.
    void write(const val& frame) {
        guard([&] {
            const std::vector<float> pos = floats(frame["positions"]);
            if (pos.size() % 3 != 0)
                throw mdz::Error(mdz::ErrorKind::InvalidInput, "positions must hold 3 values per atom");
            const std::size_t n = pos.size() / 3;
            mdz::FrameBatch b(n, 1);
            b.columns["positions"] = pos;
            auto per_atom = [&](const std::string& name, const val& v) {
                const std::size_t len = v["length"].as<std::size_t>();
                if (len == 0 || len % n != 0)
                    throw mdz::Error(mdz::ErrorKind::InvalidInput, "column '" + name + "' has " + std::to_string(len) +
                                                                       " values for " + std::to_string(n) + " atoms");
                if (v.instanceof(val::global("Int32Array")))
                    b.int_columns[name] = ints(v);
                else
                    b.columns[name] = floats(v);
            };
            for (const char* k : {"velocities", "forces", "charge"})
                if (has(frame, k)) per_atom(k, frame[k]);
            if (has(frame, "columns")) {
                const val cols = frame["columns"];
                for (const auto& k : string_list(val::global("Object").call<val>("keys", cols))) per_atom(k, cols[k]);
            }
            for (const char* k : {"time", "energy"})
                if (has(frame, k)) b.frame_props[k] = {frame[k].as<double>()};
            if (has(frame, "step")) b.int_frame_props["step"] = {frame["step"].as<std::int64_t>()};
            if (has(frame, "scalars")) {
                const val sc = frame["scalars"];
                for (const auto& k : string_list(val::global("Object").call<val>("keys", sc)))
                    b.frame_props[k] = {sc[k].as<double>()};
            }
            if (has(frame, "cell")) {
                const std::vector<double> m = doubles(frame["cell"]);
                if (m.size() != 9) throw mdz::Error(mdz::ErrorKind::InvalidInput, "cell must hold 9 values");
                mdz::Box box;
                std::copy(m.begin(), m.end(), box.begin());
                b.box = std::vector<mdz::Box>{box};
            }
            if (variable_) {
                if (has(frame, "ids")) b.atom_ids = mdz::share(ints(frame["ids"]));
                if (has(frame, "elements")) b.elements = mdz::share(string_list(frame["elements"]));
                if (has(frame, "names")) b.atom_names = mdz::share(string_list(frame["names"]));
                if (has(frame, "types")) b.atom_types = mdz::share(ints(frame["types"]));
            } else {
                if (has(frame, "ids"))
                    throw mdz::Error(mdz::ErrorKind::InvalidInput,
                                     "ids identify the atoms of each frame when the count varies: create the writer "
                                     "with variableAtoms, or give atomIds to the writer");
                b.elements = meta_.elements;
                b.atom_names = meta_.atom_names;
                b.atom_types = meta_.atom_types;
                b.atom_ids = meta_.atom_ids;
            }
            b.bodies = meta_.bodies;
            writer_->write(b);
            return 0;
        });
    }

    void flush() {
        guard([&] {
            writer_->flush();
            return 0;
        });
    }
    void close() {
        guard([&] {
            writer_->close();
            return 0;
        });
    }
    // The bytes written since the last call.
    val take() { return take_bytes(buf_); }
    double frames_written() const { return static_cast<double>(writer_->frames_written()); }
    double frames_committed() const { return static_cast<double>(writer_->frames_committed()); }
    bool closed() const { return writer_->closed(); }

private:
    // Declared before writer_, which writes into it until it is destroyed.
    std::vector<std::uint8_t> buf_;
    mdz::FrameBatch meta_;
    bool variable_ = false;
    std::unique_ptr<mdz::Writer> writer_;
};

#endif  // MDZ_JS_READ_ONLY

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
#ifndef MDZ_JS_READ_ONLY
    function("formats", &formats);
    function("detectFormat", &detect_format);
    function("parse", &parse);
    function("convert", &convert);
    class_<JsWriter>("Writer")
        .constructor<const val&>()
        .function("write", &JsWriter::write)
        .function("flush", &JsWriter::flush)
        .function("close", &JsWriter::close)
        .function("take", &JsWriter::take)
        .property("framesWritten", &JsWriter::frames_written)
        .property("framesCommitted", &JsWriter::frames_committed)
        .property("closed", &JsWriter::closed);
#endif
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
