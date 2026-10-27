#include "common.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <map>

namespace mdz::py {

void raise(PyObject* type, const std::string& msg) {
    PyErr_SetString(type, msg.c_str());
    throw nb::python_error();
}

void register_error_translator() {
    nb::register_exception_translator([](const std::exception_ptr& p, void*) {
        try {
            std::rethrow_exception(p);
        } catch (const Error& e) {
            PyObject* type = PyExc_OSError;
            switch (e.kind()) {
                case ErrorKind::InvalidInput:
                    type = PyExc_ValueError;
                    break;
                case ErrorKind::OutOfRange:
                    type = PyExc_IndexError;
                    break;
                case ErrorKind::Io:
                case ErrorKind::Corrupt:
                    type = PyExc_OSError;
                    break;
            }
            PyErr_SetString(type, e.what());
        }
    });
}

// --- numpy input ---

PyObject* numpy_attr(const char* name) {
    nb::object o = nb::module_::import_("numpy").attr(name);
    return o.release().ptr();
}

nb::object as_numpy(nb::handle obj, const char* dtype) {
    // Use asarray rather than ascontiguousarray, which turns 0-d input into
    // shape (1,). Keeping a scalar 0-d makes it fail the shape check.
    // The function is looked up once and never released, because a static
    // nb::object would be decref'd after the interpreter has shut down.
    static PyObject* asarray = numpy_attr("asarray");
    return nb::handle(asarray)(obj, nb::arg("dtype") = dtype, nb::arg("order") = "C");
}

std::string fmt_shape(const std::vector<std::size_t>& shape) {
    if (shape.size() == 1) return "(" + std::to_string(shape[0]) + ",)";
    std::string s = "(";
    for (std::size_t i = 0; i < shape.size(); ++i) s += (i ? ", " : "") + std::to_string(shape[i]);
    return s + ")";
}

void check_positions(const Array<float>& a) {
    const auto s = a.shape();
    if (s.size() != 3 || s[2] != 3) value_error("positions must have shape (nframes, natoms, 3), got " + fmt_shape(s));
    if (s[0] == 0 || s[1] == 0)
        value_error("positions must have at least one frame and one atom, got shape " + fmt_shape(s));
}

std::vector<Box> boxes_from_flat(const float* flat, std::size_t nframes) {
    for (std::size_t i = 0; i < nframes * 9; ++i)
        if (!std::isfinite(flat[i])) {
            char buf[64];
            std::snprintf(buf, sizeof buf, "%g", static_cast<double>(flat[i]));
            value_error(std::string("boxes: value ") + buf + " is not finite");
        }
    std::vector<Box> out(nframes);
    for (std::size_t f = 0; f < nframes; ++f)
        for (int k = 0; k < 9; ++k) out[f][k] = static_cast<double>(flat[f * 9 + k]);
    return out;
}

// --- numpy output ---

nb::object column_to_numpy(DecodedColumn&& d, const ColumnDesc& desc, std::size_t natoms, std::size_t nframes,
                           bool variable) {
    const std::size_t w = desc.width;
    // A per-frame column may cover fewer frames than the trajectory.
    if (desc.per_frame) nframes = d.size() / (w ? w : 1);
    std::vector<std::size_t> shape;
    if (variable && !desc.per_frame) {
        const std::size_t atoms = d.size() / (w ? w : 1);
        if (w == 1)
            shape = {atoms};
        else
            shape = {atoms, w};
    } else if (desc.per_frame) {
        if (w == 1)
            shape = {nframes};
        else if (w == 9 && desc.name == "box")
            shape = {nframes, 3, 3};
        else
            shape = {nframes, w};
    } else {
        if (w == 1)
            shape = {nframes, natoms};
        else
            shape = {nframes, natoms, w};
    }
    std::size_t expected = 1;
    for (std::size_t s : shape) expected *= s;
    if (d.size() != expected)
        value_error("decoded " + std::to_string(d.size()) + " values, expected " + std::to_string(expected) +
                    " for shape " + fmt_shape(shape));
    switch (d.dtype) {
        case Dtype::F64:
            return to_numpy(std::move(d.f64), shape);
        case Dtype::I32:
            return to_numpy(std::move(d.i32), shape);
        case Dtype::I64:
            return to_numpy(std::move(d.i64), shape);
        default:
            return to_numpy(std::move(d.f32), shape);
    }
}

nb::list split_frames(const std::vector<float>& flat, const std::vector<std::size_t>& counts) {
    nb::list out;
    std::size_t off = 0;
    for (std::size_t n : counts) {
        if (off + n * 3 > flat.size()) value_error("decoded fewer positions than the frames' atom counts");
        out.append(to_numpy(std::vector<float>(flat.begin() + static_cast<std::ptrdiff_t>(off),
                                               flat.begin() + static_cast<std::ptrdiff_t>(off + n * 3)),
                            {n, std::size_t(3)}));
        off += n * 3;
    }
    return out;
}

nb::object cell_from_box(const DecodedColumn& box) {
    const std::size_t n = box.f64.size() / 9;
    std::vector<double> out(n * 6);
    for (std::size_t i = 0; i < n; ++i) {
        Box m;
        for (int k = 0; k < 9; ++k) m[k] = box.f64[i * 9 + k];
        const CellParams p = box_to_cell_params(m);
        for (int k = 0; k < 6; ++k) out[i * 6 + k] = p[k];
    }
    return to_numpy(std::move(out), {n, std::size_t(6)});
}

nb::object read_named_column(const CompressedTrajectory& ct, const std::string& name) {
    const CompressedColumn* col = ct.column(name);
    const bool derived_cell = !col && name == "cell";
    if (derived_cell) col = ct.column("box");
    if (!col || is_internal_column(name)) raise(PyExc_KeyError, "no column named '" + name + "'");
    DecodedColumn d = nogil([&] { return decode_column(ct, *col); });
    if (derived_cell) return cell_from_box(d);
    return column_to_numpy(std::move(d), col->desc, ct.natoms, ct.gofs.nframes(), ct.variable_atoms);
}

nb::dict metadata_to_dict(const CompressedTrajectory& ct) {
    nb::dict d;
    Metadata m;
    try {
        m = ct.decoded_metadata();
    } catch (const Error&) {
        return d;  // unreadable metadata is treated as empty
    }
    if (m.atom_names) d["atom_names"] = nb::cast(*m.atom_names);
    if (m.elements) d["elements"] = nb::cast(*m.elements);
    if (m.atom_types) d["atom_types"] = nb::cast(*m.atom_types);
    if (m.atom_ids) d["atom_ids"] = nb::cast(*m.atom_ids);
    if (m.residues) d["residues"] = nb::cast(*m.residues);
    if (m.residue_ids) d["residue_ids"] = nb::cast(*m.residue_ids);
    if (m.chain_ids) d["chain_ids"] = nb::cast(*m.chain_ids);
    if (m.hetatm) {
        nb::list flags;
        for (std::int32_t v : *m.hetatm) flags.append(nb::bool_(v != 0));
        d["hetatm"] = flags;
    }
    if (m.bodies) d["bodies"] = nb::cast(*m.bodies);
    return d;
}

std::size_t count_rigid_bodies(const CompressedTrajectory& ct) {
    std::optional<std::vector<std::int32_t>> bodies;
    try {
        bodies = ct.decoded_metadata().bodies;
    } catch (const Error&) {
        return 0;
    }
    if (!bodies) return 0;
    std::map<std::int32_t, std::size_t> size;
    for (std::int32_t b : *bodies)
        if (b >= 0) ++size[b];
    std::size_t n = 0;
    for (const auto& [b, k] : size) n += k >= 2;
    return n;
}

bool is_internal_column(std::string_view name) { return name == kMembershipColumn; }

nb::object frame_natoms_array(const CompressedTrajectory& ct) {
    std::vector<std::size_t> counts = nogil([&] { return ct.frame_natoms(); });
    std::vector<std::int64_t> out(counts.begin(), counts.end());
    const std::size_t n = out.size();
    return to_numpy(std::move(out), {n});
}

std::string path_str(const std::filesystem::path& p) { return p.string(); }

// --- options ---

CompressOptions make_options(float precision, std::int64_t gof_size, std::int64_t predictor_order,
                             std::int64_t zstd_level, bool adaptive, std::int64_t anchor_step, bool lossless) {
    // The same ranges as the `mdz compress` CLI. Arguments are int64 so that a
    // negative value raises ValueError here rather than OverflowError during
    // conversion.
    if (predictor_order < 0 || predictor_order > 4)
        value_error("predictor_order must be between 0 and 4 (0 = choose per group), got " +
                    std::to_string(predictor_order));
    if (zstd_level < 1 || zstd_level > 22)
        value_error("zstd_level must be between 1 and 22, got " + std::to_string(zstd_level));
    if (gof_size < 1 || gof_size > 65535)
        value_error("gof_size must be between 1 and 65535, got " + std::to_string(gof_size));
    if (anchor_step < 1 || anchor_step > 255)
        value_error("anchor_step must be between 1 and 255, got " + std::to_string(anchor_step));
    CompressOptions o;
    o.precision = precision;
    o.gof_size = static_cast<std::uint32_t>(gof_size);
    o.predictor_order = static_cast<std::uint32_t>(predictor_order);
    o.zstd_level = static_cast<int>(zstd_level);
    o.adaptive = adaptive;
    o.anchor_step = static_cast<std::uint32_t>(anchor_step);
    o.lossless = lossless;
    // Checks that involve several parameters (precision, anchors per group).
    // As in the CLI, lossless mode ignores precision and anchor_step.
    o.validate();
    return o;
}

void add_column_precision(CompressOptions& opts, nb::handle column_precision) {
    if (column_precision.is_none()) return;
    for (auto [k, v] : nb::cast<nb::dict>(column_precision))
        opts.column_precision[nb::cast<std::string>(k)] = nb::cast<float>(v);
}

// --- atoms ---

Shared<std::int32_t> bodies_from(nb::handle bodies) {
    if (bodies.is_none()) return nullptr;
    // Convert through int64 so that a value outside the int32 range raises an
    // error instead of wrapping.
    Array<std::int64_t> a = as_array<std::int64_t>(bodies);
    if (a.ndim() != 1) value_error("bodies must have shape (natoms,), got " + fmt_shape(a.shape()));
    std::vector<std::int32_t> v(a.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        const std::int64_t b = a.data()[i];
        if (b < -1 || b > std::numeric_limits<std::int32_t>::max())
            value_error("bodies: value " + std::to_string(b) + " is not a body index (>= 0) or -1 (no body)");
        v[i] = static_cast<std::int32_t>(b);
    }
    return share(std::move(v));
}

void AtomData::set(nb::handle names_, nb::handle types_, nb::handle ids_, nb::handle bodies_) {
    if (!names_.is_none()) names = share(nb::cast<std::vector<std::string>>(names_));
    if (!types_.is_none()) types = share(nb::cast<std::vector<std::int32_t>>(types_));
    if (!ids_.is_none()) ids = share(nb::cast<std::vector<std::int32_t>>(ids_));
    if (!bodies_.is_none()) bodies = bodies_from(bodies_);
}

void AtomData::apply(FrameBatch& b) const {
    const std::size_t natoms = b.natoms;
    auto check = [&](const char* name, const auto& v) {
        if (v && v->size() != natoms)
            value_error(std::string(name) + " has length " + std::to_string(v->size()) +
                        ", expected natoms = " + std::to_string(natoms));
    };
    check("atom_names", names);
    check("atom_types", types);
    check("atom_ids", ids);
    check("bodies", bodies);
    b.atom_names = names;
    b.elements = elements;
    b.atom_types = types;
    b.atom_ids = ids;
    b.residue_names = residue_names;
    b.residue_ids = residue_ids;
    b.chain_ids = chain_ids;
    b.hetatm = hetatm;
    b.bodies = bodies;
    b.attributes = attributes;
}

// --- frames ---

FrameBatch FrameArgs::to_batch(bool* from_cell) const {
    Array<float> pos = as_array<float>(positions);
    check_positions(pos);
    const std::size_t nframes = pos.shape(0), natoms = pos.shape(1);
    FrameBatch b(natoms, nframes);
    b.columns.emplace("positions", pos.to_vector());

    const std::pair<const char*, nb::handle> xyz[] = {{"velocities", velocities}, {"forces", forces}};
    for (const auto& [name, obj] : xyz)
        if (auto a = as_optional_array<float>(obj)) {
            check_shape(name, *a, {nframes, natoms, 3});
            b.columns.emplace(name, a->to_vector());
        }
    if (auto a = as_optional_array<float>(charge)) {
        check_shape("charge", *a, {nframes, natoms});
        b.columns.emplace("charge", a->to_vector());
    }

    // The 6-parameter cell (a, gamma, b, beta, alpha, c) is stored as the
    // equivalent box matrix. If a full `boxes` array is also given, it takes
    // precedence.
    std::optional<std::vector<Box>> cell_boxes;
    if (auto a = as_optional_array<double>(cell)) {
        check_shape("cell", *a, {nframes, 6});
        std::vector<Box> v(nframes);
        for (std::size_t f = 0; f < nframes; ++f) {
            CellParams p;
            for (int k = 0; k < 6; ++k) {
                p[k] = a->data()[f * 6 + k];
                if (!std::isfinite(p[k])) value_error("cell: every value must be finite");
            }
            v[f] = cell_params_to_box(p);
        }
        cell_boxes = std::move(v);
    }
    if (auto a = as_optional_array<float>(boxes)) {
        check_shape("boxes", *a, {nframes, 3, 3});
        b.box = boxes_from_flat(a->data(), nframes);
    } else if (cell_boxes) {
        b.box = std::move(cell_boxes);
        if (from_cell) *from_cell = true;
    }

    const std::pair<const char*, nb::handle> props[] = {{"time", time}, {"energy", energy}};
    for (const auto& [name, obj] : props)
        if (auto a = as_optional_array<double>(obj)) {
            check_shape(name, *a, {nframes});
            b.frame_props.emplace(name, a->to_vector());
        }
    if (auto a = as_optional_array<std::int64_t>(step)) {
        check_shape("step", *a, {nframes});
        b.int_frame_props.emplace("step", a->to_vector());
    }
    return b;
}

}  // namespace mdz::py
