// mdz._core: the Python extension module wrapping the C++ library.

#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <system_error>

#include "common.hpp"

namespace mdz::py {

namespace {

using namespace nb::literals;
namespace fs = std::filesystem;

// --- compress / decompress ---

PyTrajectory compress_fn(nb::handle positions, nb::handle boxes, float precision, std::int64_t gof_size,
                         std::int64_t predictor_order, std::int64_t zstd_level, bool adaptive, std::int64_t anchor_step,
                         const std::string& codec, bool lossless, bool rigid, nb::handle bodies) {
    const CodecMode mode = parse_codec_mode(codec);
    CompressOptions opts =
        make_options(precision, gof_size, predictor_order, zstd_level, adaptive, anchor_step, lossless);
    opts.codec = mode;
    Shared<std::int32_t> body_of = bodies_from(bodies);
    opts.rigid = rigid || body_of;
    Array<float> pos = as_array<float>(positions);
    check_positions(pos);
    const std::size_t nframes = pos.shape(0), natoms = pos.shape(1);
    std::optional<std::vector<Box>> box;
    if (auto b = as_optional_array<float>(boxes)) {
        check_shape("boxes", *b, {nframes, 3, 3});
        box = boxes_from_flat(b->data(), nframes);
    }
    PyTrajectory t;
    if (opts.rigid) {
        // Rigid bodies need the batch API. compress_positions writes the same
        // columns but cannot take the body assignments.
        if (body_of && body_of->size() != natoms)
            value_error("bodies has length " + std::to_string(body_of->size()) +
                        ", expected natoms = " + std::to_string(natoms));
        FrameBatch batch(natoms, nframes);
        batch.columns.emplace("positions", pos.to_vector());
        batch.box = std::move(box);
        batch.bodies = body_of;
        t.ct = nogil([&] { return compress(batch, opts); });
        return t;
    }
    // Encode directly from the numpy buffer. `pos` keeps it alive while the
    // GIL is released.
    t.ct = nogil([&] { return compress_positions(pos.data(), natoms, nframes, box ? &*box : nullptr, opts); });
    return t;
}

// The stored boxes in nm, as a float32 (nframes, 3, 3) array, or None.
nb::object boxes_of(const CompressedTrajectory& ct) {
    const CompressedColumn* col = ct.column("box");
    if (!col || !col->desc.per_frame || col->desc.width != 9) return nb::none();
    std::vector<float> out = nogil([&] {
        DecodedColumn d = decode_column(ct, *col);
        const double f = ct.length_to_nm();
        std::vector<float> v(d.f64.size());
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(d.f64[i] * f);
        return v;
    });
    const std::size_t n = out.size() / 9;
    return to_numpy(std::move(out), {n, std::size_t(3), std::size_t(3)});
}

nb::tuple decompress_fn(const PyTrajectory& t) {
    const CompressedTrajectory& ct = t.ct;
    std::vector<float> pos = nogil([&] { return decompress_positions(ct); });
    if (ct.variable_atoms) return nb::make_tuple(split_frames(pos, ct.frame_natoms()), boxes_of(ct));
    nb::object p = to_numpy(std::move(pos), {ct.nframes, ct.natoms, std::size_t(3)});
    return nb::make_tuple(p, boxes_of(ct));
}

PyTrajectory compress_batch_fn(nb::handle positions, nb::handle velocities, nb::handle forces, nb::handle charge,
                               nb::handle cell, nb::handle time, nb::handle energy, nb::handle step,
                               nb::handle atom_names, nb::handle atom_types, nb::handle atom_ids, float precision,
                               nb::handle column_precision, std::int64_t gof_size, std::int64_t predictor_order,
                               std::int64_t zstd_level, bool adaptive, std::int64_t anchor_step, bool lossless,
                               nb::handle boxes, bool rigid, nb::handle bodies) {
    CompressOptions opts =
        make_options(precision, gof_size, predictor_order, zstd_level, adaptive, anchor_step, lossless);
    add_column_precision(opts, column_precision);
    opts.rigid = rigid || !bodies.is_none();
    PyTrajectory t;
    FrameArgs args{positions, velocities, forces, charge, cell, time, energy, step, boxes};
    FrameBatch batch = args.to_batch(&t.box_from_cell);
    AtomData atoms;
    atoms.set(atom_names, atom_types, atom_ids, bodies);
    atoms.apply(batch);
    t.ct = nogil([&] { return compress(batch, opts); });
    return t;
}

// --- Files ---

std::uint64_t save_fn(const fs::path& path, const PyTrajectory& t) {
    const std::string p = path_str(path);
    return nogil([&] { return save(p, t.ct); });
}

std::uint64_t save_fn_member(const PyTrajectory& t, const fs::path& path) { return save_fn(path, t); }

PyTrajectory load_fn(const fs::path& path) {
    const std::string p = path_str(path);
    PyTrajectory t;
    t.ct = nogil([&] { return load(p); });
    return t;
}

nb::object decode_boxes_fn(const fs::path& path) {
    const std::string p = path_str(path);
    // Reader::open maps the file and decodes only the box column.
    std::optional<std::vector<Reader::Box32>> boxes = nogil([&] { return Reader::open(p).boxes(); });
    if (!boxes) return nb::none();
    const std::size_t n = boxes->size();
    std::vector<float> flat(n * 9);
    for (std::size_t i = 0; i < n; ++i) std::copy(boxes->at(i).begin(), boxes->at(i).end(), flat.begin() + i * 9);
    return to_numpy(std::move(flat), {n, std::size_t(3), std::size_t(3)});
}

nb::dict info_fn(const fs::path& path) {
    const std::string p = path_str(path);
    CompressedTrajectory ct = nogil([&] { return load(p); });
    nb::dict d;
    d["version"] = 4;
    d["natoms"] = ct.natoms;
    d["nframes"] = ct.nframes;
    d["gof_size"] = ct.gof_size;
    std::size_t ncols = 0;
    nb::list names;
    for (const auto& c : ct.columns)
        if (!is_internal_column(c.desc.name)) {
            names.append(c.desc.name);
            ++ncols;
        }
    d["n_columns"] = ncols;
    d["compressed_size"] = ct.compressed_size();
    d["columns"] = names;
    d["variable_atoms"] = ct.variable_atoms;
    d["rigid_bodies"] = count_rigid_bodies(ct);
    return d;
}

// --- convert ---

constexpr const char* kWritable[] = {"xyz", "extxyz", "pdb", "lammps", "dcd", "xtc", "npy", "mdz"};

// Refuse to write over the input, or over an existing file unless overwrite
// is set. These are the same rules as `mdz convert` without --force.
void ensure_writable(const fs::path& input, const fs::path& output, bool overwrite) {
    std::error_code ec;
    if (fs::exists(input, ec) && fs::exists(output, ec) && fs::equivalent(input, output, ec) && !ec)
        value_error("input and output are the same file: " + path_str(output));
    if (fs::exists(output, ec) && !overwrite)
        raise(PyExc_FileExistsError, path_str(output) + " already exists (pass overwrite=True to replace it)");
}

// Report every read or write failure as OSError, whatever its kind.
template <class F>
auto as_os_error(F&& f) -> decltype(f()) {
    try {
        return f();
    } catch (const Error& e) {
        throw Error(ErrorKind::Io, e.what());
    }
}

void convert_fn(const fs::path& input, const fs::path& output, bool overwrite) {
    ensure_writable(input, output, overwrite);
    const std::string in = path_str(input), out = path_str(output);
    const std::string in_fmt(detect_format(in));
    const std::string out_fmt(detect_format(out));
    // Check the output format before reading the input or issuing warnings.
    if (std::find(std::begin(kWritable), std::end(kWritable), out_fmt) == std::end(kWritable)) {
        std::string list;
        for (const char* f : kWritable) list += (list.empty() ? "." : ", .") + std::string(f);
        value_error("unsupported output format '" + (out_fmt.empty() ? std::string("unknown") : out_fmt) + "' for " +
                    out + " (expected one of: " + list + ")");
    }

    std::optional<FrameBatch> batch = nogil([&] { return as_os_error([&] { return open_reader(in)->read_all(); }); });
    if (!batch || batch->nframes == 0) value_error("no frames in " + in);

    // Warn about data the output format cannot store, as the CLI does.
    std::vector<std::string> dropped = check_data_loss(*batch, out_fmt);
    if (!dropped.empty()) {
        std::string what;
        for (const auto& d : dropped) what += (what.empty() ? "" : ", ") + (d == "box" ? "box (unit cell)" : d);
        const std::string msg =
            "output format '" + out_fmt + "' cannot carry: " + what + ". This data will be dropped.";
        if (PyErr_WarnEx(PyExc_UserWarning, msg.c_str(), 1) < 0) throw nb::python_error();
    }

    nogil([&] {
        as_os_error([&] {
            if (out_fmt == "mdz") {
                CompressOptions opts;  // same defaults as the CLI
                opts.source_format = in_fmt;
                save(out, compress(*batch, opts));
            } else {
                const FrameBatch& b = *batch;
                std::optional<FrameBatch> internal;
                if (out_fmt == "npy" && b.units) internal = b.in_internal_units();
                auto w = open_writer(out, b.natoms, out_fmt);
                w->write_batch(internal ? *internal : b);
                w->finish();
            }
            return 0;
        });
    });
}

// --- Docstrings ---

constexpr const char* kCompressDoc = R"doc(Compress an MD trajectory.

With the default precision the maximum error is 0.0005 nm (0.005 A), the
same worst case as a GROMACS XTC file at default settings. With
adaptive=True (the default), the maximum error is precision * 4.

Parameters
----------
positions : array_like, shape (nframes, natoms, 3)
    Coordinates in nm. Any float dtype and memory layout is accepted.
    Values are converted to float32.
boxes : array_like, shape (nframes, 3, 3), optional
    Per-frame row-major box matrices in nm, stored losslessly.
precision : float, default 0.000125
    Base quantization step in nm. The maximum error is precision * 4 when
    adaptive is True.
gof_size : int, default 20
    Frames per group (1-65535). Groups are the unit of random access and
    parallelism.
predictor_order : int, default 0
    Polynomial predictor order, 0-4. 0 chooses an order per group.
zstd_level : int, default 3
    Zstd compression level, 1-22.
adaptive : bool, default True
    Spend more bits on atoms that move more.
anchor_step : int, default 1
    Spacing of anchor frames. 1 disables B-frames.
codec : {"auto", "temporal", "intra", "lossless"}, default "auto"
lossless : bool, default False
    Store every coordinate exactly. ``precision``, ``codec``, ``adaptive``
    and ``anchor_step`` are ignored, since they only control rounding.
    Expect a ratio of about 1.3-2.6x, compared with 10-40x for the lossy
    codecs.
rigid : bool, default False
    Also encode the positions as rigid bodies, with a centre and an
    orientation per body and frame plus small residuals. For each group,
    the smaller encoding is kept. The error bound is unchanged. Bodies are
    taken from ``bodies`` if given. Otherwise they are detected as atoms
    within 0.16 nm of each other whose distances stay within 0.002 nm.
    Ignored when ``lossless`` is True. Most useful for molecules with many
    rigidly connected atoms (rings, rigid fragments). Roughly breaks even
    for water.
bodies : array_like of int, shape (natoms,), optional
    Rigid body index of each atom, -1 for atoms in no body. Implies
    ``rigid=True``. Stored in the metadata (``metadata()["bodies"]``).

Returns
-------
Trajectory

Raises
------
ValueError
    If an array has the wrong shape, a parameter is out of range, or a
    value cannot be stored (NaN, Inf, or out of range).)doc";

constexpr const char* kDecompressDoc = R"doc(Decompress a trajectory.

Returns a tuple ``(positions, boxes)``:
  - ``positions`` is a float32 array of shape ``(nframes, natoms, 3)``, in nm.
    If the atom count varies, it is a list with one ``(natoms_f, 3)`` array
    per frame.
  - ``boxes`` is a float32 array of shape ``(nframes, 3, 3)``, in nm, or
    ``None`` if the trajectory stores no box.)doc";

constexpr const char* kCompressBatchDoc = R"doc(Compress a multi-column trajectory.

Columns other than ``positions`` are optional. Any array-like is accepted
and is converted to the dtype below, whatever its memory layout. Shapes
must match exactly:
  * ``positions`` / ``velocities`` / ``forces`` - ``(nframes, natoms, 3)``, float32
  * ``charge`` - ``(nframes, natoms)``, float32
  * ``cell`` - ``(nframes, 6)``, float64 (A, gamma, B, beta, alpha, C; nm and
    degrees), stored as the equivalent box matrix
  * ``boxes`` - ``(nframes, 3, 3)``, float32 row-major box matrices (nm),
    stored losslessly as a ``box`` column (taking precedence over ``cell``)
  * ``time`` / ``energy`` - ``(nframes,)``, float64
  * ``step`` - ``(nframes,)``, integer, stored exactly as int64
  * ``atom_names`` - list[str] of length natoms
  * ``atom_types`` / ``atom_ids`` - list[int] of length natoms

``precision`` is in each column's native units. ``column_precision``
(dict[str, float]) overrides it for individual columns. With
``lossless=True``, every float column is stored exactly and both are
ignored.

``rigid`` and ``bodies`` model rigid bodies in the positions, as in
:func:`compress`.

The tuning parameters (``gof_size=20``, ``predictor_order=0``,
``zstd_level=3``, ``adaptive=True``, ``anchor_step=1``) have the same
defaults and ranges as for :func:`compress` and the ``mdz compress`` CLI.)doc";

constexpr const char* kConvertDoc = R"doc(Convert between trajectory formats.

Supported formats are .dcd, .extxyz, .xyz, .pdb, .lammpstrj, .xtc, .npy
(output only) and .mdz. Writing .mdz compresses with the default settings
of :func:`compress`. Use :func:`compress_batch` to control the precision.

Emits a ``UserWarning`` if the output format cannot store all the data in
the input, for example velocities when writing .xyz.

Parameters
----------
input, output : str or os.PathLike
    Formats are inferred from the extensions.
overwrite : bool, default False
    Replace ``output`` if it exists. Otherwise an existing output raises
    ``FileExistsError``, as ``mdz convert`` does without ``--force``.

Raises
------
FileExistsError
    ``output`` exists and ``overwrite`` is False.
ValueError
    ``input`` and ``output`` are the same file, the output format is not
    supported, or the input is empty.
OSError
    Reading or writing failed.)doc";

constexpr const char* kDecodeBoxesDoc = R"doc(Decode only the per-frame 3x3 box matrices from a .mdz file.

Returns an ``(nframes, 3, 3)`` float32 array in nm, or ``None`` if the file
stores no box. The file is memory-mapped and only the box column is
decompressed. The positions are not decoded. For a file that is already
open, ``mdz.open(path).cells`` returns the same data at no extra cost.)doc";

constexpr const char* kSetThreadsDoc = R"doc(Set the number of worker threads used for compression, decompression
and reads. ``0`` restores the default, which is ``MDZ_NUM_THREADS`` if set
and otherwise one thread per logical core. May be called at any time.)doc";

}  // namespace

// --- Trajectory ---

void bind_trajectory(nb::module_& m) {
    nb::class_<PyTrajectory>(m, "Trajectory",
                             "A compressed trajectory, holding the columns, their descriptors and the\n"
                             "metadata.\n\n"
                             "Returned by :func:`compress`, :func:`compress_batch` and :func:`load`.\n"
                             "Write it to a file with :meth:`save`.")
        .def_prop_ro(
            "natoms", [](const PyTrajectory& t) { return t.ct.natoms; },
            "Atoms per frame, or 0 if the count varies (see ``variable_atoms``).")
        .def_prop_ro("nframes", [](const PyTrajectory& t) { return t.ct.nframes; })
        .def_prop_ro(
            "variable_atoms", [](const PyTrajectory& t) { return t.ct.variable_atoms; },
            "Whether the atom count varies from frame to frame.")
        .def_prop_ro(
            "frame_natoms", [](const PyTrajectory& t) { return frame_natoms_array(t.ct); },
            "Number of atoms in each frame, as an (nframes,) int64 array.")
        .def_prop_ro(
            "rigid_bodies", [](const PyTrajectory& t) { return count_rigid_bodies(t.ct); },
            "Number of rigid bodies recorded in the metadata (0 if none).")
        .def_prop_ro("gof_size", [](const PyTrajectory& t) { return t.ct.gof_size; })
        .def_prop_ro("compressed_size", [](const PyTrajectory& t) { return t.ct.compressed_size(); })
        .def_prop_ro(
            "precision",
            [](const PyTrajectory& t) {
                const CompressedColumn* c = t.ct.column("positions");
                return c ? c->desc.precision : 0.0f;
            },
            "Quantization step of the positions column, in nm, or 0.0 if lossless.")
        .def_prop_ro(
            "ratio",
            [](const PyTrajectory& t) {
                return static_cast<double>(t.ct.raw_size()) /
                       static_cast<double>(std::max<std::size_t>(t.ct.compressed_size(), 1));
            },
            "Ratio of uncompressed to compressed size, over all stored columns.")
        .def_prop_ro(
            "column_names",
            [](const PyTrajectory& t) {
                std::vector<std::string> names;
                for (const auto& c : t.ct.columns) {
                    if (is_internal_column(c.desc.name)) continue;
                    if (c.desc.name == "box" && t.box_from_cell) {
                        names.push_back("cell");
                        continue;
                    }
                    names.push_back(c.desc.name);
                    if (c.desc.name == "box") names.push_back("cell");
                }
                return names;
            },
            "Names of all columns that can be read from this trajectory. ``cell`` is\n"
            "derived from ``box``.")
        .def(
            "columns",
            [](const PyTrajectory& t) {
                nb::list out;
                for (const auto& c : t.ct.columns) {
                    if (is_internal_column(c.desc.name)) continue;
                    nb::dict d;
                    d["name"] = c.desc.name;
                    const char* dt = "F32";
                    switch (c.desc.dtype) {
                        case Dtype::F32:
                            dt = "F32";
                            break;
                        case Dtype::F64:
                            dt = "F64";
                            break;
                        case Dtype::I32:
                            dt = "I32";
                            break;
                        case Dtype::U32:
                            dt = "U32";
                            break;
                        case Dtype::U8:
                            dt = "U8";
                            break;
                        case Dtype::I64:
                            dt = "I64";
                            break;
                    }
                    d["dtype"] = dt;
                    d["width"] = c.desc.width;
                    d["per_frame"] = c.desc.per_frame;
                    d["lossy"] = c.desc.lossy;
                    d["precision"] = c.desc.precision;
                    d["unit"] = c.desc.unit;
                    out.append(d);
                }
                return out;
            },
            "Column descriptors (name, dtype, width, per_frame, lossy, precision, unit).")
        .def(
            "metadata", [](const PyTrajectory& t) { return metadata_to_dict(t.ct); },
            "Metadata dict (atom_names, atom_types, atom_ids, residues, bodies, ...).\n"
            "Empty if the trajectory has none.")
        .def(
            "read_column", [](const PyTrajectory& t, const std::string& name) { return read_named_column(t.ct, name); },
            "name"_a,
            "Decode a column by name. Returns a numpy array with shape:\n"
            "  - per-atom width-3 (positions, velocities, forces) -> (nframes, natoms, 3)\n"
            "  - per-atom scalar (charge, ...) -> (nframes, natoms)\n"
            "  - per-frame ``box`` -> (nframes, 3, 3) row-major matrices, float64\n"
            "  - ``cell`` -> (nframes, 6) (A, gamma, B, beta, alpha, C), float64,\n"
            "    derived from ``box``\n"
            "  - per-frame scalar (time, energy -> float64; step -> int64) -> (nframes,)\n\n"
            "When the atom count varies, per-atom columns are flat, with each frame's\n"
            "atoms following the previous frame's: (total_atoms, 3) or\n"
            "(total_atoms,). Split them with ``np.split(a, np.cumsum(frame_natoms)[:-1])``.\n"
            "The atom ids are available from ``MdzReader.atom_ids``.\n\n"
            "Raises KeyError if there is no such column.")
        .def("save", &save_fn_member, "path"_a, "Save this trajectory to a .mdz file. Returns the bytes written.")
        .def("__repr__", [](const PyTrajectory& t) {
            std::size_t ncols = 0;
            for (const auto& c : t.ct.columns) ncols += !is_internal_column(c.desc.name);
            return "Trajectory(natoms=" + (t.ct.variable_atoms ? std::string("varying") : std::to_string(t.ct.natoms)) +
                   ", nframes=" + std::to_string(t.ct.nframes) + ", columns=" + std::to_string(ncols) + ")";
        });
}

void bind_functions(nb::module_& m) {
    m.def("compress", &compress_fn, "positions"_a, "boxes"_a = nb::none(), "precision"_a = kDefaultPrecision,
          "gof_size"_a = kDefaultGofSize, "predictor_order"_a = 0, "zstd_level"_a = 3, "adaptive"_a = true,
          "anchor_step"_a = 1, "codec"_a = "auto", "lossless"_a = false, "rigid"_a = false, "bodies"_a = nb::none(),
          kCompressDoc);
    m.def("decompress", &decompress_fn, "ct"_a, kDecompressDoc);
    m.def("compress_batch", &compress_batch_fn, "positions"_a, "velocities"_a = nb::none(), "forces"_a = nb::none(),
          "charge"_a = nb::none(), "cell"_a = nb::none(), "time"_a = nb::none(), "energy"_a = nb::none(),
          "step"_a = nb::none(), "atom_names"_a = nb::none(), "atom_types"_a = nb::none(), "atom_ids"_a = nb::none(),
          "precision"_a = kDefaultPrecision, "column_precision"_a = nb::none(), "gof_size"_a = kDefaultGofSize,
          "predictor_order"_a = 0, "zstd_level"_a = 3, "adaptive"_a = true, "anchor_step"_a = 1, "lossless"_a = false,
          "boxes"_a = nb::none(), "rigid"_a = false, "bodies"_a = nb::none(), kCompressBatchDoc);
    m.def("save", &save_fn, "path"_a, "ct"_a,
          "Save a compressed trajectory to a .mdz file. Returns the bytes written.");
    m.def("load", &load_fn, "path"_a, "Load a compressed trajectory from a .mdz file.");
    m.def("decode_boxes", &decode_boxes_fn, "path"_a, kDecodeBoxesDoc);
    m.def("info", &info_fn, "path"_a,
          "Summary of a .mdz file as a dict with keys version, natoms (0 if the\n"
          "count varies), nframes, gof_size, n_columns, columns, compressed_size,\n"
          "variable_atoms (bool) and rigid_bodies (number of bodies recorded in\n"
          "the metadata).");
    m.def("convert", &convert_fn, "input"_a, "output"_a, "overwrite"_a = false, kConvertDoc);
    m.def("set_num_threads", [](unsigned n) { set_num_threads(n); }, "n"_a, kSetThreadsDoc);
}

}  // namespace mdz::py

NB_MODULE(_core, m) {
    using namespace mdz::py;
    m.doc() = "mdz - high-ratio MD trajectory compression (C++ core)";
    register_error_translator();
    bind_trajectory(m);
    bind_reader(m);
    bind_writer(m);
    bind_functions(m);
    m.attr("__version__") = std::string(mdz::version());
}
