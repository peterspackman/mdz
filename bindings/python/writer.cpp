// mdz.Writer: write a .mdz file incrementally, as frames arrive.

#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>

#include <limits>

#include "common.hpp"

namespace mdz::py {

namespace {

using namespace nb::literals;

struct PyWriter {
    std::optional<Writer> w;  // empty once closed
    AtomData atoms;
    std::string path;
    std::uint64_t closed_written = 0, closed_committed = 0;
    // `variable` is set by variable_atoms, or when appending to a file with a
    // varying atom count. Otherwise `natoms` is fixed by the first frame, or
    // by the file being appended to, and is 0 until then.
    bool variable = false;
    std::size_t natoms = 0;
    // Set while a call runs without the GIL, so that a second thread gets an
    // error instead of racing on the writer.
    bool busy = false;

    Writer& open() {
        if (!w) value_error("I/O operation on a closed mdz.Writer");
        if (busy) raise(PyExc_RuntimeError, "mdz.Writer is already in use by another thread");
        return *w;
    }

    template <class F>
    void run(F&& f) {
        busy = true;
        try {
            nogil(f);
        } catch (...) {
            busy = false;
            throw;
        }
        busy = false;
    }

    void close() {
        if (!w) return;
        if (busy) raise(PyExc_RuntimeError, "mdz.Writer is already in use by another thread");
        // The writer counts as closed from here on, even if the close fails.
        Writer local = std::move(*w);
        w.reset();
        closed_written = closed_committed = local.frames_written();
        nogil([&] {
            local.close();
            return 0;
        });
    }
};

void init_writer(PyWriter* self, const std::filesystem::path& path, nb::handle atom_names, nb::handle atom_types,
                 nb::handle atom_ids, float precision, nb::handle column_precision, std::int64_t gof_size,
                 std::int64_t predictor_order, std::int64_t zstd_level, bool adaptive, std::int64_t anchor_step,
                 const std::string& codec, bool lossless, bool append, bool variable_atoms, bool rigid,
                 nb::handle bodies) {
    const CodecMode mode = parse_codec_mode(codec);
    CompressOptions opts =
        make_options(precision, gof_size, predictor_order, zstd_level, adaptive, anchor_step, lossless);
    opts.codec = mode;
    add_column_precision(opts, column_precision);
    opts.variable_atoms = variable_atoms;
    opts.rigid = rigid || !bodies.is_none();
    opts.validate();
    if (variable_atoms) {
        if (opts.rigid)
            value_error(
                "rigid bodies need a fixed atom count: rigid and bodies cannot be used with "
                "variable_atoms=True");
        if (!atom_names.is_none() || !atom_types.is_none() || !atom_ids.is_none())
            value_error(
                "with variable_atoms=True the atoms can change from frame to frame, so there is no static "
                "atom_names, atom_types or atom_ids: pass each frame's ids to write(..., ids=...)");
    }
    const std::string p = path_str(path);

    AtomData atoms;
    std::optional<Writer> w;
    bool variable = variable_atoms;
    std::size_t natoms = 0;
    if (append) {
        // New frames must carry the file's static metadata. Load it from the
        // file so the caller does not have to pass it again.
        Metadata meta = nogil([&] { return load(p).decoded_metadata(); });
        if (meta.attributes.count(kUnitsAttr))
            value_error(p +
                        " stores a LAMMPS dump in its native units; mdz.Writer writes nm/ps and cannot append to it");
        opts.source_format = meta.source_format;
        w.emplace(nogil([&] { return Writer::append(p, opts); }));
        if (const FrameBatch* b = w->appended_metadata()) {
            if (b->variable()) {
                variable = true;  // the file's setting takes precedence
            } else if (variable_atoms) {
                value_error(p + " has a fixed atom count; it cannot be continued with variable_atoms=True");
            } else {
                natoms = b->natoms;
            }
            atoms.bodies = b->bodies;
            atoms.names = b->atom_names;
            atoms.elements = b->elements;
            atoms.types = b->atom_types;
            atoms.ids = b->atom_ids;
            atoms.residue_names = b->residue_names;
            atoms.residue_ids = b->residue_ids;
            atoms.chain_ids = b->chain_ids;
            atoms.hetatm = b->hetatm;
            atoms.attributes = b->attributes;
        }
    } else {
        w.emplace(nogil([&] { return Writer::create(p, opts); }));
    }
    atoms.set(atom_names, atom_types, atom_ids, bodies);
    new (self) PyWriter{std::move(w), std::move(atoms), p};
    self->variable = variable;
    self->natoms = natoms;
}

// `obj` with a leading frame axis added if a single frame was passed without one.
nb::object frame_arg(nb::handle obj, bool single) {
    if (obj.is_none()) return nb::none();
    if (!single) return nb::borrow(obj);
    static PyObject* expand = numpy_attr("expand_dims");
    return nb::handle(expand)(obj, 0);
}

void write_frames(PyWriter& self, nb::handle positions, nb::handle velocities, nb::handle forces, nb::handle charge,
                  nb::handle cell, nb::handle time, nb::handle energy, nb::handle step, nb::handle boxes,
                  nb::handle ids) {
    self.open();
    static PyObject* ndim = numpy_attr("ndim");
    const bool single = nb::cast<std::size_t>(nb::handle(ndim)(positions)) == 2;
    nb::object pos = frame_arg(positions, single), vel = frame_arg(velocities, single), frc = frame_arg(forces, single),
               chg = frame_arg(charge, single), cel = frame_arg(cell, single), tim = frame_arg(time, single),
               ene = frame_arg(energy, single), stp = frame_arg(step, single), box = frame_arg(boxes, single);
    FrameBatch batch = FrameArgs{pos, vel, frc, chg, cel, tim, ene, stp, box}.to_batch();
    if (!self.variable && self.natoms != 0 && batch.natoms != self.natoms)
        value_error("frame has " + std::to_string(batch.natoms) + " atoms; earlier frames had " +
                    std::to_string(self.natoms) +
                    ". To let the atom count change, open the file with mdz.Writer(..., variable_atoms=True)");
    self.atoms.apply(batch);
    if (!ids.is_none()) {
        if (!self.variable)
            value_error(
                "ids identify the atoms of a frame when the atom count varies: open the file with "
                "mdz.Writer(..., variable_atoms=True), or give static atom_ids to mdz.Writer");
        Array<std::int64_t> a = as_array<std::int64_t>(ids);
        check_shape("ids", a, {batch.natoms});
        std::vector<std::int32_t> v(a.size());
        for (std::size_t i = 0; i < v.size(); ++i) {
            const std::int64_t id = a.data()[i];
            if (id < std::numeric_limits<std::int32_t>::min() || id > std::numeric_limits<std::int32_t>::max())
                value_error("ids: atom id " + std::to_string(id) + " does not fit in 32 bits");
            v[i] = static_cast<std::int32_t>(id);
        }
        batch.atom_ids = share(std::move(v));
    }
    Writer& w = self.open();
    // Complete groups are encoded here, in parallel.
    self.run([&] {
        w.write(batch);
        return 0;
    });
    if (!self.variable) self.natoms = batch.natoms;
}

constexpr const char* kWriterDoc = R"doc(Write a .mdz file incrementally, as frames arrive.

Frames are buffered until a complete group of ``gof_size`` frames is
ready. The group is then encoded and written to the OS. If the process
dies, every complete group can still be read from the file. :meth:`close`,
or leaving the ``with`` block, encodes the remaining frames and writes the
footer.

>>> with mdz.Writer("traj.mdz", atom_names=names) as w:
...     for step, x in simulation():
...         w.write(x, step=step)       # one frame: (natoms, 3) in nm
...         if step % 10000 == 0:
...             w.flush()               # checkpoint: all frames so far are safe

Parameters
----------
path : str or os.PathLike
    File to create, truncating it if it exists. With ``append=True``, the
    existing file to continue.
atom_names, atom_types, atom_ids : list, optional
    Static per-atom data, each of length natoms. When appending, any that
    are not given are taken from the file's metadata. Cannot be used with
    ``variable_atoms``.
precision, column_precision, gof_size, predictor_order, zstd_level,
adaptive, anchor_step, codec, lossless, rigid, bodies
    As for :func:`compress` and :func:`compress_batch`. When appending,
    pass the settings the file was created with. With ``rigid`` and no
    ``bodies``, bodies are detected in the first group and used for the
    rest of the file.
append : bool, default False
    Continue an existing .mdz file after its last complete group, whether
    or not it was closed cleanly. A file with a varying atom count keeps
    that setting.
variable_atoms : bool, default False
    Allow the atom count to change between frames, for example when atoms
    are deposited, evaporated or deleted, or in grand-canonical runs. Each
    call to :meth:`write` then passes the atom ``ids`` of its frames, and
    atoms are matched across frames by id. Without this option, a frame
    with a different atom count raises ValueError. Cannot be used with
    ``rigid`` or ``bodies``.

>>> with mdz.Writer("gcmc.mdz", variable_atoms=True) as w:
...     for x, ids in frames:             # (n, 3) and (n,), n changing
...         w.write(x, ids=ids)

Raises
------
ValueError
    On invalid arguments, frames that do not match earlier ones (atom
    count or columns), or settings that do not match the file being
    appended to.
OSError
    If the file cannot be read or written.)doc";

constexpr const char* kWriteDoc = R"doc(Add one frame or several.

``positions`` is ``(natoms, 3)`` for one frame or ``(nframes, natoms, 3)``
for several, in nm. The other arrays match it. For one frame,
``velocities`` and ``forces`` are ``(natoms, 3)``, ``charge`` is
``(natoms,)``, ``cell`` is ``(6,)``, ``boxes`` is ``(3, 3)``, and ``time``,
``energy`` and ``step`` are scalars. For several frames, each has a leading
``nframes`` axis, as for :func:`compress_batch`. Every call must pass the
same set of arrays and the same atom count. ``step`` is stored exactly as
int64.

With ``variable_atoms=True``, the atom count may differ between calls, but
all frames in one call share it. ``ids`` is a ``(natoms,)`` array of
integers that fit in 32 bits, the same for every frame in the call. It
identifies the atoms so that each can be followed across frames. Without
``ids``, atoms are identified by their position in the frame. To write
frames with different atom counts, call once per frame.)doc";

}  // namespace

void bind_writer(nb::module_& m) {
    nb::class_<PyWriter>(m, "Writer", kWriterDoc)
        .def("__init__", &init_writer, "path"_a, "atom_names"_a = nb::none(), "atom_types"_a = nb::none(),
             "atom_ids"_a = nb::none(), "precision"_a = kDefaultPrecision, "column_precision"_a = nb::none(),
             "gof_size"_a = kDefaultGofSize, "predictor_order"_a = 0, "zstd_level"_a = 3, "adaptive"_a = true,
             "anchor_step"_a = 1, "codec"_a = "auto", "lossless"_a = false, "append"_a = false,
             "variable_atoms"_a = false, "rigid"_a = false, "bodies"_a = nb::none())
        .def("write", &write_frames, "positions"_a, "velocities"_a = nb::none(), "forces"_a = nb::none(),
             "charge"_a = nb::none(), "cell"_a = nb::none(), "time"_a = nb::none(), "energy"_a = nb::none(),
             "step"_a = nb::none(), "boxes"_a = nb::none(), "ids"_a = nb::none(), kWriteDoc)
        .def(
            "flush",
            [](PyWriter& self) {
                Writer& w = self.open();
                self.run([&] {
                    w.flush();
                    return 0;
                });
            },
            "Encode the buffered frames as a group, which may be short, and write\n"
            "everything to the OS. A crash after this keeps every frame written so\n"
            "far. Short groups compress slightly worse, so flush at checkpoints\n"
            "rather than after every frame.")
        .def("close", &PyWriter::close,
             "Encode any buffered frames and write the footer that marks the file\n"
             "complete. Further writes raise ValueError. Closing again does nothing.")
        .def_prop_ro(
            "frames_written",
            [](const PyWriter& self) { return self.w ? self.w->frames_written() : self.closed_written; },
            "Frames passed to :meth:`write` so far, including any still buffered.\n"
            "When appending, this includes the frames already in the file.")
        .def_prop_ro(
            "frames_committed",
            [](const PyWriter& self) { return self.w ? self.w->frames_committed() : self.closed_committed; },
            "Frames already encoded into the file, which a crash would not lose.")
        .def_prop_ro(
            "closed", [](const PyWriter& self) { return !self.w.has_value(); },
            "Whether :meth:`close` has been called.")
        .def("__enter__", [](nb::handle self) { return nb::borrow(self); })
        .def(
            "__exit__",
            [](PyWriter& self, nb::handle exc_type, nb::handle, nb::handle) {
                // If the block raised, do not let a failure to close hide that
                // exception.
                if (exc_type.is_none()) {
                    self.close();
                } else {
                    try {
                        self.close();
                    } catch (...) {
                    }
                }
                return false;
            },
            "exc_type"_a = nb::none(), "_exc_value"_a = nb::none(), "_traceback"_a = nb::none())
        .def("__repr__", [](const PyWriter& self) {
            const std::uint64_t n = self.w ? self.w->frames_written() : self.closed_written;
            return "Writer(path='" + self.path + "', frames_written=" + std::to_string(n) +
                   ", closed=" + (self.w ? "False" : "True") + ")";
        });
}

}  // namespace mdz::py
