// mdz.MdzReader: random access to a .mdz file, as a read-only sequence of
// frames.

#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstring>

#include "common.hpp"

namespace mdz::py {

namespace {

using namespace nb::literals;

struct PyReader {
    Reader r;
};

// One decoded group of frames. Consecutive single-frame reads (iteration,
// strided slices) reuse it, so each group is decoded once rather than once
// per frame.
struct GroupCache {
    std::size_t start = 0, count = 0;
    std::vector<float> data;
    // Start of each frame of the group in `data`, in atoms. Has one more
    // entry than there are frames.
    std::vector<std::size_t> offsets;

    bool contains(std::size_t i) const { return i >= start && i < start + count; }

    // Decode the group holding frame i, unless it is already cached. i must be
    // in range. Does not need the GIL.
    void fill(const Reader& r, std::size_t i) {
        if (contains(i)) return;
        const auto& gofs = r.gofs();
        const auto span = gofs.span(gofs.overlapping(i, i + 1).start);
        const std::vector<std::size_t> natoms = r.frame_natoms(span.start, span.size());
        offsets.assign(1, 0);
        for (std::size_t n : natoms) offsets.push_back(offsets.back() + n);
        data.resize(offsets.back() * 3);
        r.read_positions(span.start, span.size(), data.data());
        start = span.start;
        count = span.size();
    }

    const float* frame(std::size_t i) const { return data.data() + offsets[i - start] * 3; }
    std::size_t natoms(std::size_t i) const { return offsets[i - start + 1] - offsets[i - start]; }
    nb::object frame_array(std::size_t i) const {
        const float* f = frame(i);
        const std::size_t n = natoms(i);
        return to_numpy(std::vector<float>(f, f + n * 3), {n, std::size_t(3)});
    }
};

std::size_t normalize_index(std::int64_t i, std::size_t n) {
    const std::int64_t ni = static_cast<std::int64_t>(n);
    const std::int64_t j = i < 0 ? i + ni : i;
    if (j < 0 || j >= ni)
        raise(PyExc_IndexError, "frame index " + std::to_string(i) + " out of range for " + std::to_string(n) +
                                    " frames");
    return static_cast<std::size_t>(j);
}

nb::object read_range(const Reader& r, std::size_t start, std::size_t stop) {
    const std::size_t n = r.nframes();
    start = std::min(start, n);
    stop = std::min(stop, n);
    const std::size_t count = stop > start ? stop - start : 0;
    if (r.variable_atoms()) {
        std::vector<float> flat = nogil([&] { return r.read_positions(start, count); });
        return split_frames(flat, r.frame_natoms(start, count));
    }
    std::vector<float> out(count * r.natoms() * 3);
    nogil([&] { r.read_positions(start, count, out.data()); });
    return to_numpy(std::move(out), {count, r.natoms(), std::size_t(3)});
}

nb::object read_one(const Reader& r, std::int64_t frame) {
    const std::size_t i = normalize_index(frame, r.nframes());
    std::vector<float> out = nogil([&] { return r.read_frame(i); });
    const std::size_t natoms = out.size() / 3;
    return to_numpy(std::move(out), {natoms, std::size_t(3)});
}

// The len frames at start, start + step, and so on. step may be negative.
nb::object read_strided(const Reader& r, Py_ssize_t start, Py_ssize_t step, Py_ssize_t len) {
    const std::size_t count = static_cast<std::size_t>(len);
    if (r.variable_atoms()) {
        nb::list out;
        GroupCache cache;
        Py_ssize_t idx = start;
        for (std::size_t k = 0; k < count; ++k, idx += step) {
            const auto i = static_cast<std::size_t>(idx);
            nogil([&] { cache.fill(r, i); });
            out.append(cache.frame_array(i));
        }
        return out;
    }
    const std::size_t natoms = r.natoms(), n3 = natoms * 3;
    std::vector<float> out(count * n3);
    nogil([&] {
        GroupCache cache;
        Py_ssize_t idx = start;
        for (std::size_t k = 0; k < count; ++k, idx += step) {
            const auto i = static_cast<std::size_t>(idx);
            cache.fill(r, i);
            std::memcpy(out.data() + k * n3, cache.frame(i), n3 * sizeof(float));
        }
    });
    return to_numpy(std::move(out), {count, natoms, std::size_t(3)});
}

nb::object atom_ids_of(const Reader& r, std::int64_t frame) {
    const std::size_t i = normalize_index(frame, r.nframes());
    std::vector<std::int32_t> ids = nogil([&] { return r.atom_ids(i); });
    const std::size_t n = ids.size();
    return to_numpy(std::move(ids), {n});
}

nb::object cell_of(const Reader& r, std::int64_t frame) {
    const std::size_t i = normalize_index(frame, r.nframes());
    const auto& boxes = r.boxes();
    if (!boxes || i >= boxes->size()) return nb::none();
    const auto& b = (*boxes)[i];
    return to_numpy(std::vector<float>(b.begin(), b.end()), {std::size_t(3), std::size_t(3)});
}

nb::object cells_of(const Reader& r) {
    const auto& boxes = r.boxes();
    if (!boxes) return nb::none();
    const std::size_t n = boxes->size();
    std::vector<float> flat(n * 9);
    if (n) std::memcpy(flat.data(), boxes->data(), n * 9 * sizeof(float));
    return to_numpy(std::move(flat), {n, std::size_t(3), std::size_t(3)});
}

// Iterator for `for frame in reader`. Decodes one group at a time and returns
// its frames from memory.
struct ReaderIter {
    nb::object owner;  // keeps the MdzReader alive
    const Reader* r;
    std::size_t idx = 0, n = 0;
    GroupCache cache;
    bool busy = false;

    nb::object next() {
        if (idx >= n) {
            cache = GroupCache{};  // free the last group
            throw nb::stop_iteration();
        }
        if (busy) raise(PyExc_RuntimeError, "MdzReader iterator is already in use by another thread");
        const std::size_t i = idx;
        if (!cache.contains(i)) {
            busy = true;
            try {
                nogil([&] { cache.fill(*r, i); });
            } catch (...) {
                busy = false;
                throw;
            }
            busy = false;
        }
        ++idx;
        return cache.frame_array(i);
    }
};

constexpr const char* kReaderDoc = R"doc(Random-access reader for .mdz files.

Behaves like a read-only sequence of frames:

>>> reader = mdz.open("traj.mdz")
>>> len(reader)                 # number of frames
>>> reader[42]                  # frame 42, shape (natoms, 3)
>>> reader[-1]                  # last frame
>>> reader[10:20]               # frames 10 to 19, shape (10, natoms, 3)
>>> for frame in reader: ...    # iterate frames
>>> with mdz.open("traj.mdz") as reader: ...

The per-frame unit cell and every stored column are also available:

>>> reader.cell(42)             # (3, 3) row-major box matrix in nm, or None
>>> reader.column_names()       # ['positions', 'velocities', 'box', 'cell', 'step', ...]
>>> reader.read_column("velocities")  # whole column as numpy
>>> reader.atom_names           # list[str] or None

Only the groups that contain the requested frames are decompressed.

When the atom count varies between frames (``variable_atoms``),
``natoms`` is 0 and frame i has shape ``(frame_natoms[i], 3)``. A range of
frames (``read_frames``, ``reader[a:b]``, ``read_all``) is returned as a
list of such arrays, one per frame. ``atom_ids(i)`` gives the ids of frame
i's atoms:

>>> reader.frame_natoms         # atoms per frame, (nframes,) int64
>>> reader[7], reader.atom_ids(7)  # (n7, 3) positions, (n7,) ids)doc";

constexpr const char* kOpenDoc = R"doc(Open a .mdz file for random-access reading.

The file is memory-mapped. Frames are decompressed only when requested, one
group at a time.

Parameters
----------
path : str or os.PathLike
    Path to a .mdz file.

Returns
-------
MdzReader)doc";

}  // namespace

void bind_reader(nb::module_& m) {
    nb::class_<ReaderIter>(m, "_MdzReaderIter")
        .def("__iter__", [](nb::handle self) { return nb::borrow(self); })
        .def("__next__", &ReaderIter::next)
        .def("__length_hint__", [](const ReaderIter& it) { return it.n > it.idx ? it.n - it.idx : 0; });

    nb::class_<PyReader>(m, "MdzReader", kReaderDoc)
        .def_prop_ro("natoms", [](const PyReader& p) { return p.r.natoms(); },
                     "Atoms per frame, or 0 if the count varies (see ``variable_atoms``).")
        .def_prop_ro("nframes", [](const PyReader& p) { return p.r.nframes(); })
        .def_prop_ro(
            "variable_atoms", [](const PyReader& p) { return p.r.variable_atoms(); },
            "Whether the atom count varies from frame to frame.")
        .def_prop_ro(
            "frame_natoms", [](const PyReader& p) { return frame_natoms_array(p.r.trajectory()); },
            "Number of atoms in each frame, as an (nframes,) int64 array.")
        .def_prop_ro(
            "rigid_bodies", [](const PyReader& p) { return count_rigid_bodies(p.r.trajectory()); },
            "Number of rigid bodies recorded in the metadata (0 if none).")
        .def(
            "atom_ids", [](const PyReader& p, std::int64_t frame) { return atom_ids_of(p.r, frame); }, "frame"_a,
            "Ids of the atoms in frame `frame`, as an int32 array in the order the\n"
            "positions are returned. For a fixed atom count these are the\n"
            "``atom_ids`` metadata, or 0..natoms-1 if there is none. Negative\n"
            "indices count from the end.")
        .def_prop_ro("precision", [](const PyReader& p) { return p.r.precision(); })
        .def_prop_ro("gof_size", [](const PyReader& p) { return p.r.gof_size(); })
        .def_prop_ro("n_gofs", [](const PyReader& p) { return p.r.n_gofs(); })
        .def(
            "read_frames",
            [](const PyReader& p, std::size_t start, std::size_t stop) { return read_range(p.r, start, stop); },
            "start"_a, "stop"_a,
            "Read frames in the range [start, stop) as a (k, natoms, 3) array. The\n"
            "range is clipped to the trajectory, as for a slice. If the atom count\n"
            "varies, returns a list of k (natoms_i, 3) arrays instead.")
        .def(
            "read_frame", [](const PyReader& p, std::int64_t i) { return read_one(p.r, i); }, "frame_idx"_a,
            "Read a single frame as a (natoms, 3) array, or (frame_natoms[i], 3) if\n"
            "the count varies. Negative indices count from the end. An index out\n"
            "of range raises IndexError.")
        .def(
            "read_all", [](const PyReader& p) { return read_range(p.r, 0, p.r.nframes()); },
            "Read all frames as a (nframes, natoms, 3) array, or as a list of\n"
            "(natoms_i, 3) arrays if the count varies.")
        .def("__len__", [](const PyReader& p) { return p.r.nframes(); })
        .def(
            "__getitem__",
            [](const PyReader& p, nb::handle key) -> nb::object {
                const std::size_t n = p.r.nframes();
                if (PySlice_Check(key.ptr())) {
                    Py_ssize_t start, stop, step;
                    if (PySlice_Unpack(key.ptr(), &start, &stop, &step) < 0) throw nb::python_error();
                    const Py_ssize_t len = PySlice_AdjustIndices(static_cast<Py_ssize_t>(n), &start, &stop, step);
                    if (step == 1)
                        return read_range(p.r, static_cast<std::size_t>(start),
                                          static_cast<std::size_t>(start + len));
                    return read_strided(p.r, start, step, len);
                }
                // An integer, or anything with __index__. Negative values count
                // from the end.
                PyObject* idx = PyNumber_Index(key.ptr());
                if (!idx) throw nb::python_error();
                const long long i = PyLong_AsLongLong(idx);
                Py_DECREF(idx);
                if (i == -1 && PyErr_Occurred()) {
                    PyErr_Clear();
                    raise(PyExc_IndexError, "frame index out of range for " + std::to_string(n) + " frames");
                }
                return read_one(p.r, i);
            },
            "key"_a,
            "``reader[i]`` returns frame i as (natoms, 3). ``reader[a:b:s]`` returns\n"
            "(k, natoms, 3), or a list of k (natoms_i, 3) arrays if the count varies.")
        .def("__iter__",
             [](nb::handle self) {
                 const PyReader& p = nb::cast<const PyReader&>(self);
                 return ReaderIter{nb::borrow(self), &p.r, 0, p.r.nframes(), {}, false};
             })
        .def("__enter__", [](nb::handle self) { return nb::borrow(self); })
        .def(
            "__exit__", [](nb::handle, nb::handle, nb::handle, nb::handle) { return false; },
            "_exc_type"_a = nb::none(), "_exc_value"_a = nb::none(), "_traceback"_a = nb::none())
        .def(
            "cell", [](const PyReader& p, std::int64_t frame) { return cell_of(p.r, frame); }, "frame"_a,
            "Unit cell for frame `frame` as a (3, 3) row-major matrix in nm, or None\n"
            "if the file stores no cell.\n\n"
            "Indexes like a Python sequence. Negative values count from the end, and\n"
            "an index outside ``-nframes <= frame < nframes`` raises IndexError,\n"
            "whether or not the file has a cell.")
        .def_prop_ro(
            "cells", [](const PyReader& p) { return cells_of(p.r); },
            "All per-frame unit cells as (nframes, 3, 3) in nm, or None.")
        .def(
            "column_names",
            [](const PyReader& p) {
                std::vector<std::string> names = p.r.column_names();
                names.erase(std::remove_if(names.begin(), names.end(), is_internal_column), names.end());
                const auto& ct = p.r.trajectory();
                if (ct.has_column("box")) names.push_back("cell");
                return names;
            },
            "Names of every readable column, with 'positions' first. ``cell`` is derived from ``box``.")
        .def(
            "read_column",
            [](const PyReader& p, const std::string& name) -> nb::object {
                if (name == "positions") {
                    if (!p.r.variable_atoms()) return read_range(p.r, 0, p.r.nframes());
                    std::vector<float> flat = nogil([&] { return p.r.read_positions(0, p.r.nframes()); });
                    const std::size_t n = flat.size() / 3;
                    return to_numpy(std::move(flat), {n, std::size_t(3)});
                }
                return read_named_column(p.r.trajectory(), name);
            },
            "name"_a,
            "Decode a whole column by name as a numpy array. 'positions' is read\n"
            "through the random-access path and is in nm. Other columns are in their\n"
            "stored units. See ``Trajectory.read_column`` for the shapes. When the\n"
            "atom count varies, per-atom columns are flat, with shape\n"
            "(total_atoms, 3) or (total_atoms,). Use\n"
            "``np.split(a, np.cumsum(reader.frame_natoms)[:-1])`` to split them by frame.")
        .def(
            "metadata", [](const PyReader& p) { return metadata_to_dict(p.r.trajectory()); },
            "Metadata dict (atom_names, atom_types, residues, bodies, ...). Empty if\n"
            "the file has no metadata.")
        .def_prop_ro(
            "atom_names",
            [](const PyReader& p) -> nb::object {
                nb::dict d = metadata_to_dict(p.r.trajectory());
                return d.contains("atom_names") ? nb::object(d["atom_names"]) : nb::none();
            },
            "Per-atom names (list[str]) or None.")
        .def_prop_ro(
            "atom_types",
            [](const PyReader& p) -> nb::object {
                nb::dict d = metadata_to_dict(p.r.trajectory());
                return d.contains("atom_types") ? nb::object(d["atom_types"]) : nb::none();
            },
            "Per-atom type ids (list[int]) or None.")
        .def("__repr__", [](const PyReader& p) {
            return "MdzReader(natoms=" + std::to_string(p.r.natoms()) + ", nframes=" + std::to_string(p.r.nframes()) +
                   ", gof_size=" + std::to_string(p.r.gof_size()) + ", n_gofs=" + std::to_string(p.r.n_gofs()) + ")";
        });

    m.def(
        "open",
        [](const std::filesystem::path& path) {
            const std::string s = path_str(path);
            return PyReader{nogil([&] { return Reader::open(s); })};
        },
        "path"_a, kOpenDoc);
}

}  // namespace mdz::py
