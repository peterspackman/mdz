#pragma once
// Shared code for the mdz._core extension: numpy conversion in both
// directions, argument checks, and the wrapper types exposed to Python.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "mdz/mdz.hpp"

namespace mdz::py {

namespace nb = nanobind;

// Defaults shared by compress, compress_batch and Writer. They
// match the `mdz compress` CLI. The precision is a double so that help()
// shows 0.000125 rather than its float32 value.
inline constexpr double kDefaultPrecision = 0.000125;
inline constexpr std::int64_t kDefaultGofSize = 20;

// --- errors ---

[[noreturn]] void raise(PyObject* type, const std::string& msg);
[[noreturn]] inline void value_error(const std::string& msg) { raise(PyExc_ValueError, msg); }

// Translate mdz::Error to ValueError for invalid input, IndexError for out of
// range, and OSError for I/O failures and corrupt files.
void register_error_translator();

// Run `f` with the GIL released.
template <class F>
decltype(auto) nogil(F&& f) {
    nb::gil_scoped_release release;
    return f();
}

// --- numpy input ---

// A C-contiguous numpy view of any array-like, converted to T if needed (for
// example a float64 array, a nested list, or a strided or Fortran-ordered
// view). No copy is made when the input is already a contiguous T array.
// Keep the Array alive, and hold the GIL, for as long as data() is used.
template <class T>
struct Array {
    nb::ndarray<const T, nb::c_contig, nb::device::cpu> a;

    std::size_t ndim() const { return a.ndim(); }
    std::size_t shape(std::size_t i) const { return a.shape(i); }
    std::vector<std::size_t> shape() const {
        std::vector<std::size_t> s(a.ndim());
        for (std::size_t i = 0; i < s.size(); ++i) s[i] = a.shape(i);
        return s;
    }
    std::size_t size() const { return a.size(); }
    const T* data() const { return a.data(); }
    std::vector<T> to_vector() const { return std::vector<T>(data(), data() + size()); }
};

// numpy.<name>. The caller looks it up once and keeps it for the life of the
// process. The returned reference is never released.
PyObject* numpy_attr(const char* name);

nb::object as_numpy(nb::handle obj, const char* dtype);

template <class T>
const char* dtype_name_of();
template <>
inline const char* dtype_name_of<float>() { return "float32"; }
template <>
inline const char* dtype_name_of<double>() { return "float64"; }
template <>
inline const char* dtype_name_of<std::int64_t>() { return "int64"; }
template <>
inline const char* dtype_name_of<std::int32_t>() { return "int32"; }

template <class T>
Array<T> as_array(nb::handle obj) {
    nb::object arr = as_numpy(obj, dtype_name_of<T>());
    return Array<T>{nb::cast<nb::ndarray<const T, nb::c_contig, nb::device::cpu>>(arr)};
}

template <class T>
std::optional<Array<T>> as_optional_array(nb::handle obj) {
    if (obj.is_none()) return std::nullopt;
    return as_array<T>(obj);
}

// A shape formatted as Python prints a tuple, e.g. (3,) or (20, 10, 3).
std::string fmt_shape(const std::vector<std::size_t>& shape);

template <class T>
void check_shape(const char* name, const Array<T>& a, const std::vector<std::size_t>& want) {
    if (a.shape() != want)
        value_error(std::string(name) + " must have shape " + fmt_shape(want) + ", got " + fmt_shape(a.shape()));
}

// A (nframes, natoms, 3) positions array with at least one frame and atom.
void check_positions(const Array<float>& a);

// Per-frame 3x3 boxes from flat float32 data, rejecting NaN/Inf.
std::vector<Box> boxes_from_flat(const float* flat, std::size_t nframes);

// --- numpy output ---

// A numpy array that takes ownership of `v` without copying.
template <class T>
nb::object to_numpy(std::vector<T>&& v, std::initializer_list<std::size_t> shape) {
    if (v.capacity() == 0) v.reserve(1);  // never hand numpy a null pointer
    auto* owned = new std::vector<T>(std::move(v));
    nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    nb::ndarray<nb::numpy, T> arr(owned->data(), shape, owner);
    return nb::cast(std::move(arr), nb::rv_policy::reference);
}

template <class T>
nb::object to_numpy(std::vector<T>&& v, const std::vector<std::size_t>& shape) {
    if (v.capacity() == 0) v.reserve(1);
    auto* owned = new std::vector<T>(std::move(v));
    nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    nb::ndarray<nb::numpy, T> arr(owned->data(), shape.size(), shape.data(), owner);
    return nb::cast(std::move(arr), nb::rv_policy::reference);
}

// A decoded column as a numpy array with the stored dtype. Shapes:
//   per-atom, width 1:   (nframes, natoms)
//   per-atom, width w:   (nframes, natoms, w)
//   per-frame, width 1:  (nframes,)
//   per-frame, width w:  (nframes, w)
//   `box`:               (nframes, 3, 3)
// With a varying atom count (`variable`), per-atom columns are flat, with
// each frame's atoms following the previous frame's: (total_atoms,) or
// (total_atoms, w).
nb::object column_to_numpy(DecodedColumn&& d, const ColumnDesc& desc, std::size_t natoms, std::size_t nframes,
                           bool variable = false);

// Split flat positions into one (n, 3) array per frame, where frame i has
// counts[i] atoms. Used to return frames with a varying atom count.
nb::list split_frames(const std::vector<float>& flat, const std::vector<std::size_t>& counts);

// `cell` parameters derived from a decoded `box` column, with shape
// (nframes, 6) in the order a, gamma, b, beta, alpha, c.
nb::object cell_from_box(const DecodedColumn& box);

// Decode a column by name, for Trajectory.read_column and
// MdzReader.read_column. Accepts stored columns and `cell`, which is derived
// from `box`. Raises KeyError if the column is absent.
nb::object read_named_column(const CompressedTrajectory& ct, const std::string& name);

nb::dict metadata_to_dict(const CompressedTrajectory& ct);

// Number of rigid bodies in the metadata's `bodies`, counting only body
// indices shared by two or more atoms. A single-atom body is a free atom.
// Returns 0 if there are none.
std::size_t count_rigid_bodies(const CompressedTrajectory& ct);

// Whether a column is hidden from Python. Only the membership column used for
// a varying atom count (kMembershipColumn) is hidden, since it is not data.
bool is_internal_column(std::string_view name);

// The (nframes,) int64 array of atoms per frame.
nb::object frame_natoms_array(const CompressedTrajectory& ct);

// --- paths ---

std::string path_str(const std::filesystem::path& p);

// --- wrapper types ---

struct PyTrajectory {
    CompressedTrajectory ct;
    // Set when compress_batch built the box from a 6-parameter `cell=` alone,
    // so the box column is reported under the name the caller used.
    bool box_from_cell = false;
};

// Per-atom body indices (-1 for none) from any integer array-like.
Shared<std::int32_t> bodies_from(nb::handle bodies);

// Build CompressOptions, applying the same range checks as the CLI.
CompressOptions make_options(float precision, std::int64_t gof_size, std::int64_t predictor_order,
                             std::int64_t zstd_level, bool adaptive, std::int64_t anchor_step, bool lossless);
void add_column_precision(CompressOptions& opts, nb::handle column_precision);

// Static per-atom data attached to every batch that compress_batch or a
// Writer encodes. The arrays are shared between batches rather than copied.
struct AtomData {
    Shared<std::string> names, elements, residue_names, chain_ids;
    Shared<std::int32_t> types, ids, residue_ids;
    Shared<std::uint8_t> hetatm;
    // Rigid body of each atom (-1 for none), for CompressOptions::rigid.
    Shared<std::int32_t> bodies;
    NamedMap<std::string> attributes;

    void set(nb::handle names_, nb::handle types_, nb::handle ids_, nb::handle bodies_ = nb::none());

    // Attach to `b`, checking lengths against its atom count.
    void apply(FrameBatch& b) const;
};

// Per-frame array arguments of compress_batch and Writer.write.
struct FrameArgs {
    nb::handle positions, velocities, forces, charge, cell, time, energy, step, boxes;
    // A FrameBatch that owns copies of the data. `from_cell` is set when the
    // box was built from `cell` alone.
    FrameBatch to_batch(bool* from_cell = nullptr) const;
};

void bind_trajectory(nb::module_& m);
void bind_reader(nb::module_& m);
void bind_writer(nb::module_& m);
void bind_functions(nb::module_& m);

}  // namespace mdz::py
