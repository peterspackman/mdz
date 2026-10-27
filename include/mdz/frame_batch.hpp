#pragma once
// FrameBatch, the in-memory form of a run of frames. Trajectory readers
// produce it, and writers and compress() consume it. Values are in internal
// units (see units.hpp) unless FrameBatch::units names a LAMMPS style.

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "mdz/units.hpp"

namespace mdz {

// Immutable per-atom data. Slices and copies of a batch share it.
template <class T>
using Shared = std::shared_ptr<const std::vector<T>>;

template <class T>
Shared<T> share(std::vector<T> v) {
    return std::make_shared<const std::vector<T>>(std::move(v));
}

// Row-major 3x3 matrix whose rows are the cell vectors a, b and c.
using Box = std::array<double, 9>;
using Vec3 = std::array<double, 3>;
// Cell lengths and angles in the DCD order: a, gamma, b, beta, alpha, c.
// Angles are in degrees.
using CellParams = std::array<double, 6>;

template <class V>
using NamedMap = std::map<std::string, V, std::less<>>;

// Names of the per-atom int columns that carry the static metadata when the
// atom count varies (see FrameBatch::make_variable). String-valued fields are
// stored as indices into FrameBatch::strings.
namespace columns {
inline constexpr const char* kId = "mdz.id";          // atom_ids; matches atoms across frames
inline constexpr const char* kType = "mdz.type";      // atom_types
inline constexpr const char* kResid = "mdz.resid";    // residue_ids
inline constexpr const char* kHetatm = "mdz.hetatm";  // hetatm (0/1)
inline constexpr const char* kName = "mdz.name";      // atom_names
inline constexpr const char* kElement = "mdz.element";  // elements
inline constexpr const char* kResname = "mdz.resname";  // residue_names
inline constexpr const char* kChain = "mdz.chain";      // chain_ids
}  // namespace columns

struct FrameBatch {
    // Atoms per frame, or 0 when the count varies (frame_natoms is then set).
    std::size_t natoms = 0;
    std::size_t nframes = 0;

    // Per-atom float columns, stored frame by frame with `width` values per
    // atom: nframes * natoms * width values, or total_atoms() * width when the
    // count varies. Standard names are positions, velocities and forces
    // (width 3), and charge, occupancy and bfactor (width 1).
    NamedMap<std::vector<float>> columns;
    // Per-atom integer columns (such as image), laid out like `columns`.
    NamedMap<std::vector<std::int32_t>> int_columns;
    // Per-frame float properties (time, energy, ...), nframes values each.
    NamedMap<std::vector<double>> frame_props;
    // Per-frame integer properties (step), nframes values each.
    NamedMap<std::vector<std::int64_t>> int_frame_props;
    // Per-frame box matrices and box lower corners. Readers may supply fewer
    // than nframes entries.
    std::optional<std::vector<Box>> box;
    std::optional<std::vector<Vec3>> origin;
    // Static string attributes that a format needs to write the data back
    // faithfully, such as lammps.units and lammps.boundary.
    NamedMap<std::string> attributes;
    // The LAMMPS unit style of positions, velocities, forces, box and origin,
    // or nullptr for internal units.
    const LammpsUnits* units = nullptr;

    // Set when the atom count varies between frames. Frame f then has
    // frame_natoms[f] atoms. Atoms are matched across frames by the int column
    // columns::kId, or by position within the frame if it is absent. The
    // static per-atom metadata below is carried in the columns:: int columns
    // instead, with string values indexing `strings`.
    std::optional<std::vector<std::size_t>> frame_natoms;
    Shared<std::string> strings;

    // Static per-atom metadata, natoms entries each. Used only when the atom
    // count is fixed.
    Shared<std::string> atom_names;
    Shared<std::string> elements;
    Shared<std::int32_t> atom_types;
    Shared<std::int32_t> atom_ids;
    Shared<std::string> residue_names;
    Shared<std::int32_t> residue_ids;
    Shared<std::string> chain_ids;
    Shared<std::uint8_t> hetatm;
    // The rigid body index of each atom, or -1 for none. Used by
    // CompressOptions::rigid.
    Shared<std::int32_t> bodies;

    FrameBatch() = default;
    FrameBatch(std::size_t natoms_, std::size_t nframes_) : natoms(natoms_), nframes(nframes_) {}

    // The float column `name`, or nullptr.
    const std::vector<float>* column(std::string_view name) const;
    std::vector<float>* column(std::string_view name);
    const std::vector<float>* positions() const { return column("positions"); }

    bool variable() const noexcept { return frame_natoms.has_value(); }
    // Atoms in frame f. Requires f < nframes.
    std::size_t atoms_in(std::size_t f) const noexcept { return variable() ? (*frame_natoms)[f] : natoms; }
    // Atoms summed over all frames, which is the length of a width-1 per-atom
    // column.
    std::size_t total_atoms() const noexcept;
    // The index, in atoms, at which each frame starts in the per-atom columns.
    // Has nframes + 1 entries; the last is total_atoms().
    std::vector<std::size_t> atom_offsets() const;
    // Convert to the varying-count form. Static per-atom metadata moves into
    // the columns:: int columns and natoms becomes 0. Does nothing if the
    // count already varies. Throws Error(InvalidInput) if the batch has rigid
    // bodies, which require a fixed count.
    void make_variable();
    // Frame f as a batch with a fixed atom count, its metadata restored to the
    // static fields. Writers for formats that store one frame at a time use
    // this. Throws Error(Corrupt) if a string index is out of range.
    FrameBatch frame(std::size_t f) const;

    // Cell parameters of each box matrix, or nullopt if there is no box.
    std::optional<std::vector<CellParams>> cell_params() const;

    // Frames [start, end) as a new batch, with the range clamped to nframes.
    // Static data is shared, not copied.
    FrameBatch slice_frames(std::size_t start, std::size_t end) const;
    // Append the frames of b. For each static metadata field, the first batch
    // that has it wins. If b has a different atom count, or the same count of
    // different atoms, the result has a varying atom count. Throws
    // Error(InvalidInput) if the batches then carry different per-atom
    // columns.
    void append(const FrameBatch& b);
    // Append the frames of b without checking that the atoms match. Both
    // batches must have the same fixed atom count.
    void append_fixed(const FrameBatch& b);
    // Convert positions, velocities, forces, box and origin to the style `to`,
    // or to internal units if `to` is nullptr. Float columns are rounded back
    // to single precision.
    void convert_units(const LammpsUnits* to);
    // A copy of this batch in internal units.
    FrameBatch in_internal_units() const;

    // The kinds of data present, sorted: column and property names, "box",
    // "origin", "atom_names", and so on. Metadata columns are reported under
    // their static names (columns::kId as "atom_ids").
    std::vector<std::string> available_data() const;
};

// Convert between a box matrix and cell parameters. cell_params_to_box returns
// the lower-triangular form, with a along x and b in the xy plane. Angles of
// exactly 90 degrees give exact zeros, so an orthorhombic box round-trips
// exactly.
Box cell_params_to_box(const CellParams& p) noexcept;
CellParams box_to_cell_params(const Box& m) noexcept;

}  // namespace mdz
