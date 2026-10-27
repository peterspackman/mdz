#include <algorithm>
#include <cctype>
#include <mutex>
#include <set>

#include "internal.hpp"
#include "mdz/error.hpp"

namespace mdz {

std::optional<FrameBatch> TrajectoryReader::read_all() {
    std::optional<FrameBatch> all;
    while (auto b = read_batch(1000)) {
        if (!all)
            all = std::move(b);
        else
            all->append(*b);
    }
    return all;
}

namespace {

std::vector<FormatInfo> builtin() {
    using namespace detail;
    std::vector<FormatInfo> v;
    auto add = [&](std::string_view name, std::vector<std::string_view> ext, std::vector<std::string_view> sup,
                   bool arb, bool arb_props, bool streamable, auto reader, auto writer) {
        FormatInfo f;
        f.name = name;
        f.extensions = std::move(ext);
        f.supported = std::move(sup);
        f.arbitrary_columns = arb;
        f.arbitrary_frame_props = arb_props;
        f.streamable = streamable;
        f.open_reader = reader;
        f.open_writer = writer;
        v.push_back(std::move(f));
    };
    add("mdz", {"mdz"},
        {"positions", "velocities", "forces", "box", "origin", "atom_names", "elements", "atom_types", "atom_ids",
         "residue_names", "residue_ids", "chain_ids", "hetatm"},
        true, true, true, make_mdz_reader, make_mdz_writer);
    add("dcd", {"dcd"}, {"positions", "box"}, false, false, false, make_dcd_reader, make_dcd_writer);
    add("xtc", {"xtc"}, {"positions", "box", "time", "step"}, false, false, true, make_xtc_reader, make_xtc_writer);
    add("extxyz", {"extxyz"}, {"positions", "velocities", "forces", "box", "atom_names", "time", "energy"}, true, false,
        true, make_extxyz_reader, make_extxyz_writer);
    add("lammps", {"lammpstrj", "lammps", "dump"},
        {"positions", "velocities", "forces", "box", "atom_types", "atom_ids", "elements", "step", "time", "origin"},
        true, false, true, make_lammps_reader, make_lammps_writer);
    add("pdb", {"pdb"},
        {"positions", "box", "atom_names", "elements", "atom_ids", "occupancy", "bfactor", "residue_names",
         "residue_ids", "chain_ids", "hetatm"},
        false, false, true, make_pdb_reader, make_pdb_writer);
    add("xyz", {"xyz"}, {"positions", "atom_names", "elements"}, false, false, true, make_xyz_reader, make_xyz_writer);
    add("npy", {"npy"}, {"positions"}, false, false, false, make_npy_reader, make_npy_writer);
    return v;
}

std::mutex& table_mutex() {
    static std::mutex m;
    return m;
}

std::vector<FormatInfo>& table() {
    static std::vector<FormatInfo> t = builtin();
    return t;
}

const FormatInfo& require(std::string_view name) {
    const FormatInfo* f = find_format(name);
    if (!f) throw Error(ErrorKind::InvalidInput, "unsupported format '" + std::string(name) + "'");
    return *f;
}

std::string_view format_for(const std::string& path, std::string_view format) {
    if (!format.empty()) return format;
    std::string_view f = detect_format(path);
    if (f.empty()) {
        auto dot = path.find_last_of('.');
        throw Error(ErrorKind::InvalidInput,
                    "unsupported format '" + (dot == std::string::npos ? path : path.substr(dot)) + "'");
    }
    return f;
}

}  // namespace

const std::vector<FormatInfo>& formats() { return table(); }

void register_format(FormatInfo info) {
    std::lock_guard<std::mutex> lock(table_mutex());
    auto& t = table();
    auto it = std::find_if(t.begin(), t.end(), [&](const FormatInfo& f) { return f.name == info.name; });
    if (it != t.end())
        *it = std::move(info);
    else
        t.push_back(std::move(info));
}

const FormatInfo* find_format(std::string_view name) noexcept {
    for (const auto& f : table())
        if (f.name == name) return &f;
    return nullptr;
}

std::string_view detect_format(std::string_view path) noexcept {
    const auto slash = path.find_last_of("/\\");
    const auto dot = path.find_last_of('.');
    if (dot == std::string_view::npos || (slash != std::string_view::npos && dot < slash)) return {};
    std::string ext(path.substr(dot + 1));
    for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    for (const auto& f : table())
        for (auto e : f.extensions)
            if (e == ext) return f.name;
    return {};
}

std::unique_ptr<TrajectoryReader> open_reader(const std::string& path, std::string_view format,
                                              const FormatOptions& opts) {
    const FormatInfo& f = require(format_for(path, format));
    if (f.name == "mdz" && path != "-") return detail::make_mdz_reader_path(path);
    return f.open_reader(open_input(path), opts);
}

std::unique_ptr<TrajectoryWriter> open_writer(const std::string& path, std::size_t natoms, std::string_view format,
                                              const FormatOptions& opts) {
    const FormatInfo& f = require(format_for(path, format));
    return f.open_writer(open_output(path), natoms, opts);
}

std::vector<std::string> check_data_loss(const FrameBatch& batch, std::string_view format) {
    const FormatInfo* f = find_format(format);
    std::set<std::string_view> supported;
    if (f)
        supported.insert(f->supported.begin(), f->supported.end());
    else
        supported.insert("positions");
    // extxyz stores atom ids as the integer property "id". They survive if the
    // batch has an "id" column, or if the batch is variable, in which case the
    // writer emits the ids as "id". A fixed batch without an "id" column loses
    // its ids.
    if (f && f->name == "extxyz" && (batch.variable() || batch.int_columns.count("id"))) supported.insert("atom_ids");
    std::vector<std::string> lost;
    for (const std::string& d : batch.available_data()) {
        if (supported.count(d)) continue;
        const bool per_atom = batch.columns.count(d) || batch.int_columns.count(d);
        const bool per_frame = batch.frame_props.count(d) || batch.int_frame_props.count(d);
        if (f && f->arbitrary_columns && per_atom) continue;
        if (f && f->arbitrary_frame_props && per_frame) continue;
        lost.push_back(d);
    }
    return lost;
}

}  // namespace mdz
