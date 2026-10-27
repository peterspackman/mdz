#pragma once
// Trajectory input and output for the commands. This includes the checks
// that refuse to overwrite files, all-or-nothing writes, and stdin/stdout
// handling.

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "mdz/mdz.hpp"

namespace cli {

std::string ext_lower(const std::string& path);
bool is_stdio(const std::string& path);
// The path with its extension replaced (or added).
std::string with_extension(const std::string& path, const std::string& ext);

// Exit with an error if output already exists (unless force is set) or is
// one of the inputs.
void ensure_writable(const std::string& output, const std::vector<std::string>& inputs, bool force);

// Whether mdz can both read and write this extension, and so can record it
// as the source format.
bool recordable_source_ext(const std::string& ext);

enum class OutFormat { Extxyz, Xyz, Pdb, Lammpstrj, Dcd, Xtc, Npy, Raw };
extern const std::vector<const char*> kOutFormatNames;  // the --format values
std::optional<OutFormat> out_format_from_ext(const std::string& ext);
std::optional<OutFormat> out_format_from_name(const std::string& name);
const char* format_name(OutFormat f);  // the --format value
const char* format_ext(OutFormat f);   // extension for a default output name
const char* core_name(OutFormat f);    // the library's format name
bool is_text(OutFormat f);

struct Output {
    bool to_stdout = false;
    std::string path;
    OutFormat format;

    // For `-` (stdout), format must be given and must be a text format.
    // Otherwise format overrides the extension if given. Exits with an
    // error if no format can be determined.
    static Output resolve(const std::string& path, std::optional<OutFormat> format);
    std::string display() const { return to_stdout ? "<stdout>" : path; }
    void ensure_writable(const std::vector<std::string>& inputs, bool force) const {
        if (!to_stdout) cli::ensure_writable(path, inputs, force);
    }
    std::vector<std::string> data_loss(const mdz::FrameBatch& b) const;
};

// Set the --lammps-units value used by every LAMMPS reader and writer opened
// in this run.
void set_lammps_units(const mdz::LammpsUnits* u);
mdz::FormatOptions format_options();

// Open a reader chosen by extension. An unknown extension is read as raw
// float32, which requires natoms. Exits on error.
std::unique_ptr<mdz::TrajectoryReader> open_input(const std::string& path, std::optional<std::size_t> natoms);

// Load a .mdz file, or stdin for "-". Warns if the file was not closed.
mdz::Reader load_mdz(const std::string& path);

// Write `path` all-or-nothing. `write` fills a temporary file in the same
// directory, which is synced and renamed over `path` once complete. A
// replaced file keeps its permissions.
void write_atomically(const std::string& path, const std::function<void(const std::string& tmp)>& write);

// Write the batches returned by `next` to `out` until it returns nullopt.
// File output is atomic. Warns once about data the format cannot store.
// Returns the number of frames written. Exits on error, quietly if stdout is
// a closed pipe.
std::size_t write_batches(const Output& out, const std::function<std::optional<mdz::FrameBatch>()>& next,
                          const std::string& what);

// Positions and frame-to-frame motion statistics for --stats.
struct Stats {
    double sum_delta = 0, sum_err = 0, sum_sq_err = 0;
    float max_delta = 0, max_err = 0;
    std::size_t n_delta = 0, n_err = 0;
    std::vector<float> prev;  // last frame seen, for the motion stats
    void add(const std::vector<float>& orig, const std::vector<float>& back, std::size_t natoms);
    // For a varying atom count. Atoms are matched by id, both between orig
    // and back (`back` is in id order) and from one frame to the next.
    // Positions are in nm.
    void add_varying(const mdz::FrameBatch& orig, const mdz::FrameBatch& back);
    void print() const;

private:
    // The last frame seen, by id, for the motion stats of varying frames.
    std::vector<std::pair<std::int64_t, std::size_t>> order_, prev_order_;
    std::vector<float> prev_pos_;
};

}  // namespace cli
