#pragma once
// Trajectory file formats: DCD, XTC, extended XYZ, LAMMPS dump, PDB, XYZ, npy
// and mdz itself. Readers and writers work in batches of frames over byte
// streams (files, stdin/stdout or memory). The formats are kept in a registry,
// and register_format adds new ones.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "mdz/frame_batch.hpp"
#include "mdz/options.hpp"
#include "mdz/span.hpp"

namespace mdz {

// --- Byte streams ---

class InputStream {
public:
    virtual ~InputStream() = default;
    // Read up to n bytes into buf and return the number read. Returns 0 only
    // at the end of the stream. Throws Error(Io).
    virtual std::size_t read(void* buf, std::size_t n) = 0;
    // A name for error messages: the path, or "<stdin>".
    virtual const std::string& name() const = 0;
};

class OutputStream {
public:
    virtual ~OutputStream() = default;
    // Write all n bytes. Throws Error(Io).
    virtual void write(const void* data, std::size_t n) = 0;
    // Hand buffered bytes to the OS. This is not an fsync. Throws Error(Io).
    virtual void flush() = 0;
    // Flush and close. Throws Error(Io). A stream destroyed without close()
    // is closed and any error is ignored.
    virtual void close() = 0;
    // Whether seek() is supported. Files support it; pipes and stdout do not.
    virtual bool seekable() const { return false; }
    // Move the write position to `offset` bytes from the start. The default
    // implementation throws Error(Io).
    virtual void seek(std::uint64_t offset);
    virtual const std::string& name() const = 0;
};

// Open a file for buffered binary reading, or create or truncate one for
// writing. The path "-" means stdin or stdout. Throws Error(Io).
std::unique_ptr<InputStream> open_input(const std::string& path);
std::unique_ptr<OutputStream> open_output(const std::string& path);
// A stream that reads from `bytes`, which it takes ownership of.
std::unique_ptr<InputStream> memory_input(std::vector<std::uint8_t> bytes, std::string name = "<memory>");
// A seekable stream that writes into *out, which must outlive the stream.
std::unique_ptr<OutputStream> memory_output(std::vector<std::uint8_t>* out, std::string name = "<memory>");

// --- Readers and writers ---

class TrajectoryReader {
public:
    virtual ~TrajectoryReader() = default;
    // Atoms per frame, or 0 if the count varies between frames.
    virtual std::size_t natoms() const = 0;
    // Total number of frames, if the format records it.
    virtual std::optional<std::size_t> nframes() const { return std::nullopt; }
    // Read the next n frames, or fewer at the end of the input. Returns
    // nullopt once the input is exhausted. Throws Error(Corrupt) for malformed
    // input and Error(Io) for read failures.
    virtual std::optional<FrameBatch> read_batch(std::size_t n) = 0;
    // Read every remaining frame. Returns nullopt if none remain.
    std::optional<FrameBatch> read_all();
};

class TrajectoryWriter {
public:
    virtual ~TrajectoryWriter() = default;
    // Write the frames of `batch`. Throws Error(InvalidInput) for data the
    // format cannot hold and Error(Io) for write failures.
    virtual void write_batch(const FrameBatch& batch) = 0;
    // Write any trailer and close the stream. Throws on failure. A writer
    // destroyed without finish() attempts to finish and ignores errors.
    virtual void finish() = 0;
};

// Settings a format may need beyond the data itself.
struct FormatOptions {
    // LAMMPS dumps only. When reading, the unit style of the values in the
    // file; nullptr means "real". When writing, the style to write in;
    // nullptr means the batch's own style, or "real" if it has none.
    const LammpsUnits* lammps_units = nullptr;
    // .mdz output only: the compression settings.
    CompressOptions compress;
};

struct FormatInfo {
    std::string_view name;
    // File extensions, lower case and without the dot.
    std::vector<std::string_view> extensions;
    // The data the writer preserves, named as in FrameBatch::available_data():
    // column names, "box", "origin", "atom_names", "atom_types", and so on.
    std::vector<std::string_view> supported;
    // Whether the writer preserves per-atom columns of any name.
    bool arbitrary_columns = false;
    // Whether the writer preserves frame properties of any name.
    bool arbitrary_frame_props = false;
    // Whether the writer can write to a non-seekable stream such as stdout.
    bool streamable = true;
    std::function<std::unique_ptr<TrajectoryReader>(std::unique_ptr<InputStream>, const FormatOptions&)> open_reader;
    std::function<std::unique_ptr<TrajectoryWriter>(std::unique_ptr<OutputStream>, std::size_t natoms,
                                                    const FormatOptions&)>
        open_writer;
};

// The registered formats. Lookups do not lock, so register any custom formats
// before other threads use the registry.
const std::vector<FormatInfo>& formats();
// Add a format, replacing any registered under the same name.
void register_format(FormatInfo info);
// The format named `name`, or nullptr.
const FormatInfo* find_format(std::string_view name) noexcept;

// The name of the format matching the extension of `path`, compared case-
// insensitively (for example "lammps" for .lammpstrj, .lammps and .dump).
// Returns "" if no format matches.
std::string_view detect_format(std::string_view path) noexcept;

// Open a file in the format named by `format`, or detected from its extension
// if `format` is empty. A path of "-" uses stdin or stdout. Throws
// Error(InvalidInput) for an unknown format and Error(Io) if the file cannot
// be opened.
std::unique_ptr<TrajectoryReader> open_reader(const std::string& path, std::string_view format = {},
                                              const FormatOptions& opts = {});
std::unique_ptr<TrajectoryWriter> open_writer(const std::string& path, std::size_t natoms, std::string_view format = {},
                                              const FormatOptions& opts = {});

// The kinds of data in `batch` (as named by available_data()) that a writer
// for `format` would not preserve. Empty if nothing would be lost.
std::vector<std::string> check_data_loss(const FrameBatch& batch, std::string_view format);

}  // namespace mdz
