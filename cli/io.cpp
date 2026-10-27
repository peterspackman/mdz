#include "io.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>

#include "signals.hpp"
#include "ui.hpp"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace cli {
namespace {

const mdz::LammpsUnits* g_lammps_units = nullptr;

const char* const kOutputExtensions =
    ".extxyz, .xyz, .pdb, .lammpstrj (or .lammps, .dump), .dcd, .xtc, .npy, and .f32/.bin/.raw for raw float32";

fs::path P(const std::string& s) { return fs::u8path(s); }

// Headerless little-endian float32 positions, in nm.
class RawReader final : public mdz::TrajectoryReader {
public:
    RawReader(std::unique_ptr<mdz::InputStream> in, std::size_t natoms) : in_(std::move(in)), natoms_(natoms) {}
    std::size_t natoms() const override { return natoms_; }
    std::optional<mdz::FrameBatch> read_batch(std::size_t n) override {
        const std::size_t frame_bytes = natoms_ * 12;
        std::vector<float> pos(n * natoms_ * 3);
        auto* dst = reinterpret_cast<unsigned char*>(pos.data());
        std::size_t got = 0, want = n * frame_bytes;
        while (got < want) {
            std::size_t k = in_->read(dst + got, want - got);
            if (k == 0) break;
            got += k;
        }
        if (got % frame_bytes)
            throw mdz::Error(mdz::ErrorKind::Corrupt, in_->name() + ": file size is not a whole number of " +
                                                          std::to_string(natoms_) + "-atom frames");
        if (got == 0) return std::nullopt;
        pos.resize(got / 4);
        mdz::FrameBatch b(natoms_, got / frame_bytes);
        b.columns["positions"] = std::move(pos);
        return b;
    }

private:
    std::unique_ptr<mdz::InputStream> in_;
    std::size_t natoms_;
};

class RawWriter final : public mdz::TrajectoryWriter {
public:
    explicit RawWriter(std::unique_ptr<mdz::OutputStream> out) : out_(std::move(out)) {}
    void write_batch(const mdz::FrameBatch& b) override {
        // The format has no header, so every frame must have the same atom
        // count.
        if (b.variable())
            throw mdz::Error(mdz::ErrorKind::InvalidInput,
                             "raw float32 holds one atom count per file and cannot store frames whose atom count "
                             "varies; write xtc, extxyz, xyz, pdb or a LAMMPS dump instead");
        if (natoms_ && b.natoms != *natoms_)
            throw mdz::Error(mdz::ErrorKind::InvalidInput, "raw float32 holds one atom count per file: this batch has " +
                                                               std::to_string(b.natoms) + " atoms per frame, the file " +
                                                               std::to_string(*natoms_));
        natoms_ = b.natoms;
        if (b.units) {
            write_batch(b.in_internal_units());
            return;
        }
        const auto* p = b.positions();
        if (!p) throw mdz::Error(mdz::ErrorKind::InvalidInput, "no positions column");
        out_->write(p->data(), p->size() * 4);
    }
    void finish() override { out_->close(); }

private:
    std::unique_ptr<mdz::OutputStream> out_;
    std::optional<std::size_t> natoms_;
};

std::unique_ptr<mdz::TrajectoryWriter> make_writer(OutFormat f, std::unique_ptr<mdz::OutputStream> out,
                                                   std::size_t natoms) {
    if (f == OutFormat::Raw) return std::make_unique<RawWriter>(std::move(out));
    const mdz::FormatInfo* info = mdz::find_format(core_name(f));
    return info->open_writer(std::move(out), natoms, format_options());
}

// Make a just-written file durable before it is renamed over its destination.
void sync_file(const std::string& path) {
#if defined(_WIN32)
    int fd = _open(path.c_str(), 0x0002 /*_O_RDWR*/);
    if (fd >= 0) {
        _commit(fd);
        _close(fd);
    }
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd >= 0) {
        ::fsync(fd);
        ::close(fd);
    }
#endif
}

}  // namespace

std::string ext_lower(const std::string& path) {
    std::string e = P(path).extension().u8string();
    if (!e.empty() && e[0] == '.') e.erase(0, 1);
    for (char& c : e) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return e;
}

bool is_stdio(const std::string& path) { return path == "-"; }

std::string with_extension(const std::string& path, const std::string& ext) {
    fs::path p = P(path);
    p.replace_extension(ext);
    return p.u8string();
}

void ensure_writable(const std::string& output, const std::vector<std::string>& inputs, bool force) {
    if (is_stdio(output)) return;
    std::error_code ec;
    if (fs::exists(P(output), ec)) {
        for (const auto& in : inputs)
            if (!is_stdio(in) && fs::equivalent(P(in), P(output), ec))
                fail("input and output are the same file: " + output);
        if (!force) fail(output + " already exists (use --force to overwrite)");
    }
}

bool recordable_source_ext(const std::string& e) {
    static const char* ok[] = {"dcd", "xtc", "extxyz", "xyz", "pdb", "lammpstrj", "lammps", "dump", "npy"};
    return std::any_of(std::begin(ok), std::end(ok), [&](const char* x) { return e == x; });
}

const std::vector<const char*> kOutFormatNames = {"extxyz", "xyz", "pdb", "lammpstrj", "lammps",
                                                  "dcd",    "xtc", "npy", "raw"};

std::optional<OutFormat> out_format_from_ext(const std::string& e) {
    if (e == "extxyz") return OutFormat::Extxyz;
    if (e == "xyz") return OutFormat::Xyz;
    if (e == "pdb") return OutFormat::Pdb;
    if (e == "lammpstrj" || e == "lammps" || e == "dump") return OutFormat::Lammpstrj;
    if (e == "dcd") return OutFormat::Dcd;
    if (e == "xtc") return OutFormat::Xtc;
    if (e == "npy") return OutFormat::Npy;
    if (e == "f32" || e == "bin" || e == "raw") return OutFormat::Raw;
    return std::nullopt;
}

std::optional<OutFormat> out_format_from_name(const std::string& n) {
    if (n == "raw") return OutFormat::Raw;
    if (n == "lammps") return OutFormat::Lammpstrj;
    return out_format_from_ext(n);
}

const char* format_name(OutFormat f) { return f == OutFormat::Raw ? "raw" : format_ext(f); }

const char* format_ext(OutFormat f) {
    switch (f) {
        case OutFormat::Extxyz: return "extxyz";
        case OutFormat::Xyz: return "xyz";
        case OutFormat::Pdb: return "pdb";
        case OutFormat::Lammpstrj: return "lammpstrj";
        case OutFormat::Dcd: return "dcd";
        case OutFormat::Xtc: return "xtc";
        case OutFormat::Npy: return "npy";
        case OutFormat::Raw: return "f32";
    }
    return "";
}

const char* core_name(OutFormat f) {
    if (f == OutFormat::Lammpstrj) return "lammps";
    if (f == OutFormat::Raw) return "npy";  // same data as npy: positions only
    return format_ext(f);
}

bool is_text(OutFormat f) {
    return f == OutFormat::Extxyz || f == OutFormat::Xyz || f == OutFormat::Pdb || f == OutFormat::Lammpstrj;
}

Output Output::resolve(const std::string& path, std::optional<OutFormat> format) {
    Output o;
    if (is_stdio(path)) {
        if (!format) fail("writing to stdout (-o -) needs --format, e.g. --format extxyz");
        if (!is_text(*format))
            fail(std::string(format_name(*format)) +
                 " is a binary format and cannot be written to stdout; use extxyz, xyz, pdb or lammpstrj, or write "
                 "to a file");
        o.to_stdout = true;
        o.format = *format;
        return o;
    }
    const std::string e = ext_lower(path);
    const auto from_ext = out_format_from_ext(e);
    if (format && from_ext && *format != *from_ext)
        warn(std::string("writing ") + format_name(*format) + " to '" + path + "', whose extension says " +
             format_name(*from_ext));
    if (format) o.format = *format;
    else if (from_ext) o.format = *from_ext;
    else if (e == "mdz") fail("cannot write .mdz to '" + path + "' here; use `mdz compress` or `mdz convert`");
    else
        fail("cannot tell the output format of '" + path + "': " +
             (e.empty() ? std::string("it has no extension") : "." + e + " is not a format mdz writes") +
             ".\n       Use one of " + kOutputExtensions + ", or pass --format");
    o.path = path;
    return o;
}

std::vector<std::string> Output::data_loss(const mdz::FrameBatch& b) const {
    return mdz::check_data_loss(b, core_name(format));
}

void set_lammps_units(const mdz::LammpsUnits* u) { g_lammps_units = u; }

mdz::FormatOptions format_options() {
    mdz::FormatOptions o;
    o.lammps_units = g_lammps_units;
    return o;
}

std::unique_ptr<mdz::TrajectoryReader> open_input(const std::string& path, std::optional<std::size_t> natoms) {
    std::string_view fmt = mdz::detect_format(path);
    try {
        if (fmt.empty()) {
            if (!natoms) fail("for raw binary input, --natoms is required");
            return std::make_unique<RawReader>(mdz::open_input(path), *natoms);
        }
        mdz::FormatOptions opts = format_options();
        if (fmt == "lammps" && !opts.lammps_units) {
            // LAMMPS unit styles differ most for velocities and forces. If the
            // dump has either, warn about the units being assumed.
            auto in = mdz::open_input(path);
            char head[16384];
            std::size_t n = in->read(head, sizeof head);
            std::string_view h(head, n);
            auto at = h.find("ITEM: ATOMS");
            if (at != std::string_view::npos) {
                auto line = h.substr(at, h.find('\n', at) - at);
                for (const char* c : {" vx", " vy", " vz", " fx", " fy", " fz"})
                    if (line.find(c) != std::string_view::npos) {
                        warn(path +
                             ": assuming LAMMPS `real` units (\xC3\x85, \xC3\x85/fs, kcal/mol/\xC3\x85) for velocities "
                             "and forces; pass --lammps-units if the run used another style");
                        break;
                    }
            }
        }
        return mdz::open_reader(path, fmt, opts);
    } catch (const mdz::Error& err) {
        fail("failed to open " + path + ": " + err.what());
    }
}

mdz::Reader load_mdz(const std::string& path) {
    try {
        mdz::Reader r = [&] {
            if (!is_stdio(path)) return mdz::Reader::open(path);
            std::vector<std::uint8_t> bytes, chunk(1 << 20);
            auto in = mdz::open_input("-");
            while (std::size_t n = in->read(chunk.data(), chunk.size()))
                bytes.insert(bytes.end(), chunk.begin(), chunk.begin() + static_cast<std::ptrdiff_t>(n));
            return mdz::Reader::from_bytes(std::move(bytes));
        }();
        if (!r.complete())
            warn(path + " was not closed cleanly (its writer crashed or is still running); read the " +
                 std::to_string(r.nframes()) + " frames written in full");
        return r;
    } catch (const mdz::Error& e) {
        fail("reading " + (is_stdio(path) ? std::string("<stdin>") : path) + ": " + e.what());
    }
}

void write_atomically(const std::string& path, const std::function<void(const std::string& tmp)>& write) {
    fs::path dest = P(path);
    fs::path dir = dest.parent_path();
    if (dir.empty()) dir = ".";
#if defined(_WIN32)
    const long pid = static_cast<long>(GetCurrentProcessId());
#else
    const long pid = static_cast<long>(getpid());
#endif
    TempFile tmp((dir / ("." + dest.filename().u8string() + ".tmp-" + std::to_string(pid))).u8string());
    write(tmp.path());
    std::error_code ec;
    auto st = fs::status(dest, ec);
    if (!ec && fs::exists(st)) {
        fs::permissions(P(tmp.path()), st.permissions(), ec);
        if (ec) warn("could not copy the permissions of " + path + " to its replacement: " + ec.message());
    }
    sync_file(tmp.path());
    tmp.rename_into(path);
}

std::size_t write_batches(const Output& out, const std::function<std::optional<mdz::FrameBatch>()>& next,
                          const std::string& what) {
    std::size_t frames = 0;
    auto run = [&](std::unique_ptr<mdz::OutputStream> stream) {
        std::unique_ptr<mdz::TrajectoryWriter> writer;
        while (auto b = next()) {
            if (!writer) {
                auto lost = out.data_loss(*b);
                if (!lost.empty()) {
                    std::string list;
                    for (std::size_t i = 0; i < lost.size(); ++i) list += (i ? ", " : "") + lost[i];
                    warn(std::string("output format '") + format_name(out.format) + "' cannot carry: " + list);
                }
                writer = make_writer(out.format, std::move(stream), b->natoms);
            }
            writer->write_batch(*b);
            frames += b->nframes;
        }
        if (!writer) throw mdz::Error(mdz::ErrorKind::Corrupt, "no frames read from " + what + " (empty input)");
        writer->finish();
    };
    try {
        if (out.to_stdout) run(mdz::open_output("-"));
        else write_atomically(out.path, [&](const std::string& tmp) { run(mdz::open_output(tmp)); });
    } catch (const mdz::Error& e) {
        if (out.to_stdout && is_broken_pipe(e.what())) std::exit(0);
        fail("writing " + out.display() + ": " + e.what());
    }
    return frames;
}

void Stats::add(const std::vector<float>& orig, const std::vector<float>& back, std::size_t natoms) {
    const std::size_t n = std::min(orig.size(), back.size()), n3 = natoms * 3;
    for (std::size_t i = 0; i < n; ++i) {
        const float e = std::fabs(orig[i] - back[i]);
        sum_err += e;
        sum_sq_err += double(e) * e;
        max_err = std::max(max_err, e);
    }
    n_err += n;
    if (n3 == 0) return;
    for (std::size_t f = 0; f * n3 + n3 <= orig.size(); ++f) {
        const float* cur = orig.data() + f * n3;
        const float* p = f == 0 ? (prev.empty() ? nullptr : prev.data()) : cur - n3;
        if (p) {
            for (std::size_t i = 0; i < n3; ++i) {
                const float d = std::fabs(cur[i] - p[i]);
                sum_delta += d;
                max_delta = std::max(max_delta, d);
            }
            n_delta += n3;
        }
    }
    if (orig.size() >= n3) prev.assign(orig.end() - static_cast<std::ptrdiff_t>(n3), orig.end());
}

void Stats::add_varying(const mdz::FrameBatch& orig, const mdz::FrameBatch& back) {
    if (!orig.variable()) {
        // A fixed-count batch from a varying file. Convert it so atoms are
        // matched by id like the rest.
        mdz::FrameBatch v = orig;
        v.make_variable();
        add_varying(v, back);
        return;
    }
    const std::vector<float>* op = orig.positions();
    const std::vector<float>* bp = back.positions();
    if (!op || !bp) return;
    // Atom ids as a reader assigns them: the id column if present, otherwise
    // the atom's index in the frame.
    auto ids_of = [](const mdz::FrameBatch& b) -> const std::vector<std::int32_t>* {
        auto it = b.int_columns.find(mdz::columns::kId);
        return it == b.int_columns.end() ? nullptr : &it->second;
    };
    const std::vector<std::int32_t>* oid = ids_of(orig);
    const std::vector<std::int32_t>* bid = ids_of(back);
    const auto oo = orig.atom_offsets(), bo = back.atom_offsets();
    for (std::size_t f = 0; f < orig.nframes && f < back.nframes; ++f) {
        // This frame's atoms sorted by id, as (id, atom index in the batch).
        const std::size_t n = orig.atoms_in(f);
        order_.resize(n);
        for (std::size_t a = 0; a < n; ++a)
            order_[a] = {oid ? (*oid)[oo[f] + a] : static_cast<std::int64_t>(a), oo[f] + a};
        if (!std::is_sorted(order_.begin(), order_.end())) std::sort(order_.begin(), order_.end());
        // Error, by merging against the decoded atoms on id.
        std::size_t i = 0;
        for (std::size_t k = 0; k < back.atoms_in(f) && i < n; ++k) {
            const std::int64_t id = bid ? (*bid)[bo[f] + k] : static_cast<std::int64_t>(k);
            while (i < n && order_[i].first < id) ++i;
            if (i == n || order_[i].first != id) continue;
            for (std::size_t c = 0; c < 3; ++c) {
                const float e = std::fabs((*op)[3 * order_[i].second + c] - (*bp)[3 * (bo[f] + k) + c]);
                sum_err += e;
                sum_sq_err += double(e) * e;
                max_err = std::max(max_err, e);
            }
            n_err += 3;
        }
        // Motion of the atoms present in this frame and the one before.
        std::size_t j = 0;
        for (const auto& [id, at] : order_) {
            while (j < prev_order_.size() && prev_order_[j].first < id) ++j;
            if (j == prev_order_.size()) break;
            if (prev_order_[j].first != id) continue;
            for (std::size_t c = 0; c < 3; ++c) {
                const float d = std::fabs((*op)[3 * at + c] - prev_pos_[3 * j + c]);
                sum_delta += d;
                max_delta = std::max(max_delta, d);
            }
            n_delta += 3;
        }
        prev_order_.swap(order_);
        prev_pos_.resize(3 * n);
        for (std::size_t a = 0; a < n; ++a)
            for (std::size_t c = 0; c < 3; ++c) prev_pos_[3 * a + c] = (*op)[3 * prev_order_[a].second + c];
    }
}

void Stats::print() const {
    if (n_delta) {
        status("Mean |delta|:     " + fixed(sum_delta / double(n_delta), 4) + " nm/frame");
        status("Max |delta|:      " + fixed(max_delta, 4) + " nm");
    }
    if (n_err) {
        const double mean = sum_err / double(n_err), rmse = std::sqrt(sum_sq_err / double(n_err));
        status("Max error:        " + fixed(max_err, 6) + " nm (" + fixed(max_err * 1000.0, 2) + " pm)");
        status("Mean error:       " + fixed(mean, 6) + " nm (" + fixed(mean * 1000.0, 2) + " pm)");
        status("RMSE:             " + fixed(rmse, 6) + " nm (" + fixed(rmse * 1000.0, 2) + " pm)");
    }
}

}  // namespace cli
