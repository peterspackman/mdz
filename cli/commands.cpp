// The compress, decompress, extract, convert and info subcommands. The
// completions command is handled in main.cpp.

#include "commands.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>

#include "io.hpp"
#include "paths.hpp"
#include "ui.hpp"

namespace fs = std::filesystem;

namespace cli {
namespace {

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

std::uint64_t file_size(const std::string& path) {
    std::error_code ec;
    auto n = fs::file_size(utf8_path(path), ec);
    return ec ? 0 : n;
}

std::optional<OutFormat> format_arg(const Args& a) {
    if (!a.has("format")) return std::nullopt;
    return out_format_from_name(a.get("format"));
}

std::string mb(double bytes, int prec) { return fixed(bytes / 1e6, prec) + " MB"; }

// The source format recorded in a .mdz file, if any.
std::optional<std::string> source_format(const mdz::Reader& r) {
    try {
        return r.metadata().source_format;
    } catch (const mdz::Error&) {
        return std::nullopt;
    }
}

// The smallest and largest atom count over all frames.
std::pair<std::size_t, std::size_t> atom_range(const std::vector<std::size_t>& counts) {
    if (counts.empty()) return {0, 0};
    const auto [lo, hi] = std::minmax_element(counts.begin(), counts.end());
    return {*lo, *hi};
}

// "N atoms", or "N-M atoms (varying)" when the atom count varies.
std::string atoms_text(std::size_t lo, std::size_t hi, bool variable) {
    if (!variable) return std::to_string(lo) + " atoms";
    return (lo == hi ? std::to_string(lo) : std::to_string(lo) + "-" + std::to_string(hi)) + " atoms (varying)";
}

std::string atoms_of(const mdz::Reader& r) {
    if (!r.variable_atoms()) return atoms_text(r.natoms(), r.natoms(), false);
    const auto [lo, hi] = atom_range(r.frame_natoms());
    return atoms_text(lo, hi, true);
}

// Formats that store a single atom count per file.
bool fixed_count_only(OutFormat f) { return f == OutFormat::Dcd || f == OutFormat::Npy || f == OutFormat::Raw; }

// Write frames [start, end) of `r` to `out`, one chunk at a time.
std::size_t stream_frames(const mdz::Reader& r, std::size_t start, std::size_t end, const Output& out,
                          const std::string& what) {
    // A file stored with a varying atom count can still be written to a
    // fixed-count format if every frame in the range has the same number of
    // atoms (for example, a file made with --variable-atoms). Atoms are then
    // matched by position in the frame.
    bool as_fixed = false;
    if (r.variable_atoms() && fixed_count_only(out.format)) {
        const auto [lo, hi] = atom_range(r.frame_natoms(start, end - start));
        as_fixed = lo == hi;
        if (!as_fixed)
            fail(std::string(format_name(out.format)) + " holds one atom count per file, and " + what +
                 "'s atom count varies from frame to frame (" + std::to_string(lo) + "-" + std::to_string(hi) +
                 " atoms); write .xtc, .extxyz, .xyz, .pdb or .lammpstrj instead");
    }
    const std::size_t chunk = std::max<std::size_t>(r.gof_size(), 1) * std::max(4u, 2 * mdz::num_threads());
    std::size_t next = start;
    Progress pb("decompressing", end - start);
    std::size_t n = write_batches(
        out,
        [&]() -> std::optional<mdz::FrameBatch> {
            if (next >= end) return std::nullopt;
            const std::size_t count = std::min(chunk, end - next);
            try {
                mdz::FrameBatch b = r.read_batch(next, count);
                if (as_fixed && b.variable()) {
                    // Treat the batch as fixed-count. The columns that hold
                    // static metadata are kept so that the data-loss warning
                    // reports them, since these formats cannot store them.
                    b.natoms = b.nframes ? b.atoms_in(0) : 0;
                    b.frame_natoms.reset();
                }
                next += count;
                pb.set(next - start);
                return b;
            } catch (const mdz::Error& e) {
                pb.finish();
                fail(std::string("decoding columns: ") + e.what());
            }
        },
        what);
    pb.finish();
    return n;
}

// --- compress ---

// Whether two fixed-count batches describe the same atoms, i.e. have the same
// static per-atom metadata. Readers usually share these arrays between
// batches, so pointer equality is checked before comparing contents.
bool same_atoms(const mdz::FrameBatch& a, const mdz::FrameBatch& b) {
    auto same = [](const auto& x, const auto& y) { return x == y || (x && y && *x == *y); };
    return same(a.atom_ids, b.atom_ids) && same(a.atom_types, b.atom_types) && same(a.atom_names, b.atom_names) &&
           same(a.elements, b.elements) && same(a.residue_names, b.residue_names) &&
           same(a.residue_ids, b.residue_ids) && same(a.chain_ids, b.chain_ids) && same(a.hetatm, b.hetatm);
}

std::map<std::string, float, std::less<>> parse_precision_overrides(const std::vector<std::string>& entries) {
    std::map<std::string, float, std::less<>> out;
    for (const auto& e : entries) {
        auto eq = e.find('=');
        if (eq == std::string::npos) fail("invalid --precision-col '" + e + "': expected NAME=VALUE");
        std::string name = e.substr(0, eq), value = e.substr(eq + 1);
        auto trim = [](std::string s) {
            s.erase(0, s.find_first_not_of(" \t"));
            s.erase(s.find_last_not_of(" \t") + 1);
            return s;
        };
        char* end = nullptr;
        value = trim(value);
        float v = std::strtof(value.c_str(), &end);
        if (value.empty() || *end) fail("invalid precision value in '" + e + "': expected a number");
        out[trim(name)] = v;
    }
    return out;
}

}  // namespace

int cmd_compress(const Args& a) {
    const auto& inputs = a.positional();
    const bool lossless = a.has("lossless");
    const bool no_adaptive = a.has("no-adaptive");
    const std::size_t gof_size = static_cast<std::size_t>(a.get_int("gof-size", 20, 0, 1ll << 40));
    const std::size_t anchor_step = static_cast<std::size_t>(a.get_int("anchor-step", 1, 0, 1ll << 40));

    // Command-line values are in Angstroms and are converted to nm here.
    // --precision sets the quantization step directly. --max-error sets an
    // error budget from which the step is derived. In adaptive mode the step
    // is a quarter of the budget, leaving room for four refinement levels.
    float precision;
    if (a.has("precision")) {
        precision = static_cast<float>(a.get_double("precision", 0)) / 10.0f;
    } else {
        const float max_error_nm = static_cast<float>(a.get_double("max-error", 0.005)) / 10.0f;
        precision = no_adaptive ? max_error_nm * 2.0f : max_error_nm / 4.0f;
    }

    if (lossless) {
        status("Target: exact \xE2\x80\x94 every coordinate stored bit for bit");
    } else {
        try {
            mdz::CompressOptions check;
            check.precision = precision;
            check.gof_size = static_cast<std::uint32_t>(std::min<std::size_t>(gof_size, UINT32_MAX));
            check.anchor_step = static_cast<std::uint32_t>(std::min<std::size_t>(anchor_step, UINT32_MAX));
            if (gof_size > 65535) check.gof_size = 65536;
            check.validate();
        } catch (const mdz::Error& e) {
            fail(e.what());
        }
        const float max_err_a = 10.0f * (no_adaptive ? precision / 2.0f : precision * 4.0f);
        status("Target: \xE2\x89\xA4 " + fmt_small(max_err_a) + " \xC3\x85 error, " + fmt_small(precision * 10.0f) +
               " \xC3\x85 step" + (no_adaptive ? "" : ", adaptive"));
    }

    if (inputs.empty()) fail("no input files specified");
    const std::string output = a.has("output") ? a.get("output") : with_extension(inputs[0], "mdz");
    if (is_stdio(output)) fail("compress cannot write to stdout; give an output file with -o");
    for (const auto& p : inputs)
        if (is_stdio(p)) fail("compress cannot read from stdin ('" + p + "'); give an input file");
    if (ext_lower(output) != "mdz")
        warn("output '" + output + "' does not end in .mdz; mdz-format bytes will be written there regardless.");
    ensure_writable(output, inputs, a.has("force"));

    const std::string first_ext = ext_lower(inputs[0]);
    for (const auto& p : inputs)
        if (ext_lower(p) != first_ext)
            fail("all inputs must have the same format to be joined; got ." + first_ext + " and ." + ext_lower(p));
    std::optional<std::size_t> natoms;
    if (a.has("natoms")) {
        if (recordable_source_ext(first_ext))
            fail("--natoms applies only to raw float32 input; ." + first_ext + " files declare their own atom count");
        natoms = static_cast<std::size_t>(a.get_int("natoms", 0, 1, 1ll << 40));
    }

    mdz::CompressOptions opts;
    opts.gof_size = static_cast<std::uint32_t>(gof_size);
    opts.precision = precision;
    opts.column_precision = parse_precision_overrides(a.all("precision-col"));
    opts.column_precision.emplace("positions", precision);
    opts.predictor_order = static_cast<unsigned>(a.get_int("predictor-order", 0, 0, 4));
    opts.zstd_level = static_cast<int>(a.get_int("zstd-level", 3, 1, 22));
    opts.adaptive = !no_adaptive;
    opts.anchor_step = static_cast<std::uint32_t>(anchor_step);
    opts.codec = mdz::parse_codec_mode(a.get("codec", "auto"));
    opts.spatial_delta = !a.has("no-intradelta");
    opts.lossless = lossless;
    if (recordable_source_ext(first_ext)) opts.source_format = first_ext;
    try {
        opts.resolved().validate();
    } catch (const mdz::Error& e) {
        fail(e.what());
    }

    opts.variable_atoms = a.has("variable-atoms");
    opts.rigid = a.has("rigid");

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::unique_ptr<mdz::TrajectoryReader>> readers;
    std::optional<std::size_t> total = std::size_t(0);
    auto open_readers = [&] {
        readers.clear();
        total = std::size_t(0);
        for (const auto& p : inputs) {
            readers.push_back(open_input(p, natoms));
            auto n = readers.back()->nframes();
            if (total && n)
                *total += *n;
            else
                total.reset();
        }
    };
    open_readers();

    // Read a few groups per thread at a time, so that groups are encoded in
    // parallel while memory use stays bounded.
    //
    // A file is either fixed-count or varying from its first group, but a
    // change in the atoms may only show up after many groups have been
    // written. When that happens, compression restarts in varying mode.
    // Inputs are always files, so they can be read again, and the restart
    // happens at most once. If the first batch already varies, the file is
    // varying from the start.
    struct AtomsChanged {
        std::string why;
    };
    const std::size_t chunk = std::max<std::size_t>(gof_size, 1) * std::max(4u, 2 * mdz::num_threads());
    std::size_t frames = 0, columns = 0, raw = 0;
    std::size_t min_atoms = SIZE_MAX, max_atoms = 0;
    bool variable = opts.variable_atoms;
    for (;;) {
        frames = 0;
        min_atoms = SIZE_MAX;
        max_atoms = 0;
        Progress pb("compressing", total.value_or(0));
        try {
            write_atomically(output, [&](const std::string& tmp) {
                mdz::Writer w = mdz::Writer::create(tmp, opts);
                mdz::FrameBatch first;  // atoms of the first batch, for a fixed-count file
                for (std::size_t i = 0; i < readers.size(); ++i) {
                    while (auto b = readers[i]->read_batch(chunk)) {
                        if (frames == 0) {
                            // Excludes the "mdz." columns that hold static
                            // metadata in a varying batch.
                            columns = b->columns.size() + b->frame_props.size() + b->int_frame_props.size() +
                                      (b->box ? 1 : 0) + (b->origin ? 1 : 0);
                            for (const auto& kv : b->int_columns) columns += kv.first.rfind("mdz.", 0) != 0;
                            if (b->variable()) variable = true;
                            first = b->slice_frames(0, 0);
                            first.natoms = b->natoms;
                        } else if (!variable) {
                            std::string why;
                            if (b->variable())
                                why = inputs[i] + ": the atoms change";
                            else if (b->natoms != first.natoms)
                                why = inputs[i] + ": " + std::to_string(b->natoms) + " atoms per frame, after " +
                                      std::to_string(first.natoms);
                            else if (!same_atoms(first, *b))
                                why = inputs[i] + ": the atoms change";
                            if (!why.empty()) throw AtomsChanged{why};
                        }
                        for (std::size_t f = 0; f < b->nframes; ++f) {
                            min_atoms = std::min(min_atoms, b->atoms_in(f));
                            max_atoms = std::max(max_atoms, b->atoms_in(f));
                        }
                        w.write(*b);
                        frames += b->nframes;
                        pb.set(frames);
                    }
                }
                if (frames == 0) throw mdz::Error(mdz::ErrorKind::Corrupt, "no frames read from " + inputs[0]);
                w.close();
            });
            break;
        } catch (const AtomsChanged& c) {
            pb.finish();
            if (opts.rigid)
                fail("the input's atom count varies (" + c.why + " at frame " + std::to_string(frames) +
                     "), and --rigid needs a fixed one");
            status("Atom count varies (" + c.why + " at frame " + std::to_string(frames) +
                   "); compressing again as a varying-atom-count file");
            opts.variable_atoms = variable = true;
            open_readers();
        } catch (const mdz::Error& e) {
            pb.finish();
            fail(e.what());
        }
    }
    const double secs = seconds_since(t0);

    mdz::Reader written = mdz::Reader::open(output);
    raw = written.trajectory().raw_size();
    status("Input: " + atoms_text(min_atoms, max_atoms, variable) + ", " + std::to_string(frames) + " frames, " +
           mb(double(raw), 2) + " (" + cols(columns) + ")");

    if (a.has("stats")) {
        // Error is measured against the decoded output, in nm.
        Stats st;
        Progress pb("measuring error");
        try {
            std::size_t at = 0;
            for (const auto& p : inputs) {
                auto r = open_input(p, natoms);
                while (auto b = r->read_batch(chunk)) {
                    mdz::FrameBatch in = b->units ? b->in_internal_units() : std::move(*b);
                    const std::vector<float>* orig = in.positions();
                    if (!orig) break;
                    if (variable) {
                        // Decoded frames have their atoms in id order.
                        st.add_varying(in, written.read_batch(at, in.nframes).in_internal_units());
                    } else {
                        st.add(*orig, written.read_positions(at, in.nframes), in.natoms);
                    }
                    at += in.nframes;
                }
            }
        } catch (const mdz::Error& e) {
            pb.finish();
            warn(std::string("could not measure error: ") + e.what());
        }
        pb.finish();
        st.print();
    }

    const std::uint64_t size = file_size(output);
    status("Output: " + fixed(double(size) / 1e6, 4) +
           " MB, ratio: " + fixed(double(raw) / double(std::max<std::uint64_t>(size, 1)), 1) +
           "x, time: " + fixed(secs, 3) + "s (" + fixed(double(raw) / 1e6 / std::max(secs, 1e-9), 1) + " MB/s)");
    status(ok("Saved " + output));
    return 0;
}

// --- decompress / extract ---

int cmd_decompress(const Args& a) {
    const std::string input = a.positional()[0];
    const auto format = format_arg(a);
    if (!a.has("output") && is_stdio(input)) fail("reading from stdin needs -o to say where to write");
    const auto t0 = std::chrono::steady_clock::now();
    mdz::Reader r = load_mdz(input);

    // Without -o, write the format the file was compressed from. If none was
    // recorded, use extxyz, which can store every column.
    const auto recorded = source_format(r);
    const bool defaulted = !a.has("output");
    std::string output = a.get("output");
    if (defaulted) output = with_extension(input, format ? format_ext(*format) : recorded.value_or("extxyz"));
    Output out = Output::resolve(output, format);
    if (!out.to_stdout) {
        std::error_code ec;
        if (fs::exists(utf8_path(out.path), ec) && !a.has("force") && defaulted && !format && recorded) {
            fail(out.path + " already exists (use --force to overwrite)\n       (this .mdz was compressed from " +
                 *recorded + ", so that is the default output format; pass -o to write somewhere else)");
        }
        out.ensure_writable({input}, a.has("force"));
    }
    if (recorded) detail("Source format recorded as ." + *recorded);
    status("Input: " + atoms_of(r) + ", " + std::to_string(r.nframes()) + " frames, " +
           cols(r.trajectory().columns.size()));
    stream_frames(r, 0, r.nframes(), out, input);
    status(ok("Saved " + out.display() + " (" + fixed(seconds_since(t0), 3) + "s)"));
    return 0;
}

namespace {

// Parse START:END, where END is exclusive and either side may be omitted. A
// single index selects one frame. A range past the end is an error and is
// not clamped.
std::pair<std::size_t, std::size_t> parse_frame_range(const std::string& spec, std::size_t nframes) {
    auto num = [&](std::string part, const char* what) -> std::size_t {
        part.erase(0, part.find_first_not_of(" \t"));
        part.erase(part.find_last_not_of(" \t") + 1);
        if (part.empty() || part.find_first_not_of("0123456789") != std::string::npos)
            fail("invalid " + std::string(what) + " in frame range '" + spec + "': expected a number");
        return static_cast<std::size_t>(std::stoull(part));
    };
    std::size_t start, end;
    auto colon = spec.find(':');
    if (colon == std::string::npos) {
        start = num(spec, "frame index");
        end = start + 1;
    } else {
        std::string lhs = spec.substr(0, colon), rhs = spec.substr(colon + 1);
        auto blank = [](const std::string& s) { return s.find_first_not_of(" \t") == std::string::npos; };
        start = blank(lhs) ? 0 : num(lhs, "range start");
        end = blank(rhs) ? nframes : num(rhs, "range end");
    }
    if (end <= start)
        fail("empty frame range '" + spec + "': end (" + std::to_string(end) + ") must be greater than start (" +
             std::to_string(start) + ")");
    if (start >= nframes)
        fail("frame range '" + spec + "' starts at " + std::to_string(start) + ", but the trajectory has " +
             std::to_string(nframes) + " frames (0.." + std::to_string(nframes ? nframes - 1 : 0) + ")");
    if (end > nframes)
        fail("frame range '" + spec + "' ends at " + std::to_string(end) + ", but the trajectory has only " +
             std::to_string(nframes) + " frames (use '" + std::to_string(start) + ":' to run to the end)");
    return {start, end};
}

}  // namespace

int cmd_extract(const Args& a) {
    const std::string input = a.positional()[0];
    const auto format = format_arg(a);
    const auto t0 = std::chrono::steady_clock::now();
    if (!a.has("output") && is_stdio(input)) fail("reading from stdin needs -o to say where to write");
    mdz::Reader r = load_mdz(input);
    const auto [start, end] = parse_frame_range(a.positional()[1], r.nframes());

    std::string output = a.get("output");
    if (!a.has("output")) {
        const std::string ext = format ? format_ext(*format) : source_format(r).value_or("extxyz");
        fs::path p = utf8_path(input);
        std::string stem = utf8_string(p.stem());
        if (stem.empty()) stem = "frames";
        output =
            utf8_string(p.parent_path() / (stem + "-" + std::to_string(start) + "-" + std::to_string(end) + "." + ext));
    }
    Output out = Output::resolve(output, format);
    out.ensure_writable({input}, a.has("force"));
    const auto g = r.gofs().overlapping(start, end);
    detail("Decoding " + std::to_string(g.size()) + " of " + std::to_string(r.n_gofs()) + " groups for frames " +
           std::to_string(start) + ".." + std::to_string(end));
    stream_frames(r, start, end, out, input);
    status(ok("Saved " + out.display() + " (frames " + std::to_string(start) + ".." + std::to_string(end) + " of " +
              std::to_string(r.nframes()) + ", " + fixed(seconds_since(t0), 3) + "s)"));
    return 0;
}

// --- convert ---

int cmd_convert(const Args& a) {
    const std::string input = a.positional()[0];
    const auto t0 = std::chrono::steady_clock::now();
    // Converting to .mdz is compression with the default settings.
    if (!a.has("format") && ext_lower(a.positional()[1]) == "mdz") {
        Args c = a;
        c.pos_ = {input};
        c.values_["output"] = {a.positional()[1]};
        return cmd_compress(c);
    }
    Output out = Output::resolve(a.positional()[1], format_arg(a));
    // Check before opening anything. Convert streams its input, so writing
    // over the input while reading it would corrupt it.
    out.ensure_writable({input}, a.has("force"));
    const std::string in_ext = ext_lower(input);
    std::unique_ptr<mdz::TrajectoryReader> reader;
    if (in_ext == "mdz") {
        mdz::Reader r = load_mdz(input);
        status("Decompressed: " + atoms_of(r) + ", " + std::to_string(r.nframes()) + " frames, " +
               cols(r.trajectory().columns.size()));
        stream_frames(r, 0, r.nframes(), out, input);
        status(ok("Saved " + out.display()));
        return 0;
    }
    reader = open_input(input, std::nullopt);
    status("Converting " + input + " (" + (in_ext.empty() ? std::string("unknown") : in_ext) + ") \xE2\x86\x92 " +
           out.display() + " (" + format_name(out.format) + ")");
    std::size_t frames = 0, seen = 0, min_atoms = SIZE_MAX, max_atoms = 0;
    bool variable = false;
    {
        Progress pb("converting", reader->nframes().value_or(0));
        frames = write_batches(
            out,
            [&]() -> std::optional<mdz::FrameBatch> {
                std::optional<mdz::FrameBatch> b;
                try {
                    b = reader->read_batch(100);
                } catch (const mdz::Error& e) {
                    pb.finish();
                    throw mdz::Error(e.kind(), std::string("read error: ") + e.what());
                }
                if (b) {
                    const auto* p = b->positions();
                    if (!p) throw mdz::Error(mdz::ErrorKind::Corrupt, "input frames carry no positions");
                    if (p->size() != b->total_atoms() * 3)
                        throw mdz::Error(mdz::ErrorKind::Corrupt,
                                         "malformed input: " + std::to_string(b->nframes) + " frames of " +
                                             std::to_string(b->total_atoms()) + " atoms in all need " +
                                             std::to_string(b->total_atoms() * 3) + " coordinates, got " +
                                             std::to_string(p->size()) + " (truncated or corrupt frame?)");
                    // The atoms may change between batches, which a
                    // fixed-count format cannot store.
                    variable = variable || b->variable() || (seen > 0 && b->natoms != min_atoms);
                    if (variable && fixed_count_only(out.format))
                        throw mdz::Error(mdz::ErrorKind::InvalidInput,
                                         std::string(format_name(out.format)) + " holds one atom count per file, and " +
                                             input +
                                             "'s atom count varies from frame to frame; write .xtc, "
                                             ".extxyz, .xyz, .pdb or .lammpstrj instead");
                    for (std::size_t f = 0; f < b->nframes; ++f) {
                        min_atoms = std::min(min_atoms, b->atoms_in(f));
                        max_atoms = std::max(max_atoms, b->atoms_in(f));
                    }
                    seen += b->nframes;
                    pb.set(seen);
                }
                return b;
            },
            input);
    }
    status(ok("Saved " + out.display() + " (" + atoms_text(min_atoms, max_atoms, variable) + ", " +
              std::to_string(frames) + " frames, " + fixed(seconds_since(t0), 3) + "s)"));
    return 0;
}

// --- info ---

namespace {

struct DcdHeader {
    std::int32_t nframes = 0, istart = 0, nsavc = 0, natoms = 0;
    float delta = 0;
    bool has_unitcell = false;
    std::vector<std::string> titles;
};

// Read only the DCD header. The DCD reader returns frames, not these fields.
DcdHeader read_dcd_header(const std::string& path) {
    std::ifstream f(utf8_path(path), std::ios::binary);
    std::vector<unsigned char> b(8192);
    f.read(reinterpret_cast<char*>(b.data()), static_cast<std::streamsize>(b.size()));
    b.resize(static_cast<std::size_t>(f.gcount()));
    auto bad = [&](const std::string& m) -> DcdHeader { fail("reading .dcd file: " + m); };
    if (b.size() < 100) return bad("file too short for a DCD header");
    std::uint32_t m;
    std::memcpy(&m, b.data(), 4);
    const bool swap = m != 84;
    if (swap && m != 0x54000000u) return bad("not a DCD file (bad first record marker)");
    auto i32 = [&](std::size_t off) {
        std::uint32_t v;
        std::memcpy(&v, b.data() + off, 4);
        if (swap) v = (v >> 24) | ((v >> 8) & 0xFF00) | ((v << 8) & 0xFF0000) | (v << 24);
        return static_cast<std::int32_t>(v);
    };
    if (std::memcmp(b.data() + 4, "CORD", 4) != 0) return bad("expected CORD magic");
    DcdHeader h;
    h.nframes = i32(8);
    h.istart = i32(12);
    h.nsavc = i32(16);
    std::int32_t dbits = i32(44);
    std::memcpy(&h.delta, &dbits, 4);
    h.has_unitcell = i32(48) != 0;
    std::size_t off = 92;
    const std::int32_t ntitle = i32(off + 4);
    if (ntitle < 0 || ntitle > 10000) return bad("implausible title count");
    off += 8;
    for (std::int32_t t = 0; t < ntitle && off + 80 <= b.size(); ++t, off += 80) {
        std::string s(reinterpret_cast<const char*>(b.data() + off), 80);
        s.erase(s.find_last_not_of(std::string(" \0", 2)) + 1);
        s.erase(0, s.find_first_not_of(' '));
        h.titles.push_back(s);
    }
    off += 4;
    if (off + 12 > b.size()) return bad("truncated header");
    h.natoms = i32(off + 4);
    return h;
}

}  // namespace

int cmd_info(const Args& a) {
    const std::string input = a.positional()[0];
    const std::string ext = ext_lower(input);
    std::string o;
    auto line = [&](const std::string& label, const std::string& v) {
        std::string l = label + ":";
        l.resize(18, ' ');
        o += l + v + "\n";
    };
    if (ext == "dcd") {
        DcdHeader h = read_dcd_header(input);
        line("File format", "DCD (CHARMM/NAMD)");
        line("Atoms", std::to_string(h.natoms));
        line("Frames", std::to_string(h.nframes));
        line("Timestep", shortest(h.delta) + " ps");
        line("Start step", std::to_string(h.istart));
        line("Save frequency", std::to_string(h.nsavc));
        line("Unit cell", h.has_unitcell ? "true" : "false");
        for (const auto& t : h.titles) line("Title", t);
        line("Raw coords size", mb(double(h.nframes) * h.natoms * 12, 2));
        line("File size", mb(double(file_size(input)), 2));
    } else if (ext == "mdz") {
        mdz::Reader r = [&] {
            try {
                return mdz::Reader::open(input);
            } catch (const mdz::Error& e) {
                fail("reading " + input + ": " + e.what());
            }
        }();
        const auto& ct = r.trajectory();
        const std::size_t raw = ct.raw_size();
        const std::uint64_t size = file_size(input);
        line("File format", "MDZ v4 (columnar)");
        line("Status", r.complete()
                           ? "complete"
                           : "incomplete: not closed by its writer; " + std::to_string(r.status().trailing_bytes) +
                                 " trailing bytes after the last complete group");
        if (r.variable_atoms()) {
            const auto [lo, hi] = atom_range(r.frame_natoms());
            line("Atoms", "varying, " +
                              (lo == hi ? std::to_string(lo) : std::to_string(lo) + "-" + std::to_string(hi)) +
                              " per frame");
        } else {
            line("Atoms", std::to_string(ct.natoms));
        }
        line("Frames", std::to_string(ct.nframes));
        bool uniform = ct.gofs.size() <= 1;
        std::size_t lo = SIZE_MAX;
        if (!uniform) {
            uniform = true;
            for (std::size_t i = 0; i < ct.gofs.size(); ++i) {
                uniform &= ct.gofs.span(i).size() == ct.gof_size;
                lo = std::min(lo, ct.gofs.span(i).size());
            }
        }
        if (uniform)
            line("GoF size", std::to_string(ct.gof_size));
        else
            line("GoF size", std::to_string(ct.gof_size) + " nominal (" + std::to_string(ct.gofs.size()) +
                                 " groups of " + std::to_string(lo) + "-" + std::to_string(ct.gofs.max_len()) +
                                 " frames)");
        line("Columns", std::to_string(ct.columns.size()));
        for (const auto& c : ct.columns) {
            const auto& d = c.desc;
            std::string prec = "lossless";
            if (d.lossy) {
                prec = "step=" + fmt_small(d.precision) + " " + d.unit;
                if (d.unit == "nm") prec += " (" + fmt_small(d.precision * 10.0f) + " \xC3\x85)";
            }
            std::string dt(mdz::dtype_name(d.dtype));
            dt[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(dt[0])));
            o += "  " + d.name + ": " + dt + " x" + std::to_string(d.width) + " " +
                 (d.per_frame ? "per-frame" : "per-atom") + " " + prec + "\n";
            o += "    GoFs: " + std::to_string(c.n_gofs()) + ", compressed: " + std::to_string(c.compressed_size()) +
                 " bytes\n";
        }
        line("Metadata", ct.metadata ? "yes" : "no");
        line("Raw size", mb(double(raw), 2));
        line("File size", fixed(double(size) / 1e6, 4) + " MB");
        line("Ratio", fixed(double(raw) / double(std::max<std::uint64_t>(size, 1)), 1) + "x");
    } else {
        std::string_view fmt = mdz::detect_format(input);
        if (fmt.empty() || fmt == "npy")
            fail("cannot show info for '" + input +
                 "': unsupported format; expected .mdz, .dcd, .xtc, .extxyz, .xyz, .pdb or .lammpstrj");
        auto reader = open_input(input, std::nullopt);
        std::optional<mdz::FrameBatch> b;
        try {
            b = reader->read_batch(1);
        } catch (const mdz::Error& e) {
            fail(std::string("reading file: ") + e.what());
        }
        if (!b) {
            o += "Empty file\n";
        } else {
            const char* name = fmt == "extxyz"   ? "Extended XYZ"
                               : fmt == "xyz"    ? "XYZ"
                               : fmt == "pdb"    ? "PDB"
                               : fmt == "lammps" ? "LAMMPS dump"
                               : fmt == "xtc"    ? "XTC (GROMACS)"
                                                 : "Unknown";
            line("File format", name);
            line("Atoms", std::to_string(b->natoms));
            line("Frames (scanned)", ">= 1");
            if (auto n = reader->nframes()) line("Frames (total)", std::to_string(*n));
            std::string names = "[";
            bool first = true;
            for (const auto& kv : b->columns) {
                names += (first ? "\"" : ", \"") + kv.first + "\"";
                first = false;
            }
            line("Columns", names + "]");
            if (b->box) line("Unit cell", "yes");
            if (b->atom_names) {
                std::set<std::string> u(b->atom_names->begin(), b->atom_names->end());
                std::string s = "{";
                first = true;
                for (const auto& e : u) {
                    s += (first ? "\"" : ", \"") + e + "\"";
                    first = false;
                }
                line("Elements", s + "}");
            }
            line("File size", mb(double(file_size(input)), 2));
        }
    }
    write_stdout(o);
    return 0;
}

}  // namespace cli
