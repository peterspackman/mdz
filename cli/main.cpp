// mdz: molecular dynamics trajectory compression with fast random access.

#include <cstdio>
#include <exception>

#include "args.hpp"
#include "commands.hpp"
#include "io.hpp"
#include "mdz/mdz.hpp"
#include "signals.hpp"
#include "ui.hpp"

namespace {

using cli::Opt;

const char* const kGlobal = "Global options";
const char* const kTuning = "Tuning (advanced)";

const std::vector<const char*> kUnitStyles = {"real", "metal", "si", "cgs", "electron", "micro", "nano", "lj"};

std::vector<Opt> output_options(const char* output_help) {
    return {
        {"output", 'o', "OUTPUT", output_help},
        {"format", 0, "FMT",
         "Output format. Overrides the extension. Required with `-o -` or when\nmdz does not recognise the "
         "extension",
         nullptr, nullptr, cli::kOutFormatNames},
        {"force", 'f', nullptr, "Overwrite the output file if it already exists"},
    };
}

cli::Program program() {
    cli::Program p;
    p.name = "mdz";
    p.version = MDZ_CLI_VERSION;
    p.about = "Molecular dynamics trajectory compression with fast random access";
    p.global = {
        {"threads", 0, "N", "Number of worker threads (default: all logical cores; 0 also means all)", kGlobal},
        {"quiet", 'q', nullptr, "Suppress informational output (errors only)", kGlobal},
        {"verbose", 'v', nullptr, "Print extra detail", kGlobal},
        {"color", 0, "WHEN", "When to colorize output", kGlobal, "auto", {"auto", "always", "never"}},
        {"lammps-units", 0, "STYLE",
         "LAMMPS unit style of dumps that are read (default real) and written\n(default: the style the "
         "trajectory was read in)",
         kGlobal, nullptr, kUnitStyles},
    };

    cli::Command compress{
        "compress",
        "Compress trajectory file(s) to .mdz",
        "Compress trajectory file(s) to .mdz.\n\nReads .dcd, .xtc, .extxyz, .xyz, .pdb, .lammpstrj, .npy or raw "
        "float32.\nSeveral files of the same format are joined into one continuous trajectory.",
        {{"INPUT", "Input file(s)", true, true}},
        {
            {"output", 'o', "OUTPUT",
             "Output .mdz file [default: <input>.mdz]. It is written to a temporary\nfile and renamed into "
             "place, so a failed run never leaves a partial file."},
            {"natoms", 0, "NATOMS", "Number of atoms. Required for raw float32 input; other formats store it"},
            {"max-error", 0, "ANGSTROM", "Maximum coordinate error, in Angstroms (the default matches XTC)", nullptr,
             "0.005"},
            {"lossless", 0, nullptr,
             "Store every coordinate exactly, with no rounding.\nExpect a ratio of about 1.3-3x, compared with "
             "2-10x for the lossy\ncodecs at the default error. The low mantissa bits of float32\n"
             "coordinates are close to random, so they compress poorly."},
            {"stats", 0, nullptr, "Print error statistics after compressing (requires a decompression pass)"},
            {"variable-atoms", 0, nullptr,
             "Store the file in a format that allows the atom count to vary between\nframes. This is chosen "
             "automatically when the input's atoms change\n(atoms inserted or deleted, or frames of different "
             "sizes). The option\nforces it for any input."},
            {"rigid", 0, nullptr,
             "Model rigid bodies (rigid fragments, clusters, coarse-grained units)\nand store each as a centre "
             "and an orientation per frame. Bodies are\ndetected from the input (atomistic data, nm). Each group "
             "keeps\nwhichever encoding is smaller. Most useful for large bodies.\nCompression is slower."},
            {"force", 'f', nullptr, "Overwrite the output file if it already exists"},
            {"precision", 0, "ANGSTROM",
             "Quantization step, in Angstroms. Sets the step directly rather than\nderiving it from --max-error",
             kTuning},
            {"precision-col",
             0,
             "NAME=STEP",
             "Quantization step for one column, as NAME=STEP. May be repeated.\nSTEP is in the column's internal "
             "unit: nm for positions, nm/ps for\nvelocities, kJ/mol/nm for forces. Example: "
             "--precision-col forces=1.0",
             kTuning,
             nullptr,
             {},
             true},
            {"codec",
             0,
             "CODEC",
             "Encoding to use for each group of frames",
             kTuning,
             "auto",
             {"auto", "temporal", "intra"}},
            {"gof-size", 0, "GOF_SIZE", "Frames per group, the unit of random access and parallelism", kTuning, "20"},
            {"predictor-order", 0, "ORDER", "Polynomial predictor order, or 0 to choose per group", kTuning, "0"},
            {"zstd-level", 0, "LEVEL", "Zstd compression level (1-22)", kTuning, "3"},
            {"no-adaptive", 0, nullptr,
             "Use the same precision for every atom, rather than spending more\nbits on atoms that move", kTuning},
            {"anchor-step", 0, "STEP", "Spacing of anchor frames. 1 disables B-frames", kTuning, "1"},
            {"no-intradelta", 0, nullptr, "Disable intra-frame sequential delta coding (XTC-style)", kTuning},
        },
        {{"lossless", "max-error"},
         {"lossless", "rigid"},
         {"rigid", "variable-atoms"},
         {"lossless", "precision"},
         {"lossless", "codec"},
         {"lossless", "no-adaptive"},
         {"lossless", "anchor-step"},
         {"lossless", "no-intradelta"},
         {"lossless", "precision-col"},
         {"precision", "max-error"}},
    };

    cli::Command decompress{
        "decompress",
        "Decompress a .mdz file back to a trajectory (or to stdout with -o -)",
        nullptr,
        {{"INPUT", "Input .mdz file, or - to read it from stdin"}},
        output_options("Output file, or - for stdout (requires --format) [default: <input>\nwith the extension "
                       "of the format it was compressed from, or .extxyz].\nThe format is taken from the extension "
                       "(case-insensitive): .extxyz,\n.xyz, .pdb, .lammpstrj, .dcd, .xtc, .npy, or .f32/.bin/.raw "
                       "for raw\nfloat32."),
    };

    cli::Command convert{
        "convert",
        "Convert directly between trajectory formats",
        "Convert directly between trajectory formats.\n\nFrames are streamed from reader to writer without "
        "compression, so no\nvalues are quantized. A warning is printed if the output format cannot\nstore "
        "everything in the input. An OUTPUT ending in .mdz is compressed\nwith the defaults of `mdz compress`.",
        {{"INPUT", "Input file (.mdz, .dcd, .xtc, .extxyz, .xyz, .pdb, .lammpstrj, .npy)"},
         {"OUTPUT", "Output file, or - for stdout (requires --format)"}},
        {output_options("")[1], output_options("")[2]},
    };

    cli::Command extract{
        "extract",
        "Write out a range of frames from a .mdz file",
        "Write out a range of frames from a .mdz file.\n\nOnly the groups that cover the range are decoded, so "
        "the cost depends\non the number of frames extracted, not on the size of the file.",
        {{"INPUT", "Input .mdz file, or - to read it from stdin"},
         {"RANGE",
          "Frames to extract, as START:END with END exclusive. `100:200` selects\n100 frames, `100:` runs to the "
          "end, `:200` starts at the beginning,\nand `42` selects a single frame."}},
        output_options("Output file, or - for stdout (requires --format)\n[default: <input>-<start>-<end>.<source "
                       "format>]"),
    };

    cli::Command info{"info",
                      "Show what a trajectory file contains",
                      nullptr,
                      {{"INPUT", "Input file (.mdz, .dcd, .xtc, .extxyz, .xyz, .pdb, .lammpstrj)"}},
                      {}};

    cli::Command completions{
        "completions",
        "Generate a shell completion script",
        "Generate a shell completion script and write it to stdout.\n\nExample: `mdz completions zsh > "
        "~/.zfunc/_mdz`",
        {{"SHELL", "Shell to generate completions for: bash, zsh or fish"}},
        {}};

    p.commands = {compress, decompress, convert, extract, info, completions};
    return p;
}

}  // namespace

int main(int argc, char** argv) {
    static const cli::Program prog = program();
    cli::Args a = cli::parse(prog, argc, argv);

    if (a.has("quiet") && a.has("verbose"))
        cli::usage_error(a, "the argument '--quiet' cannot be used with '--verbose'");
    const std::string color = a.get("color", "auto");
    cli::set_color(color == "always" ? cli::Color::Always : color == "never" ? cli::Color::Never : cli::Color::Auto);
    cli::set_verbosity(a.has("quiet") ? 0 : a.has("verbose") ? 2 : 1);
    cli::install_signal_handlers();
    if (a.has("lammps-units")) cli::set_lammps_units(mdz::LammpsUnits::from_style(a.get("lammps-units")));
    if (a.has("threads")) {
        const auto n = static_cast<unsigned>(a.get_int("threads", 0, 0, 1 << 16));
        mdz::set_num_threads(n);
        cli::detail("Using " + std::to_string(mdz::num_threads()) + " worker thread(s)");
    }

    const std::string cmd = a.command->name;
    try {
        if (cmd == "compress") return cli::cmd_compress(a);
        if (cmd == "decompress") return cli::cmd_decompress(a);
        if (cmd == "convert") return cli::cmd_convert(a);
        if (cmd == "extract") return cli::cmd_extract(a);
        if (cmd == "info") return cli::cmd_info(a);
        if (cmd == "completions") {
            const std::string shell = a.positional()[0];
            std::string script = cli::completion_script(prog, shell);
            if (script.empty())
                cli::usage_error(a,
                                 "invalid value '" + shell + "' for '<SHELL>'\n  [possible values: bash, zsh, fish]");
            cli::write_stdout(script);
            return 0;
        }
    } catch (const std::exception& e) {
        cli::fail(e.what());
    }
    return 0;
}
