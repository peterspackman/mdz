#pragma once
// Constructors for the built-in formats, which registry.cpp puts in the
// format table. Each format lives in its own file.

#include <memory>

#include "mdz/formats.hpp"

namespace mdz::detail {

#define MDZ_FORMAT(name)                                                                                        \
    std::unique_ptr<TrajectoryReader> make_##name##_reader(std::unique_ptr<InputStream>, const FormatOptions&); \
    std::unique_ptr<TrajectoryWriter> make_##name##_writer(std::unique_ptr<OutputStream>, std::size_t natoms,   \
                                                           const FormatOptions&);
MDZ_FORMAT(dcd)
MDZ_FORMAT(xtc)
MDZ_FORMAT(npy)
MDZ_FORMAT(extxyz)
MDZ_FORMAT(lammps)
MDZ_FORMAT(pdb)
MDZ_FORMAT(xyz)
MDZ_FORMAT(mdz)
#undef MDZ_FORMAT

// Opens a .mdz file by path. The file is memory-mapped rather than read
// through a stream.
std::unique_ptr<TrajectoryReader> make_mdz_reader_path(const std::string& path);

// Calls read_frame up to n times and merges the single-frame batches it returns
// into one. Stops early when read_frame returns nullopt. The text readers use
// this to implement read_batch.
template <class F>
std::optional<FrameBatch> collect_frames(std::size_t n, F&& read_frame) {
    std::optional<FrameBatch> all;
    for (std::size_t i = 0; i < n; ++i) {
        std::optional<FrameBatch> f = read_frame();
        if (!f) break;
        if (!all)
            all = std::move(f);
        else
            all->append(*f);
    }
    return all;
}

// Appends a frame to *out as a variable-atom-count batch. A reader calls this
// for every frame once it has seen the atom count change. The reader stays in
// variable mode from then on, so later batches are marked variable even if all
// their frames have the same atom count.
inline void append_varying(std::optional<FrameBatch>& out, FrameBatch frame) {
    if (!out || out->nframes == 0) {
        frame.make_variable();
        out = std::move(frame);
        return;
    }
    out->make_variable();
    out->append(frame);
}

// Removes the last frame from a fixed-count batch and returns it. Used by
// readers that decode frames directly into a batch and only find out at the
// end of a frame that its atom count differs. This copies the batch, which
// happens at most once per reader.
inline FrameBatch pop_last_frame(FrameBatch& b) {
    FrameBatch last = b.slice_frames(b.nframes - 1, b.nframes);
    b = b.slice_frames(0, b.nframes - 1);
    return last;
}

}  // namespace mdz::detail
