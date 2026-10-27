#pragma once
// Incremental writing of .mdz files. Frames are buffered until a group is
// complete, and each group is then encoded and appended as a self-contained
// record. If the process dies, the file still holds every complete group and
// can be read or appended to.

#include <cstdint>
#include <memory>
#include <string>

#include "mdz/frame_batch.hpp"
#include "mdz/options.hpp"

namespace mdz {

class OutputStream;

// A Writer is not thread-safe. Errors are reported by throwing mdz::Error.
class Writer {
public:
    // Create `path`, or truncate it if it exists. Throws Error(Io) if it
    // cannot be opened and Error(InvalidInput) for invalid options.
    static Writer create(const std::string& path, CompressOptions opts = {});
    // Write to any stream. The stream does not need to be seekable.
    static Writer create(std::unique_ptr<OutputStream> out, CompressOptions opts = {});
    // Reopen an existing .mdz file and continue after its last complete group,
    // discarding any partial record and the footer. `opts` must produce the
    // same columns and settings the file was created with; otherwise encoding
    // the first new group throws Error(InvalidInput). Throws
    // Error(InvalidInput) if the file is not a .mdz file, Error(Corrupt) if it
    // is damaged and Error(Io) if it cannot be read or truncated.
    static Writer append(const std::string& path, CompressOptions opts = {});

    Writer(Writer&&) noexcept;
    // Closes this writer first if it is still open, ignoring errors.
    Writer& operator=(Writer&&) noexcept;
    // Closes the writer if it is still open, unless it is a new file that
    // never received a frame. Errors are ignored, so call close() to see them.
    ~Writer();

    // Add the frames of `batch`. Every batch must carry the same columns,
    // frame properties and static metadata as the first, and each per-frame
    // array must have exactly batch.nframes entries. Throws
    // Error(InvalidInput) for a mismatched batch or a closed writer, and
    // Error(Io) if writing a completed group fails.
    void write(const FrameBatch& batch);
    // Encode buffered frames as a group, which may be short, and hand all
    // written bytes to the OS. Short groups compress slightly worse, so flush
    // at checkpoints rather than after every frame. Throws
    // Error(InvalidInput) if the writer is closed and Error(Io) if writing
    // fails.
    void flush();
    // Encode buffered frames, write the footer and close the stream. Calling
    // it again does nothing. Throws Error(InvalidInput) if no frames were ever
    // written, since the file then has no columns to describe.
    void close();

    // Frames received so far, including buffered ones and, when appending,
    // those already in the file.
    std::uint64_t frames_written() const noexcept;
    // Frames already encoded into the file, which is what a crash at this
    // point would preserve.
    std::uint64_t frames_committed() const noexcept;
    bool closed() const noexcept;
    // Static metadata of the file being appended to, or nullptr for a new
    // file.
    const FrameBatch* appended_metadata() const noexcept;

    struct Impl;

private:
    explicit Writer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace mdz
