#pragma once
// Cleanup of temporary files on SIGINT, SIGTERM and SIGHUP.
//
// Every output is written to a temporary file and then renamed into place.
// While a temporary file exists, its path is registered here. On a signal, the
// handler deletes every registered path, prints "interrupted" (unless -q is
// given) and exits with status 130. Destination files are never modified
// before the rename, so an existing file stays complete. Registration and the
// final rename run with these signals blocked, so the handler never sees a
// half-finished rename.

#include <string>

namespace cli {

constexpr int kExitInterrupted = 130;

void install_signal_handlers();

// A registered temporary file. If it is destroyed without a call to
// rename_into() (for example, after an error), the file is deleted and
// unregistered.
class TempFile {
public:
    explicit TempFile(std::string path);
    ~TempFile();
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
    const std::string& path() const { return path_; }
    // Rename over dest and unregister. Throws mdz::Error(Io).
    void rename_into(const std::string& dest);

private:
    std::string path_;
    int slot_;
    bool done_ = false;
};

}  // namespace cli
