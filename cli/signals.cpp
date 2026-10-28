#include "signals.hpp"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>

#include "mdz/error.hpp"
#include "paths.hpp"
#include "ui.hpp"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <signal.h>
#include <unistd.h>
#endif

namespace cli {
namespace {

// Fixed-size storage, so the signal handler can read it without allocating.
// A run has at most a few temporary files open at once.
constexpr int kSlots = 16;
constexpr std::size_t kMaxPath = 4096;
char g_paths[kSlots][kMaxPath];
std::atomic<bool> g_used[kSlots];
std::atomic<bool> g_interrupted{false};
std::mutex g_lock;  // serializes registration against the Windows handler thread

#if defined(_WIN32)
[[noreturn]] void cleanup_and_exit() {
    std::lock_guard<std::mutex> l(g_lock);
    g_interrupted.store(true);
    for (int i = 0; i < kSlots; ++i)
        if (g_used[i].load()) DeleteFileA(g_paths[i]);
    if (verbosity() >= 1) std::fputs("\ninterrupted\n", stderr);
    std::_Exit(kExitInterrupted);
}
BOOL WINAPI on_console(DWORD) { cleanup_and_exit(); }
struct Block {
    Block() {}  // user-provided, so MSVC doesn't report it as an unused local
};
#else
void on_signal(int) {
    g_interrupted.store(true);
    for (int i = 0; i < kSlots; ++i)
        if (g_used[i].load()) unlink(g_paths[i]);
    if (verbosity() >= 1) {
        static const char msg[] = "\ninterrupted\n";
        ssize_t r = write(2, msg, sizeof msg - 1);
        (void)r;
    }
    _exit(kExitInterrupted);
}
// Block the handled signals for the lifetime of the object.
struct Block {
    sigset_t old;
    Block() {
        sigset_t s;
        sigemptyset(&s);
        sigaddset(&s, SIGINT);
        sigaddset(&s, SIGTERM);
        sigaddset(&s, SIGHUP);
        pthread_sigmask(SIG_BLOCK, &s, &old);
    }
    ~Block() { pthread_sigmask(SIG_SETMASK, &old, nullptr); }
};
#endif

}  // namespace

void install_signal_handlers() {
#if defined(_WIN32)
    SetConsoleCtrlHandler(on_console, TRUE);
#else
    struct sigaction sa;
    std::memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sigaddset(&sa.sa_mask, SIGINT);
    sigaddset(&sa.sa_mask, SIGTERM);
    sigaddset(&sa.sa_mask, SIGHUP);
    for (int s : {SIGINT, SIGTERM, SIGHUP}) sigaction(s, &sa, nullptr);
    // Ignore SIGPIPE so that a closed stdout pipe surfaces as a write error,
    // which the commands handle.
    signal(SIGPIPE, SIG_IGN);
#endif
}

TempFile::TempFile(std::string path) : path_(std::move(path)), slot_(-1) {
    if (path_.size() >= kMaxPath) throw mdz::Error(mdz::ErrorKind::Io, path_ + ": path too long");
    Block b;
    std::lock_guard<std::mutex> l(g_lock);
    for (int i = 0; i < kSlots; ++i)
        if (!g_used[i].load()) {
            std::memcpy(g_paths[i], path_.c_str(), path_.size() + 1);
            g_used[i].store(true);
            slot_ = i;
            break;
        }
    if (slot_ < 0) throw mdz::Error(mdz::ErrorKind::Io, "too many temporary files");
}

TempFile::~TempFile() {
    if (done_) return;
    Block b;
    std::lock_guard<std::mutex> l(g_lock);
    std::remove(path_.c_str());
    g_used[slot_].store(false);
}

void TempFile::rename_into(const std::string& dest) {
    Block b;
    std::lock_guard<std::mutex> l(g_lock);
    if (g_interrupted.load()) throw mdz::Error(mdz::ErrorKind::Io, "interrupted");
    std::error_code ec;
    std::filesystem::rename(utf8_path(path_), utf8_path(dest), ec);
    if (ec) throw mdz::Error(mdz::ErrorKind::Io, "renaming " + path_ + " to " + dest + ": " + ec.message());
    g_used[slot_].store(false);
    done_ = true;
}

}  // namespace cli
