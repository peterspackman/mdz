#include "ui.hpp"

#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#include <io.h>
#define MDZ_ISATTY _isatty
#define MDZ_FILENO _fileno
#else
#include <unistd.h>
#define MDZ_ISATTY isatty
#define MDZ_FILENO fileno
#endif

namespace cli {
namespace {

std::atomic<int> g_verbosity{1};
Color g_color = Color::Auto;

bool stderr_tty() { return MDZ_ISATTY(MDZ_FILENO(stderr)) != 0; }

bool color_on() {
    if (g_color == Color::Always) return true;
    if (g_color == Color::Never) return false;
    const char* nc = std::getenv("NO_COLOR");
    return stderr_tty() && !(nc && *nc);
}

std::string paint(const char* code, const std::string& s) {
    return color_on() ? std::string("\x1b[") + code + "m" + s + "\x1b[0m" : s;
}

}  // namespace

void set_verbosity(int level) { g_verbosity.store(level); }
int verbosity() { return g_verbosity.load(); }
void set_color(Color c) { g_color = c; }

void status(const std::string& msg) {
    if (verbosity() >= 1) std::fprintf(stderr, "%s\n", msg.c_str());
}

void detail(const std::string& msg) {
    if (verbosity() >= 2) std::fprintf(stderr, "%s\n", msg.c_str());
}

void warn(const std::string& msg) {
    if (verbosity() >= 1) std::fprintf(stderr, "%s %s\n", paint("1;33", "warning:").c_str(), msg.c_str());
}

void fail(const std::string& msg) {
    std::fprintf(stderr, "%s %s\n", paint("1;31", "error:").c_str(), msg.c_str());
    std::exit(1);
}

std::string ok(const std::string& msg) { return paint("32", msg); }

std::string fixed(double v, int prec) {
    char b[64];
    std::snprintf(b, sizeof b, "%.*f", prec, v);
    return b;
}

std::string fmt_small(float v) {
    if (v == 0.0f || !std::isfinite(v)) {
        char b[32];
        std::snprintf(b, sizeof b, "%g", static_cast<double>(v));
        return b;
    }
    int decimals = 5 - static_cast<int>(std::floor(std::log10(std::fabs(v))));
    decimals = decimals < 0 ? 0 : (decimals > 15 ? 15 : decimals);
    std::string s = fixed(v, decimals);
    if (s.find('.') != std::string::npos) {
        while (!s.empty() && s.back() == '0') s.pop_back();
        if (!s.empty() && s.back() == '.') s.pop_back();
    }
    return s;
}

std::string shortest(float v) {
    char b[64];
    auto r = std::to_chars(b, b + sizeof b, v);
    return std::string(b, static_cast<std::size_t>(r.ptr - b));
}

std::string cols(std::size_t n) { return std::to_string(n) + (n == 1 ? " column" : " columns"); }

Progress::Progress(std::string msg, std::uint64_t total) : msg_(std::move(msg)), total_(total) {
    if (verbosity() >= 1 && stderr_tty()) th_ = std::thread([this] { run(); });
}

Progress::~Progress() { finish(); }

void Progress::finish() {
    if (!th_.joinable()) return;
    stop_.store(true);
    th_.join();
    std::fprintf(stderr, "\r\x1b[2K");
    std::fflush(stderr);
}

void Progress::run() {
    static const char* spin = "|/-\\";
    const auto t0 = std::chrono::steady_clock::now();
    for (unsigned tick = 0; !stop_.load(); ++tick) {
        const std::uint64_t pos = pos_.load(std::memory_order_relaxed);
        if (total_) {
            const int width = 28;
            const int done = static_cast<int>(width * std::min<double>(1.0, double(pos) / double(total_)));
            std::string bar(static_cast<std::size_t>(done), '=');
            if (done < width) bar += '>';
            bar.resize(width, ' ');
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const double rate = secs > 0 ? double(pos) / secs : 0;
            std::fprintf(stderr, "\r\x1b[2K%s [%s] %llu/%llu frames (%.0f/s)", msg_.c_str(),
                         paint("36", bar).c_str(), static_cast<unsigned long long>(pos),
                         static_cast<unsigned long long>(total_), rate);
        } else if (pos) {
            std::fprintf(stderr, "\r\x1b[2K%s %s (%llu frames)", paint("36", std::string(1, spin[tick % 4])).c_str(),
                         msg_.c_str(), static_cast<unsigned long long>(pos));
        } else {
            std::fprintf(stderr, "\r\x1b[2K%s %s", paint("36", std::string(1, spin[tick % 4])).c_str(), msg_.c_str());
        }
        std::fflush(stderr);
        for (int i = 0; i < 10 && !stop_.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

bool is_broken_pipe(const std::string& what) { return what.find(std::strerror(EPIPE)) != std::string::npos; }

void write_stdout(std::string_view s) {
    if (std::fwrite(s.data(), 1, s.size(), stdout) != s.size() || std::fflush(stdout) != 0) {
        if (errno == EPIPE) std::exit(0);
        fail(std::string("writing to stdout: ") + std::strerror(errno));
    }
}

}  // namespace cli
