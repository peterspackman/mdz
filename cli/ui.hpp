#pragma once
// Terminal output. Messages for people go to stderr, so that stdout carries
// only data. `mdz info` and `mdz completions` write their output to stdout.

#include <atomic>
#include <cstdint>
#include <string>
#include <string_view>
#include <thread>

namespace cli {

enum class Color { Auto, Always, Never };

// 0 = quiet (errors only), 1 = normal, 2 = verbose.
void set_verbosity(int level);
int verbosity();
void set_color(Color c);

// Informational line, hidden by -q.
void status(const std::string& msg);
// Extra detail, shown only with -v.
void detail(const std::string& msg);
// A yellow "warning:" line, hidden by -q.
void warn(const std::string& msg);
// Print a red "error:" line (always shown) and exit with status 1.
[[noreturn]] void fail(const std::string& msg);
// The message in green, when colour is enabled.
std::string ok(const std::string& msg);

// About 6 significant figures, without float rounding noise
// (0.00024999998 becomes "0.00025").
std::string fmt_small(float v);
// "%.<prec>f"
std::string fixed(double v, int prec);
// The shortest decimal string that parses back to v, e.g. "0.002".
std::string shortest(float v);
// "1 column" / "N columns".
std::string cols(std::size_t n);

// A spinner or progress bar drawn on stderr while work runs. It is drawn only
// when stderr is a terminal and -q is not given. finish() clears it.
class Progress {
public:
    // With total == 0, shows a spinner with the message and a running count.
    Progress(std::string msg, std::uint64_t total = 0);
    ~Progress();
    Progress(const Progress&) = delete;
    Progress& operator=(const Progress&) = delete;
    void set(std::uint64_t pos) { pos_.store(pos, std::memory_order_relaxed); }
    void finish();

private:
    void run();
    std::string msg_;
    std::uint64_t total_;
    std::atomic<std::uint64_t> pos_{0};
    std::atomic<bool> stop_{false};
    std::thread th_;
};

// Write data to stdout. If the pipe is closed, exits quietly with status 0.
void write_stdout(std::string_view s);
// Whether an error message came from writing to a closed pipe.
bool is_broken_pipe(const std::string& what);

}  // namespace cli
