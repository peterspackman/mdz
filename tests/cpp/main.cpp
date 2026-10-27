#include <chrono>
#include <cstring>
#include <exception>
#include <filesystem>
#include <random>

#include "test.hpp"

namespace mdz_test {

std::vector<Case>& registry() {
    static std::vector<Case> r;
    return r;
}

int g_failures = 0;

void fail(const char* file, int line, const std::string& what, bool fatal) {
    ++g_failures;
    std::fprintf(stderr, "  %s:%d: %s\n", file, line, what.c_str());
    if (fatal) throw Failure{what};
}

std::string scratch_dir() {
    static std::string dir = [] {
        auto base = std::filesystem::temp_directory_path() /
                    ("mdz_tests_" + std::to_string(std::random_device{}()) + "_" +
                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(base);
        return base.string();
    }();
    return dir;
}

}  // namespace mdz_test

int main(int argc, char** argv) {
    using namespace mdz_test;
    int failed_cases = 0, run = 0;
    for (const auto& c : registry()) {
        if (argc > 1 && !std::strstr(c.name, argv[1])) continue;
        ++run;
        const int before = g_failures;
        auto t0 = std::chrono::steady_clock::now();
        try {
            c.fn();
        } catch (const Failure&) {
        } catch (const std::exception& e) {
            ++g_failures;
            std::fprintf(stderr, "  unexpected exception: %s\n", e.what());
        }
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const bool ok = g_failures == before;
        failed_cases += !ok;
        std::printf("%s %s (%.1f ms)\n", ok ? "ok  " : "FAIL", c.name, ms);
    }
    std::error_code ec;
    std::filesystem::remove_all(scratch_dir(), ec);
    std::printf("\n%d/%d passed\n", run - failed_cases, run);
    return failed_cases ? 1 : 0;
}
