// Replacement main() for compilers without libFuzzer. It runs the target over
// each file given on the command line, recursing into directories, which is
// useful for replaying a saved corpus or reproducing a crash. With -mutate=N
// it also runs N random mutations of each input (bit flips, overwrites,
// truncation and insertion). This is a crude fuzzer, but combined with the
// sanitizers it still finds real bugs.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size);

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    std::size_t n = 0;
    long mutations = 0;
    std::mt19937_64 rng(12345);
    auto run_one = [&](std::vector<std::uint8_t> b) {
        LLVMFuzzerTestOneInput(b.data(), b.size());
        ++n;
        for (long m = 0; m < mutations; ++m) {
            std::vector<std::uint8_t> v = b;
            const int edits = 1 + static_cast<int>(rng() % 4);
            for (int e = 0; e < edits && !v.empty(); ++e) {
                const std::size_t at = rng() % v.size();
                switch (rng() % 5) {
                    case 0:
                        v[at] ^= static_cast<std::uint8_t>(1u << (rng() % 8));
                        break;
                    case 1:
                        v[at] = static_cast<std::uint8_t>(rng());
                        break;
                    case 2:
                        v.resize(at);
                        break;
                    case 3:
                        v.insert(v.begin() + static_cast<std::ptrdiff_t>(at), static_cast<std::uint8_t>(rng()));
                        break;
                    case 4: {  // an interesting 32-bit value
                        static const std::uint32_t vals[] = {0, 1, 0x7FFFFFFF, 0x80000000u, 0xFFFFFFFFu, 65535, 256};
                        std::uint32_t x = vals[rng() % 7];
                        if (at + 4 <= v.size()) std::memcpy(v.data() + at, &x, 4);
                        break;
                    }
                }
            }
            LLVMFuzzerTestOneInput(v.data(), v.size());
            ++n;
        }
    };
    auto run = [&](const fs::path& p) {
        std::ifstream f(p, std::ios::binary);
        run_one(std::vector<std::uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()));
    };
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "-mutate=", 8) == 0) {
            mutations = std::strtol(argv[i] + 8, nullptr, 10);
            continue;
        }
        if (fs::is_directory(argv[i])) {
            for (const auto& e : fs::recursive_directory_iterator(argv[i]))
                if (e.is_regular_file()) run(e.path());
        } else {
            run(argv[i]);
        }
    }
    std::printf("ran %zu inputs\n", n);
    return 0;
}
