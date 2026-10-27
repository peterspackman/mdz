// Writes a few frames one at a time, as an MD engine would, and reads them back.
#include <cmath>
#include <cstdio>

#include "mdz/mdz.hpp"

int main() {
    const std::size_t natoms = 100;
    {
        auto w = mdz::Writer::create("consumer.mdz");
        for (int step = 0; step < 50; ++step) {
            mdz::FrameBatch f(natoms, 1);
            auto& p = f.columns["positions"];
            for (std::size_t i = 0; i < natoms * 3; ++i) p.push_back(std::sin(float(i)) + 0.001f * float(step));
            f.int_frame_props["step"] = {step * 100};
            f.box = std::vector<mdz::Box>{{3, 0, 0, 0, 3, 0, 0, 0, 3}};
            w.write(f);
        }
        w.close();
    }
    auto r = mdz::Reader::open("consumer.mdz");
    auto frame = r.read_frame(42);
    const float want = std::sin(7.0f) + 0.042f;
    if (r.nframes() != 50 || std::fabs(frame[7] - want) > 0.001f) {
        std::fprintf(stderr, "read back the wrong data\n");
        return 1;
    }
    std::printf("mdz %s: ok\n", std::string(mdz::version()).c_str());
    return 0;
}
