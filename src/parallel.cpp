#include "parallel.hpp"

#include <atomic>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

#include "mdz/mdz.hpp"

namespace mdz {
namespace {

std::atomic<unsigned> g_threads{0};

unsigned default_threads() noexcept {
    if (const char* env = std::getenv("MDZ_NUM_THREADS")) {
        long v = std::strtol(env, nullptr, 10);
        if (v > 0) return static_cast<unsigned>(v);
    }
    unsigned hw = std::thread::hardware_concurrency();
    return hw ? hw : 1;
}

}  // namespace

void set_num_threads(unsigned n) noexcept { g_threads.store(n, std::memory_order_relaxed); }

unsigned num_threads() noexcept {
    unsigned n = g_threads.load(std::memory_order_relaxed);
    if (n) return n;
    static const unsigned def = default_threads();
    return def;
}

namespace detail {

void parallel_for(std::size_t n, const std::function<void(std::size_t)>& fn) {
    std::size_t workers = std::min<std::size_t>(num_threads(), n);
    if (workers <= 1) {
        for (std::size_t i = 0; i < n; ++i) fn(i);
        return;
    }
    std::atomic<std::size_t> next{0};
    std::exception_ptr error;
    std::mutex error_mutex;
    auto work = [&] {
        for (;;) {
            std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= n) return;
            try {
                fn(i);
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!error) error = std::current_exception();
                next.store(n, std::memory_order_relaxed);  // start no further items
            }
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(workers - 1);
    for (std::size_t t = 1; t < workers; ++t) pool.emplace_back(work);
    work();
    for (auto& th : pool) th.join();
    if (error) std::rethrow_exception(error);
}

}  // namespace detail
}  // namespace mdz
