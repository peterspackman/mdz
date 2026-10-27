#pragma once
// A minimal test harness: TEST(name) { CHECK(...); REQUIRE(...); }.

#include <cstdio>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace mdz_test {

struct Case {
    const char* name;
    void (*fn)();
};

std::vector<Case>& registry();
struct Reg {
    Reg(const char* n, void (*f)()) { registry().push_back({n, f}); }
};

struct Failure {
    std::string msg;
};
extern int g_failures;

void fail(const char* file, int line, const std::string& what, bool fatal);
std::string scratch_dir();  // a fresh temporary directory per run

}  // namespace mdz_test

#define MDZ_CAT2(a, b) a##b
#define MDZ_CAT(a, b) MDZ_CAT2(a, b)
#define TEST(name)                                                       \
    static void name();                                                  \
    static ::mdz_test::Reg MDZ_CAT(reg_, name)(#name, name);             \
    static void name()

#define CHECK(cond) \
    ((cond) ? (void)0 : ::mdz_test::fail(__FILE__, __LINE__, "CHECK(" #cond ")", false))
#define REQUIRE(cond) \
    ((cond) ? (void)0 : ::mdz_test::fail(__FILE__, __LINE__, "REQUIRE(" #cond ")", true))
#define CHECK_EQ(a, b)                                                                   \
    do {                                                                                 \
        auto&& va_ = (a);                                                                \
        auto&& vb_ = (b);                                                                \
        if (!(va_ == vb_)) {                                                             \
            std::ostringstream os_;                                                      \
            os_ << "CHECK_EQ(" #a ", " #b "): " << va_ << " != " << vb_;                 \
            ::mdz_test::fail(__FILE__, __LINE__, os_.str(), false);                      \
        }                                                                                \
    } while (0)
#define CHECK_THROWS(expr)                                                                       \
    do {                                                                                         \
        bool threw_ = false;                                                                     \
        try {                                                                                    \
            (void)(expr);                                                                        \
        } catch (...) {                                                                          \
            threw_ = true;                                                                       \
        }                                                                                        \
        if (!threw_) ::mdz_test::fail(__FILE__, __LINE__, "CHECK_THROWS(" #expr ")", false);   \
    } while (0)
