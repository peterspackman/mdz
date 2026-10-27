#include "bytes.hpp"

namespace mdz::detail {

void corrupt(std::string_view what) {
    std::string msg = "corrupt .mdz file: ";
    msg += what;
    throw Error(ErrorKind::Corrupt, msg);
}

std::size_t checked_add(std::size_t a, std::size_t b, std::string_view what) {
    if (a > SIZE_MAX - b) corrupt(what);
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, std::string_view what) {
    if (b != 0 && a > SIZE_MAX / b) corrupt(what);
    return a * b;
}

}  // namespace mdz::detail
