#pragma once
// Error reporting. Every failure in mdz is reported by throwing mdz::Error,
// whose kind() says what went wrong.

#include <stdexcept>
#include <string>

namespace mdz {

enum class ErrorKind {
    InvalidInput,  // A bad argument, or a value the format cannot represent.
    Corrupt,       // Input that is not valid .mdz data (or valid data of another format).
    Io,            // The operating system reported a failed open, read or write.
    OutOfRange,    // A frame or index past the end.
};

class Error : public std::runtime_error {
public:
    Error(ErrorKind kind, const std::string& what) : std::runtime_error(what), kind_(kind) {}
    ErrorKind kind() const noexcept { return kind_; }

private:
    ErrorKind kind_;
};

// Throw an Error of the corresponding kind with message `what`.
[[noreturn]] void throw_invalid(const std::string& what);
[[noreturn]] void throw_corrupt(const std::string& what);
[[noreturn]] void throw_io(const std::string& what);
[[noreturn]] void throw_range(const std::string& what);

}  // namespace mdz
