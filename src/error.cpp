#include "mdz/error.hpp"

namespace mdz {

void throw_invalid(const std::string& what) { throw Error(ErrorKind::InvalidInput, what); }
void throw_corrupt(const std::string& what) { throw Error(ErrorKind::Corrupt, what); }
void throw_io(const std::string& what) { throw Error(ErrorKind::Io, what); }
void throw_range(const std::string& what) { throw Error(ErrorKind::OutOfRange, what); }

}  // namespace mdz
