#pragma once
// The parallel loop used for all of mdz's parallel work, which always
// consists of independent items such as groups of frames.
//
// parallel_for runs fn(i) for each i in [0, n) on up to num_threads()
// threads, starting a fresh set of threads per call. It runs inline when
// there is one item or one thread. Callers write results by index, so output
// order does not depend on scheduling. If fn throws, no further items are
// started, and the first exception is rethrown once every worker has
// finished.

#include <cstddef>
#include <functional>

namespace mdz::detail {

void parallel_for(std::size_t n, const std::function<void(std::size_t)>& fn);

}  // namespace mdz::detail
