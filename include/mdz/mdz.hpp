#pragma once
// mdz compresses molecular dynamics trajectories while keeping fast random
// access to any frame. This header includes the whole C++ API.

#include <string_view>

#include "mdz/column.hpp"
#include "mdz/error.hpp"
#include "mdz/formats.hpp"
#include "mdz/frame_batch.hpp"
#include "mdz/metadata.hpp"
#include "mdz/options.hpp"
#include "mdz/reader.hpp"
#include "mdz/trajectory.hpp"
#include "mdz/units.hpp"
#include "mdz/writer.hpp"

namespace mdz {

// The library version, for example "0.4.0".
std::string_view version() noexcept;

// Set the number of worker threads used for bulk compression and
// decompression. 0 restores the default, which is MDZ_NUM_THREADS if set and
// otherwise one thread per hardware thread. Applications that call mdz from
// inside their own parallel regions should set 1. Safe to call from any
// thread.
void set_num_threads(unsigned n) noexcept;
// The number of worker threads currently in effect.
unsigned num_threads() noexcept;

}  // namespace mdz
