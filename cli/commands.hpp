#pragma once

#include "args.hpp"

namespace cli {

int cmd_compress(const Args& a);
int cmd_decompress(const Args& a);
int cmd_extract(const Args& a);
int cmd_convert(const Args& a);
int cmd_info(const Args& a);

}  // namespace cli
