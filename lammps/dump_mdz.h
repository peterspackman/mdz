/* -*- c++ -*- ----------------------------------------------------------
   dump mdz: write LAMMPS trajectories as compressed, crash-tolerant,
   appendable .mdz files through the mdz C API (mdz/mdz.h).

   This file is distributed under the terms of the GNU General Public
   License, like LAMMPS itself.
------------------------------------------------------------------------- */

#ifdef DUMP_CLASS
// clang-format off
DumpStyle(mdz,DumpMDZ);
// clang-format on
#else

#ifndef LMP_DUMP_MDZ_H
#define LMP_DUMP_MDZ_H

#include "dump_custom.h"

#include <map>
#include <string>
#include <vector>

struct mdz_writer;

namespace LAMMPS_NS {

// The dump command's arguments, with the default fields filled in. This is a
// base class so that it is built before DumpCustom, which keeps pointers
// into it.
struct DumpMDZArgs {
  DumpMDZArgs(int narg, char **arg);
  std::vector<std::string> strings;
  std::vector<char *> argv;
};

class DumpMDZ : private DumpMDZArgs, public DumpCustom {
 public:
  DumpMDZ(class LAMMPS *, int, char **);
  ~DumpMDZ() override;

  double memory_usage() override;

 protected:
  bigint natoms;    // atoms per frame, fixed for the file unless variable
  bigint nframe;    // atoms in the current frame
  bigint nfilled;   // atoms of the current frame received so far
  int frame_open;   // header received, but not all atoms yet
  int skip_frame;   // current frame does not match the file and is dropped
  bigint nskipped;  // frames dropped so far

  // settings, fixed once the file is opened
  int lossless_flag;
  int variable_flag;    // atom count may change between frames
  int rigid_flag;       // 0 off, 1 detect rigid bodies, 2 one body per molecule
  double precision, vel_precision, force_precision;
  std::map<std::string, double> column_precision;    // general float columns
  int gof_size;

  // where each quantity is in a packed per-atom record, -1 if absent
  int f_id, f_type, f_x[3], f_v[3], f_f[3], f_image[3], f_mol;
  int unwrapped;    // positions are xu yu zu
  struct Column {
    std::string name;
    int field;
    int is_int;
  };
  std::vector<Column> extra;    // every other field, stored as a general column

  // the frame being assembled on the writing proc
  bigint step;
  double time, origin[3], box[9];
  std::vector<double> xs, vs, fs;
  std::vector<int> images;
  std::vector<int64_t> frame_ids;
  std::vector<int> frame_types;
  std::vector<std::vector<double>> extra_float;
  std::vector<std::vector<int>> extra_int;
  std::vector<int64_t> first_ids;    // atom ID order stored in the file

  mdz_writer *writer;
  int nwritten;    // frames handed to the writer by this dump

  void init_style() override;
  int modify_param(int, char **) override;
  void openfile() override;
  void write_header(bigint) override;
  void write_data(int, double *) override;

  void map_fields();
  void size_frame(bigint);
  void skip(const std::string &);
  void write_frame();
  void check(int, const char *);
};

}    // namespace LAMMPS_NS

#endif
#endif
