/* Adds dump style mdz to a LAMMPS binary built with the PLUGIN package:
     plugin load /path/to/mdzplugin.so

   LAMMPS's plugin loader does not handle dump styles, so the plugin adds
   itself to the dump style table directly. It is then not listed by
   "plugin list" and cannot be unloaded. */

#include "lammps.h"
#include "output.h"
#include "utils.h"

#include "dump_mdz.h"

#if __has_include("creator_registry.h")
#include "creator_registry.h"
#define MDZ_STYLE_REGISTRY 1
#endif

using namespace LAMMPS_NS;

static Dump* mdzcreator(LAMMPS* lmp, int argc, char** argv) { return new DumpMDZ(lmp, argc, argv); }

extern "C" void lammpsplugin_init(void* ptr, void*, void*) {
    auto lmp = (LAMMPS*)ptr;
#ifdef MDZ_STYLE_REGISTRY
    // newer LAMMPS versions keep one dump style table per process
    Output::dump_styles().set_plugin("mdz", &mdzcreator);
#else
    // older versions keep one table per LAMMPS instance
    (*lmp->output->dump_map)["mdz"] = &mdzcreator;
#endif
    utils::logmesg(lmp, "Loaded dump style mdz (compressed, crash-tolerant .mdz trajectories)\n");
}
