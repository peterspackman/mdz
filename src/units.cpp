#include "mdz/units.hpp"

namespace mdz {
namespace units {

double length_to_nm(double v, std::string_view u) noexcept {
    if (u == "A" || u == "Ang" || u == "Angstrom" || u == "angstrom") return v * kAngstromToNm;
    if (u == "bohr" || u == "Bohr") return v * kBohrToNm;
    if (u == "pm") return v * 0.001;
    return v;
}

double length_from_nm(double v, std::string_view u) noexcept {
    if (u == "A" || u == "Ang" || u == "Angstrom" || u == "angstrom") return v * kNmToAngstrom;
    if (u == "bohr" || u == "Bohr") return v * (1.0 / kBohrToNm);
    if (u == "pm") return v * 1000.0;
    return v;
}

double energy_to_kjmol(double v, std::string_view u) noexcept {
    if (u == "kcal/mol" || u == "kcal") return v * kKcalToKj;
    if (u == "eV") return v * kEvToKjmol;
    if (u == "Ha" || u == "hartree" || u == "Hartree") return v * kHartreeToKjmol;
    return v;
}

double velocity_to_nm_ps(double v, std::string_view u) noexcept {
    if (u == "A/ps" || u == "Ang/ps") return v * kAngstromToNm;
    if (u == "A/fs" || u == "Ang/fs") return v * kAngstromToNm * 1000.0;
    if (u == "nm/fs") return v * 1000.0;
    return v;
}

double force_to_kjmol_nm(double v, std::string_view u) noexcept {
    if (u == "kJ/mol/A" || u == "kJ/mol/Ang") return v * kNmToAngstrom;
    if (u == "eV/A" || u == "eV/Ang") return v * kEvToKjmol * kNmToAngstrom;
    if (u == "Ha/bohr" || u == "hartree/bohr") return v * kHartreeToKjmol * (1.0 / kBohrToNm);
    return v;
}

}  // namespace units

namespace {

constexpr double kNA = 6.02214076e23;
// CODATA 2018.
constexpr double kEvToKjmolExact = 1.602176634e-19 * kNA / 1000.0;
constexpr double kBohrToNmExact = 0.0529177210903;
constexpr double kHartreeToKjmolExact = 4.3597447222071e-18 * kNA / 1000.0;

// A style defined in SI-derived per-particle units, with energy in J and
// force in N. Do not reorder this arithmetic. The precisions stored in files
// are derived from these factors, and any change in rounding would change
// them.
LammpsUnits si_like(std::string_view style, double length, double time, double force_n, double energy_j,
                    std::string_view lu, std::string_view vu, std::string_view fu) {
    const double j = kNA / 1000.0;
    return {style, length, time, length / time, force_n * j * 1e-9, energy_j * j, lu, vu, fu};
}

const std::array<LammpsUnits, 8>& table() {
    static const std::array<LammpsUnits, 8> t = {
        LammpsUnits{"real", 0.1, 1e-3, 0.1 / 1e-3, units::kKcalToKj / 0.1, units::kKcalToKj, "\xC3\x85",
                    "\xC3\x85/fs", "kcal/mol/\xC3\x85"},
        LammpsUnits{"metal", 0.1, 1.0, 0.1, kEvToKjmolExact / 0.1, kEvToKjmolExact, "\xC3\x85", "\xC3\x85/ps",
                    "eV/\xC3\x85"},
        si_like("si", 1e9, 1e12, 1.0, 1.0, "m", "m/s", "N"),
        si_like("cgs", 1e7, 1e12, 1e-5, 1e-7, "cm", "cm/s", "dyne"),
        LammpsUnits{"electron", kBohrToNmExact, 1e-3, kBohrToNmExact / 1.03275e-3,
                    kHartreeToKjmolExact / kBohrToNmExact, kHartreeToKjmolExact, "Bohr", "Bohr/atu", "Hartree/Bohr"},
        si_like("micro", 1e3, 1e6, 1e-9, 1e-15, "\xC2\xB5m", "\xC2\xB5m/\xC2\xB5s",
                "pg\xC2\xB7\xC2\xB5m/\xC2\xB5s\xC2\xB2"),
        si_like("nano", 1.0, 1e3, 1e-12, 1e-21, "nm", "nm/ns", "ag\xC2\xB7nm/ns\xC2\xB2"),
        LammpsUnits{"lj", 1.0, 1.0, 1.0, 1.0, 1.0, "\xCF\x83", "\xCF\x83/\xCF\x84", "\xCE\xB5/\xCF\x83"},
    };
    return t;
}

}  // namespace

const LammpsUnits* LammpsUnits::from_style(std::string_view name) noexcept {
    for (const auto& u : table())
        if (u.style == name) return &u;
    return nullptr;
}

const std::array<std::string_view, 8>& LammpsUnits::styles() noexcept {
    static const std::array<std::string_view, 8> s = {"real", "metal", "si", "cgs", "electron", "micro", "nano", "lj"};
    return s;
}

}  // namespace mdz
