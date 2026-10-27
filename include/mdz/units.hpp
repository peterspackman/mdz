#pragma once
// Unit conversions. mdz's internal units are nm, ps, nm/ps, kJ/mol/nm,
// kJ/mol and e. A trajectory may instead be stored in a LAMMPS unit style, so
// that a LAMMPS dump round-trips exactly.

#include <array>
#include <string_view>

namespace mdz {

namespace units {
inline constexpr double kAngstromToNm = 0.1;
inline constexpr double kNmToAngstrom = 10.0;
inline constexpr double kBohrToNm = 0.052917721;
inline constexpr double kKcalToKj = 4.184;
inline constexpr double kEvToKjmol = 96.4853;
inline constexpr double kHartreeToKjmol = 2625.5;

// Convert a value given in the named unit to internal units, or from nm for
// length_from_nm. An unrecognized unit name is assumed to be internal already,
// and the value is returned unchanged.
double length_to_nm(double v, std::string_view unit) noexcept;
double length_from_nm(double v, std::string_view unit) noexcept;
double energy_to_kjmol(double v, std::string_view unit) noexcept;
double velocity_to_nm_ps(double v, std::string_view unit) noexcept;
double force_to_kjmol_nm(double v, std::string_view unit) noexcept;
}  // namespace units

// How one LAMMPS unit style relates to the internal units. Multiplying a
// value in the style's units by the matching factor gives internal units.
// The *_unit fields are UTF-8 display names.
struct LammpsUnits {
    std::string_view style;
    double length;
    double time;
    double velocity;
    double force;
    double energy;
    std::string_view length_unit;
    std::string_view velocity_unit;
    std::string_view force_unit;

    // The style named `name`, or nullptr if LAMMPS has no such style. The
    // returned object has static storage duration.
    static const LammpsUnits* from_style(std::string_view name) noexcept;
    // The names of every supported style.
    static const std::array<std::string_view, 8>& styles() noexcept;
};

inline bool operator==(const LammpsUnits& a, const LammpsUnits& b) noexcept { return a.style == b.style; }

}  // namespace mdz
