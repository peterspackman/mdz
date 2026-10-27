#pragma once
// A small table-driven command-line parser. It supports subcommands, global
// options that may appear anywhere, the forms --long=value, --long value and
// -s value, repeatable options, generated --help and shell completions.
// Usage errors exit with status 2.

#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace cli {

struct Opt {
    const char* name;               // long name, without "--"
    char short_name;                // 0 for none
    const char* value_name;         // nullptr for a flag
    const char* help;               // first line is the short help
    const char* heading = nullptr;  // help section; nullptr = "Options"
    const char* default_text = nullptr;
    std::vector<const char*> choices = {};  // allowed values, if a closed set
    bool repeat = false;
};

struct Positional {
    const char* name;
    const char* help;
    bool required = true;
    bool many = false;
};

struct Command {
    const char* name;
    const char* about;       // one line
    const char* long_about;  // shown by --help; nullptr = about
    std::vector<Positional> positionals;
    std::vector<Opt> options;
    // Pairs of options that cannot be given together.
    std::vector<std::pair<const char*, const char*>> conflicts = {};
};

struct Program {
    const char* name;
    const char* version;
    const char* about;
    std::vector<Opt> global;
    std::vector<Command> commands;
};

class Args {
public:
    bool has(std::string_view name) const { return values_.count(std::string(name)) > 0; }
    // The last value given, or `def`.
    std::string get(std::string_view name, std::string_view def = {}) const;
    const std::vector<std::string>& all(std::string_view name) const;
    const std::vector<std::string>& positional() const { return pos_; }

    // Typed access. A value that does not parse, or is out of range, is a
    // usage error.
    long long get_int(std::string_view name, long long def, long long lo, long long hi) const;
    double get_double(std::string_view name, double def) const;

    std::map<std::string, std::vector<std::string>> values_;
    std::vector<std::string> pos_;
    const Command* command = nullptr;
    const Program* program = nullptr;
};

// Parse argv. --help, -h and --version are handled here and exit.
Args parse(const Program& p, int argc, char** argv);

[[noreturn]] void usage_error(const Args& a, const std::string& msg);

// A completion script for "bash", "zsh" or "fish"; empty for anything else.
std::string completion_script(const Program& p, std::string_view shell);

}  // namespace cli
