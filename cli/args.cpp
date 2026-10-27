#include "args.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace cli {
namespace {

const std::vector<std::string> kNone;

const Opt* find_long(const std::vector<Opt>& v, std::string_view name) {
    for (const auto& o : v)
        if (name == o.name) return &o;
    return nullptr;
}

const Opt* find_short(const std::vector<Opt>& v, char c) {
    for (const auto& o : v)
        if (c && o.short_name == c) return &o;
    return nullptr;
}

std::string first_line(const char* s) {
    const char* nl = std::strchr(s, '\n');
    return nl ? std::string(s, nl) : std::string(s);
}

std::string usage_line(const Program& p, const Command* c) {
    std::string u = std::string("Usage: ") + p.name;
    if (!c) return u + " [OPTIONS] <COMMAND>";
    u += std::string(" ") + c->name + " [OPTIONS]";
    for (const auto& pos : c->positionals) {
        std::string n = std::string(pos.required ? "<" : "[") + pos.name + (pos.required ? ">" : "]");
        if (pos.many) n += "...";
        u += " " + n;
    }
    return u;
}

void wrap(std::string& out, const std::string& text, std::size_t indent) {
    // Help text is wrapped by hand, so only indent the continuation lines.
    std::size_t start = 0;
    bool first = true;
    while (start <= text.size()) {
        std::size_t nl = text.find('\n', start);
        std::string line = text.substr(start, nl == std::string::npos ? std::string::npos : nl - start);
        if (!first) out += std::string(indent, ' ');
        out += line + "\n";
        first = false;
        if (nl == std::string::npos) break;
        start = nl + 1;
    }
}

std::string opt_label(const Opt& o) {
    std::string s = o.short_name ? std::string("-") + o.short_name + ", " : "    ";
    s += std::string("--") + o.name;
    if (o.value_name) s += std::string(" <") + o.value_name + ">";
    return s;
}

void print_options(std::string& out, const std::vector<Opt>& opts, const char* only_heading, bool all_default) {
    for (const auto& o : opts) {
        const char* h = o.heading;
        if (all_default ? h != nullptr : (h == nullptr || std::strcmp(h, only_heading) != 0)) continue;
        std::string label = "  " + opt_label(o);
        std::string help = o.help;
        if (!o.choices.empty()) {
            help += " [possible values: ";
            for (std::size_t i = 0; i < o.choices.size(); ++i) help += (i ? ", " : "") + std::string(o.choices[i]);
            help += "]";
        }
        if (o.default_text) help += std::string(" [default: ") + o.default_text + "]";
        if (label.size() < 30) {
            label.resize(30, ' ');
            out += label;
            wrap(out, help, 30);
        } else {
            out += label + "\n" + std::string(30, ' ');
            wrap(out, help, 30);
        }
    }
}

std::vector<const char*> headings(const std::vector<Opt>& opts) {
    std::vector<const char*> hs;
    for (const auto& o : opts) {
        if (!o.heading) continue;
        bool seen = false;
        for (auto h : hs) seen |= std::strcmp(h, o.heading) == 0;
        if (!seen) hs.push_back(o.heading);
    }
    return hs;
}

[[noreturn]] void print_help(const Program& p, const Command* c) {
    std::string out;
    if (c) {
        out += std::string(c->long_about ? c->long_about : c->about) + "\n\n";
        out += usage_line(p, c) + "\n";
        if (!c->positionals.empty()) {
            out += "\nArguments:\n";
            for (const auto& pos : c->positionals) {
                std::string label = std::string("  ") + (pos.required ? "<" : "[") + pos.name +
                                    (pos.required ? ">" : "]") + (pos.many ? "..." : "");
                if (label.size() < 30)
                    label.resize(30, ' ');
                else
                    label += "\n" + std::string(30, ' ');
                out += label;
                wrap(out, pos.help, 30);
            }
        }
        out += "\nOptions:\n";
        print_options(out, c->options, nullptr, true);
        out += "  -h, --help                  Print help\n";
        for (auto h : headings(c->options)) {
            out += std::string("\n") + h + ":\n";
            print_options(out, c->options, h, false);
        }
        for (auto h : headings(p.global)) {
            out += std::string("\n") + h + ":\n";
            print_options(out, p.global, h, false);
        }
    } else {
        out += std::string(p.about) + "\n\n" + usage_line(p, nullptr) + "\n\nCommands:\n";
        for (const auto& cmd : p.commands) {
            std::string label = std::string("  ") + cmd.name;
            label.resize(16, ' ');
            out += label + cmd.about + "\n";
        }
        out += "  help          Print this message or the help of the given subcommand(s)\n";
        out += "\nOptions:\n  -h, --help                  Print help\n  -V, --version               Print version\n";
        for (auto h : headings(p.global)) {
            out += std::string("\n") + h + ":\n";
            print_options(out, p.global, h, false);
        }
    }
    std::fputs(out.c_str(), stdout);
    std::exit(0);
}

[[noreturn]] void fail_usage(const Program& p, const Command* c, const std::string& msg) {
    std::fprintf(stderr, "error: %s\n\n%s\n\nFor more information, try '--help'.\n", msg.c_str(),
                 usage_line(p, c).c_str());
    std::exit(2);
}

}  // namespace

std::string Args::get(std::string_view name, std::string_view def) const {
    auto it = values_.find(std::string(name));
    return it == values_.end() || it->second.empty() ? std::string(def) : it->second.back();
}

const std::vector<std::string>& Args::all(std::string_view name) const {
    auto it = values_.find(std::string(name));
    return it == values_.end() ? kNone : it->second;
}

long long Args::get_int(std::string_view name, long long def, long long lo, long long hi) const {
    if (!has(name)) return def;
    std::string v = get(name);
    char* end = nullptr;
    errno = 0;
    long long x = std::strtoll(v.c_str(), &end, 10);
    if (v.empty() || *end || errno)
        usage_error(*this, "invalid value '" + v + "' for '--" + std::string(name) + "': expected an integer");
    if (x < lo || x > hi)
        usage_error(*this, "invalid value '" + v + "' for '--" + std::string(name) + "': " + std::to_string(x) +
                               " is not in " + std::to_string(lo) + ".." + std::to_string(hi));
    return x;
}

double Args::get_double(std::string_view name, double def) const {
    if (!has(name)) return def;
    std::string v = get(name);
    char* end = nullptr;
    double x = std::strtod(v.c_str(), &end);
    if (v.empty() || *end) usage_error(*this, "invalid value '" + v + "' for '--" + std::string(name) + "'");
    return x;
}

void usage_error(const Args& a, const std::string& msg) { fail_usage(*a.program, a.command, msg); }

Args parse(const Program& p, int argc, char** argv) {
    Args a;
    a.program = &p;
    std::vector<std::string> rest;
    bool only_positional = false;
    auto take_value = [&](const Opt& o, int& i, const char* inline_value) {
        std::string v;
        if (inline_value) {
            v = inline_value;
        } else {
            if (i + 1 >= argc)
                fail_usage(p, a.command,
                           std::string("a value is required for '--") + o.name + " <" + o.value_name +
                               ">' but none was supplied");
            v = argv[++i];
        }
        if (!o.choices.empty()) {
            bool ok = false;
            for (auto c : o.choices) ok |= v == c;
            if (!ok) {
                std::string list;
                for (std::size_t k = 0; k < o.choices.size(); ++k) list += (k ? ", " : "") + std::string(o.choices[k]);
                fail_usage(p, a.command,
                           "invalid value '" + v + "' for '--" + o.name + "'\n  [possible values: " + list + "]");
            }
        }
        auto& slot = a.values_[o.name];
        if (!o.repeat && !slot.empty())
            fail_usage(p, a.command, std::string("the argument '--") + o.name + "' cannot be used multiple times");
        slot.push_back(std::move(v));
    };

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (only_positional || arg == "-" || arg.empty() || arg[0] != '-') {
            if (!a.command) {
                if (arg == "help") {
                    if (i + 1 < argc)
                        for (const auto& c : p.commands)
                            if (c.name == std::string_view(argv[i + 1])) print_help(p, &c);
                    print_help(p, nullptr);
                }
                for (const auto& c : p.commands)
                    if (arg == c.name) a.command = &c;
                if (!a.command) fail_usage(p, nullptr, "unrecognized subcommand '" + std::string(arg) + "'");
                continue;
            }
            a.pos_.emplace_back(arg);
            continue;
        }
        if (arg == "--") {
            only_positional = true;
            continue;
        }
        if (arg == "--help" || arg == "-h") print_help(p, a.command);
        if (arg == "--version" || arg == "-V") {
            std::printf("%s %s\n", p.name, p.version);
            std::exit(0);
        }
        if (arg.substr(0, 2) == "--") {
            std::string name(arg.substr(2));
            const char* inline_value = nullptr;
            auto eq = name.find('=');
            if (eq != std::string::npos) {
                inline_value = argv[i] + 2 + eq + 1;
                name.resize(eq);
            }
            const Opt* o = a.command ? find_long(a.command->options, name) : nullptr;
            if (!o) o = find_long(p.global, name);
            if (!o) fail_usage(p, a.command, "unexpected argument '--" + name + "' found");
            if (o->value_name) {
                take_value(*o, i, inline_value);
            } else {
                if (inline_value) fail_usage(p, a.command, "unexpected value for flag '--" + name + "'");
                a.values_[o->name].emplace_back();
            }
            continue;
        }
        // Short options, possibly bundled (-qf) or with a value (-o out).
        for (std::size_t k = 1; k < arg.size(); ++k) {
            const char c = arg[k];
            const Opt* o = a.command ? find_short(a.command->options, c) : nullptr;
            if (!o) o = find_short(p.global, c);
            if (!o) fail_usage(p, a.command, std::string("unexpected argument '-") + c + "' found");
            if (o->value_name) {
                take_value(*o, i, k + 1 < arg.size() ? argv[i] + k + 1 : nullptr);
                break;
            }
            a.values_[o->name].emplace_back();
        }
    }
    if (!a.command) print_help(p, nullptr);

    const auto& specs = a.command->positionals;
    std::size_t required = 0;
    bool many = false;
    for (const auto& s : specs) {
        required += s.required;
        many |= s.many;
    }
    if (a.pos_.size() < required) {
        std::string missing;
        for (std::size_t k = a.pos_.size(); k < specs.size(); ++k)
            if (specs[k].required) missing += std::string("\n  <") + specs[k].name + ">";
        fail_usage(p, a.command, "the following required arguments were not provided:" + missing);
    }
    if (!many && a.pos_.size() > specs.size())
        fail_usage(p, a.command, "unexpected argument '" + a.pos_[specs.size()] + "' found");

    for (const auto& [x, y] : a.command->conflicts)
        if (a.has(x) && a.has(y))
            fail_usage(p, a.command, std::string("the argument '--") + x + "' cannot be used with '--" + y + "'");
    return a;
}

std::string completion_script(const Program& p, std::string_view shell) {
    std::string out;
    auto opts_of = [&](const Command* c) {
        std::vector<const Opt*> v;
        if (c)
            for (const auto& o : c->options) v.push_back(&o);
        for (const auto& o : p.global) v.push_back(&o);
        return v;
    };
    if (shell == "bash") {
        out += std::string("_") + p.name +
               "() {\n  local cur prev cmd\n  cur=\"${COMP_WORDS[COMP_CWORD]}\"\n"
               "  prev=\"${COMP_WORDS[COMP_CWORD-1]}\"\n  cmd=\"\"\n"
               "  for w in \"${COMP_WORDS[@]:1}\"; do case \"$w\" in";
        for (const auto& c : p.commands) out += std::string(" ") + c.name + (&c == &p.commands.back() ? "" : " |");
        out += ") cmd=\"$w\"; break;; esac; done\n  case \"$cmd\" in\n";
        auto emit = [&](const Command* c) {
            for (const Opt* o : opts_of(c))
                if (!o->choices.empty()) {
                    out +=
                        std::string("      if [ \"$prev\" = \"--") + o->name + "\" ]; then COMPREPLY=($(compgen -W \"";
                    for (auto ch : o->choices) out += std::string(ch) + " ";
                    out += "\" -- \"$cur\")); return; fi\n";
                }
            std::string words;
            for (const Opt* o : opts_of(c)) words += std::string("--") + o->name + " ";
            if (!c)
                for (const auto& cmd : p.commands) words += std::string(cmd.name) + " ";
            out += "      if [[ \"$cur\" == -* ]]";
            if (!c) out += " || [ \"$COMP_CWORD\" -eq 1 ]";
            out += "; then COMPREPLY=($(compgen -W \"" + words + "\" -- \"$cur\")); fi\n";
        };
        for (const auto& c : p.commands) {
            out += std::string("    ") + c.name + ")\n";
            emit(&c);
            out += "      ;;\n";
        }
        out += "    *)\n";
        emit(nullptr);
        out += "      ;;\n  esac\n}\ncomplete -o default -F _" + std::string(p.name) + " " + p.name + "\n";
    } else if (shell == "zsh") {
        out += std::string("#compdef ") + p.name + "\n\n_" + p.name + "() {\n  local -a commands\n  commands=(\n";
        for (const auto& c : p.commands) out += std::string("    '") + c.name + ":" + c.about + "'\n";
        out +=
            "  )\n  if (( CURRENT == 2 )); then\n    _describe 'command' commands\n    return\n  fi\n"
            "  case ${words[2]} in\n";
        for (const auto& c : p.commands) {
            out += std::string("    ") + c.name + ")\n      _arguments";
            for (const Opt* o : opts_of(&c)) {
                std::string h = first_line(o->help);
                for (auto& ch : h)
                    if (ch == '\'' || ch == '[' || ch == ']' || ch == ':') ch = ' ';
                out += std::string(" \\\n        '--") + o->name + "[" + h + "]";
                if (o->value_name) {
                    out += ":" + std::string(o->value_name) + ":";
                    if (!o->choices.empty()) {
                        out += "(";
                        for (auto ch : o->choices) out += std::string(ch) + " ";
                        out += ")";
                    } else {
                        out += "_files";
                    }
                }
                out += "'";
            }
            out += " \\\n        '*:file:_files'\n      ;;\n";
        }
        out += "  esac\n}\n\n_" + std::string(p.name) + " \"$@\"\n";
    } else if (shell == "fish") {
        for (const auto& c : p.commands)
            out += std::string("complete -c ") + p.name + " -n '__fish_use_subcommand' -f -a " + c.name + " -d '" +
                   c.about + "'\n";
        for (const auto& c : p.commands)
            for (const Opt* o : opts_of(&c)) {
                std::string h = first_line(o->help);
                for (auto& ch : h)
                    if (ch == '\'') ch = ' ';
                out += std::string("complete -c ") + p.name + " -n '__fish_seen_subcommand_from " + c.name + "' -l " +
                       o->name;
                if (o->short_name) out += std::string(" -s ") + o->short_name;
                if (o->value_name) out += " -r";
                if (!o->choices.empty()) {
                    out += " -f -a '";
                    for (auto ch : o->choices) out += std::string(ch) + " ";
                    out += "'";
                }
                out += " -d '" + h + "'\n";
            }
    }
    return out;
}

}  // namespace cli
