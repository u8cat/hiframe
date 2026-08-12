#include <string>
#include <iostream>
#include <cstdlib>
#include <cctype>
#include <format>
#include <algorithm>
#include <stdexcept>

#include <boost/program_options.hpp>

#include "arguments.hxx"

using std::string, std::vector;
using std::clog, std::endl;
namespace po = boost::program_options;
namespace fs = std::filesystem;

// Parse one dimension of --size: a length, optionally suffixed with `~` to mark
// it as an upper bound rather than an exact size. Throws if it is not a number.
int parse_length(string length, bool &flexible) {
    flexible = length.ends_with('~');
    if (flexible) length.pop_back();

    if (length.empty() || !std::all_of(length.begin(), length.end(),
                                       [](unsigned char c) { return std::isdigit(c); }))
        throw std::invalid_argument(length);

    return std::stoi(length); // may still throw out_of_range
}

fs::path format_output(const fs::path &input, const string &pattern) {
    auto filename = input.filename();
    auto stem = filename.stem().string();
    auto extension = filename.extension();

    fs::path new_filename = std::vformat(pattern, std::make_format_args(stem));
    new_filename+=extension;

    auto output = input;
    return output.replace_filename(new_filename);
}

std::variant<CLIArgs,int> parse_arguments(int argc, char **argv) {
    CLIArgs args;
    string output_file, output_pattern, image_size;

    po::positional_options_description op_positional;
    op_positional.add("input", -1);

    po::options_description op_basic("Options");
    op_basic.add_options()
        ("output,o", po::value<string>(&output_file), "output file")
        ("output-pattern,O", po::value<string>(&output_pattern)->default_value("framed/{}"), "pattern of output files");

    po::options_description op_image("Image Options");
    op_image.add_options()
        ("size,s", po::value<string>(&image_size)->default_value("1080x1350~"),
                   "size; a dimension suffixed with ~ shrinks to the photo")
        ("quality,q", po::value<int>(&args.quality)->default_value(90), "quality")
        ("font-size,f", po::value<int>(&args.fontsize)->default_value(26), "font size")
        ("margin,m",po::value<int>(&args.margin)->default_value(0), "frame margin");

    po::options_description op_other("Other Options");
    op_other.add_options()
        ("help", "display this help")
        ("version", "output version information")
        ("verbose", po::value<bool>(&args.verbose)->default_value(false), "increase verbosity");

    po::options_description hidden_options("");
    hidden_options.add_options()
        ("input",po::value<vector<string>>(),"");

    po::options_description visible_options;
    visible_options.add(op_basic).add(op_image).add(op_other);
    po::options_description all_options;
    all_options.add(visible_options).add(hidden_options);

    auto help = [&](int x) {
        clog << "Usage:" << argv[0] << "<input> -o <output>\n";
        clog << "      " << argv[0] << "<input>\n";
        clog << "      " << argv[0] << "<input>.. -O <output pattern>\n";
        clog << visible_options << endl;
        return x;
    };

    po::variables_map vm;
    po::store(po::command_line_parser(argc,argv).
        options(all_options).positional(op_positional).run(), vm);
    po::notify(vm);

    if(vm.contains("help"))
        return help(0);

    // parse file option
    if (!vm.contains("input"))
        return help(2);
    auto &inputs = vm["input"].as<vector<string>>();
    try {
        switch (inputs.size())
        {
        case 0: return help(2);
        case 1:
            if(output_file != "")
                args.files.emplace_back(inputs[0], output_file);
            else
                args.files.emplace_back(inputs[0], format_output(inputs[0], output_pattern));
            break;
        default:
            if(output_file != "")
                return help(2);

            for (auto &input: vm["input"].as<vector<string>>())
                args.files.emplace_back(input, format_output(input, output_pattern));
        }
    } catch (const std::format_error &e) {
        // std::format rejects unbalanced braces, and accepts {} only once
        clog << "Wrong --output-pattern \"" << output_pattern << "\": " << e.what() << "\n"
                "The pattern is a format string whose only argument is the input file "
                "name: {} inserts it once, {0} may be repeated.\n\n";
        return help(2);
    }

    // parse image size
    std::transform(image_size.begin(), image_size.end(), image_size.begin(), tolower);
    {
        auto p = image_size.find('x');
        if (p == string::npos) {
            clog << "Wrong --size format, expect: [0-9]+~?x[0-9]+~?\n\n";
            return help(2);
        }

        bool flexible_width, flexible_height;
        try {
            args.width = parse_length(image_size.substr(0,p), flexible_width);
            args.height = parse_length(image_size.substr(p+1), flexible_height);
        } catch (...) {
            clog << "Wrong --size format, expect: [0-9]+~?x[0-9]+~?\n\n";
            return help(2);
        }

        if (args.width <= 0 || args.height <= 0) {
            clog << "Wrong --size, expect positive integers\n\n";
            return help(2);
        }

        if (flexible_width && flexible_height) {
            clog << "Wrong --size, only one dimension may be followed by ~, "
                    "as the other one fixes the scale of the photo\n\n";
            return help(2);
        }

        args.shrink = flexible_width ? Shrink::Width :
                      flexible_height ? Shrink::Height : Shrink::None;
    }

    if (args.fontsize <= 0) {
        clog << "Wrong --font-size, expect a positive integer\n\n";
        return help(2);
    }

    return args;
}
