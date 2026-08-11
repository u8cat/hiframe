#include <string>
#include <iostream>
#include <cstdlib>
#include <format>
#include <algorithm>

#include <boost/program_options.hpp>

#include "arguments.hxx"

using std::string, std::vector;
using std::clog, std::endl;
namespace po = boost::program_options;
namespace fs = std::filesystem;

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
        ("output-pattern,O", po::value<string>(&output_pattern)->default_value("{}.frame"), "pattern of output files");

    po::options_description op_image("Image Options");
    op_image.add_options()
        ("size,s", po::value<string>(&image_size)->default_value("2160x2700"), "size")
        ("quality,q", po::value<int>(&args.quality)->default_value(90), "quality")
        ("font-size,f", po::value<int>(&args.fontsize)->default_value(DEFAULT_FONT_SIZE), "font size")
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

    // parse image size
    std::transform(image_size.begin(), image_size.end(), image_size.begin(), tolower);
    {
        auto p = image_size.find('x');
        if (p == string::npos) {
            clog << "Wrong --size format, expect: [0-9]+x[0-9]+\n\n";
            return help(2);
        }

        try {
            args.width = std::stoi(image_size.substr(0,p));
            args.height = std::stoi(image_size.substr(p+1));
        } catch (...) {
            clog << "Wrong --size format, expect: [0-9]+x[0-9]+\n\n";
            return help(2);
        }
    }

    if (args.fontsize <= 0) {
        clog << "Wrong --font-size, expect a positive integer\n\n";
        return help(2);
    }

    return args;
}
