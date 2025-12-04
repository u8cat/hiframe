#include <filesystem>
#include <vector>
#include <variant>

struct CLIArgs
{
    int quality;   // JPEG quality, 95 by default
    // List of files to process, each file is a pair of (input,output)
    std::vector<std::pair<std::filesystem::path,std::filesystem::path>> files;
    int width, height; // size of the output, default: 2160×2700
    // TODO: int mainfontsize, subfontsize; // font size, default: 52, 40
    int margin; // width of the white frame, default: 80

    bool verbose;
};

std::variant<CLIArgs,int> parse_arguments(int argc, char **argv);
