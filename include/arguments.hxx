#include <filesystem>
#include <vector>
#include <variant>

// Font size the frame layout was designed against: every length in the frame is
// a fixed proportion of the main font size, expressed in this file and in
// main.cxx as the pixel length it takes at this reference size.
// Note that for historical reason, this is not the same as the default font size
// in the CLI argument (which is 26).
constexpr int DEFAULT_FONT_SIZE = 52;

struct CLIArgs
{
    int quality;   // JPEG quality, 90 by default
    // List of files to process, each file is a pair of (input,output)
    std::vector<std::pair<std::filesystem::path,std::filesystem::path>> files;
    int width, height; // size of the output, default: 1080×1350
    int fontsize; // font size of the main text, default: 26; the sub text and
                  // every other length in the frame scale along with it
    int margin; // width of the white frame, default: 0

    bool verbose;
};

std::variant<CLIArgs,int> parse_arguments(int argc, char **argv);
