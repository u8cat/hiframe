#include <filesystem>
#include <vector>
#include <variant>

// Dimension that the `~` suffix of --size turns into an upper bound: the frame
// shrinks to the photo in that direction, instead of padding it with white
// space. Only one dimension may shrink, as the other one fixes the scale.
enum class Shrink { None, Width, Height };

struct CLIArgs
{
    int quality;   // JPEG quality, 90 by default
    // List of files to process, each file is a pair of (input,output)
    std::vector<std::pair<std::filesystem::path,std::filesystem::path>> files;
    int width, height; // size of the output, default: 1080×1350~
    Shrink shrink; // dimension marked with `~` in --size, default: none
    int fontsize; // font size of the main text, default: 26; the sub text and
                  // every other length in the frame scale along with it
    int margin; // width of the white frame, default: 0

    bool verbose;
};

std::variant<CLIArgs,int> parse_arguments(int argc, char **argv);
