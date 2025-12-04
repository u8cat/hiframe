#include <stdexcept>

#include "text_renderer.hxx"
#include "string.hxx"

using std::string;
using namespace std::string_literals;

TextRenderer::TextRenderer(std::initializer_list<std::string_view> fontPaths, int fontSize) {
    if (FT_Init_FreeType(&ft)) {
        throw std::runtime_error("ERROR: Could not init FreeType Library");
        return;
    }
    for (auto fontPath: fontPaths) {
        FT_Face face;
        if (FT_New_Face(ft, fontPath.data(), 0, &face)) {
            throw std::runtime_error("ERROR: Failed to load font: "s + fontPath.data());
            return;
        }
        FT_Set_Pixel_Sizes(face, 0, fontSize);
        faces.push_back(face);
    }
}

TextRenderer::~TextRenderer() {
    for (auto &face: faces)
        FT_Done_Face(face);
    FT_Done_FreeType(ft);
}

void TextRenderer::render(cv::Mat& img, const string& text, cv::Point pos, cv::Scalar color) {
    auto pen_x = pos.x;
    auto pen_y = pos.y;

    for (auto it = utf8_iterator::begin(text); it != utf8_iterator::end(text); ++it) {
        // Try rendering the codepoint in all avaliable fonts
        for (auto &face : faces) {
            if (FT_Get_Char_Index(face, *it) == 0 || FT_Load_Char(face, *it, FT_LOAD_RENDER)) continue;
            FT_Bitmap& bitmap = face->glyph->bitmap;
            auto top = pen_y - face->glyph->bitmap_top;
            auto left = pen_x + face->glyph->bitmap_left;

            for (int r = 0; r < bitmap.rows; r++) {
                for (int c = 0; c < bitmap.width; c++) {
                    auto y = top + r;
                    auto x = left + c;

                    if (y < 0 || y >= img.rows || x < 0 || x >= img.cols) continue;

                    double alpha = bitmap.buffer[r * bitmap.width + c] / 255.0;
                    if (alpha > 0) {
                        // Handle Multi-channel generic
                        auto channels = img.channels();
                        if (img.depth() == CV_8U) {
                            cv::Vec3b& pixel = img.at<cv::Vec3b>(y, x);
                            for (int i = 0; i < 3; i++)
                                pixel[i] = (uchar)(pixel[i] * (1.0 - alpha) + color[i] * alpha);
                        } else if (img.depth() == CV_32F) {
                            cv::Vec3f& pixel = img.at<cv::Vec3f>(y, x);
                            for (int i = 0; i < 3; i++)
                                pixel[i] = (float)(pixel[i] * (1.0 - alpha) + color[i] * alpha);
                        }
                    }
                }
            }
            pen_x += (face->glyph->advance.x >> 6);
            break;
        }
    }
}

int TextRenderer::width(const std::string& text) {
    int width = 0;
    for (auto it = utf8_iterator::begin(text); it != utf8_iterator::end(text); ++it) {
        for (auto &face : faces) {
            if (FT_Get_Char_Index(face, *it) == 0 || FT_Load_Char(face, *it, FT_LOAD_RENDER)) continue;
            width += (face->glyph->advance.x >> 6);
            break;
        }
    }
    return width;
}
