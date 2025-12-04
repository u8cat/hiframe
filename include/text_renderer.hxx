#pragma once

#include <vector>
#include <initializer_list>
#include <string>
#include <string_view>

#include <opencv2/opencv.hpp>
#include <ft2build.h>
#include <freetype/freetype.h>

class TextRenderer {
    FT_Library ft;
    std::vector<FT_Face> faces;

public:
    TextRenderer(std::initializer_list<std::string_view> fontPaths, int fontSize);
    ~TextRenderer();

    void render(cv::Mat& img, const std::string& text, cv::Point pos, cv::Scalar color);
    int width(const std::string& text);
};
