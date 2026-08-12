#pragma once
#include <string>
#include <vector>

#include <exiv2/exiv2.hpp>

struct Metadata {
    std::string make, model, lens, iso, aperture,
        shutter, focal, date, coordinate;
};

Metadata parseExif(const Exiv2::ExifData &exifData);
Exiv2::ExifData getExif(const std::vector<char> &buf);
// ICC profile of the image, empty if it does not carry one
std::vector<uint8_t> getIcc(const std::vector<char> &buf);
