#include <format>
#include <cmath>
#include <cstdio>

#include "exif.hxx"
#include "string.hxx"

using namespace std::string_literals;
using std::format, std::string, std::vector, Exiv2::ExifKey;

Metadata parseExif(const Exiv2::ExifData &exifData) {
    Metadata meta;
    if (!exifData.empty()) {
        if (auto key = exifData.findKey(ExifKey("Exif.Image.Make")); key != exifData.end())
            meta.make = key->toString();
        if (auto key = exifData.findKey(ExifKey("Exif.Image.Model")); key != exifData.end()) {
            meta.model = key->toString();
            if (meta.make == "") {
                // If Make is none, try inferring it from Model
                meta.make = meta.model.substr(0,meta.model.find(" "));
            }
            auto m = meta.model; std::transform(m.begin(), m.end(), m.begin(), ::tolower);
            if (m.ends_with(" digital camera"))
                meta.model.resize(meta.model.length()-" digital camera"s.length());
        }
        if (auto key = exifData.findKey(ExifKey("Exif.Photo.FNumber")); key != exifData.end()) {
            auto val = key->toFloat();
            auto s = format("{:.3f}", val);
            while (s.back() == '0') s.pop_back();
            if (s.back() == '.') s += '0';
            meta.aperture = "f/" + s;
        }
        if (auto key = exifData.findKey(ExifKey("Exif.Photo.ExposureTime")); key != exifData.end()) {
            auto r = key->toRational();
            if (r.first >= r.second) meta.shutter = format("{}s",r.first / r.second);
            else meta.shutter = format("1/{}s", int(round((double)r.second/r.first)));
        }
        if (auto key = exifData.findKey(ExifKey("Exif.Photo.ISOSpeedRatings")); key != exifData.end())
            meta.iso = "ISO" + key->toString();
        if (auto key = exifData.findKey(ExifKey("Exif.Photo.FocalLength")); key != exifData.end()) {
            float val = key->toFloat();
            auto s = format("{:.3f}", val);
            while (s.back() == '0') s.pop_back();
            if (s.back() == '.') s += '0';
            meta.focal = s + "mm";
        }
        if (auto key = exifData.findKey(ExifKey("Exif.Photo.LensModel")); key != exifData.end()) {
            meta.lens = key->toString();
            if (meta.model != "" && meta.lens.starts_with(meta.model+" ")) {
                meta.lens = meta.lens.substr(meta.model.length()+1);
            }
        }
        else
            meta.lens = "builtin lens";
        if (auto key = exifData.findKey(ExifKey("Exif.Photo.DateTimeOriginal")); key != exifData.end()) {
            string d = key->toString();
            d[4] = '-'; d[7] = '-'; d[10]='T';
            meta.date = d;
            string offset;
            if (auto key = exifData.findKey(ExifKey("Exif.Photo.OffsetTimeOriginal")); key != exifData.end()) {
                offset = key->toString();
                if (offset == "+00:00") meta.date += "Z";
                else meta.date += offset;
            }

            // Move the date to UTC, so that it can be compared with the moment a
            // logo took effect. An absent offset leaves it as it is, that is,
            // takes the date to be in UTC already.
            meta.taken = parse_datetime(d);
            if (int h, m; meta.taken && sscanf(offset.c_str(), "%3d:%2d", &h, &m) == 2)
                *meta.taken -= std::chrono::hours{h} + std::chrono::minutes{h < 0 ? -m : m};
        }
        if (auto key = exifData.findKey(ExifKey("Exif.GPSInfo.GPSLatitude")); key != exifData.end()) {
            // assume GPS data is complete
            auto latitude = key->toFloat(0) + key->toFloat(1)/60.0f + key->toFloat(2)/3600.0f;
            auto key2 = exifData.findKey(ExifKey("Exif.GPSInfo.GPSLongitude"));
            auto longitude = key2->toFloat(0) + key2->toFloat(1)/60.0f + key2->toFloat(2)/3600.0f;
            auto latitude_ref = exifData.findKey(ExifKey("Exif.GPSInfo.GPSLatitudeRef"))->toString();
            auto longitude_ref = exifData.findKey(ExifKey("Exif.GPSInfo.GPSLongitudeRef"))->toString();
            meta.coordinate = format("{:.5f}{},{:.5f}{}", latitude, latitude_ref, longitude, longitude_ref);
            if (auto key = exifData.findKey(ExifKey("Exif.GPSInfo.GPSAltitude")); key != exifData.end()) {
                int height = std::round(key->toFloat());
                if (auto key = exifData.findKey(ExifKey("Exif.GPSInfo.GPSAltitudeRef")); key != exifData.end()) {
                    if (key->toLong() == 1) height = -height;
                }
                meta.coordinate += format(",{:+}m",height);
            }
        }
    }
    return meta;
}

Exiv2::ExifData getExif(const vector<char> &buf) {
    auto image = Exiv2::ImageFactory::open(reinterpret_cast<const Exiv2::byte*>(buf.data()), buf.size());
    image->readMetadata();
    return image->exifData();
}

vector<uint8_t> getIcc(const vector<char> &buf) {
    auto image = Exiv2::ImageFactory::open(reinterpret_cast<const Exiv2::byte*>(buf.data()), buf.size());
    image->readMetadata();
    if (!image->iccProfileDefined()) return {};

    auto profile = image->iccProfile();
    return {profile->pData_, profile->pData_ + profile->size_};
}
