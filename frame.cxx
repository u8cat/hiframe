#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <sstream>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

// Exiv2
#include <exiv2/exiv2.hpp>

// Freetype
#include <ft2build.h>
#include FT_FREETYPE_H

// libultrahdr
#include <ultrahdr_api.h>

using namespace std;
using namespace cv;

// --- Constants ---
const int JPEG_QUALITY = 95;

// Iterator class to traverse UTF-8 strings as char32_t codepoints
class unicode_iterator {
public:
    // Iterator traits for STL compatibility
    using iterator_category = std::forward_iterator_tag;
    using value_type        = char32_t;
    using difference_type   = std::ptrdiff_t;
    using pointer           = const char32_t*;
    using reference         = const char32_t; // Returns by value, conceptually const

    unicode_iterator() = default;

    unicode_iterator(std::string::const_iterator it, std::string::const_iterator end)
        : m_it(it), m_end(end) {}

    // Dereference operator: Decodes the UTF-8 sequence at current position
    char32_t operator*() const {
        if (m_it == m_end) return 0;

        unsigned char c = static_cast<unsigned char>(*m_it);
        uint32_t codepoint = 0;

        if ((c & 0x80) == 0) {
            // 1-byte sequence (ASCII)
            return static_cast<char32_t>(c);
        } 
        else if ((c & 0xE0) == 0xC0) {
            // 2-byte sequence
            codepoint = c & 0x1F;
            return decode_continuation(1, codepoint);
        } 
        else if ((c & 0xF0) == 0xE0) {
            // 3-byte sequence
            codepoint = c & 0x0F;
            return decode_continuation(2, codepoint);
        } 
        else if ((c & 0xF8) == 0xF0) {
            // 4-byte sequence
            codepoint = c & 0x07;
            return decode_continuation(3, codepoint);
        } 
        else {
            // Invalid start byte
            return 0xFFFD; // Replacement Character
        }
    }

    // Prefix increment
    unicode_iterator& operator++() {
        if (m_it == m_end) return *this;

        unsigned char c = static_cast<unsigned char>(*m_it);
        size_t len = 1;

        if ((c & 0x80) == 0) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        else len = 1; // Invalid, skip 1 byte

        // Ensure we don't go past end
        for (size_t i = 0; i < len && m_it != m_end; ++i) {
            ++m_it;
        }
        return *this;
    }

    // Postfix increment
    unicode_iterator operator++(int) {
        unicode_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    bool operator==(const unicode_iterator& other) const {
        return m_it == other.m_it;
    }

    bool operator!=(const unicode_iterator& other) const {
        return !(*this == other);
    }

private:
    std::string::const_iterator m_it;
    std::string::const_iterator m_end;

    // Helper to decode trailing bytes
    char32_t decode_continuation(int remaining_bytes, uint32_t current_cp) const {
        auto temp_it = m_it;
        
        for (int i = 0; i < remaining_bytes; ++i) {
            if (++temp_it == m_end) return 0xFFFD; // Unexpected end of string
            
            unsigned char next_c = static_cast<unsigned char>(*temp_it);
            if ((next_c & 0xC0) != 0x80) return 0xFFFD; // Invalid continuation
            
            current_cp = (current_cp << 6) | (next_c & 0x3F);
        }
        return static_cast<char32_t>(current_cp);
    }
};

// --- Helper Functions ---

// Non-const version
unicode_iterator utf8_begin(std::string &str) {
    return unicode_iterator(str.begin(), str.end());
}

unicode_iterator utf8_end(std::string &str) {
    return unicode_iterator(str.end(), str.end());
}

// Const version
unicode_iterator utf8_begin(const std::string &str) {
    return unicode_iterator(str.begin(), str.end());
}

unicode_iterator utf8_end(const std::string &str) {
    return unicode_iterator(str.end(), str.end());
}

// --- Helper: High Quality Text Renderer ---
class TextRenderer {
    FT_Library ft;
    FT_Face face;
    bool initialized = false;

public:
    TextRenderer(const string& fontPath, int fontSize) {
        if (FT_Init_FreeType(&ft)) {
            cerr << "ERROR: Could not init FreeType Library" << endl;
            return;
        }
        if (FT_New_Face(ft, fontPath.c_str(), 0, &face)) {
            cerr << "ERROR: Failed to load font: " << fontPath << endl;
            return;
        }
        FT_Set_Pixel_Sizes(face, 0, fontSize);
        initialized = true;
    }

    ~TextRenderer() {
        if(initialized) {
            FT_Done_Face(face);
            FT_Done_FreeType(ft);
        }
    }

    // Templated to handle CV_8U (SDR) and CV_32F (HDR)
    template <typename T>
    void putText(Mat& img, const string& text, Point pos, Scalar color) {
        if (!initialized) return;

        int pen_x = pos.x;
        int pen_y = pos.y;

        for (char c : text) {
            if (FT_Load_Char(face, c, FT_LOAD_RENDER)) continue;

            FT_Bitmap& bitmap = face->glyph->bitmap;
            int top = pen_y - face->glyph->bitmap_top;
            int left = pen_x + face->glyph->bitmap_left;

            for (int r = 0; r < bitmap.rows; r++) {
                for (int c = 0; c < bitmap.width; c++) {
                    int y = top + r;
                    int x = left + c;

                    if (y < 0 || y >= img.rows || x < 0 || x >= img.cols) continue;

                    double alpha = bitmap.buffer[r * bitmap.width + c] / 255.0;
                    if (alpha > 0) {
                        // Handle Multi-channel generic
                        int channels = img.channels();
                        if (img.depth() == CV_8U) {
                            Vec3b& pixel = img.at<Vec3b>(y, x);
                            for (int i = 0; i < 3; i++) pixel[i] = (uchar)(pixel[i] * (1.0 - alpha) + color[i] * alpha);
                        } else if (img.depth() == CV_32F) {
                            Vec3f& pixel = img.at<Vec3f>(y, x);
                            for (int i = 0; i < 3; i++) pixel[i] = (float)(pixel[i] * (1.0 - alpha) + color[i] * alpha);
                        }
                    }
                }
            }
            pen_x += (face->glyph->advance.x >> 6);
        }
    }

    int getTextWidth(const string& text) {
        if (!initialized) return 0;
        int width = 0;
        for (char c : text) {
            if (FT_Load_Char(face, c, FT_LOAD_DEFAULT)) continue;
            width += (face->glyph->advance.x >> 6);
        }
        return width;
    }
};

// --- Helper: Metadata Parser ---
struct PhotoMeta {
    string make;
    string model;
    string lens;
    string iso;
    string f_number;
    string shutter;
    string focal;
    string date;
};

PhotoMeta parseExif(const string& path) {
    PhotoMeta meta;
    try {
        Exiv2::Image::AutoPtr image = Exiv2::ImageFactory::open(path);
        image->readMetadata();
        Exiv2::ExifData &exifData = image->exifData();

        if (!exifData.empty()) {
            if (exifData.findKey(Exiv2::ExifKey("Exif.Image.Make")) != exifData.end())
                meta.make = exifData["Exif.Image.Make"].toString();
            if (exifData.findKey(Exiv2::ExifKey("Exif.Image.Model")) != exifData.end())
                meta.model = exifData["Exif.Image.Model"].toString();
            if (exifData.findKey(Exiv2::ExifKey("Exif.Photo.FNumber")) != exifData.end()) {
                float val = exifData["Exif.Photo.FNumber"].toFloat();
                stringstream ss; ss << fixed << setprecision(1) << val;
                meta.f_number = "f/" + ss.str();
            }
            if (exifData.findKey(Exiv2::ExifKey("Exif.Photo.ExposureTime")) != exifData.end()) {
                Exiv2::Rational r = exifData["Exif.Photo.ExposureTime"].toRational();
                if (r.first >= r.second) meta.shutter = to_string(r.first / r.second) + "s";
                else meta.shutter = "1/" + to_string(int(0.5 + (double)r.second/r.first)) + "s";
            }
            if (exifData.findKey(Exiv2::ExifKey("Exif.Photo.ISOSpeedRatings")) != exifData.end())
                meta.iso = "ISO" + exifData["Exif.Photo.ISOSpeedRatings"].toString();
            if (exifData.findKey(Exiv2::ExifKey("Exif.Photo.FocalLength")) != exifData.end())
                meta.focal = exifData["Exif.Photo.FocalLength"].toString() + "mm";
            if (exifData.findKey(Exiv2::ExifKey("Exif.Photo.LensModel")) != exifData.end())
                meta.lens = exifData["Exif.Photo.LensModel"].toString();
            if (exifData.findKey(Exiv2::ExifKey("Exif.Photo.DateTimeOriginal")) != exifData.end()) {
                string d = exifData["Exif.Photo.DateTimeOriginal"].toString(); 
                if(d.length() >= 10) meta.date = d.substr(0, 4) + "-" + d.substr(5, 2) + "-" + d.substr(8, 2);
            }
        }
    } catch (Exiv2::Error& e) {
        cerr << "Exif Parsing Error: " << e.what() << endl;
    }
    return meta;
}

vector<uint8_t> getRawExif(const string& path) {
    try {
        Exiv2::Image::AutoPtr image = Exiv2::ImageFactory::open(path);
        image->readMetadata();
        Exiv2::ExifData &exifData = image->exifData();
        if (exifData.empty()) return {};
        Exiv2::Blob blob;
        Exiv2::ExifParser::encode(blob, Exiv2::littleEndian, exifData);
        return blob;
    } catch (...) { return {}; }
}

// --- Helper: LibUltraHDR Utils ---
bool checkUhdr(uhdr_error_info_t status, const string& msg) {
    if (status.error_code != UHDR_CODEC_OK) {
        cerr << "[UltraHDR] " << msg << " Failed: " << status.error_code;
        if (status.has_detail) cerr << " (" << status.detail << ")";
        cerr << endl;
        return false;
    }
    return true;
}

Mat wrapUhdrImage(uhdr_raw_image_t* img) {
    if (!img) return Mat();
    if (img->fmt == UHDR_IMG_FMT_32bppRGBA8888)
        return Mat(img->h, img->w, CV_8UC4, img->planes[UHDR_PLANE_PACKED]);
    if (img->fmt == UHDR_IMG_FMT_64bppRGBAHalfFloat)
        return Mat(img->h, img->w, CV_16FC4, img->planes[UHDR_PLANE_PACKED]);
    return Mat();
}

// --- Main Processing Logic ---
int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: ./exif_framer <input.jpg> [output.jpg]" << endl;
        return 1;
    }
    string inputPath = argv[1];
    string outputPath = (argc > 2) ? argv[2] : "framed_output.jpg";

    // 1. Read Input
    ifstream file(inputPath, ios::binary | ios::ate);
    if (!file.good()) { cerr << "File error." << endl; return -1; }
    size_t size = file.tellg();
    file.seekg(0, ios::beg);
    vector<char> buffer(size);
    file.read(buffer.data(), size);

    // 2. Decode (Dual Pass if UltraHDR)
    Mat sdrMat, hdrMat; // sdrMat is BGR (8u), hdrMat is BGR (32f Linear)
    bool hasHDR = false;

    if (is_uhdr_image(buffer.data(), size)) {
        cout << "Detected UltraHDR. Decoding SDR and HDR planes..." << endl;
        
        // Pass 1: SDR
        {
            uhdr_codec_private_t* dec = uhdr_create_decoder();
            uhdr_enable_gpu_acceleration(dec, 0);
            uhdr_compressed_image_t input_img = { buffer.data(), size, size, UHDR_CG_UNSPECIFIED, UHDR_CT_UNSPECIFIED, UHDR_CR_UNSPECIFIED };
            uhdr_dec_set_image(dec, &input_img);
            uhdr_dec_set_out_img_format(dec, UHDR_IMG_FMT_32bppRGBA8888);
            uhdr_dec_set_out_color_transfer(dec, UHDR_CT_SRGB);
            uhdr_dec_probe(dec);
            if (uhdr_decode(dec).error_code == UHDR_CODEC_OK) {
                Mat raw = wrapUhdrImage(uhdr_get_decoded_image(dec));
                cvtColor(raw, sdrMat, COLOR_RGBA2BGR);
            }
            uhdr_release_decoder(dec);
        }

        // Pass 2: HDR (Reconstructed)
        {
            uhdr_codec_private_t* dec = uhdr_create_decoder();
            uhdr_enable_gpu_acceleration(dec, 0);
            uhdr_compressed_image_t input_img = { buffer.data(), size, size, UHDR_CG_UNSPECIFIED, UHDR_CT_UNSPECIFIED, UHDR_CR_UNSPECIFIED };
            uhdr_dec_set_image(dec, &input_img);
            uhdr_dec_set_out_img_format(dec, UHDR_IMG_FMT_64bppRGBAHalfFloat);
            uhdr_dec_set_out_color_transfer(dec, UHDR_CT_LINEAR); // Essential for raw linear data
            uhdr_dec_probe(dec);
            if (uhdr_decode(dec).error_code == UHDR_CODEC_OK) {
                Mat raw16 = wrapUhdrImage(uhdr_get_decoded_image(dec)); // CV_16FC4
                
                // Convert 16F -> 32F first to avoid OpenCV cvtColor issues with 16-bit float
                Mat raw32;
                raw16.convertTo(raw32, CV_32F); 
                
                // Now safe to convert Color space on 32F image
                cvtColor(raw32, hdrMat, COLOR_RGBA2BGR);
                
                hasHDR = true;
            }
            uhdr_release_decoder(dec);
        }
    }

    // Fallback for standard JPEG
    if (sdrMat.empty()) {
        sdrMat = imdecode(buffer, IMREAD_COLOR);
        if (sdrMat.empty()) { cerr << "Decode failed." << endl; return -1; }
    }

    // 3. Resize & Pad
    int targetW = 2160;
    int targetH = targetW * 1.25;
    int margin = 80;
    int bottomPad = 300;

    auto layout = [&](const Mat& src, Mat& dst, Scalar padColor, int interp) {
        double scale = min((double)(targetW - margin*2) / src.cols, (double)(targetH - margin*2 - bottomPad) / src.rows);
        int dw = src.cols * scale;
        int dh = src.rows * scale;
        int x = (targetW - dw) / 2;
        int y = margin + (targetH - margin*2 - bottomPad) / 2;
        
        dst.setTo(padColor);
        Mat resized;
        resize(src, resized, Size(dw, dh), 0, 0, interp);
        resized.copyTo(dst(Rect(x, y, dw, dh)));
    };

    Mat sdrCanvas(targetH, targetW, CV_8UC3);
    layout(sdrMat, sdrCanvas, Scalar(255, 255, 255), INTER_LANCZOS4);

    Mat hdrCanvas;
    if (hasHDR) {
        hdrCanvas.create(targetH, targetW, CV_32FC3);
        // Pad with 1.0 (SDR White in Linear HDR)
        layout(hdrMat, hdrCanvas, Scalar(1.0f, 1.0f, 1.0f), INTER_LANCZOS4); 
    }

    // 4. Draw Metadata
    PhotoMeta meta = parseExif(inputPath);
    TextRenderer fontMain("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf", 52);
    TextRenderer fontSub("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf", 40);

    string params = meta.f_number + " " + meta.shutter + " " + meta.focal + " " + meta.iso;
    string subtext = meta.date; 
    int footerY = targetH - bottomPad + 60;

    // SDR Colors
    Scalar sdrText(0,0,0);
    Scalar sdrSub(100,100,100);
    
    // HDR Colors (Linear)
    // Black is 0. Gray #666 (0.4 sRGB) is approx 0.133 Linear
    Scalar hdrText(0,0,0);
    Scalar hdrSub(0.133, 0.133, 0.133);

    // Draw SDR
    fontMain.putText<uchar>(sdrCanvas, params, Point(margin, footerY + 52), sdrText);
    fontSub.putText<uchar>(sdrCanvas, subtext, Point(margin, footerY + 122), sdrSub);

    // Draw HDR
    if (hasHDR) {
        fontMain.putText<float>(hdrCanvas, params, Point(margin, footerY + 52), hdrText);
        fontSub.putText<float>(hdrCanvas, subtext, Point(margin, footerY + 122), hdrSub);
    }

    // Logos
    string logoPath = "logo/default.png";
    string m = meta.make; transform(m.begin(), m.end(), m.begin(), ::tolower);
    if (m.find("nikon") != string::npos) logoPath = "logo/nikon.png";
    else if (m.find("google") != string::npos) logoPath = "logo/google.png";

    Mat logo = imread(logoPath, IMREAD_UNCHANGED);
    if (!logo.empty()) {
        if (logo.channels() < 4) cvtColor(logo, logo, logo.channels()==1 ? COLOR_GRAY2BGRA : COLOR_BGR2BGRA);
        
        int sz = 120;
        resize(logo, logo, Size(sz, sz), 0, 0, INTER_AREA);
        int lx = targetW - margin - sz;
        int ly = footerY;

        // Blend SDR
        for(int r=0; r<logo.rows; r++) {
            for(int c=0; c<logo.cols; c++) {
                Vec4b p = logo.at<Vec4b>(r,c);
                float a = p[3]/255.f;
                if(a>0) {
                    Vec3b& b = sdrCanvas.at<Vec3b>(ly+r, lx+c);
                    for(int k=0;k<3;k++) b[k] = saturate_cast<uchar>(b[k]*(1-a) + p[k]*a);
                }
            }
        }

        // Blend HDR (Linear)
        if (hasHDR) {
            for(int r=0; r<logo.rows; r++) {
                for(int c=0; c<logo.cols; c++) {
                    Vec4b p = logo.at<Vec4b>(r,c);
                    float a = p[3]/255.f;
                    if(a>0) {
                        Vec3f& b = hdrCanvas.at<Vec3f>(ly+r, lx+c);
                        for(int k=0;k<3;k++) {
                            // Convert logo sRGB to Linear: pow(x/255, 2.2)
                            float val = pow(p[k]/255.f, 2.2f); 
                            b[k] = b[k]*(1-a) + val*a;
                        }
                    }
                }
            }
        }
        
        string cam = meta.model;
        int w = fontMain.getTextWidth(cam);
        fontMain.putText<uchar>(sdrCanvas, cam, Point(lx - 40 - w, footerY + 52), sdrText);
        if (hasHDR) fontMain.putText<float>(hdrCanvas, cam, Point(lx - 40 - w, footerY + 52), hdrText);
        
        if(!meta.lens.empty()) {
            int w2 = fontSub.getTextWidth(meta.lens);
            fontSub.putText<uchar>(sdrCanvas, meta.lens, Point(lx - 40 - w2, footerY + 122), sdrSub);
            if(hasHDR) fontSub.putText<float>(hdrCanvas, meta.lens, Point(lx - 40 - w2, footerY + 122), hdrSub);
        }
    }

    // 5. Encode (Raw SDR + Raw HDR)
    cout << "Encoding..." << endl;
    
    if (hasHDR) {
        // Prepare Raw Images
        // SDR: Convert BGR to RGBA
        Mat sdrRaw; cvtColor(sdrCanvas, sdrRaw, COLOR_BGR2RGBA);
        
        // HDR: Convert BGR Linear Float to RGBA Half Float (16F)
        Mat hdrLinear; cvtColor(hdrCanvas, hdrLinear, COLOR_BGR2RGBA);
        Mat hdrHalf; hdrLinear.convertTo(hdrHalf, CV_16F);

        uhdr_codec_private_t* enc = uhdr_create_encoder();
        uhdr_enable_gpu_acceleration(enc, 0);

        uhdr_raw_image_t sdr_img = { UHDR_IMG_FMT_32bppRGBA8888, UHDR_CG_BT_709, UHDR_CT_SRGB, UHDR_CR_FULL_RANGE, 
                                     (unsigned)sdrRaw.cols, (unsigned)sdrRaw.rows };
        sdr_img.planes[UHDR_PLANE_PACKED] = sdrRaw.data;
        sdr_img.stride[UHDR_PLANE_PACKED] = sdrRaw.cols; // Stride in pixels

        uhdr_raw_image_t hdr_img = { UHDR_IMG_FMT_64bppRGBAHalfFloat, UHDR_CG_BT_709, UHDR_CT_LINEAR, UHDR_CR_FULL_RANGE,
                                     (unsigned)hdrHalf.cols, (unsigned)hdrHalf.rows };
        hdr_img.planes[UHDR_PLANE_PACKED] = hdrHalf.data;
        hdr_img.stride[UHDR_PLANE_PACKED] = hdrHalf.cols;

        checkUhdr(uhdr_enc_set_raw_image(enc, &sdr_img, UHDR_SDR_IMG), "Set SDR");
        checkUhdr(uhdr_enc_set_raw_image(enc, &hdr_img, UHDR_HDR_IMG), "Set HDR");
        uhdr_enc_set_quality(enc, JPEG_QUALITY, UHDR_BASE_IMG);
        uhdr_enc_set_quality(enc, JPEG_QUALITY, UHDR_GAIN_MAP_IMG);

        // Inject EXIF
        vector<uint8_t> exif = getRawExif(inputPath);
        if (!exif.empty()) {
            uhdr_mem_block_t eb = { exif.data(), exif.size(), exif.size() };
            uhdr_enc_set_exif_data(enc, &eb);
        }

        if (checkUhdr(uhdr_encode(enc), "Encode")) {
            uhdr_compressed_image_t* out = uhdr_get_encoded_stream(enc);
            ofstream f(outputPath, ios::binary);
            f.write((char*)out->data, out->data_sz);
            cout << "Saved UltraHDR: " << outputPath << endl;
        }
        uhdr_release_encoder(enc);
    } else {
        // Standard JPEG
        vector<uchar> buf;
        vector<int> p = {IMWRITE_JPEG_QUALITY, JPEG_QUALITY};
        imencode(".jpg", sdrCanvas, buf, p);
        ofstream f(outputPath, ios::binary);
        f.write((char*)buf.data(), buf.size());
        f.close();
        
        // Copy Exif Legacy
        try {
            Exiv2::Image::AutoPtr src = Exiv2::ImageFactory::open(inputPath); src->readMetadata();
            Exiv2::Image::AutoPtr dst = Exiv2::ImageFactory::open(outputPath); dst->readMetadata();
            dst->setExifData(src->exifData());
            dst->writeMetadata();
        } catch(...) {}
        cout << "Saved SDR: " << outputPath << endl;
    }

    return 0;
}
