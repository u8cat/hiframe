# HDR Image Frame

HDR EXIF Frame is a command line tool that creates a frame around a photo displaying its EXIF metadata. It distinguishes itself from other similar tools in supporting HDR output.

## Building

This tool has been built and tested on Ubuntu 24.04.

### Prerequisites

- [CMake](https://cmake.org/) v3.15 or later.
- C++ compiler, supporting at least C++ 20.
- [OpenCV 4](https://opencv.org/).
- [FreeType 2](http://freetype.org/).
- [Exiv 2](https://exiv2.org/).
- libjpeg.
- [libultrahdr](https://github.com/google/libultrahdr/tree/5ed39d67cd31d254e84ebf76b03d4b7bcc12e2f7).

Ubuntu users can install most prerequisites by running
```bash
sudo apt install cmake pkg-config
sudo apt install libopencv-dev libfreetype-dev libexiv2-dev libjpeg-dev
```

But libultrahdr must be built from source. To build libultrahdr, clone its repository by running
```bash
git clone https://github.com/google/libultrahdr.git
cd libultrahdr
git checkout 5ed39d67cd31d254e84ebf76b03d4b7bcc12e2f7
```

And then build and install the library to `/usr/local/lib` by running
```bash
mkdir build && cd build
cmake .. -DUHDR_MAX_DIMENSION=16384 -DUHDR_WRITE_XMP=on -DUHDR_WRITE_ISO=on
make -j`nproc`
sudo make install
```

This configuration outputs HDR gain maps in both XMP and ISO 21496-1 format to maximize compatability. Please refer to [building.md](https://github.com/google/libultrahdr/blob/main/docs/building.md) for further detail.

### Build Steps

Clone the source code:
```bash
git clone https://github.com/u8cat/hiframe.git
cd hiframe
```

Compile:
```bash
mkdir build && cd build
cmake ..
make
```

The output executable is `build/hiframe`.

## Usage

Create a symbolic link of `logo` to working directory, and then run

```
exif_framer <input.jpg> [output.jpg]
```
