// Standalone harness for pil_resize: reads a raw PPM (P6) and writes a raw
// PPM, so it can be diffed byte-for-byte against Pillow without OpenCV.
//
//   pil_resize_cli in.ppm out.ppm 640 640
//
// Not part of the board binary; it exists so the resize port can be verified
// against PIL on the host before it is trusted on RK3588.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "pil_resize.h"

namespace {

bool read_ppm(const char* path, ec::RgbImage* img) {
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr) {
        std::fprintf(stderr, "cannot open %s\n", path);
        return false;
    }
    char magic[3] = {0, 0, 0};
    if (std::fscanf(f, "%2s", magic) != 1 || std::strcmp(magic, "P6") != 0) {
        std::fprintf(stderr, "%s: expected binary PPM (P6)\n", path);
        std::fclose(f);
        return false;
    }
    int w = 0, h = 0, maxval = 0;
    if (std::fscanf(f, "%d %d %d", &w, &h, &maxval) != 3 || maxval != 255 || w <= 0 || h <= 0) {
        std::fprintf(stderr, "%s: bad PPM header\n", path);
        std::fclose(f);
        return false;
    }
    std::fgetc(f);  // single whitespace after the header
    img->width = w;
    img->height = h;
    img->data.resize(static_cast<size_t>(w) * h * 3);
    const size_t got = std::fread(img->data.data(), 1, img->data.size(), f);
    std::fclose(f);
    if (got != img->data.size()) {
        std::fprintf(stderr, "%s: truncated pixel data (%zu of %zu)\n", path, got, img->data.size());
        return false;
    }
    return true;
}

bool write_ppm(const char* path, const ec::RgbImage& img) {
    FILE* f = std::fopen(path, "wb");
    if (f == nullptr) {
        std::fprintf(stderr, "cannot write %s\n", path);
        return false;
    }
    std::fprintf(f, "P6\n%d %d\n255\n", img.width, img.height);
    const size_t put = std::fwrite(img.data.data(), 1, img.data.size(), f);
    std::fclose(f);
    return put == img.data.size();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s in.ppm out.ppm W H\n", argv[0]);
        return 2;
    }
    const int out_w = std::atoi(argv[3]);
    const int out_h = std::atoi(argv[4]);

    ec::RgbImage src;
    if (!read_ppm(argv[1], &src)) return 1;

    ec::PilResampler r;
    if (!r.prepare(src.width, src.height, out_w, out_h)) {
        std::fprintf(stderr, "prepare failed\n");
        return 1;
    }
    ec::RgbImage dst;
    if (!r.resize(src, &dst)) {
        std::fprintf(stderr, "resize failed\n");
        return 1;
    }
    if (!write_ppm(argv[2], dst)) return 1;

    std::fprintf(stderr, "%dx%d -> %dx%d (horizontal_first=%d)\n", src.width, src.height, dst.width,
                 dst.height, r.horizontal_first() ? 1 : 0);
    return 0;
}
