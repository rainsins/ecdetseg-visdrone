// Bit-exact port of Pillow's 8-bit-per-channel BILINEAR resampler.
//
// The deployment contract requires the board preprocessing to be byte-identical
// to `PIL.Image.resize((640, 640), Image.BILINEAR)` on the RGB view.  OpenCV's
// INTER_LINEAR / INTER_AREA are both measurably different (see README 3.2), so
// "close enough" is not acceptable here: a 1-LSB difference is a different
// input tensor, and the whole point of this port is that the board and the
// simulator see the same bytes.
//
// Source of truth: Pillow src/libImaging/Resample.c
//   * PRECISION_BITS = 32 - 8 - 2 = 22
//   * coefficients rounded with (int)(+/-0.5 + w * (1 << 22))
//   * accumulator seeded with 1 << (PRECISION_BITS - 1)
//   * clip8 saturates the arithmetic right shift into 0..255
//   * two separable passes; horizontal first unless the vertical downscale
//     exceeds twice the horizontal downscale
#pragma once

#include <cstdint>
#include <vector>

namespace ec {

// Interleaved RGB, 8 bits per channel, tightly packed rows.
struct RgbImage {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> data;

    RgbImage() = default;
    RgbImage(int w, int h) : width(w), height(h), data(static_cast<size_t>(w) * h * 3, 0) {}

    bool empty() const { return width <= 0 || height <= 0; }
    size_t stride() const { return static_cast<size_t>(width) * 3; }
    uint8_t* row(int y) { return data.data() + static_cast<size_t>(y) * stride(); }
    const uint8_t* row(int y) const { return data.data() + static_cast<size_t>(y) * stride(); }
};

// One axis of the separable filter: Pillow's precompute_coeffs() output after
// normalize_coeffs_8bpc().  Reused across frames because a video stream has a
// fixed source size, so the 22-bit coefficients are computed once.
struct AxisCoeffs {
    int in_size = 0;
    int out_size = 0;
    int ksize = 0;
    std::vector<int> bounds;      // [xmin, xmax] per output index
    std::vector<int32_t> kernel;  // out_size * ksize, Q22 fixed point

    bool valid() const { return ksize > 0 && out_size > 0; }
};

// Build the Q22 coefficient table for one axis.  `in_size` samples mapped onto
// `out_size` samples.  Returns false on invalid sizes.
bool build_pil_axis_coeffs(int in_size, int out_size, AxisCoeffs* out);

// Reusable resampler.  Prepare once per (source size, target size) pair, then
// call resize() for every frame.
class PilResampler {
public:
    bool prepare(int src_w, int src_h, int dst_w, int dst_h);
    bool ready() const { return prepared_; }
    bool resize(const RgbImage& src, RgbImage* dst) const;

    // Exposed for tests and for mirroring Pillow's pass order in logs.
    bool horizontal_first() const { return horizontal_first_; }

private:
    void pass_horizontal(const RgbImage& src, int offset, int height, RgbImage* dst) const;
    void pass_vertical(const RgbImage& src, const AxisCoeffs& ax, RgbImage* dst) const;

    bool prepared_ = false;
    int src_w_ = 0, src_h_ = 0, dst_w_ = 0, dst_h_ = 0;
    bool need_horizontal_ = false;
    bool need_vertical_ = false;
    bool horizontal_first_ = true;
    AxisCoeffs h_axis_;
    AxisCoeffs v_axis_;
};

// Convenience wrapper: builds a one-shot resampler.  Use PilResampler directly
// when resizing many frames of the same size.
bool pil_resize_bilinear(const RgbImage& src, int out_w, int out_h, RgbImage* dst);

}  // namespace ec
