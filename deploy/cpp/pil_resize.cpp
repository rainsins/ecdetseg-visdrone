#include "pil_resize.h"

#include <cmath>
#include <utility>

namespace ec {
namespace {

// Pillow src/libImaging/Resample.c: 8 bits for the result, plus two extra bits
// for overflow because filter sums can exceed 1.0.
constexpr int kPrecisionBits = 32 - 8 - 2;  // 22
constexpr int32_t kOne = 1 << kPrecisionBits;
constexpr int32_t kSeed = 1 << (kPrecisionBits - 1);

// Bilinear triangle kernel, support 1.0 (Pillow's `BILINEAR` filter).
inline double bilinear_filter(double x) {
    if (x < 0.0) x = -x;
    if (x < 1.0) return 1.0 - x;
    return 0.0;
}

// Pillow clip8() indexes a 1280-entry table with `in >> PRECISION_BITS`, valid
// for -640..639, and the table holds clamp(i - 640, 0, 255) at raw index i.
// With the pointer offset the lookup reduces to clamp(idx, 0, 255).
inline uint8_t clip8(int32_t in) {
    int idx = in >> kPrecisionBits;  // arithmetic shift, matches Pillow
    if (idx < 0) return 0;
    if (idx > 255) return 255;
    return static_cast<uint8_t>(idx);
}

// Pillow normalize_coeffs_8bpc: round half away from zero, implemented as a
// truncating cast of (0.5 + v * 2^22) for positives and (-0.5 + v * 2^22) for
// negatives.  The asymmetry matters for exactness.
inline int32_t quantize_coeff(double v) {
    if (v < 0) return static_cast<int32_t>(-0.5 + v * kOne);
    return static_cast<int32_t>(0.5 + v * kOne);
}

}  // namespace

bool build_pil_axis_coeffs(int in_size, int out_size, AxisCoeffs* out) {
    if (out == nullptr || in_size <= 0 || out_size <= 0) return false;

    // precompute_coeffs with the full box: scale == in_size / out_size.
    const double scale = static_cast<double>(in_size) / static_cast<double>(out_size);
    const double filterscale = scale < 1.0 ? 1.0 : scale;

    const double support = 1.0 * filterscale;  // BILINEAR support is 1.0
    const int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
    if (ksize <= 0) return false;

    out->in_size = in_size;
    out->out_size = out_size;
    out->ksize = ksize;
    out->bounds.assign(static_cast<size_t>(out_size) * 2, 0);
    out->kernel.assign(static_cast<size_t>(out_size) * ksize, 0);

    const double inv_filterscale = 1.0 / filterscale;
    std::vector<double> raw(static_cast<size_t>(ksize), 0.0);

    for (int xx = 0; xx < out_size; ++xx) {
        const double center = (xx + 0.5) * scale;

        // Truncating double->int casts, exactly as Pillow writes them.
        int xmin = static_cast<int>(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;
        if (xmax < 0) xmax = 0;

        // Pass 1: raw weights, accumulated in output order (Pillow sums first,
        // normalizes second, rounds last -- the order is observable in the
        // rounding of the 22-bit fixed-point coefficients).
        double ww = 0.0;
        for (int x = 0; x < xmax; ++x) {
            raw[x] = bilinear_filter((x + xmin - center + 0.5) * inv_filterscale);
            ww += raw[x];
        }

        int32_t* k = out->kernel.data() + static_cast<size_t>(xx) * ksize;
        for (int x = 0; x < xmax; ++x) {
            k[x] = quantize_coeff(ww != 0.0 ? raw[x] / ww : raw[x]);
        }
        for (int x = xmax; x < ksize; ++x) {
            k[x] = 0;  // Pillow zero-fills the unused tail
        }

        out->bounds[static_cast<size_t>(xx) * 2 + 0] = xmin;
        out->bounds[static_cast<size_t>(xx) * 2 + 1] = xmax;
    }
    return true;
}

bool PilResampler::prepare(int src_w, int src_h, int dst_w, int dst_h) {
    prepared_ = false;
    h_axis_ = AxisCoeffs{};
    v_axis_ = AxisCoeffs{};
    if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) return false;

    src_w_ = src_w;
    src_h_ = src_h;
    dst_w_ = dst_w;
    dst_h_ = dst_h;

    need_horizontal_ = (dst_w != src_w);
    need_vertical_ = (dst_h != src_h);

    // Pillow: run the vertical pass first only when the height is being scaled
    // down more than twice as much as the width.  For 1360x765 -> 640x640 both
    // differences are positive and (765-640) < (1360-640)*2, so horizontal runs
    // first and the ybox row restriction applies.
    horizontal_first_ = !((src_h - dst_h) > 0 && (src_h - dst_h) > (src_w - dst_w) * 2);

    if (!need_horizontal_ && !need_vertical_) {
        prepared_ = true;
        return true;
    }

    if (need_horizontal_) {
        if (!build_pil_axis_coeffs(src_w, dst_w, &h_axis_)) return false;
    }
    // Pillow computes the vertical coefficients whenever the vertical pass will
    // run, and also when horizontal-first needs bounds_vert for the row window.
    if (need_vertical_ || (need_horizontal_ && horizontal_first_)) {
        if (!build_pil_axis_coeffs(src_h, dst_h, &v_axis_)) return false;
    }

    prepared_ = true;
    return true;
}

void PilResampler::pass_horizontal(const RgbImage& src, int offset, int height,
                                   RgbImage* dst) const {
    const AxisCoeffs& ax = h_axis_;
    const int ksize = ax.ksize;
    const int out_w = dst->width;

    for (int yy = 0; yy < height; ++yy) {
        const uint8_t* line_in = src.row(yy + offset);
        uint8_t* line_out = dst->row(yy);
        for (int xx = 0; xx < out_w; ++xx) {
            const int xmin = ax.bounds[static_cast<size_t>(xx) * 2 + 0];
            const int xmax = ax.bounds[static_cast<size_t>(xx) * 2 + 1];
            const int32_t* k = ax.kernel.data() + static_cast<size_t>(xx) * ksize;

            int32_t ss0 = kSeed, ss1 = kSeed, ss2 = kSeed;
            for (int x = 0; x < xmax; ++x) {
                const uint8_t* px = line_in + static_cast<size_t>(x + xmin) * 3;
                const int32_t kv = k[x];
                ss0 += static_cast<int32_t>(px[0]) * kv;
                ss1 += static_cast<int32_t>(px[1]) * kv;
                ss2 += static_cast<int32_t>(px[2]) * kv;
            }
            uint8_t* out = line_out + static_cast<size_t>(xx) * 3;
            out[0] = clip8(ss0);
            out[1] = clip8(ss1);
            out[2] = clip8(ss2);
        }
    }
}

void PilResampler::pass_vertical(const RgbImage& src, const AxisCoeffs& ax,
                                 RgbImage* dst) const {
    const int ksize = ax.ksize;
    const int width = dst->width;  // the intermediate may be src_w_ wide

    for (int yy = 0; yy < ax.out_size; ++yy) {
        const int ymin = ax.bounds[static_cast<size_t>(yy) * 2 + 0];
        const int ymax = ax.bounds[static_cast<size_t>(yy) * 2 + 1];
        const int32_t* k = ax.kernel.data() + static_cast<size_t>(yy) * ksize;
        uint8_t* line_out = dst->row(yy);

        for (int xx = 0; xx < width; ++xx) {
            int32_t ss0 = kSeed, ss1 = kSeed, ss2 = kSeed;
            for (int y = 0; y < ymax; ++y) {
                const uint8_t* px = src.row(y + ymin) + static_cast<size_t>(xx) * 3;
                const int32_t kv = k[y];
                ss0 += static_cast<int32_t>(px[0]) * kv;
                ss1 += static_cast<int32_t>(px[1]) * kv;
                ss2 += static_cast<int32_t>(px[2]) * kv;
            }
            uint8_t* out = line_out + static_cast<size_t>(xx) * 3;
            out[0] = clip8(ss0);
            out[1] = clip8(ss1);
            out[2] = clip8(ss2);
        }
    }
}

bool PilResampler::resize(const RgbImage& src, RgbImage* dst) const {
    if (!prepared_ || dst == nullptr) return false;
    if (src.width != src_w_ || src.height != src_h_) return false;
    if (src.data.size() < static_cast<size_t>(src_w_) * src_h_ * 3) return false;

    if (!need_horizontal_ && !need_vertical_) {
        // Pillow returns a copy with identical bytes.
        dst->width = src_w_;
        dst->height = src_h_;
        dst->data = src.data;
        return true;
    }

    if (!need_horizontal_) {
        dst->width = dst_w_;
        dst->height = dst_h_;
        dst->data.assign(static_cast<size_t>(dst_w_) * dst_h_ * 3, 0);
        pass_vertical(src, v_axis_, dst);
        return true;
    }

    if (horizontal_first_) {
        // Pillow runs the horizontal pass only over the source rows the
        // vertical pass will read, then rebases the vertical bounds by that
        // offset.  Replicating the window matters: it changes which rows are
        // filtered and therefore the exact output bytes.
        int ybox_first = need_vertical_ ? v_axis_.bounds[0] : 0;
        int ybox_last = need_vertical_
                            ? v_axis_.bounds[static_cast<size_t>(dst_h_) * 2 - 2] +
                                  v_axis_.bounds[static_cast<size_t>(dst_h_) * 2 - 1]
                            : src_h_;
        if (ybox_first < 0) ybox_first = 0;
        if (ybox_last > src_h_) ybox_last = src_h_;
        int rows = ybox_last - ybox_first;
        if (rows <= 0) {
            ybox_first = 0;
            rows = src_h_;
        }

        RgbImage temp(dst_w_, rows);
        pass_horizontal(src, ybox_first, rows, &temp);

        if (!need_vertical_) {
            temp.height = dst_h_;
            *dst = std::move(temp);
            return true;
        }

        dst->width = dst_w_;
        dst->height = dst_h_;
        dst->data.assign(static_cast<size_t>(dst_w_) * dst_h_ * 3, 0);

        AxisCoeffs shifted = v_axis_;
        if (ybox_first != 0) {
            for (int i = 0; i < shifted.out_size; ++i) {
                shifted.bounds[static_cast<size_t>(i) * 2] -= ybox_first;
            }
        }
        pass_vertical(temp, shifted, dst);
        return true;
    }

    // Vertical first, then horizontal over the full width.
    RgbImage temp(src_w_, dst_h_);
    pass_vertical(src, v_axis_, &temp);

    dst->width = dst_w_;
    dst->height = dst_h_;
    dst->data.assign(static_cast<size_t>(dst_w_) * dst_h_ * 3, 0);
    pass_horizontal(temp, 0, dst_h_, dst);
    return true;
}

bool pil_resize_bilinear(const RgbImage& src, int out_w, int out_h, RgbImage* dst) {
    PilResampler r;
    if (!r.prepare(src.width, src.height, out_w, out_h)) return false;
    return r.resize(src, dst);
}

}  // namespace ec
