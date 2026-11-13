#include "grid_sample_cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <atomic>
#include <chrono>

namespace ec {
namespace {
using Clock = std::chrono::steady_clock;
std::atomic<bool> profile_enabled{false};
std::atomic<uint64_t> profile_calls{0}, profile_failures{0}, callback_ns{0}, sampling_ns{0};
uint64_t elapsed_ns(Clock::time_point start) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now() - start).count());
}
class SamplingTimer {
public:
    explicit SamplingTimer(bool enabled) : enabled_(enabled) {
        if (enabled_) start_ = Clock::now();
    }
    ~SamplingTimer() {
        if (enabled_) sampling_ns.fetch_add(elapsed_ns(start_), std::memory_order_relaxed);
    }
private:
    bool enabled_;
    Clock::time_point start_{};
};
int fail(const char* message) {
    std::fprintf(stderr, "EC GridSample CPU: %s\n", message);
    return RKNN_ERR_PARAM_INVALID;
}

bool text_attr(rknn_custom_op_context* ctx, const char* name, const char* expected) {
    rknn_custom_op_attr attr{};
    rknn_custom_op_get_op_attr(ctx, name, &attr);
    const size_t len = std::strlen(expected);
    if (!attr.data || (attr.dtype != RKNN_TENSOR_UINT8 && attr.dtype != RKNN_TENSOR_INT8)) return false;
    const char* p = static_cast<const char*>(attr.data);
    // The runtime wraps string attributes in double quotes, e.g. "bilinear".
    const bool quoted = (attr.n_elems == len + 2) && p[0] == '"' && p[len + 1] == '"';
    const bool bare = (attr.n_elems == len) || (attr.n_elems == len + 1 && p[len] == '\0');
    if (!quoted && !bare) return false;
    return std::memcmp(quoted ? p + 1 : p, expected, len) == 0;
}

bool alignment(rknn_custom_op_context* ctx, bool* align) {
    rknn_custom_op_attr attr{};
    rknn_custom_op_get_op_attr(ctx, "align_corners", &attr);
    if (!attr.data || attr.n_elems != 1) return false;
    int64_t value = -1;
    if (attr.dtype == RKNN_TENSOR_INT64) {
        std::memcpy(&value, attr.data, sizeof(value));
    } else if (attr.dtype == RKNN_TENSOR_INT32) {
        int32_t v;
        std::memcpy(&v, attr.data, sizeof(v));
        value = v;
    } else if (attr.dtype == RKNN_TENSOR_INT8 || attr.dtype == RKNN_TENSOR_UINT8 ||
               attr.dtype == RKNN_TENSOR_BOOL) {
        value = *static_cast<const unsigned char*>(attr.data);
    }
    if (value != 0 && value != 1) return false;
    *align = value != 0;
    return true;
}

struct Tensor {
    unsigned char* data = nullptr;
    size_t n = 0, c = 0, h = 0, w = 0;
    bool nhwc = false;
    size_t index(size_t b, size_t ch, size_t y, size_t x) const {
        return nhwc ? ((b * h + y) * w + x) * c + ch
                    : ((b * c + ch) * h + y) * w + x;
    }
    float get(size_t b, size_t ch, size_t y, size_t x) const {
        float value;
        std::memcpy(&value, data + index(b, ch, y, x) * sizeof(float), sizeof(value));
        return value;
    }
    void put(size_t b, size_t ch, size_t y, size_t x, float value) const {
        std::memcpy(data + index(b, ch, y, x) * sizeof(float), &value, sizeof(value));
    }
};

bool tensor_view(rknn_custom_op_tensor& tensor, Tensor* view) {
    const auto& a = tensor.attr;
    const auto& m = tensor.mem;
    if (a.n_dims != 4 || a.type != RKNN_TENSOR_FLOAT32 || a.qnt_type != RKNN_TENSOR_QNT_NONE ||
        (a.fmt != RKNN_TENSOR_NCHW && a.fmt != RKNN_TENSOR_NHWC) ||
        !m.virt_addr || m.offset < 0) return false;
    size_t count = 1;
    for (size_t i = 0; i < 4; ++i) {
        if (!a.dims[i] || count > std::numeric_limits<uint32_t>::max() / a.dims[i]) return false;
        count *= a.dims[i];
    }
    if (count > std::numeric_limits<uint32_t>::max() / sizeof(float)) return false;
    const size_t bytes = count * sizeof(float);
    view->nhwc = a.fmt == RKNN_TENSOR_NHWC;
    view->n = a.dims[0];
    view->c = a.dims[view->nhwc ? 3 : 1];
    view->h = a.dims[view->nhwc ? 1 : 2];
    view->w = a.dims[view->nhwc ? 2 : 3];
    if (a.n_elems != count || a.size != bytes ||
        (a.w_stride && a.w_stride != view->w) ||
        (a.h_stride && a.h_stride != view->h) ||
        (a.size_with_stride && a.size_with_stride != bytes) ||
        static_cast<size_t>(m.offset) > m.size || bytes > m.size - static_cast<size_t>(m.offset)) return false;
    view->data = static_cast<unsigned char*>(m.virt_addr) + m.offset;
    return true;
}

int compute(rknn_custom_op_context* ctx, rknn_custom_op_tensor* inputs, uint32_t ni,
            rknn_custom_op_tensor* outputs, uint32_t no, bool profile) {
    if (!ctx || !inputs || !outputs || ni != 2 || no != 1) return fail("expected 2 inputs and 1 output");
    bool align = false;
    if (!text_attr(ctx, "mode", "bilinear") || !text_attr(ctx, "padding_mode", "zeros") ||
        !alignment(ctx, &align)) return fail(
            "require explicit bilinear/zeros string attributes and align_corners=0/1; "
            "missing or numeric mode encodings need board inspection, not guessed defaults");
    Tensor feature, grid, out;
    // Resolve the known board ABI using output geometry, before stride checks.
    // A standard NCHW grid may also end in 2 (OW=2); do not transpose it.
    auto grid_tensor = inputs[1];
    const bool output_valid = tensor_view(outputs[0], &out);
    const auto& ga = grid_tensor.attr;
    if (output_valid && ga.n_dims == 4 && ga.fmt == RKNN_TENSOR_NCHW &&
        !(ga.dims[1] == 2 && ga.dims[2] == out.h && ga.dims[3] == out.w) &&
        ga.dims[3] == 2 && ga.dims[1] == out.h && ga.dims[2] == out.w) {
        grid_tensor.attr.fmt = RKNN_TENSOR_NHWC;
    }
    if (!output_valid || !tensor_view(inputs[0], &feature) ||
        !tensor_view(grid_tensor, &grid)) {
        for (uint32_t i = 0; i < ni + no; ++i) {
            const auto& t = i < ni ? inputs[i] : outputs[i - ni];
            std::fprintf(stderr, "  tensor=%u dtype=%d fmt=%d qnt=%d dims=%u,%u,%u,%u "
                         "bytes=%u stride_bytes=%u w_stride=%u h_stride=%u mem=%u offset=%d\n",
                         i, t.attr.type, t.attr.fmt, t.attr.qnt_type, t.attr.dims[0],
                         t.attr.dims[1], t.attr.dims[2], t.attr.dims[3], t.attr.size,
                         t.attr.size_with_stride, t.attr.w_stride, t.attr.h_stride, t.mem.size, t.mem.offset);
        }
        return fail("only packed unquantized FLOAT32 NCHW/NHWC callback tensors supported");
    }
    // ONNX grid is semantically [N,OH,OW,2]. Accept the runtime's explicit
    // NHWC representation, or an explicitly transposed NCHW [N,2,OH,OW].
    if (grid.c != 2 || feature.n != grid.n || out.n != feature.n || out.c != feature.c ||
        out.h != grid.h || out.w != grid.w) return fail("feature/grid/output shapes disagree");
    const SamplingTimer sampling_timer(profile);
    for (size_t n = 0; n < grid.n; ++n) {
        for (size_t y = 0; y < grid.h; ++y) {
            for (size_t x = 0; x < grid.w; ++x) {
                const float gx = grid.get(n, 0, y, x), gy = grid.get(n, 1, y, x);
                if (!std::isfinite(gx) || !std::isfinite(gy)) return fail("non-finite grid coordinate");
                const float px = align ? ((gx + 1.f) / 2.f) * (feature.w - 1)
                                       : ((gx + 1.f) * feature.w - 1.f) * .5f;
                const float py = align ? ((gy + 1.f) / 2.f) * (feature.h - 1)
                                       : ((gy + 1.f) * feature.h - 1.f) * .5f;
                // Short-circuit far-outside coordinates BEFORE any float->int cast.
                if (px <= -1.f || py <= -1.f || px >= feature.w || py >= feature.h) {
                    for (size_t c = 0; c < feature.c; ++c) out.put(n, c, y, x, 0.f);
                    continue;
                }
                const int64_t x0 = static_cast<int64_t>(std::floor(px));
                const int64_t y0 = static_cast<int64_t>(std::floor(py));
                const float dx = px - x0, dy = py - y0;
                for (size_t c = 0; c < feature.c; ++c) {
                    auto at = [&](int64_t yy, int64_t xx) {
                        return xx < 0 || yy < 0 || static_cast<size_t>(xx) >= feature.w ||
                               static_cast<size_t>(yy) >= feature.h ? 0.f :
                               feature.get(n, c, static_cast<size_t>(yy), static_cast<size_t>(xx));
                    };
                    const float value = at(y0, x0) * ((1.f - dx) * (1.f - dy)) +
                        at(y0, x0 + 1) * (dx * (1.f - dy)) +
                        at(y0 + 1, x0) * ((1.f - dx) * dy) + at(y0 + 1, x0 + 1) * (dx * dy);
                    if (!std::isfinite(value)) return fail("non-finite sampled feature");
                    out.put(n, c, y, x, value);
                }
            }
        }
    }
    return RKNN_SUCC;
}
int compute_callback(rknn_custom_op_context* ctx, rknn_custom_op_tensor* inputs, uint32_t ni,
                     rknn_custom_op_tensor* outputs, uint32_t no) {
    if (!profile_enabled.load(std::memory_order_relaxed))
        return compute(ctx, inputs, ni, outputs, no, false);
    const auto start = Clock::now();
    const int status = compute(ctx, inputs, ni, outputs, no, true);
    callback_ns.fetch_add(elapsed_ns(start), std::memory_order_relaxed);
    profile_calls.fetch_add(1, std::memory_order_relaxed);
    if (status != RKNN_SUCC) profile_failures.fetch_add(1, std::memory_order_relaxed);
    return status;
}
}  // namespace

void enable_grid_sample_profile(bool enabled) { profile_enabled.store(enabled); }
void reset_grid_sample_profile() {
    profile_calls.store(0); profile_failures.store(0);
    callback_ns.store(0); sampling_ns.store(0);
}
GridSampleProfile grid_sample_profile() {
    return {profile_calls.load(), profile_failures.load(), callback_ns.load(), sampling_ns.load()};
}

rknn_custom_op make_grid_sample_cpu_op() {
    rknn_custom_op op{};
    op.version = 1;
    op.target = RKNN_TARGET_TYPE_CPU;
    std::strcpy(op.op_type, "GridSample");
    op.compute = compute_callback;
    return op;
}
}  // namespace ec
