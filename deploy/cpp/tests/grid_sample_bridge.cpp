// Host test adapter: mocks attribute delivery, NOT the RKNN runtime/backend.
#include "grid_sample_cpu.h"
#include <cstring>
#include <vector>

struct Attributes { int64_t align; int fault; };
extern "C" void profile_enable(int enabled) {
    ec::enable_grid_sample_profile(enabled != 0); ec::reset_grid_sample_profile();
}
extern "C" void profile_read(uint64_t* values) {
    const auto p = ec::grid_sample_profile();
    values[0] = p.calls; values[1] = p.failures;
    values[2] = p.callback_ns; values[3] = p.sampling_ns;
}
extern "C" void rknn_custom_op_get_op_attr(rknn_custom_op_context* ctx, const char* name,
                                            rknn_custom_op_attr* attr) {
    auto* a = static_cast<Attributes*>(ctx->priv_data);
    if (a->fault == 1) return;
    if (std::strcmp(name, "align_corners") == 0) {
        attr->dtype = RKNN_TENSOR_INT64; attr->n_elems = 1; attr->data = &a->align;
    } else {
        const char* text = std::strcmp(name, "mode") == 0 ?
            (a->fault == 2 ? "nearest" : "bilinear") : "zeros";
        if (a->fault >= 8) text = std::strcmp(name, "mode") == 0 ?
            "\"bilinear\"" : "\"zeros\"";
        attr->dtype = RKNN_TENSOR_UINT8; attr->n_elems = std::strlen(text);
        attr->data = const_cast<char*>(text);
    }
}

extern "C" int sample(const float* x, const float* g, float* y, int n, int c,
                       int h, int w, int oh, int ow, int align, int layout, int fault) {
    // Vary each layout independently; use unaligned nonzero offsets deliberately.
    std::vector<unsigned char> storage[3];
    rknn_custom_op_tensor tensors[3]{};
    const int dims[3][4] = {{n,c,h,w},{n,2,oh,ow},{n,c,oh,ow}};
    for (int i = 0; i < 3; ++i) {
        auto& t = tensors[i];
        const bool nhwc = layout & (1 << i);
        t.attr.n_dims = 4; t.attr.type = RKNN_TENSOR_FLOAT32;
        t.attr.fmt = nhwc ? RKNN_TENSOR_NHWC : RKNN_TENSOR_NCHW;
        t.attr.qnt_type = RKNN_TENSOR_QNT_NONE;
        t.attr.dims[0] = dims[i][0];
        t.attr.dims[nhwc ? 3 : 1] = dims[i][1];
        t.attr.dims[nhwc ? 1 : 2] = dims[i][2];
        t.attr.dims[nhwc ? 2 : 3] = dims[i][3];
        t.attr.n_elems = dims[i][0]*dims[i][1]*dims[i][2]*dims[i][3];
        t.attr.size = t.attr.n_elems * sizeof(float);
        storage[i].resize(t.attr.size + 7, 0xA5);
        t.mem.virt_addr = storage[i].data(); t.mem.offset = 3; t.mem.size = storage[i].size();
    }
    std::memcpy(storage[0].data()+3, x, tensors[0].attr.size);
    std::memcpy(storage[1].data()+3, g, tensors[1].attr.size);
    if (fault == 3) tensors[0].attr.type = RKNN_TENSOR_FLOAT16;
    if (fault == 4) tensors[0].attr.w_stride = w + 1;
    if (fault == 5) tensors[1].mem.size = 1;
    if (fault == 6) tensors[0].attr.fmt = RKNN_TENSOR_NC1HWC2;
    if (fault == 7) tensors[0].attr.qnt_type = RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC;
    // 8: quoted attrs; 9: board's mislabeled channel-last grid; 10: packed
    // logical strides; 11: actual padding must still fail. Physical data stay NHWC.
    if (fault >= 9) tensors[1].attr.fmt = RKNN_TENSOR_NCHW;
    if (fault >= 10) {
        tensors[1].attr.w_stride = ow;
        tensors[1].attr.h_stride = oh;
        tensors[1].attr.size_with_stride = tensors[1].attr.size;
    }
    if (fault == 11) tensors[1].attr.w_stride = ow + 1;
    Attributes attrs{align, fault};
    rknn_custom_op_context ctx{}; ctx.priv_data = &attrs;
    const auto op = ec::make_grid_sample_cpu_op();
    const int ret = op.compute(&ctx, tensors, 2, tensors+2, 1);
    for (const auto& s : storage) {
        for (size_t j : {size_t(0),size_t(1),size_t(2),s.size()-1})
            if (s[j] != 0xA5) return -999;
    }
    if (ret == 0) std::memcpy(y, storage[2].data()+3, tensors[2].attr.size);
    return ret;
}
