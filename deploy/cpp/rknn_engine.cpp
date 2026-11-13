#include "rknn_engine.h"

#include <cstdio>
#include <cstring>
#include <fstream>

#include "rknn_api.h"
#ifdef EC_ENABLE_GRID_SAMPLE_CPU
#include "grid_sample_cpu.h"
#endif

namespace ec {
namespace {

std::string rknn_error_string(int ret) {
    switch (ret) {
        case RKNN_SUCC: return "RKNN_SUCC";
        case RKNN_ERR_FAIL: return "RKNN_ERR_FAIL";
        case RKNN_ERR_TIMEOUT: return "RKNN_ERR_TIMEOUT";
        case RKNN_ERR_DEVICE_UNAVAILABLE: return "RKNN_ERR_DEVICE_UNAVAILABLE";
        case RKNN_ERR_MALLOC_FAIL: return "RKNN_ERR_MALLOC_FAIL";
        case RKNN_ERR_PARAM_INVALID: return "RKNN_ERR_PARAM_INVALID";
        case RKNN_ERR_MODEL_INVALID: return "RKNN_ERR_MODEL_INVALID";
        case RKNN_ERR_CTX_INVALID: return "RKNN_ERR_CTX_INVALID";
        case RKNN_ERR_INPUT_INVALID: return "RKNN_ERR_INPUT_INVALID";
        case RKNN_ERR_OUTPUT_INVALID: return "RKNN_ERR_OUTPUT_INVALID";
        case RKNN_ERR_DEVICE_UNMATCH: return "RKNN_ERR_DEVICE_UNMATCH";
        case RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL: return "RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL";
        case RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION:
            return "RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION";
        case RKNN_ERR_TARGET_PLATFORM_UNMATCH: return "RKNN_ERR_TARGET_PLATFORM_UNMATCH";
        case RKNN_ERR_KEY_INVALID: return "RKNN_ERR_KEY_INVALID";
        case RKNN_ERR_DECRYPT_FAILED: return "RKNN_ERR_DECRYPT_FAILED";
        case RKNN_ERR_SOC_BIND_FAILED: return "RKNN_ERR_SOC_BIND_FAILED";
        default: return "unknown";
    }
}

bool check(int ret, const char* what, std::string* error) {
    if (ret == RKNN_SUCC) return true;
    if (error != nullptr) {
        *error = std::string(what) + " failed: " + rknn_error_string(ret) + " (" +
                 std::to_string(ret) + ")";
    }
    return false;
}

TensorInfo to_info(const rknn_tensor_attr& attr) {
    TensorInfo info;
    info.name = attr.name;
    info.dims.assign(attr.dims, attr.dims + attr.n_dims);
    info.n_elems = attr.n_elems;
    info.size = attr.size;
    info.size_with_stride = attr.size_with_stride;
    info.w_stride = attr.w_stride;
    info.dtype = static_cast<int>(attr.type);
    info.fmt = static_cast<int>(attr.fmt);
    info.qnt_type = static_cast<int>(attr.qnt_type);
    info.zp = attr.zp;
    info.scale = attr.scale;
    return info;
}

bool read_file(const std::string& path, std::vector<uint8_t>* out, std::string* error) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        *error = "cannot open model: " + path;
        return false;
    }
    const std::streamsize size = f.tellg();
    if (size <= 0) {
        *error = "model is empty: " + path;
        return false;
    }
    f.seekg(0, std::ios::beg);
    out->resize(static_cast<size_t>(size));
    if (!f.read(reinterpret_cast<char*>(out->data()), size)) {
        *error = "short read on model: " + path;
        return false;
    }
    return true;
}

}  // namespace

RknnEngine::~RknnEngine() { release(); }

void RknnEngine::release() {
    if (ctx_valid_) {
        rknn_destroy(static_cast<rknn_context>(ctx_));
        ctx_valid_ = false;
    }
    ctx_ = 0;
    n_input_ = 0;
    n_output_ = 0;
    logits_index_ = -1;
    boxes_index_ = -1;
}

bool RknnEngine::init(const std::string& model_path, int core_mask, std::string* error,
                      bool grid_sample_cpu, bool fallback_gpu, bool collect_perf) {
    release();
    if (grid_sample_cpu && fallback_gpu) {
        *error = "choose either custom CPU or GPU-priority fallback for isolated validation";
        return false;
    }
#ifndef EC_ENABLE_GRID_SAMPLE_CPU
    if (grid_sample_cpu) {
        *error = "rebuild with -DEC_ENABLE_GRID_SAMPLE_CPU=ON and rknn_custom_op.h";
        return false;
    }
#endif

    std::vector<uint8_t> model;
    if (!read_file(model_path, &model, error)) return false;

    rknn_context ctx = 0;
    uint32_t flags = collect_perf ? RKNN_FLAG_COLLECT_PERF_MASK : 0;
    if (fallback_gpu) flags |= RKNN_FLAG_EXECUTE_FALLBACK_PRIOR_DEVICE_GPU;
    if (!check(rknn_init(&ctx, model.data(), static_cast<uint32_t>(model.size()), flags, nullptr),
               "rknn_init", error)) {
        return false;
    }
    ctx_ = static_cast<unsigned long long>(ctx);
    ctx_valid_ = true;

#ifdef EC_ENABLE_GRID_SAMPLE_CPU
    if (grid_sample_cpu) {
        static rknn_custom_op op = make_grid_sample_cpu_op();
        if (!check(rknn_register_custom_ops(ctx, &op, 1), "rknn_register_custom_ops(GridSample)", error)) {
            release();
            return false;
        }
        std::fprintf(stderr, "GridSample : registered strict FLOAT32 CPU fallback (board validation required)\n");
    }
#endif

    if (core_mask != 0 &&
        !check(rknn_set_core_mask(ctx, static_cast<rknn_core_mask>(core_mask)),
               "rknn_set_core_mask", error)) {
        release();
        return false;
    }

    rknn_sdk_version version{};
    if (check(rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(version)),
              "rknn_query(SDK_VERSION)", error)) {
        sdk_version_ = version.api_version;
        driver_version_ = version.drv_version;
    }

    rknn_input_output_num io_num{};
    if (!check(rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)),
               "rknn_query(IN_OUT_NUM)", error)) {
        release();
        return false;
    }
    n_input_ = io_num.n_input;
    n_output_ = io_num.n_output;

    if (n_input_ != 1) {
        *error = "model has " + std::to_string(n_input_) + " inputs, expected 1";
        release();
        return false;
    }
    if (n_output_ != 2) {
        *error = "model has " + std::to_string(n_output_) + " outputs, expected 2 (logits, boxes)";
        release();
        return false;
    }

    rknn_tensor_attr attr{};
    attr.index = 0;
    if (!check(rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr)),
               "rknn_query(INPUT_ATTR)", error)) {
        release();
        return false;
    }
    input_info_ = to_info(attr);

    // Output order is not guaranteed to be (logits, boxes), so identify by the
    // last dimension rather than by index -- the same rule postprocess.py uses.
    // Guessing here would silently swap scores and geometry.
    rknn_tensor_attr attr_l{};
    attr_l.index = 0;
    if (!check(rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &attr_l, sizeof(attr_l)),
               "rknn_query(OUTPUT_ATTR 0)", error)) {
        release();
        return false;
    }
    rknn_tensor_attr attr_b{};
    attr_b.index = 1;
    if (!check(rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &attr_b, sizeof(attr_b)),
               "rknn_query(OUTPUT_ATTR 1)", error)) {
        release();
        return false;
    }

    // Identify by element count (300x10 vs 300x4).  RKNN output *ordering* is
    // not part of the contract, and swapping scores with geometry still yields
    // syntactically valid detections -- the worst kind of silent bug.
    const TensorInfo info0 = to_info(attr_l);
    const TensorInfo info1 = to_info(attr_b);
    const uint32_t kLogitsElems = 300u * 10u;
    const uint32_t kBoxesElems = 300u * 4u;

    if (info0.n_elems == kLogitsElems && info1.n_elems == kBoxesElems) {
        logits_info_ = info0;
        boxes_info_ = info1;
        logits_index_ = 0;
        boxes_index_ = 1;
    } else if (info0.n_elems == kBoxesElems && info1.n_elems == kLogitsElems) {
        logits_info_ = info1;
        boxes_info_ = info0;
        logits_index_ = 1;
        boxes_index_ = 0;
    } else {
        *error = "output tensors are not (300x10 logits, 300x4 boxes): got [" +
                 std::to_string(info0.n_elems) + "] '" + info0.name + "' and [" +
                 std::to_string(info1.n_elems) + "] '" + info1.name + "'";
        release();
        return false;
    }

    return true;
}

bool RknnEngine::write_perf_detail(const std::string& path, std::string* error) const {
    if (!ctx_valid_) { *error = "engine not initialised"; return false; }
    rknn_perf_detail perf{};
    if (!check(rknn_query(static_cast<rknn_context>(ctx_), RKNN_QUERY_PERF_DETAIL,
                          &perf, sizeof(perf)), "rknn_query(PERF_DETAIL)", error)) return false;
    if (!perf.perf_data || !perf.data_len) {
        *error = "empty performance report; runtime/model may not expose per-op detail";
        return false;
    }
    std::ofstream file(path, std::ios::binary);
    file.write(perf.perf_data, static_cast<std::streamsize>(perf.data_len));
    file.close();
    if (!file) { *error = "cannot write performance report: " + path; return false; }
    return true;
}

bool RknnEngine::validate_contract(std::string* error) const {
    if (!ctx_valid_) {
        *error = "engine not initialised";
        return false;
    }

    const TensorInfo& in = input_info_;
    // INPUT_ATTR describes the model, not the packed caller buffer.
    // infer() requests UINT8 NCHW conversion with pass_through=0.
    if (in.dtype != RKNN_TENSOR_UINT8 && in.dtype != RKNN_TENSOR_INT8 &&
        in.dtype != RKNN_TENSOR_FLOAT16 && in.dtype != RKNN_TENSOR_FLOAT32) {
        *error = "input dtype is " + std::to_string(in.dtype) +
                 ", expected a FLOAT32/FLOAT16/INT8/UINT8 image model";
        return false;
    }
    const bool nchw = in.fmt == RKNN_TENSOR_NCHW &&
                      in.dims == std::vector<uint32_t>{1, 3, 640, 640};
    const bool nhwc = in.fmt == RKNN_TENSOR_NHWC &&
                      in.dims == std::vector<uint32_t>{1, 640, 640, 3};
    if ((!nchw && !nhwc) || in.n_elems != 3u * 640u * 640u) {
        std::string dims;
        for (uint32_t d : in.dims) dims += " " + std::to_string(d);
        *error = "input dims/layout are [" + dims + "], fmt=" +
                 std::to_string(in.fmt) +
                 ", expected NCHW [1 3 640 640] or NHWC [1 640 640 3]";
        return false;
    }
    // Native strides matter for set_io_mem/pass-through, not inputs_set's
    // packed input. The runtime handles packing for this conversion path.
    if (logits_info_.n_elems != 300u * 10u) {
        *error = "pred_logits has " + std::to_string(logits_info_.n_elems) +
                 " elements, expected 3000";
        return false;
    }
    if (boxes_info_.n_elems != 300u * 4u) {
        *error = "pred_boxes has " + std::to_string(boxes_info_.n_elems) +
                 " elements, expected 1200";
        return false;
    }
    return true;
}

bool RknnEngine::infer(const uint8_t* data, size_t bytes, RawOutputs* out, std::string* error) {
    if (!ctx_valid_) {
        *error = "engine not initialised";
        return false;
    }
    if (data == nullptr || out == nullptr) {
        *error = "infer: null argument";
        return false;
    }
    const size_t expected = input_n_elems();
    if (bytes != expected) {
        *error = "infer: input is " + std::to_string(bytes) + " bytes, expected " +
                 std::to_string(expected);
        return false;
    }

    rknn_context ctx = static_cast<rknn_context>(ctx_);

    // rknn_inputs_set applies the driver's layout conversion for us, so the
    // caller can hand over a plain NCHW uint8 tensor.  The zero-copy path
    // (rknn_create_mem + rknn_set_io_mem) avoids one 1.2 MB copy per frame but
    // requires matching the *native* tensor layout reported by
    // RKNN_QUERY_NATIVE_INPUT_ATTR, which may be NC1HWC2; that layout cannot be
    // validated off-board, so it is deliberately not used here.
    rknn_input input{};
    input.index = 0;
    input.buf = const_cast<uint8_t*>(data);
    input.size = static_cast<uint32_t>(bytes);
    input.type = RKNN_TENSOR_UINT8;
    input.fmt = RKNN_TENSOR_NCHW;
    input.pass_through = 0;
    if (!check(rknn_inputs_set(ctx, 1, &input), "rknn_inputs_set", error)) return false;

    if (!check(rknn_run(ctx, nullptr), "rknn_run", error)) return false;

    rknn_output outputs[2]{};
    outputs[0].index = 0;
    outputs[0].want_float = 1;  // let the runtime apply zp/scale, if any
    outputs[1].index = 1;
    outputs[1].want_float = 1;
    if (!check(rknn_outputs_get(ctx, 2, outputs, nullptr), "rknn_outputs_get", error)) {
        return false;
    }

    struct Guard {
        rknn_context ctx;
        rknn_output* outputs;
        ~Guard() { rknn_outputs_release(ctx, 2, outputs); }
    } guard{ctx, outputs};

    // want_float=1 means the runtime has already applied scale/zero_point and
    // both buffers are float32 of n_elems each.  A different size means a
    // quantized-output build, which this decoder does not handle -- say so
    // rather than reinterpreting bytes.
    const uint32_t want_l = static_cast<uint32_t>(logits_info_.n_elems * sizeof(float));
    const uint32_t want_b = static_cast<uint32_t>(boxes_info_.n_elems * sizeof(float));
    if (outputs[logits_index_].size != want_l) {
        *error = "pred_logits buffer is " + std::to_string(outputs[logits_index_].size) +
                 " bytes, expected " + std::to_string(want_l) +
                 " (float32); if the model was built with quantized outputs, dequantize with "
                 "zp/scale before postprocessing";
        return false;
    }
    if (outputs[boxes_index_].size != want_b) {
        *error = "pred_boxes buffer is " + std::to_string(outputs[boxes_index_].size) +
                 " bytes, expected " + std::to_string(want_b) + " (float32)";
        return false;
    }

    out->num_queries = 300;
    out->num_classes = 10;
    out->logits.resize(logits_info_.n_elems);
    out->boxes.resize(boxes_info_.n_elems);
    std::memcpy(out->logits.data(), outputs[logits_index_].buf,
                out->logits.size() * sizeof(float));
    std::memcpy(out->boxes.data(), outputs[boxes_index_].buf, out->boxes.size() * sizeof(float));

    for (float v : out->logits) {
        if (!(v == v) || v > 3.4e38f || v < -3.4e38f) {
            *error = "pred_logits contains a non-finite value; check the .rknn and the input tensor";
            return false;
        }
    }
    for (float v : out->boxes) {
        if (!(v == v) || v > 3.4e38f || v < -3.4e38f) {
            *error = "pred_boxes contains a non-finite value; check the .rknn and the input tensor";
            return false;
        }
    }
    return true;
}

}  // namespace ec
