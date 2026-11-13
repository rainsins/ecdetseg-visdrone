// RKNPU2 engine wrapper for ECDet-S INT8 on RK3588.
//
// Only the symbols declared in rknn_api.h are used; nothing is invented.  Every
// call is checked and every failure returns a message instead of a silent zero,
// because a half-initialised NPU context produces plausible-looking garbage.
//
// The contract this class implements (see deploy/MODEL_CONTRACT.md):
//   * one uint8 RGB NCHW (1,3,640,640) input, no mean/std/255 in user code --
//     normalisation lives inside the .rknn (mean_values/std_values at convert
//     time). The compiled tensor may be FP16/FP32/INT8/UINT8;
//     rknn_inputs_set(pass_through=0) converts the caller's UINT8 buffer.
//   * two float32 outputs, [1,300,10] logits and [1,300,4] boxes.
//
// Input goes through rknn_inputs_set rather than the zero-copy
// rknn_create_mem / rknn_set_io_mem pair.  The zero-copy path requires filling
// whatever *native* layout the driver reports (potentially NC1HWC2), which
// cannot be validated off-board; a plain NCHW buffer handed to
// rknn_inputs_set costs one 1.2 MB copy per frame and is verifiable by
// inspection.  If profiling ever shows that copy matters, revisit with
// RKNN_QUERY_NATIVE_INPUT_ATTR on real hardware.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ec {

// One inferred frame, dequantised to float32 host order.
struct RawOutputs {
    std::vector<float> logits;  // 300*10, row-major [query][class]
    std::vector<float> boxes;   // 300*4,  row-major [query][cx,cy,w,h]
    int num_queries = 0;
    int num_classes = 0;
};

// Shape/dtype description of one model tensor, captured at init for logging and
// for the contract assertions in main.cpp.
struct TensorInfo {
    std::string name;
    std::vector<uint32_t> dims;  // for NCHW input / NHWC output as reported
    uint32_t n_elems = 0;
    uint32_t size = 0;          // bytes, packed
    uint32_t size_with_stride = 0;
    uint32_t w_stride = 0;
    int dtype = 0;              // rknn_tensor_type
    int fmt = 0;                // rknn_tensor_format (queried model layout)
    int qnt_type = 0;           // rknn_tensor_qnt_type
    int32_t zp = 0;
    float scale = 0.0f;
};

// Owns one loaded .rknn context.  Construction does not touch the NPU; call
// init() and check the result.
class RknnEngine {
public:
    RknnEngine() = default;
    ~RknnEngine();

    RknnEngine(const RknnEngine&) = delete;
    RknnEngine& operator=(const RknnEngine&) = delete;

    // Load `model_path`, query tensor attributes and optionally pin the context
    // to `core_mask` (pass 0 to keep RKNN_NPU_CORE_AUTO).
    // Returns false and fills `error` on any failure.
    bool init(const std::string& model_path, int core_mask, std::string* error,
              bool grid_sample_cpu = false, bool fallback_gpu = false,
              bool collect_perf = false);
    bool write_perf_detail(const std::string& path, std::string* error) const;

    // Feed a uint8 (3,640,640) NCHW tensor into the NPU and dequantise the two
    // outputs into `out`.  `data` must hold exactly input_n_elems() bytes.
    bool infer(const uint8_t* data, size_t bytes, RawOutputs* out, std::string* error);

    const TensorInfo& input_info() const { return input_info_; }
    const TensorInfo& logits_info() const { return logits_info_; }
    const TensorInfo& boxes_info() const { return boxes_info_; }
    const std::string& sdk_version() const { return sdk_version_; }
    const std::string& driver_version() const { return driver_version_; }

    size_t input_n_elems() const { return static_cast<size_t>(input_info_.n_elems); }

    // Convenience contract check: the loaded model must match the frozen
    // ECDet-S deployment shape (uint8 RGB NCHW 1x3x640x640 -> 300x10, 300x4).
    bool validate_contract(std::string* error) const;

private:
    void release();

    // rknn_context is uint32_t on 32-bit ARM and uint64_t elsewhere; the header
    // typedefs it, so keep it opaque here and only touch it from the .cpp.
    unsigned long long ctx_ = 0;
    bool ctx_valid_ = false;

    uint32_t n_input_ = 0;
    uint32_t n_output_ = 0;

    TensorInfo input_info_;
    TensorInfo logits_info_;
    TensorInfo boxes_info_;
    // Which rknn_output index holds each head; resolved at init from n_elems.
    int logits_index_ = -1;
    int boxes_index_ = -1;

    std::string sdk_version_;
    std::string driver_version_;
};

}  // namespace ec
