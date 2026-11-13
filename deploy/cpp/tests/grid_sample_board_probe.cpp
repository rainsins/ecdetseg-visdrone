// Isolated compiler/runtime probe; this is NOT the UINT8 full-model caller.
#include "rknn_api.h"
#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

namespace {
struct Context {
    rknn_context value = 0;
    bool valid = false;
    ~Context() { if (valid) rknn_destroy(value); }
};
std::vector<char> read(const char* path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
bool ok(int result, const char* call) {
    std::fprintf(stderr, "%s=%d\n", call, result);
    return result == RKNN_SUCC;
}
}
int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: probe MODEL FEATURE.f32 GRID.f32 OUTPUT.f32\n");
        return 2;
    }
    auto model = read(argv[1]);
    if (model.empty()) return 2;
    Context ctx;
    if (!ok(rknn_init(&ctx.value,model.data(),static_cast<uint32_t>(model.size()),
                     RKNN_FLAG_EXECUTE_FALLBACK_PRIOR_DEVICE_GPU,nullptr),
            "rknn_init GPU-priority")) return 1;
    ctx.valid = true;
    rknn_input_output_num io{};
    if (!ok(rknn_query(ctx.value,RKNN_QUERY_IN_OUT_NUM,&io,sizeof(io)),
            "query io") || io.n_input != 2 || io.n_output != 1) return 1;
    std::vector<char> data[2] = {read(argv[2]),read(argv[3])};
    rknn_input inputs[2]{};
    const uint32_t expected[2][4] = {{8,24,80,80},{8,300,3,2}};
    for (uint32_t i = 0; i < 2; ++i) {
        rknn_tensor_attr a{}; a.index = i;
        if (!ok(rknn_query(ctx.value,RKNN_QUERY_INPUT_ATTR,&a,sizeof(a)),
                "query input")) return 1;
        std::fprintf(stderr, "input %u %s dtype=%d fmt=%d dims=%u,%u,%u,%u\n",
            i,a.name,a.type,a.fmt,a.dims[0],a.dims[1],a.dims[2],a.dims[3]);
        if (a.n_dims != 4 || (a.fmt != RKNN_TENSOR_NCHW && a.fmt != RKNN_TENSOR_NHWC) ||
            data[i].size() != static_cast<size_t>(a.n_elems)*sizeof(float)) return 2;
        const int nhwc_order[4] = {0,2,3,1};
        for (int j=0;j<4;++j) {
            const int k = a.fmt == RKNN_TENSOR_NHWC ? nhwc_order[j] : j;
            if (a.dims[j] != expected[i][k]) return 2;
        }
        inputs[i].index = i; inputs[i].buf = data[i].data();
        inputs[i].size = static_cast<uint32_t>(data[i].size());
        inputs[i].type = RKNN_TENSOR_FLOAT32;
        inputs[i].fmt = RKNN_TENSOR_NCHW; inputs[i].pass_through = 0;
    }
    if (!ok(rknn_inputs_set(ctx.value,2,inputs),"inputs_set")) return 1;
    if (!ok(rknn_run(ctx.value,nullptr),"run")) return 1;
    rknn_tensor_attr attr{};
    if (!ok(rknn_query(ctx.value,RKNN_QUERY_OUTPUT_ATTR,&attr,sizeof(attr)),
            "query output")) return 1;
    std::fprintf(stderr,"output dtype=%d fmt=%d dims=%u,%u,%u,%u\n",attr.type,attr.fmt,
                 attr.dims[0],attr.dims[1],attr.dims[2],attr.dims[3]);
    rknn_output output{}; output.want_float = 1;
    if (!ok(rknn_outputs_get(ctx.value,1,&output,nullptr),"outputs_get")) return 1;
    bool success = output.buf && output.size == 8*24*300*3*sizeof(float) &&
        attr.fmt == RKNN_TENSOR_NCHW && attr.n_dims == 4 &&
        attr.dims[0] == 8 && attr.dims[1] == 24 && attr.dims[2] == 300 && attr.dims[3] == 3;
    if (success) {
        std::ofstream file(argv[4],std::ios::binary);
        file.write(static_cast<const char*>(output.buf),output.size);
        file.close(); success = static_cast<bool>(file);
    }
    const bool released = ok(rknn_outputs_release(ctx.value,1,&output),"outputs_release");
    return success && released ? 0 : 1;
}
