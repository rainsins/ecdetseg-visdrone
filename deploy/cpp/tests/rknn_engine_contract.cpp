// Mock the C API to verify caller buffers independently of model dtype/layout.
// This does not emulate normalization, quantization, or NPU execution.
#include "rknn_engine.h"
#include "rknn_api.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {
rknn_tensor_type model_type = RKNN_TENSOR_FLOAT16;
rknn_tensor_format model_fmt = RKNN_TENSOR_NCHW;
bool bad_shape = false;
int inputs_seen = 0;
uint32_t init_flags = 0;
bool empty_perf = false;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
}
extern "C" {
int rknn_init(rknn_context* ctx, void*, uint32_t, uint32_t flags, rknn_init_extend*) {
    init_flags=flags; *ctx=1; return 0;
}
int rknn_destroy(rknn_context) { return 0; }
int rknn_set_core_mask(rknn_context, rknn_core_mask) { return 0; }
int rknn_query(rknn_context, rknn_query_cmd cmd, void* p, uint32_t) {
    if (cmd == RKNN_QUERY_SDK_VERSION) {
        auto* v=static_cast<rknn_sdk_version*>(p);
        std::strcpy(v->api_version, "mock"); std::strcpy(v->drv_version, "mock");
    } else if (cmd == RKNN_QUERY_IN_OUT_NUM) {
        auto* v=static_cast<rknn_input_output_num*>(p); v->n_input=1; v->n_output=2;
    } else if (cmd == RKNN_QUERY_INPUT_ATTR) {
        auto* a=static_cast<rknn_tensor_attr*>(p);
        a->n_dims=4; a->fmt=model_fmt; a->type=model_type;
        a->dims[0]=1; a->dims[1]=model_fmt==RKNN_TENSOR_NHWC?640:3;
        a->dims[2]=640; a->dims[3]=model_fmt==RKNN_TENSOR_NHWC?3:640;
        if(bad_shape) a->dims[0]=2;
        a->n_elems=3*640*640; a->size=a->n_elems*2;
        a->w_stride=672; a->size_with_stride=3*640*672*2;
    } else if (cmd == RKNN_QUERY_OUTPUT_ATTR) {
        auto* a=static_cast<rknn_tensor_attr*>(p);
        // Reverse the conventional order deliberately.
        a->n_dims=3; a->dims[0]=1; a->dims[1]=300; a->dims[2]=a->index?10:4;
        a->n_elems=300*a->dims[2]; a->type=RKNN_TENSOR_FLOAT32;
        a->size=a->n_elems*sizeof(float);
    } else if (cmd == RKNN_QUERY_PERF_DETAIL) {
        if (!(init_flags & RKNN_FLAG_COLLECT_PERF_MASK)) return RKNN_ERR_PARAM_INVALID;
        static char report[] = "mock per-op report";
        auto* perf=static_cast<rknn_perf_detail*>(p);
        perf->perf_data=empty_perf?nullptr:report;
        perf->data_len=empty_perf?0:sizeof(report)-1;
    } else return RKNN_ERR_PARAM_INVALID;
    return 0;
}
int rknn_inputs_set(rknn_context, uint32_t count, rknn_input* in) {
    if(count!=1 || in->type!=RKNN_TENSOR_UINT8 || in->fmt!=RKNN_TENSOR_NCHW ||
       in->pass_through!=0 || in->size!=3*640*640 ||
       static_cast<const uint8_t*>(in->buf)[0]!=255) return RKNN_ERR_INPUT_INVALID;
    ++inputs_seen; return 0;
}
int rknn_run(rknn_context, rknn_run_extend*) { return 0; }
int rknn_outputs_get(rknn_context, uint32_t count, rknn_output* out, rknn_output_extend*) {
    static float boxes[1200]={.5f}; static float logits[3000]={2.f};
    if(count!=2 || !out[0].want_float || !out[1].want_float) return RKNN_ERR_OUTPUT_INVALID;
    out[0].buf=boxes; out[0].size=sizeof(boxes);
    out[1].buf=logits; out[1].size=sizeof(logits); return 0;
}
int rknn_outputs_release(rknn_context, uint32_t, rknn_output*) { return 0; }
}
int main(int argc, char** argv) {
    if(argc!=2) return 2;
    std::ofstream(argv[1],std::ios::binary).put('x');
    try {
        for(auto type : {RKNN_TENSOR_FLOAT16,RKNN_TENSOR_FLOAT32,RKNN_TENSOR_INT8,RKNN_TENSOR_UINT8}) {
            for(auto fmt : {RKNN_TENSOR_NCHW,RKNN_TENSOR_NHWC}) {
                model_type=type; model_fmt=fmt;
                ec::RknnEngine engine; std::string err; ec::RawOutputs out;
                require(engine.init(argv[1],0,&err),err.c_str());
                require(engine.validate_contract(&err),err.c_str());
                std::vector<uint8_t> input(3*640*640,255);
                require(engine.infer(input.data(),input.size(),&out,&err),err.c_str());
                require(out.logits[0]==2.f && out.boxes[0]==.5f,"output identity");
                require(!engine.infer(input.data(),input.size()-1,&out,&err),"reject short input");
            }
        }
        require(inputs_seen==8,"all caller buffer cases exercised");
        for(int fault=0;fault<3;++fault) {
            model_type=fault==0?RKNN_TENSOR_INT64:RKNN_TENSOR_FLOAT16;
            model_fmt=fault==1?RKNN_TENSOR_NC1HWC2:RKNN_TENSOR_NCHW; bad_shape=fault==2;
            ec::RknnEngine engine; std::string err;
            require(engine.init(argv[1],0,&err),err.c_str());
            require(!engine.validate_contract(&err),"reject invalid model contract");
        }
        model_type=RKNN_TENSOR_FLOAT16; model_fmt=RKNN_TENSOR_NCHW; bad_shape=false;
        {
            ec::RknnEngine engine; std::string err;
            require(engine.init(argv[1],0,&err,false,true,true),err.c_str());
            require(init_flags==(RKNN_FLAG_COLLECT_PERF_MASK |
                RKNN_FLAG_EXECUTE_FALLBACK_PRIOR_DEVICE_GPU),"GPU/perf flags");
            const std::string path=std::string(argv[1])+".perf";
            require(engine.write_perf_detail(path,&err),err.c_str());
            std::ifstream report(path); std::string text; std::getline(report,text);
            require(text=="mock per-op report","performance report bytes");
            empty_perf=true;
            require(!engine.write_perf_detail(path,&err),"reject empty report");
            require(!engine.init(argv[1],0,&err,true,true,false),"reject conflicting fallback flags");
        }
        std::puts("PASS: input contracts, output order, GPU/perf flags and report handling");
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
    return 0;
}
