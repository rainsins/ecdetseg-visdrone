// Board-side entry point for ECDet-S INT8 on RK3588.
//
// One frame goes through exactly three stages, mirroring the Python pipeline:
//
//   decode image (I/O only)
//     -> PilResampler::resize to 640x640, RGB        (byte-identical to PIL)
//     -> RknnEngine::infer, uint8 NCHW, no normalise (mean/std live in .rknn)
//     -> postprocess(), global top-k over [query][class]
//
// Usage:
//   ec_infer --model m.rknn --image frame.jpg [options]
//
// Modes:
//   (default)      run warmup+measure loop, print latency percentiles
//   --dump-npy F   write the uint8 CHW input tensor to F for
//                  `deploy/preprocess.py --compare-candidate F`
//   --dump-dets F  write detections as text, one per line
//
// Exit codes: 0 ok, 1 runtime failure, 2 bad usage.  Any failure is printed to
// stderr with the originating call name; a bare non-zero exit would make a
// board deployment impossible to debug.

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>  // cvtColor: imgcodecs links imgproc anyway

#include "pil_resize.h"
#include "postprocess.h"
#include "rknn_engine.h"
#ifdef EC_ENABLE_GRID_SAMPLE_CPU
#include "grid_sample_cpu.h"
#endif

namespace {

constexpr int kInputSize = 640;

struct Options {
    std::string model;
    std::string image;
    std::string image_list;  // optional: one path per line, sizes may vary
    std::string dump_npy;
    std::string dump_dets;
    std::string dump_raw;
    float score_thresh = 0.0f;
    int max_det = 300;
    int core_mask = 0;  // 0 -> RKNN_NPU_CORE_AUTO
    int warmup = 20;
    int iters = 500;
    bool timing = false;
    bool grid_sample_cpu = false;
    bool grid_sample_profile = false;
    bool fallback_gpu = false;
    std::string perf_detail;
};

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s --model MODEL.rknn --image IMAGE [options]\n"
                 "\n"
                 "  --model PATH        .rknn file (required)\n"
                 "  --image PATH        one image (required unless --image-list)\n"
                 "  --image-list PATH   text file, one image path per line\n"
                 "  --timing            warmup/measure loop with percentiles\n"
                 "  --grid-sample-cpu   register strict FLOAT32 CPU GridSample fallback\n"
                 "  --grid-sample-profile  report callback/sampling timing (requires CPU fallback)\n"
                 "  --fallback-gpu      prefer runtime GPU fallback (board support required)\n"
                 "  --perf-detail PATH  save first-frame runtime per-op report (adds overhead)\n"
                 "  --warmup N          warmup iterations (default 20)\n"
                 "  --iters N           measured iterations (default 500)\n"
                 "  --score-thresh F    drop detections below F (default 0.0)\n"
                 "  --max-det N         keep at most N detections (default 300)\n"
                 "  --core-mask N       RKNN core mask (default 0 = AUTO)\n"
                 "  --dump-npy PATH     write the uint8 CHW input for preprocess.py\n"
                 "  --dump-dets PATH    write detections as text\n"
                 "  --dump-raw PREFIX   first-frame logits/boxes as native float32 .f32 files\n"
                 "\n"
                 "Per-frame output goes to stdout; diagnostics go to stderr.\n",
                 argv0);
}

bool parse_int(const char* s, int* out) {
    char* end = nullptr;
    errno = 0;
    const long v = std::strtol(s, &end, 10);
    // Range-check before narrowing: strtol saturates at LONG_MAX/LONG_MIN and a
    // silent (long)->(int) truncation would turn --iters 4294967296 into 0.
    if (end == s || *end != '\0' || errno == ERANGE) return false;
    if (v < INT_MIN || v > INT_MAX) return false;
    *out = static_cast<int>(v);
    return true;
}

bool parse_float(const char* s, float* out) {
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(s, &end);
    // atof() silently yields 0.0 for "--score-thresh abc", which reads as a
    // valid "no threshold" run and hides the typo.
    if (end == s || *end != '\0' || errno == ERANGE) return false;
    if (!std::isfinite(v)) return false;
    *out = static_cast<float>(v);
    return true;
}

bool parse_options(int argc, char** argv, Options* opt, std::string* error) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                *error = std::string("missing value for ") + name;
                return nullptr;
            }
            return argv[++i];
        };

        if (a == "--model") {
            const char* v = need("--model");
            if (v == nullptr) return false;
            opt->model = v;
        } else if (a == "--image") {
            const char* v = need("--image");
            if (v == nullptr) return false;
            opt->image = v;
        } else if (a == "--image-list") {
            const char* v = need("--image-list");
            if (v == nullptr) return false;
            opt->image_list = v;
        } else if (a == "--dump-npy") {
            const char* v = need("--dump-npy");
            if (v == nullptr) return false;
            opt->dump_npy = v;
        } else if (a == "--dump-dets") {
            const char* v = need("--dump-dets");
            if (v == nullptr) return false;
            opt->dump_dets = v;
        } else if (a == "--dump-raw") {
            const char* v = need("--dump-raw");
            if (v == nullptr) return false;
            opt->dump_raw = v;
        } else if (a == "--timing") {
            opt->timing = true;
        } else if (a == "--grid-sample-cpu") {
            opt->grid_sample_cpu = true;
        } else if (a == "--grid-sample-profile") {
            opt->grid_sample_profile = true;
        } else if (a == "--fallback-gpu") {
            opt->fallback_gpu = true;
        } else if (a == "--perf-detail") {
            const char* v = need("--perf-detail");
            if (!v) return false;
            opt->perf_detail = v;
        } else if (a == "--score-thresh") {
            const char* v = need("--score-thresh");
            if (v == nullptr || !parse_float(v, &opt->score_thresh)) {
                *error = "invalid --score-thresh";
                return false;
            }
        } else if (a == "--max-det") {
            const char* v = need("--max-det");
            if (v == nullptr || !parse_int(v, &opt->max_det)) {
                *error = "invalid --max-det";
                return false;
            }
        } else if (a == "--core-mask") {
            const char* v = need("--core-mask");
            if (v == nullptr || !parse_int(v, &opt->core_mask)) {
                *error = "invalid --core-mask";
                return false;
            }
        } else if (a == "--warmup") {
            const char* v = need("--warmup");
            if (v == nullptr || !parse_int(v, &opt->warmup)) {
                *error = "invalid --warmup";
                return false;
            }
        } else if (a == "--iters") {
            const char* v = need("--iters");
            if (v == nullptr || !parse_int(v, &opt->iters)) {
                *error = "invalid --iters";
                return false;
            }
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else {
            *error = "unknown argument: " + a;
            return false;
        }
    }
    if (opt->grid_sample_profile && !opt->grid_sample_cpu) {
        *error = "--grid-sample-profile requires --grid-sample-cpu";
        return false;
    }
    if (opt->grid_sample_cpu && opt->fallback_gpu) {
        *error = "--grid-sample-cpu and --fallback-gpu are mutually exclusive";
        return false;
    }
    if (opt->model.empty()) {
        *error = "--model is required";
        return false;
    }
    if (opt->image.empty() && opt->image_list.empty()) {
        *error = "--image or --image-list is required";
        return false;
    }
    // Non-positive values would make the timing loop a silent no-op that still
    // reports a table of zeros, which reads like a fast model.
    if (opt->warmup < 0) {
        *error = "--warmup must be >= 0";
        return false;
    }
    if (opt->iters <= 0) {
        *error = "--iters must be >= 1";
        return false;
    }
    if (opt->max_det <= 0) {
        *error = "--max-det must be >= 1";
        return false;
    }
    return true;
}

// Load an image as interleaved RGB, matching cv2.imread + cvtColor(BGR2RGB).
bool load_rgb(const std::string& path, ec::RgbImage* out, std::string* error) {
    cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        *error = "cannot decode image: " + path;
        return false;
    }
    if (bgr.channels() != 3) {
        *error = "expected a 3-channel image: " + path;
        return false;
    }
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    out->width = rgb.cols;
    out->height = rgb.rows;
    out->data.resize(static_cast<size_t>(rgb.cols) * rgb.rows * 3);
    // cv::Mat rows may be padded, so copy row by row rather than the whole block.
    for (int y = 0; y < rgb.rows; ++y) {
        std::memcpy(out->row(y), rgb.ptr<uint8_t>(y),
                    static_cast<size_t>(rgb.cols) * 3);
    }
    return true;
}

// Planar CHW uint8, the layout RKNN_TENSOR_NCHW expects.
void rgb_to_chw(const ec::RgbImage& img, std::vector<uint8_t>* chw) {
    const size_t plane = static_cast<size_t>(img.width) * img.height;
    chw->resize(plane * 3);
    for (int y = 0; y < img.height; ++y) {
        const uint8_t* row = img.row(y);
        const size_t off = static_cast<size_t>(y) * img.width;
        for (int x = 0; x < img.width; ++x) {
            (*chw)[off + x] = row[static_cast<size_t>(x) * 3 + 0];
            (*chw)[plane + off + x] = row[static_cast<size_t>(x) * 3 + 1];
            (*chw)[plane * 2 + off + x] = row[static_cast<size_t>(x) * 3 + 2];
        }
    }
}

// Minimal .npy writer for a uint8 array, so `np.load` can read it without a
// numpy/C++ dependency.  Header format per numpy's format v1.0 spec.
bool write_npy_u8(const std::string& path, const uint8_t* data, size_t count,
                  const std::vector<int>& shape, std::string* error) {
    std::string shape_str = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) shape_str += ", ";
        shape_str += std::to_string(shape[i]);
    }
    // A 1-D array needs the trailing comma to stay a tuple; 0-D would need
    // "()", but this writer is only used for the CHW input tensor.
    if (shape.size() == 1) shape_str += ",";
    shape_str += ")";
    std::string dict = "{'descr': '|u1', 'fortran_order': False, 'shape': " + shape_str + ", }";

    // Total header (magic + version + len + dict + padding + '\n') must be a
    // multiple of 64 bytes.
    const size_t preamble = 10;
    size_t pad = 64 - ((preamble + dict.size() + 1) % 64);
    if (pad == 64) pad = 0;
    dict.append(pad, ' ');
    dict.push_back('\n');

    FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        *error = "cannot open for writing: " + path;
        return false;
    }
    const char magic[] = "\x93NUMPY\x01\x00";
    const uint16_t hlen = static_cast<uint16_t>(dict.size());
    bool ok = std::fwrite(magic, 1, 8, f) == 8;
    ok = ok && std::fwrite(&hlen, 2, 1, f) == 1;
    ok = ok && std::fwrite(dict.data(), 1, dict.size(), f) == dict.size();
    ok = ok && std::fwrite(data, 1, count, f) == count;
    if (std::fclose(f) != 0) ok = false;
    if (!ok) *error = "short write: " + path;
    return ok;
}

bool write_f32(const std::string& path, const std::vector<float>& values) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(values.data(), sizeof(float), values.size(), f) == values.size();
    const int closed = std::fclose(f);
    return ok && closed == 0;
}

bool write_dets(const std::string& path, const std::vector<ec::Detection>& dets,
                int orig_w, int orig_h, std::string* error) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (f == nullptr) {
        *error = "cannot open for writing: " + path;
        return false;
    }
    std::fprintf(f, "# %dx%d %zu detections\n", orig_w, orig_h, dets.size());
    std::fprintf(f, "# label x1 y1 x2 y2 score\n");
    for (const auto& d : dets) {
        std::fprintf(f, "%d %.6f %.6f %.6f %.6f %.8f\n", d.label, d.x1, d.y1, d.x2, d.y2,
                     d.score);
    }
    const bool ok = std::fclose(f) == 0;
    if (!ok) *error = "short write: " + path;
    return ok;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    const auto d = std::chrono::steady_clock::now() - t0;
    return std::chrono::duration<double, std::milli>(d).count();
}

struct Stats {
    double mean = 0.0;
    double median = 0.0;
    double p95 = 0.0;
    double min = 0.0;
    double max = 0.0;
};

Stats summarise(std::vector<double> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();

    double sum = 0.0;
    for (double x : v) sum += x;
    s.mean = sum / static_cast<double>(n);
    s.min = v.front();
    s.max = v.back();

    // Linear interpolation between order statistics, matching numpy.median and
    // numpy.percentile's default method so the numbers can be compared with a
    // Python-side benchmark without converting conventions first.
    auto linear = [&v, n](double q) -> double {
        if (n == 1) return v[0];
        const double pos = q * static_cast<double>(n - 1);
        const size_t lo = static_cast<size_t>(std::floor(pos));
        const size_t hi = std::min(lo + 1, n - 1);
        const double frac = pos - static_cast<double>(lo);
        return v[lo] * (1.0 - frac) + v[hi] * frac;
    };
    s.median = linear(0.50);
    s.p95 = linear(0.95);
    return s;
}

// One image end to end, with optional timing.  Returns false on any failure.
struct FrameResult {
    std::vector<ec::Detection> dets;
    double pre_ms = 0.0;
    double npu_ms = 0.0;
    double post_ms = 0.0;
    double total_ms = 0.0;
    int orig_w = 0;
    int orig_h = 0;
};

bool run_frame(ec::RknnEngine* engine, const ec::PilResampler& resampler,
               const ec::RgbImage& src, const ec::PostConfig& pcfg, FrameResult* out,
               std::vector<uint8_t>* chw_scratch, ec::RawOutputs* raw_scratch,
               std::string* error) {
    using clock = std::chrono::steady_clock;
    const auto t_start = clock::now();

    ec::RgbImage resized;
    const auto t_pre0 = clock::now();
    if (!resampler.resize(src, &resized)) {
        *error = "PilResampler::resize failed";
        return false;
    }
    rgb_to_chw(resized, chw_scratch);
    out->pre_ms = ms_since(t_pre0);

    const auto t_npu0 = clock::now();
    if (!engine->infer(chw_scratch->data(), chw_scratch->size(), raw_scratch, error)) {
        return false;
    }
    out->npu_ms = ms_since(t_npu0);

    const auto t_post0 = clock::now();
    if (!ec::postprocess(raw_scratch->logits.data(), raw_scratch->boxes.data(), src.width,
                         src.height, pcfg, &out->dets, error)) {
        return false;
    }
    out->post_ms = ms_since(t_post0);

    out->orig_w = src.width;
    out->orig_h = src.height;
    out->total_ms = ms_since(t_start);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    std::string error;
    if (!parse_options(argc, argv, &opt, &error)) {
        std::fprintf(stderr, "error: %s\n\n", error.c_str());
        usage(argv[0]);
        return 2;
    }

#ifdef EC_ENABLE_GRID_SAMPLE_CPU
    ec::enable_grid_sample_profile(opt.grid_sample_profile);
    ec::reset_grid_sample_profile();
#endif
    ec::RknnEngine engine;
    if (!engine.init(opt.model, opt.core_mask, &error, opt.grid_sample_cpu,
                     opt.fallback_gpu, !opt.perf_detail.empty())) {
        std::fprintf(stderr, "engine init failed: %s\n", error.c_str());
        return 1;
    }
    if (!engine.validate_contract(&error)) {
        std::fprintf(stderr, "model does not match the deployment contract: %s\n", error.c_str());
        return 1;
    }

    // Print what was actually loaded.  A board log with these numbers is enough
    // to tell a wrong .rknn from a wrong preprocessing from a wrong decoder.
    std::fprintf(stderr, "model      : %s\n", opt.model.c_str());
    std::fprintf(stderr, "sdk        : api=%s driver=%s\n", engine.sdk_version().c_str(),
                 engine.driver_version().c_str());
    {
        auto dims = [](const ec::TensorInfo& t) {
            std::string s;
            for (uint32_t d : t.dims) s += " " + std::to_string(d);
            return s.empty() ? std::string(" ?") : s.substr(1);
        };
        std::fprintf(stderr, "input      : '%s' [%s] n_elems=%u w_stride=%u model_dtype=%d fmt=%d\n",
                     engine.input_info().name.c_str(), dims(engine.input_info()).c_str(),
                     engine.input_info().n_elems, engine.input_info().w_stride,
                     engine.input_info().dtype, engine.input_info().fmt);
        std::fprintf(stderr, "caller     : UINT8 RGB NCHW, pass_through=0, no host normalization\n");
        std::fprintf(stderr, "pred_logits: [%s] n_elems=%u\n", dims(engine.logits_info()).c_str(),
                     engine.logits_info().n_elems);
        std::fprintf(stderr, "pred_boxes : [%s] n_elems=%u\n", dims(engine.boxes_info()).c_str(),
                     engine.boxes_info().n_elems);
    }

    ec::PostConfig pcfg;
    pcfg.score_thresh = opt.score_thresh;
    pcfg.max_det = opt.max_det;

    // Collect the frame list up front so a list-file typo fails before the NPU
    // is warmed up.
    std::vector<std::string> paths;
    if (!opt.image_list.empty()) {
        FILE* f = std::fopen(opt.image_list.c_str(), "r");
        if (f == nullptr) {
            std::fprintf(stderr, "cannot open image list: %s\n", opt.image_list.c_str());
            return 1;
        }
        char line[4096];
        while (std::fgets(line, sizeof(line), f) != nullptr) {
            std::string s(line);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) {
                s.pop_back();
            }
            if (!s.empty() && s[0] != '#') paths.push_back(s);
        }
        std::fclose(f);
    } else {
        paths.push_back(opt.image);
    }
    if (paths.empty()) {
        std::fprintf(stderr, "no images to process\n");
        return 1;
    }

    // Resampler and scratch buffers are reused across frames; preparing the
    // coefficient tables once is the whole point of PilResampler.
    ec::PilResampler resampler;
    bool resampler_ready = false;
    int prepared_w = 0, prepared_h = 0;

    std::vector<uint8_t> chw;
    ec::RawOutputs raw;
    std::vector<double> pre_t, npu_t, post_t, total_t;
    [[maybe_unused]] double profile_infer_ms = 0.0;
    [[maybe_unused]] size_t profile_frames = 0;

    for (size_t i = 0; i < paths.size(); ++i) {
        ec::RgbImage src;
        if (!load_rgb(paths[i], &src, &error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        if (!resampler_ready || src.width != prepared_w || src.height != prepared_h) {
            if (!resampler.prepare(src.width, src.height, kInputSize, kInputSize)) {
                std::fprintf(stderr, "resampler prepare failed for %dx%d\n", src.width,
                             src.height);
                return 1;
            }
            prepared_w = src.width;
            prepared_h = src.height;
            resampler_ready = true;
            std::fprintf(stderr, "resample   : %dx%d -> %dx%d (%s first)\n", src.width,
                         src.height, kInputSize, kInputSize,
                         resampler.horizontal_first() ? "horizontal" : "vertical");
        }

        FrameResult fr;
        if (!run_frame(&engine, resampler, src, pcfg, &fr, &chw, &raw, &error)) {
            std::fprintf(stderr, "%s: %s\n", paths[i].c_str(), error.c_str());
            return 1;
        }

        // The parity oracle only needs one tensor, written from the first frame.
        if (i == 0 && !opt.perf_detail.empty() &&
            !engine.write_perf_detail(opt.perf_detail, &error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        if (!opt.dump_npy.empty() && i == 0) {
            const std::vector<int> shape{3, kInputSize, kInputSize};
            if (!write_npy_u8(opt.dump_npy, chw.data(), chw.size(), shape, &error)) {
                std::fprintf(stderr, "%s\n", error.c_str());
                return 1;
            }
            std::fprintf(stderr, "wrote      : %s (%zu bytes, uint8 CHW)\n",
                         opt.dump_npy.c_str(), chw.size());
        }
        if (!opt.dump_raw.empty() && i == 0) {
            if (!write_f32(opt.dump_raw + ".logits.f32", raw.logits) ||
                !write_f32(opt.dump_raw + ".boxes.f32", raw.boxes)) {
                std::fprintf(stderr, "cannot write raw outputs: %s\n", opt.dump_raw.c_str());
                return 1;
            }
        }
        if (!opt.dump_dets.empty()) {
            // With a list file, one output path would be overwritten per frame
            // and the last image would be the only evidence.  Suffix instead.
            std::string dets_path = opt.dump_dets;
            if (paths.size() > 1) {
                dets_path = opt.dump_dets + "." + std::to_string(i);
            }
            if (!write_dets(dets_path, fr.dets, fr.orig_w, fr.orig_h, &error)) {
                std::fprintf(stderr, "%s\n", error.c_str());
                return 1;
            }
        }

        profile_infer_ms += fr.npu_ms;
        ++profile_frames;
        std::printf("%s %dx%d dets=%zu pre=%.3f infer=%.3f post=%.3f total=%.3f ms\n",
                    paths[i].c_str(), fr.orig_w, fr.orig_h, fr.dets.size(), fr.pre_ms,
                    fr.npu_ms, fr.post_ms, fr.total_ms);

        if (opt.timing) {
            pre_t.push_back(fr.pre_ms);
            npu_t.push_back(fr.npu_ms);
            post_t.push_back(fr.post_ms);
            total_t.push_back(fr.total_ms);
        }
    }

    // Warmup/measure loop over the first frame.  Timing the real pipeline (not
    // rknn_run alone) is what a deployment cares about: the copy into the NPU
    // and the resize are on the critical path for a live stream.
    if (opt.timing) {
        ec::RgbImage src;
        if (!load_rgb(paths[0], &src, &error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        if (!resampler_ready || src.width != prepared_w || src.height != prepared_h) {
            if (!resampler.prepare(src.width, src.height, kInputSize, kInputSize)) {
                std::fprintf(stderr, "resampler prepare failed\n");
                return 1;
            }
        }

        for (int i = 0; i < opt.warmup; ++i) {
            FrameResult fr;
            if (!run_frame(&engine, resampler, src, pcfg, &fr, &chw, &raw, &error)) {
                std::fprintf(stderr, "warmup: %s\n", error.c_str());
                return 1;
            }
        }

        pre_t.clear();
#ifdef EC_ENABLE_GRID_SAMPLE_CPU
        ec::reset_grid_sample_profile();
#endif
        profile_infer_ms = 0.0;
        profile_frames = 0;
        npu_t.clear();
        post_t.clear();
        total_t.clear();
        pre_t.reserve(static_cast<size_t>(opt.iters));
        npu_t.reserve(static_cast<size_t>(opt.iters));
        post_t.reserve(static_cast<size_t>(opt.iters));
        total_t.reserve(static_cast<size_t>(opt.iters));

        for (int i = 0; i < opt.iters; ++i) {
            FrameResult fr;
            if (!run_frame(&engine, resampler, src, pcfg, &fr, &chw, &raw, &error)) {
                std::fprintf(stderr, "iter %d: %s\n", i, error.c_str());
                return 1;
            }
            pre_t.push_back(fr.pre_ms);
            profile_infer_ms += fr.npu_ms;
            ++profile_frames;
            npu_t.push_back(fr.npu_ms);
            post_t.push_back(fr.post_ms);
            total_t.push_back(fr.total_ms);
        }

        const Stats s_pre = summarise(pre_t);
        const Stats s_npu = summarise(npu_t);
        const Stats s_post = summarise(post_t);
        const Stats s_tot = summarise(total_t);

        std::printf("\n");
        std::printf("stage      mean    median      p95       min       max   (ms, n=%d)\n",
                    opt.iters);
        auto row = [](const char* name, const Stats& s) {
            std::printf("%-8s %8.3f %8.3f %8.3f %9.3f %9.3f\n", name, s.mean, s.median, s.p95,
                        s.min, s.max);
        };
        row("pre", s_pre);
        row("infer", s_npu);
        row("post", s_post);
        row("total", s_tot);
        if (s_tot.mean > 0.0) {
            std::printf("\nfps (total mean): %.2f\n", 1000.0 / s_tot.mean);
        }
    }

    #ifdef EC_ENABLE_GRID_SAMPLE_CPU
    if (opt.grid_sample_profile && profile_frames) {
        const auto p = ec::grid_sample_profile();
        const double frames = static_cast<double>(profile_frames);
        const double callback_ms = p.callback_ns / 1e6;
        std::printf("GridSample profile: frames=%zu calls=%llu failures=%llu calls/frame=%.2f\n"
                    "  infer=%.3f callback_sum=%.3f sampling_sum=%.3f ms/frame\n"
                    "  infer-minus-callback-sum=%.3f ms/frame (NOT transfer time; sums may overlap)\n",
                    profile_frames, static_cast<unsigned long long>(p.calls),
                    static_cast<unsigned long long>(p.failures), p.calls / frames,
                    profile_infer_ms / frames, callback_ms / frames,
                    p.sampling_ns / 1e6 / frames, (profile_infer_ms - callback_ms) / frames);
    }
    #endif
    return 0;
}
