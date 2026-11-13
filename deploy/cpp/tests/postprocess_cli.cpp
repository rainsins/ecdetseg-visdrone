// Standalone harness for postprocess: reads raw float32 logits/boxes dumps and
// prints detections, so the C++ decoder can be diffed against
// deploy/postprocess.py on the host.
//
//   postprocess_cli logits.f32 boxes.f32 ORIG_W ORIG_H [score_thresh]
//
// Output: one line per detection, "%d %.6f %.6f %.6f %.6f %.8f" (label x1 y1 x2
// y2 score), in the same order as the numpy reference.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "postprocess.h"

namespace {

bool read_f32(const char* path, size_t count, std::vector<float>* out) {
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr) {
        std::fprintf(stderr, "cannot open %s\n", path);
        return false;
    }
    out->resize(count);
    const size_t got = std::fread(out->data(), sizeof(float), count, f);
    std::fclose(f);
    if (got != count) {
        std::fprintf(stderr, "%s: read %zu of %zu floats\n", path, got, count);
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5 && argc != 6) {
        std::fprintf(stderr, "usage: %s logits.f32 boxes.f32 ORIG_W ORIG_H [score_thresh]\n",
                     argv[0]);
        return 2;
    }
    const int ow = std::atoi(argv[3]);
    const int oh = std::atoi(argv[4]);

    std::vector<float> logits, boxes;
    if (!read_f32(argv[1], 300u * 10u, &logits)) return 1;
    if (!read_f32(argv[2], 300u * 4u, &boxes)) return 1;

    ec::PostConfig cfg;
    if (argc == 6) cfg.score_thresh = static_cast<float>(std::atof(argv[5]));

    std::vector<ec::Detection> dets;
    std::string err;
    if (!ec::postprocess(logits.data(), boxes.data(), ow, oh, cfg, &dets, &err)) {
        std::fprintf(stderr, "postprocess failed: %s\n", err.c_str());
        return 1;
    }

    for (const auto& d : dets) {
        std::printf("%d %.6f %.6f %.6f %.6f %.8f\n", d.label, d.x1, d.y1, d.x2, d.y2,
                    d.score);
    }
    std::fprintf(stderr, "%zu detections\n", dets.size());
    return 0;
}
