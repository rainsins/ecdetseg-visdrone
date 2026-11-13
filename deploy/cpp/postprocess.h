// Detector postprocessing, mirroring deploy/postprocess.py exactly.
//
// Board and simulator must produce the same detections from the same logits, so
// this file deliberately copies the numpy reference rather than inventing a
// tidier variant.  Two details are load-bearing:
//
//   1. Ranking is a *global* top-k over the flattened [query][class] grid, with
//      ties broken by ascending flat index.  One query can therefore emit
//      several classes, which is what the PyTorch PostProcessor does and what
//      the reported mAP was measured with.  Per-query argmax would silently
//      change the numbers.
//   2. Boxes are clipped to the source image and min/max-normalised before the
//      degenerate filter.  That is stricter than the upstream postprocessor and
//      is why the simulated mAP is not a pure quantisation delta.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ec {

struct Detection {
    float x1 = 0.0f, y1 = 0.0f, x2 = 0.0f, y2 = 0.0f;  // source-image pixels
    float score = 0.0f;
    int label = 0;  // 0..9, no category remap
};

struct PostConfig {
    int num_classes = 10;
    int num_queries = 300;
    float score_thresh = 0.0f;  // applied after top-k, like the reference
    int max_det = 300;
    bool clip = true;
    // Boxes whose width or height falls below this after inversion are dropped.
    // COCO's loadRes rejects zero-area boxes, and float32 rounding can produce
    // them from a box that was just clipped to a degenerate extent.
    float min_wh = 1e-6f;
};

// Decode int8/int32-free float32 decoder heads into source-space detections.
// `logits` is [num_queries][num_classes], `boxes` is [num_queries][4] in
// normalised cxcywh (stretch: x*orig_w, y*orig_h, no pad, no scale).
bool postprocess(const float* logits, const float* boxes, int orig_w, int orig_h,
                 const PostConfig& cfg, std::vector<Detection>* out, std::string* error);

}  // namespace ec
