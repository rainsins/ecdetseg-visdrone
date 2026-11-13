#include "postprocess.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace ec {
namespace {

// Same numerical split as postprocess.py::sigmoid: the positive and negative
// branches are evaluated separately so large |x| cannot overflow exp().
inline float sigmoid(float x) {
    if (x >= 0.0f) {
        return 1.0f / (1.0f + std::exp(-x));
    }
    const float e = std::exp(x);
    return e / (1.0f + e);
}

}  // namespace

bool postprocess(const float* logits, const float* boxes, int orig_w, int orig_h,
                 const PostConfig& cfg, std::vector<Detection>* out, std::string* error) {
    out->clear();
    if (logits == nullptr || boxes == nullptr) {
        *error = "postprocess: null input";
        return false;
    }
    if (orig_w <= 0 || orig_h <= 0) {
        *error = "postprocess: invalid source size " + std::to_string(orig_w) + "x" +
                 std::to_string(orig_h);
        return false;
    }
    if (cfg.num_classes <= 0 || cfg.num_queries <= 0) {
        *error = "postprocess: invalid config";
        return false;
    }

    const int nc = cfg.num_classes;
    const int nq = cfg.num_queries;
    const size_t total = static_cast<size_t>(nc) * nq;

    // Flatten and score.  scores[flat] with flat = query*nc + class, matching
    // numpy's reshape(-1) on a [query][class] array.
    std::vector<float> scores(total);
    for (size_t i = 0; i < total; ++i) {
        scores[i] = sigmoid(logits[i]);
    }

    const int k = std::min(cfg.max_det > 0 ? cfg.max_det : nq * nc,
                           static_cast<int>(std::min<size_t>(total, 1u << 30)));
    if (k <= 0) {
        *error = "postprocess: max_det is zero";
        return false;
    }

    std::vector<int> order(total);
    std::iota(order.begin(), order.end(), 0);
    // Descending score, then ascending flat index: stable_sort with a strict
    // weak order that is exactly numpy's (-score, index) stable argsort.
    std::stable_sort(order.begin(), order.end(), [&scores](int a, int b) {
        if (scores[a] != scores[b]) return scores[a] > scores[b];
        return a < b;
    });
    order.resize(static_cast<size_t>(k));

    const float ow = static_cast<float>(orig_w);
    const float oh = static_cast<float>(orig_h);

    out->reserve(order.size());
    for (int flat : order) {
        const float score = scores[flat];
        // score_thresh is applied after top-k in the reference, so the cut can
        // leave the list shorter than max_det but never re-ranks it.
        if (cfg.score_thresh > 0.0f && score < cfg.score_thresh) continue;

        const int label = flat % nc;
        const int query = flat / nc;
        const float* b = boxes + static_cast<size_t>(query) * 4;
        const float cx = b[0], cy = b[1], w = b[2], h = b[3];

        float x1 = (cx - 0.5f * w) * ow;
        float y1 = (cy - 0.5f * h) * oh;
        float x2 = (cx + 0.5f * w) * ow;
        float y2 = (cy + 0.5f * h) * oh;

        if (cfg.clip) {
            x1 = std::min(std::max(x1, 0.0f), ow);
            x2 = std::min(std::max(x2, 0.0f), ow);
            y1 = std::min(std::max(y1, 0.0f), oh);
            y2 = std::min(std::max(y2, 0.0f), oh);
        }

        // Clipping can invert a box that poked out of an edge; the reference
        // normalises rather than dropping, so a legitimately coarse box is kept.
        const float nx1 = std::min(x1, x2);
        const float nx2 = std::max(x1, x2);
        const float ny1 = std::min(y1, y2);
        const float ny2 = std::max(y1, y2);

        if (nx2 - nx1 <= cfg.min_wh || ny2 - ny1 <= cfg.min_wh) continue;

        Detection d;
        d.x1 = nx1;
        d.y1 = ny1;
        d.x2 = nx2;
        d.y2 = ny2;
        d.score = score;
        d.label = label;
        out->push_back(d);
    }
    return true;
}

}  // namespace ec
