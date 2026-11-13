#pragma once

#include "rknn_custom_op.h"
#include <cstdint>

namespace ec {
// Strict reference fallback: FLOAT32, packed logical tensors only. The model's
// external input dtype does NOT determine the custom callback tensor dtype.
// Unsupported runtime representations fail with a diagnostic, never reinterpret.
rknn_custom_op make_grid_sample_cpu_op();
struct GridSampleProfile {
    uint64_t calls = 0, failures = 0, callback_ns = 0, sampling_ns = 0;
};
// Process-wide counters for the single-engine synchronous CLI. Reset/read only
// between completed inferences; sums are not wall time if callbacks overlap.
void enable_grid_sample_profile(bool enabled);
void reset_grid_sample_profile();
GridSampleProfile grid_sample_profile();
}  // namespace ec
