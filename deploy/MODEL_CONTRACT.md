# ECDet-S / RK3588 Deployment Model Contract

Status: **CPU custom-op path validated on RK3588; performance investigation active**

2026-11-07: original RKNN + `--grid-sample-cpu` passed 548-image board validation:
AP 28.7691625 / AP50 48.3768373 / AP75 28.6381307; drop 0.051596 vs FP32.
See `artifacts/board_eval_summary.json` and `RK3588_REPAIR.md`. The UINT8
contract describes the caller, while the board reports FP16 NHWC model input.
GPU-priority runtime flag alone still fails; Gather full-model compilation
hung for 22h46m and was manually killed. Simulator AP describes a rebuild.
Pipeline: PyTorch → ONNX (opset 17) → RKNN INT8/FP16-hybrid → RK3588  
Primary artifact: `deploy/artifacts/ecdet_s_p150u_640_rk3588_int8.rknn`

This document is the single source of truth for preprocessing, inference,
postprocessing, calibration, runtime, and acceptance. It is intentionally
path-agnostic (`<PROJECT_ROOT>`, `<VISDRONE_ROOT>`, `<ARTIFACTS>`) so it can be
copied into another AI / project as a prompt seed.

---

## 0. Copy-paste contract summary

> The deployed model is a static 1×3×640×640 RK3588 INT8 model for VisDrone
> 10-class detection. Feed it **uint8 RGB NCHW, 0..255**, produced by a
> **direct stretch resize to 640×640 using PIL bilinear** (no letterbox, no
> padding, no /255, no mean/std in Python). RKNN normalization is baked into the
> model with RGB mean `[123.675,116.28,103.53]` and std
> `[58.395,57.12,57.375]`. Outputs are `pred_logits [1,300,10]` (pre-sigmoid)
> and `pred_boxes [1,300,4]` (normalized cxcywh). Postprocess in float32:
> sigmoid → flatten 300×10 → top-300 → `label=idx%10`, `query=idx//10` →
> cxcywh→xyxy → multiply x by `orig_w` and y by `orig_h` → clip to image →
> COCO `[x,y,w,h]`. No NMS. Do **not** use OpenCV `INTER_AREA`/`INTER_LINEAR`
> as a substitute for the PIL resize without re-validating: it loses ~0.25–0.5
> mAP. The final simulator result is AP 28.71 vs FP32 28.82 (drop 0.11, gate
> passes); the archived build measured 28.6887.
> The frozen model keeps LQE and uses `auto_hybrid_cos_thresh=0.9999`; the
> default `export_onnx.py` without `--keep-lqe` is **not** this model.

---

## 1. Model identity

| Field | Value |
|---|---|
| Architecture | ECDet-S (D-FINE variant): ViTAdapter + HybridEncoder + ECTransformer |
| Task | Object detection, VisDrone 10 classes |
| Config | `ecdetseg/configs/ecdet/ecdet_s_p150u.yml` |
| Checkpoint | `ecdetseg/outputs/p150u/e3_s42/best.pth` (EMA container) |
| Input | 1×3×640×640, static |
| Queries | 300 |
| Classes | 0..9, order below |
| ONNX opset | 17 |
| RKNN target | `rk3588` |
| RKNN toolkit | 2.5.0 |
| Final model size | 49,772,463 bytes (~48 MiB) |
| Calibration | 300 VisDrone train images, seed=42, stratified |

Class order (must not change):

```text
0 pedestrian
1 people
2 bicycle
3 car
4 van
5 truck
6 tricycle
7 awning-tricycle
8 bus
9 motor
```

Artifact hashes. Rebuilds are not byte-reproducible, so treat these as records of
specific builds and never as regression gates (compare AP instead, §9):

```text
ONNX sha256: 3e2a20507eee52d0aafb3a5ca261d2333a5f352bb3a6696c247848554d551be8
             (stable — same ONNX produced both `.rknn` builds)

RKNN sha256: ff61f31c8ef1dd565c613c1bb59a7c40fb17a5e955e44503db19f928ede35130
             49,837,410 B — current artifact, AP 28.7075 (2026-10-27 rebuild)

RKNN sha256: 1d44dc17ce736e807d21b517265a35498bd0ff6e83bb0cc265654bde439e5b40
             49,772,463 B — archived build, AP 28.6887 (superseded, kept as record)
```

---

## 2. Pipeline and frozen commands

The reproducible pipeline is `deploy/run_all.sh`. It performs:

1. `export_onnx.py --keep-lqe` → static opset-17 ONNX
2. `calib.py` → 300 stretch PNG + `calib.txt` + non-finite filter
3. `preprocess.py` → contract smoke test
4. `convert.py --hybrid-cos-thresh 0.9999` → `.rknn`
5. `sim_eval.py --fp32-baseline --hybrid-cos-thresh 0.9999` → mAP

Frozen invocation:

```bash
TRAIN_ENV=default RKNN_ENV=rknn ./deploy/run_all.sh
```

`CALIB_ENV` 未显式设置时等于 `RKNN_ENV`，即校准 PNG 与仿真评测共用同一份 Pillow
（见 README 3）。这属于冻结调用的一部分：校准分布与推理分布必须由同一重采样器生成。
`run_all.sh` 会检查两个环境的 `PIL.__version__` 并在不一致时拒绝启动。

Equivalent individual commands (paths relative to project root):

```bash
# ONNX (LQE must be kept for the final model)
python deploy/export_onnx.py --keep-lqe

# calibration
python deploy/calib.py

# preprocess contract
python deploy/preprocess.py --image <calib-image> \
  --output-npy /tmp/input.npy --output-meta /tmp/meta.json

# RKNN
python deploy/convert.py --hybrid-cos-thresh 0.9999

# simulator
python deploy/sim_eval.py --fp32-baseline --hybrid-cos-thresh 0.9999
```

Important:

- `export_onnx.py` without `--keep-lqe` exports a **no-LQE** graph. That graph
  has an FP32 ceiling of ~28.29 AP and must not be used for the board artifact.
- `convert.py` and `sim_eval.py` both default `--hybrid-cos-thresh` to `0.9999`,
  so the frozen invocation reproduces if the flag is omitted. `run_all.sh`
  passes it explicitly anyway. The threshold is a **cosine distance** bound
  (RKNN's own default is `0.98`): larger means more aggressive FP16 fallback,
  not more similarity. See README 4.9.
- The simulator rebuilds the INT8 model in-process because rknn-toolkit2 2.5
  cannot run `load_rknn` on the simulator; this is expected.

---

## 3. Raw input and preprocessing contract

### 3.1 Raw input

| Item | Contract |
|---|---|
| Source | Camera / image / video frame |
| Color | BGR uint8 as returned by `cv2.imread` |
| Channels | 3 |
| Orientation | Upright, no EXIF rotation unless explicitly applied |
| Original size | Arbitrary; `orig_w`, `orig_h` must be preserved |

### 3.2 Preprocess (normative)

```text
BGR uint8
  -> RGB
  -> direct stretch resize to 640x640 (no aspect preservation, no pad)
  -> HWC -> CHW
  -> uint8, 0..255
  -> shape (1,3,640,640)
```

- Resize backend: **PIL bilinear** (`PIL.Image.resize((640,640),
  Image.BILINEAR)`), matching `torchvision.transforms.v2.Resize` used by the
  training/val pipeline.
- No letterbox.
- No pad.
- No `/255`.
- No mean/std in Python; RKNN does it (see §4).
- Do not use `cv2.INTER_AREA` / `cv2.INTER_LINEAR` as a silent replacement.
  Measured with the same ONNX: PIL 28.82 FP32 vs cv2 `INTER_AREA` 28.65; the
  final INT8 difference (archived build) was 28.69 vs 28.44. If the board
  cannot run PIL, the replacement resize must be validated against the PIL
  tensor, not assumed equivalent.

### 3.3 Caller tensor contract (RKNN API, not ONNX input dtype)

| Field | Value |
|---|---|
| Name (ONNX) | `images` |
| Shape | `[1, 3, 640, 640]` |
| Dtype | `uint8` |
| Range | 0..255 |
| Layout | NCHW |
| Color | RGB |
| Normalization | none in Python |

### 3.4 Preprocess metadata

`preprocess.py` returns/writes:

```json
{
  "orig_w": 1360,
  "orig_h": 765,
  "resized_w": 640,
  "resized_h": 640,
  "input_w": 640,
  "input_h": 640,
  "pad_x": 0,
  "pad_y": 0,
  "scale_x": 1.0,
  "scale_y": 1.0,
  "pad_value": 0,
  "color_order": "RGB",
  "layout": "CHW",
  "normalized": false
}
```

`pad_x/pad_y/scale_x/scale_y` are deliberately neutral. Postprocessing does
**not** use them; it maps normalized boxes directly with `orig_w/orig_h`.

Required smoke check:

```text
deploy/artifacts/preprocess_reference.json:
pad_x == 0
pad_y == 0
scale_x == 1.0
scale_y == 1.0
orig_w > 640 or orig_h > 640   # otherwise the resize path was never exercised
```

The reference tensor must be produced from a **full-resolution** frame. A
640x640 input makes ``PIL.Image.resize`` return a byte-identical copy, so the
comparison passes regardless of what the board does. ``preprocess.py
--require-resize`` enforces this.

Board-side minimum implementation:

```python
from deploy.preprocess import preprocess_path
pre = preprocess_path("frame.jpg", size=640)
tensor = pre.as_batch()          # uint8 (1,3,640,640)
meta = pre.meta_dict()           # orig_w/orig_h used by postprocess
```

If Pillow is not available on the board, install it or implement a byte-exact
equivalent. Do not silently switch to OpenCV.

---

## 4. RKNN normalization contract

The normalization is **inside** the `.rknn`; the caller buffer stays uint8.
The queried model input may be FP16/FP32/INT8/UINT8. C++ supplies
`rknn_input.type=RKNN_TENSOR_UINT8`, `fmt=RKNN_TENSOR_NCHW`, `pass_through=0`.
Do not reject a model solely because `RKNN_QUERY_INPUT_ATTR.type` is FP16.
Shape validation uses the reported NCHW/NHWC layout. Native padding is not
the packing requirement for this non-zero-copy caller buffer.

```python
rknn.config(
    mean_values=[[123.675, 116.28, 103.53]],  # RGB, 0..255 domain
    std_values=[[58.395, 57.12, 57.375]],     # RGB, 0..255 domain
    quant_img_RGB2BGR=False,
    target_platform="rk3588",
    quantized_dtype="w8a8",
    quantized_method="channel",
    quantized_algorithm="normal",
    optimization_level=3,
    auto_hybrid_cos_thresh=0.9999,
)
rknn.build(do_quantization=True, dataset="calib.txt", auto_hybrid=True)
```

- Mean/std are ImageNet ×255 in **RGB** order.
- `quant_img_RGB2BGR=False` because preprocessing already outputs RGB.
- Do **not** apply mean/std again on the board.
- Do **not** feed float32 `[0,1]` unless `inputs_pass_through` is explicitly
  configured and validated; the frozen contract is uint8 0..255.

---

## 5. Output contract

The ONNX/RKNN graph ends at the raw decoder outputs:

| Name | Shape | Dtype | Semantics |
|---|---|---|---|
| `pred_logits` | `[1, 300, 10]` | float32 runtime | **Pre-sigmoid** class logits |
| `pred_boxes` | `[1, 300, 4]` | float32 runtime | Normalized `cxcywh` in the 640 stretch frame |

- The final detection TopK/gather is **not** in the graph.
- Output order is not guaranteed by RKNN Lite. Identify by the last dimension:
  `10` → logits, `4` → boxes. `deploy/postprocess.py::split_outputs` does this.
- `split_outputs` refuses ambiguous cases instead of guessing:
  `num_classes == 4` raises (both tensors would end in 4), and more than one
  4-dimensional candidate for either role raises. A model with 4 classes is
  therefore unsupported by this helper, not silently mis-split.
- `pred_boxes` are already in the same normalized space as the PyTorch
  reference; no anchor decoding is needed on the board.
- The model has 300 queries; each query has 10 class logits.

If the board API returns int8/uint8 outputs, dequantize with the RKNN output
attributes first. The simulator returns float32; the Python `postprocess`
rejects non-floating outputs to avoid silent wrong results.

---

## 6. Postprocess contract (normative)

Inputs:

```text
logits: float32 [1,300,10]
boxes:  float32 [1,300,4]   normalized cxcywh
meta:   {orig_w, orig_h}
```

Algorithm:

```python
scores = sigmoid(logits)                 # per class
flat = scores.reshape(-1)                # 3000 values
order = stable_topk(flat, k=300)
labels = order % 10
queries = order // 10
selected = boxes[queries]                # [300,4] normalized cxcywh
xyxy = cxcywh_to_xyxy(selected)
xyxy[..., 0::2] *= orig_w                # stretch inverse: x
xyxy[..., 1::2] *= orig_h                # stretch inverse: y
xyxy = clip(xyxy, [0,0,orig_w,orig_h])   # default on
drop boxes with w <= 1e-6 or h <= 1e-6
```

Rules:

- One query may appear under multiple classes; this is intentional and matches
  the official ECDet/D-FINE postprocessor.
- No NMS.
- For COCO mAP: `score_thresh=0.0`, `max_det=300`.
- For a product: choose a score threshold (e.g. 0.25–0.4); no extra NMS unless
  the product spec explicitly requires it.
- COCO record:

```json
{
  "image_id": 0,
  "category_id": 3,
  "bbox": [x, y, w, h],
  "score": 0.83
}
```

`category_id` is **0..9** (VisDrone), not 1..10.

### 6.1 Why stretch inverse, not letterbox inverse

`postprocess.py` multiplies normalized x by `orig_w` and y by `orig_h`. It does
**not** subtract padding or divide by `scale_x/scale_y`. This matches both the
PyTorch reference postprocessor and the stretch preprocessing. The old
letterbox inverse was removed.

### 6.2 Why TopK is outside the graph

RKNN INT8 handles dynamic TopK/Gather poorly, and the final selection is a
policy knob (score threshold / max detections). Keeping it in float32 numpy
also makes the decoder output easy to compare with PyTorch and to diagnose.

---

## 7. Calibration contract

| Field | Value |
|---|---|
| Dataset | VisDrone train |
| Count | 300 |
| Seed | 42 |
| Stratification | Dominant class per image; D'Hondt quotas proportional to per-class instance counts; minimum 1 per class |
| Sampling | `random.Random(42)` within each stratum |
| Overlap | Assert train/val file-name disjoint |
| Image transform | `preprocess.py` direct stretch to 640×640 (PIL bilinear), no pad, uint8 RGB PNG |
| Non-finite filter | Run FP32 ONNX, check **every** decoder `GatherElements` output (resolved structurally, all matches not just one); replace inf/NaN images with another image from the same dominant-class stratum, probing each candidate |
| Current replacement | `calib_imgs/0000790_9999999_00576_d_0000260.png` → `calib_imgs/0001635_0000135_00828_d_0000150.png` |
| Calib list | `calib.txt`, one relative path per line |
| Manifest | `calib_manifest.json` |

Why the non-finite filter:

A single calibration image can make the query-selection gather produce `+inf`.
RKNN then sets the layer scale to `inf`, and the entire INT8 model emits
near-zero scores. The filter is not optional for a valid build. `calib.py`
logs the removed image and the replacement; the final 300 must have no
non-finite query-selection activations.

The graph has two decoder gathers and only the first one overflows:

```text
/model/decoder/GatherElements_output_0      <- produces +inf on 1 of 300 images
/model/decoder/GatherElements_1_output_0    <- clean on all 300
```

Probing a single node — even one resolved structurally — is therefore unsound.
An earlier version took the last match, so it watched the clean node, reported
"all 300 images clean", and left the bad image in the set: the build succeeded
and the simulator returned AP 0.00 while FP32 scored 35.08 on the same frames.
The filter now probes all of them and treats any non-finite output as bad.

Calibration images go through the same `preprocess.py` resampler as board
inference, so the calibration and inference distributions are identical by
construction. An earlier version used `cv2.INTER_AREA` here while inference used
PIL bilinear; on a 1360×765 frame those differ on 75% of pixels (max 34/255),
which shifts the activation ranges without any visible symptom. Do not
re-letterbox calibration images.

---

## 8. Quantization and export contract

| Field | Value |
|---|---|
| ONNX opset | 17 |
| Static shape | 1×3×640×640 |
| LQE | **kept** (`--keep-lqe`) |
| Score head input clamp | ±10 |
| Score head output clamp | ±30 |
| Final logits clamp | ±30 |
| Quantization | `w8a8` |
| Auto hybrid | `True` |
| Hybrid cosine threshold | `0.9999` |
| Target | `rk3588` |
| Toolkit | rknn-toolkit2 2.5.0 |
| Runtime | matching librknnrt / RKNPU2 2.5.0 |

Why LQE is kept:

LQE is part of the trained model. With letterbox and naive quantization it
collapsed, which is why an earlier workaround disabled it. With stretch + PIL +
`auto_hybrid=0.9999`, LQE quantizes correctly and recovers the AP ceiling
(no-LQE FP32 28.29 vs LQE FP32 28.82). The frozen board artifact keeps LQE.

Why score-head clamp:

Some VisDrone frames produce FP32 encoder logits down to about -310. Without a
clamp, RKNN calibrates the score-head range on that tail and clips the rare
positive peaks (~+2.6), destroying query selection. Clamping is part of the
exported graph, not a runtime postprocess step.

Do not change any of the following without re-running the full accuracy
protocol:

- input size 640×640
- RGB channel order
- mean/std values
- LQE enabled/disabled
- `auto_hybrid` / threshold
- calibration seed/count/filter
- output semantics (pre-sigmoid logits, normalized cxcywh)
- TopK/label/query mapping

---

## 9. Accuracy contract

**Board result (2026-11-07):** original artifact with the CPU custom callback,
548 images / 164400 detections: AP **28.76916252162901**, AP50 **48.37683725216193**,
AP75 **28.638130657815363**. One-sided 1.0 AP gate passes (drop **0.051596**).
The reported evaluation took 347.2 seconds, 633.577 ms/image, 1.578 FPS. This
is evaluation-average time, not isolated callback or NPU time. The board report
does not include a measured model hash; its path alone cannot verify identity.
Board-vs-simulator AP difference does not by itself establish its cause.
The table below remains the separately measured **simulator rebuild** result.

Evaluation protocol: VisDrone val 548 images, COCOeval bbox, `maxDets=[1,10,100,300]`,
custom 12-stat summary reading the trailing `maxDets` slot, `score_thresh=0`,
`clip=True`, stretch preprocessing, LQE kept.

| Metric | PyTorch(stretch) | FP32 ONNX(stretch) | RKNN INT8(stretch) | INT8 drop |
|---|---:|---:|---:|---:|
| AP | 28.82 | 28.8208 | **28.7075** | **0.1133** |
| AP50 | 48.54 | 48.5410 | 48.5089 | 0.0321 |
| AP75 | 28.63 | 28.6290 | 28.3918 | 0.2372 |
| APs | 18.67 | 18.6712 | 18.5602 | 0.1111 |
| APm | 41.47 | 41.4749 | 41.3606 | 0.1143 |
| APl | 61.67 | 61.6736 | 61.4027 | 0.2708 |
| detections | 164400 | 164400 | 164400 | — |

Above is an ONNX-based in-process simulator rebuild cross-checked against
the current artifact's conversion configuration (`ff61f31c…`). It does not
execute that exported file (`describes_shipped_artifact: false`) and does not
validate board execution. The archived
build (`1d44dc17…`) measured AP 28.6887 / AP50 48.5519 / AP75 28.2911; its
row is kept in the rebuild note below, not in this table — mixing two builds
in one column invites quoting a number that no current artifact produces.

Acceptance (single-sided, enforced by `sim_eval.py`):

```text
fail when: reference_AP - INT8_AP > fail_drop      (default fail_drop = 1.0)
pass when: INT8_AP >= reference_AP - fail_drop
```

The gate is one-sided on purpose. INT8 scoring *above* the FP32 baseline is a
quantisation artefact, not a regression, so an absolute-difference test would
fail a better-than-reference build. `reference_AP` is the FP32 stretch baseline
from the same run, not the PyTorch number and not the archived `28.6887`.
The current run measures `INT8 28.7075` against `reference_AP = 28.8208`, so
`drop = 0.1133 <= 1.0` and the gate passes.

`sim_eval.py` writes the verdict into the summary so downstream steps do not
have to re-derive it:

```json
"gate": {
  "passed": true,
  "against": "FP32 stretch baseline",
  "reference_AP": 28.820758052964102,
  "drop": 0.11329371972247415,
  "fail_drop": 1.0,
  "one_sided": true,
  "reason": "INT8 AP 28.71 is 0.11 points below FP32 stretch baseline 28.82 (within limit 1.00)"
}
```

`run_all.sh` prints its final `RESULT` line from this same `gate.reference_AP`,
so the reported drop and the gate verdict cannot contradict each other. An
earlier revision documented a fixed `AP >= 28.6` plus `drop <= 0.3`; the `0.3`
bound is no longer enforced anywhere — do not treat it as a live threshold.

Rebuild note: re-running the pipeline on 2026-10-27 after the non-finite filter
fix produced a new `.rknn` (`ff61f31c…`, 49,837,410 B) measuring **28.7075** AP
(AP50 48.5089, AP75 28.3918), i.e. 0.11 below the FP32 baseline of 28.8208 on the
same 548 frames. That is the same calibration set (identical PNG bytes for
8/8 sampled images regenerated from source) and the same seed, so the 0.02 AP
difference against the archived 28.6887 is run-to-run quantisation noise, not a
config change. The archived hash stays as the record of the original build; a
rebuilt artifact is not byte-reproducible and its hash must not be treated as a
substitute.

The simulator timing (~1.21 s/img) is **not** board latency; measure on RK3588.

---

## 10. Board runtime contract

### 10.1 Versions

- `rknn-toolkit-lite2` 2.5.0 (Python board path)
- `librknnrt.so` / RKNPU2 runtime 2.5.0 — required by both the Python and the C++ path
- Kernel NPU driver compatible with RK3588
- Python 3.10/3.11
- `numpy<2`
- `opencv-python`
- `pillow` (for the PIL resize)
- `pycocotools` only for board-side mAP validation
- C++ path: a C++17 compiler, CMake >= 3.16, `rknn_api.h` + `librknnrt.so`,
  and OpenCV core/imgproc/imgcodecs

A runtime older than the toolkit may fail to load the `.rknn` or may mis-handle
the FP16-hybrid graph.

### 10.2 Load / init

The following Python snippets illustrate the caller contract, not the validated
GridSample backend. The shipped graph requires the C++ custom callback on the
tested board: build with `EC_ENABLE_GRID_SAMPLE_CPU=ON` and run with
`--grid-sample-cpu`. Bare init/run and GPU-priority-only paths failed there.

```python
from rknnlite.api import RKNNLite

rknn = RKNNLite()
assert rknn.load_rknn("ecdet_s_p150u_640_rk3588_int8.rknn") == 0
assert rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0
```

`core_mask` is optional; measure single-core vs all-core. RK3588 has 3 NPU
cores.

### 10.3 Inference

```python
outs = rknn.inference(
    inputs=[pre.as_batch()],      # uint8 (1,3,640,640)
    data_format="nchw",
)
logits, boxes = split_outputs(outs, num_classes=10)
dets = postprocess(logits, boxes, pre.meta_dict())
records = to_coco_detections(dets, image_id=0)
```

- If the board API only accepts NHWC, transpose to `(1,640,640,3)` and pass
  `data_format="nhwc"`. Validate the tensor and outputs against the NCHW
  simulator path.
- Do not normalize before inference.
- Do not cast uint8 to int8 manually; the runtime handles the internal
  quantized input.
- If outputs are int8/uint8, dequantize with the runtime output attributes
  before postprocessing.

### 10.4 C++ runtime (RKNPU2 C API)

`deploy/cpp/` is the second reference implementation of this contract, and it
is the one to copy if the board is C++. It uses the RKNPU2 C API directly:

```c
rknn_init(&ctx, model_bytes, model_size, 0, NULL);          // whole file in memory
rknn_set_core_mask(ctx, RKNN_NPU_CORE_AUTO);                // or CORE_0 / CORE_0_1_2
rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver));
rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));   // must be 1 in, 2 out

rknn_input in = {0};
in.index = 0; in.type = RKNN_TENSOR_UINT8; in.fmt = RKNN_TENSOR_NCHW;
in.buf = tensor; in.size = 640*640*3; in.pass_through = 0;
rknn_inputs_set(ctx, 1, &in);
rknn_run(ctx, NULL);

rknn_output outs[2] = {0};
outs[0].index = 0; outs[0].want_float = 1;
outs[1].index = 1; outs[1].want_float = 1;
rknn_outputs_get(ctx, 2, outs, NULL);
/* ... decode ... */
rknn_outputs_release(ctx, 2, outs);
rknn_destroy(ctx);
```

Rules that are part of the contract, not style preferences:

- **Identify outputs by element count, not by index.** `3000` is `pred_logits`,
  `1200` is `pred_boxes`. RKNN does not guarantee output ordering, and a
  swapped pair still produces structurally valid detections — the worst class
  of silent bug. Query both `RKNN_QUERY_OUTPUT_ATTR` entries during init and
  fail loudly if the counts are neither (3000, 1200) nor (1200, 3000).
- **Use `rknn_inputs_set` with `RKNN_TENSOR_NCHW` + `RKNN_TENSOR_UINT8`.**
  The zero-copy path (`rknn_create_mem` + `rknn_set_io_mem`) saves one 1.2 MB
  copy per frame but requires the *native* layout from
  `RKNN_QUERY_NATIVE_INPUT_ATTR`, which may be `NC1HWC2`. That layout cannot
  be validated off-board, so the contract does not use it.
- **Set `want_float = 1`** and let the runtime apply scale/zero_point. Verify
  the returned buffer is `n_elems * sizeof(float)`; a different size means a
  quantized-output build and must be dequantized explicitly, not reinterpreted.
- **`pass_through = 0`.** Do not hand the runtime a pre-normalized or FP32
  buffer; normalization lives inside the `.rknn` (§4).
- **Reject non-finite logits/boxes** before postprocessing. The same failure
  that produced AP 0.00 in calibration (§7) can show up as `inf`/`NaN` here.
- `core_mask` is a runtime choice, not a model property: measure single-core
  vs all-core. `0` / `RKNN_NPU_CORE_AUTO` is the default when unspecified.

Build: `-DEC_BUILD_BOARD=ON` (default) for `ec_infer`, `-DEC_BUILD_TESTS=ON`
(default off) for the host-only parity harnesses that need neither RKNN nor
OpenCV. CMake finds RKNN through `EC_RKNN_ROOT` (a prefix containing
`include/rknn_api.h` and `lib/librknnrt.so`), falling back to a system search,
with `EC_RKNN_INCLUDE_DIR` / `EC_RKNN_LIBRARY` as explicit overrides. When
`EC_BUILD_BOARD=ON` and RKNN is not found, configuration fails on purpose —
a silently different backend would defeat the point of the target.

---

## 11. Board validation protocol

Do these in order; do not debug quantization before step 3 passes.

### 11.1 Single-image tensor parity

On host and board, save the preprocessed uint8 tensor for the same image:

```text
tensor shape: (1,3,640,640)
tensor dtype: uint8
max_abs_diff(host, board): 0 (or explain any nonzero diff)
```

Also compare the tensor statistics (min/max/mean/std). A nonzero tensor diff
means the board preprocessing is not the frozen contract.

The delivered tooling does this in two commands (details in §11.5):

```bash
./build/ec_infer --model m.rknn --image frame.jpg --dump-npy /tmp/board.npy
pixi run -e rknn python deploy/preprocess.py \
  --image frame.jpg --compare-candidate /tmp/board.npy --require-resize
```

`--require-resize` matters: a 640×640 source makes PIL return a byte-identical
copy, so the resize path is skipped and the comparison passes no matter what the
board does. Use a non-640 frame when validating a new resize implementation.

### 11.2 Single-image output parity

Compare `pred_logits` and `pred_boxes` with the simulator for the same image:

- logits range/mean/top-k ordering
- boxes range and top-300 selected queries
- decoded boxes after postprocess

If the tensor is identical but outputs differ materially, check:

1. `data_format` / NCHW vs NHWC
2. output order (identify by shape)
3. runtime version
4. output dtype/dequantization
5. accidental double normalization

### 11.3 Board COCO mAP

Use 50–100 val images for diagnosis, then all 548 for final acceptance.
Compare a subset only with references evaluated on exactly the same image IDs:

1. Board writes `detections_board.json` with `image_id/category_id/bbox/score`
2. Host runs the same COCOeval script/statistics
3. Compare with `gate.reference_AP` from `artifacts/sim_eval_summary.json`
   (currently `28.8208`), not with a hard-coded constant. The archived
   `28.6887` and the rebuilt `28.7075` are records of specific builds, not
   thresholds.

Acceptance:

```text
board AP >= gate.reference_AP - 1.0   (same fail_drop the simulator uses)
```

If board AP is lower, re-check §11.1 first. Only after that consider
re-calibration or hybrid-quantization changes.

### 11.4 Latency and stability

Measure with warmup:

```text
warmup: 20 runs
measure: 100 runs
report: preprocess ms, inference ms, postprocess ms, end-to-end ms, FPS
also: peak memory, NPU core mask, board temperature, power
```

Run a 30–60 minute sustained test to detect thermal throttling.

### 11.5 Executable harness (delivered)

`deploy/cpp/` implements §11.1 and §11.4 directly, so the protocol above does
not have to be re-derived by hand:

```bash
# §11.1: board writes the uint8 CHW tensor, host Python is the oracle.
./build/ec_infer --model m.rknn --image frame.jpg --dump-npy /tmp/board.npy
pixi run -e rknn python deploy/preprocess.py \
  --image frame.jpg --compare-candidate /tmp/board.npy --require-resize

# §11.2: dump raw logits/boxes (little-endian float32 on RK3588).
./build/ec_infer --model m.rknn --image frame.jpg --dump-raw /tmp/board
# /tmp/board.logits.f32 = [1,300,10]; /tmp/board.boxes.f32 = [1,300,4].
# --dump-dets writes decoded detections, not raw outputs.

# §11.4: percentile table for measured iterations after warmup, pre/infer/post/total.
./build/ec_infer --model m.rknn --image frame.jpg --timing --warmup 20 --iters 100
```

`--require-resize` is not optional: a 640x640 input makes PIL return a
byte-identical copy, so the resize path is skipped and the check passes
regardless of board behaviour.

What the stub-based host validation does NOT prove: NPU numerics and real
latency. It proves call sequence, tensor shapes, data flow, and that the
decoder matches `postprocess.py` on identical float inputs. §11.3 and the
temperature/power half of §11.4 require an actual RK3588.

---

## 12. Failure matrix / common mistakes

| Symptom | Likely cause | Correct action |
|---|---|---|
| Input dtype is 1, expected UINT8 | Model dtype confused with caller dtype | Use fixed UINT8 NCHW caller with pass_through=0; permit FP16 model input |
| Unsupport CPU op: GridSample | This runtime lacks the selected CPU implementation | Use the validated C++ `--grid-sample-cpu` path; GPU routing/Gather alternatives remain experimental |
| All scores ~0.01 | LQE quantized badly or missing clamps | Use the frozen export (`--keep-lqe`, clamps) and `auto_hybrid=0.9999` |
| Board AP ~0 | Calibration set contains non-finite gather image | Re-run `calib.py`; verify replacement log; final 300 clean |
| Board AP ~28.4 instead of 28.71 | cv2 resize instead of PIL bilinear | Use PIL or validate a byte-equivalent resize |
| Boxes shifted/scaled | letterbox inverse still applied | Use `x*=orig_w`, `y*=orig_h`; pad/scale are neutral |
| Boxes in 640-space | forgot orig-size multiplication | Multiply normalized xyxy by orig_w/orig_h |
| Channel swap / weird colors | BGR fed as RGB | Preprocess BGR→RGB; `quant_img_RGB2BGR=False` |
| Double normalization | mean/std applied on board | Remove Python normalization; it is inside `.rknn` |
| Output shape mismatch | NCHW/NHWC mix | Match `data_format` and transpose correctly |
| `load_rknn` simulator error | rknn-toolkit2 2.5 limitation | Simulator must `load_onnx + build`; `.rknn` is only for board/NPU |
| `pkg_resources` missing | setuptools >=81 | Pin `setuptools<80.1` in the RKNN env |
| Model loads but slow | FP16-hybrid ops on CPU / core mask | Measure per-stage; try core mask; inspect RKNN perf |
| Accuracy differs from host ONNX | runtime/version or output dtype | Validate tensor first; compare output dtype; update librknnrt |
| C++ build passes but board AP still low | stub-tested harness, real NPU differs | The stub proves call sequence and decoder parity only; re-check §11.1 on the board before touching the decoder |
| Rebuilt `.rknn` hash equals the archived one | impossible — builds are not byte-reproducible | Do not use the hash as a regression gate; compare AP against `gate.reference_AP` |
| `split_outputs` raises on a 4-class model | `num_classes == 4` makes "last dim == 4" match both tensors | Pass the real `num_classes`; 4 is rejected by design instead of guessing |

---

## 13. Invariants (do not change silently)

1. Input is 640×640, static, uint8 RGB NCHW 0..255.
2. Preprocessing is direct stretch, **PIL bilinear**, no pad, no letterbox.
3. Mean/std are RGB ×255 values inside `.rknn`; no Python normalization.
4. Outputs are `[1,300,10]` pre-sigmoid logits and `[1,300,4]` normalized
   cxcywh boxes.
5. Postprocess TopK is outside the graph; mapping is `label=idx%10`,
   `query=idx//10`.
6. Box inverse is `x*=orig_w`, `y*=orig_h`; no pad/scale subtraction.
7. No NMS by default.
8. Category IDs are 0..9 in the order listed in §1.
9. Calibration is 300 images, seed 42, with the non-finite filter.
10. The frozen model keeps LQE and uses `auto_hybrid_cos_thresh=0.9999`.
11. `summary JSON` field names `fp32_letterbox_baseline` /
    `drop_vs_fp32_letterbox` are retained for compatibility even though the
    baseline is now stretch.
12. The accuracy gate is single-sided on `fail_drop` (default 1.0) against
    `gate.reference_AP`; there is no fixed AP floor any more.
13. RKNN output tensors are identified by element count (`3000` logits,
    `1200` boxes), never by index — on both the Python and C++ paths.

---

## 14. Artifact map

```text
deploy/
├── MODEL_CONTRACT.md              # this file
├── preprocess.py                  # stretch + PIL + metadata
├── postprocess.py                 # sigmoid/topk/stretch inverse
├── calib.py                       # calibration entry point
├── make_calib.py                  # implementation
├── export_onnx.py                 # ONNX exporter (use --keep-lqe)
├── convert.py                     # ONNX → RKNN
├── sim_eval.py                    # simulator + FP32 baseline + COCOeval
├── accuracy_analysis.py           # layer-wise diagnostic
├── eval_pytorch.py                # PyTorch reference with deploy preprocessing
├── run_all.sh                     # frozen end-to-end pipeline
├── cpp/                           # board reference implementation (§10.4, §11.5)
│   ├── main.cpp                   # ec_infer CLI, timing, --dump-npy / --dump-dets
│   ├── rknn_engine.h / .cpp       # rknn_init/inputs_set/run/outputs_get
│   ├── pil_resize.h / .cpp        # byte-exact PIL bilinear resampler
│   ├── postprocess.h / .cpp       # sigmoid/topk/stretch inverse
│   ├── CMakeLists.txt             # EC_BUILD_BOARD / EC_BUILD_TESTS
│   └── tests/                     # host-only parity harnesses
└── artifacts/
    ├── ecdet_s_p150u_640.onnx
    ├── ecdet_s_p150u_640_meta.json
    ├── ecdet_s_p150u_640_rk3588_int8.rknn
    ├── ecdet_s_p150u_640_rk3588_int8.rknn.json
    ├── calib.txt
    ├── calib_imgs/                # 300 stretch 640 PNG
    ├── calib_manifest.json
    ├── preprocess_reference.npy / .json
    ├── detections_sim.json
    └── sim_eval_summary.json
```

The board needs at minimum: `.rknn`, plus either `deploy/cpp/` (compiled) or
`preprocess.py` + `postprocess.py` + `deploy/__init__.py` (Python `rknnlite`).
Keep the rest for reproducibility/audit.

---

## 15. Prompt seed for another AI

Use this block when asking another AI to generate board code:

```text
Implement an RK3588 board inference wrapper for this frozen model contract:

- Artifact: ecdet_s_p150u_640_rk3588_int8.rknn (rknn-toolkit2/lite2 2.5.0)
- Input: uint8 RGB NCHW (1,3,640,640), 0..255
- Preprocess: BGR frame -> RGB -> direct stretch 640x640 with PIL bilinear;
  no letterbox, no pad, no /255, no mean/std in Python
- RKNN normalization is baked in: RGB mean [123.675,116.28,103.53],
  std [58.395,57.12,57.375], quant_img_RGB2BGR=False
- Outputs: pred_logits [1,300,10] pre-sigmoid, pred_boxes [1,300,4]
  normalized cxcywh
- Postprocess: sigmoid, flatten 300x10, top-300, label=idx%10,
  query=idx//10, cxcywh->xyxy, x*=orig_w, y*=orig_h, clip, COCO xywh
- No NMS
- VisDrone classes 0..9
- Validate: preprocessed tensor byte-identical to host; board AP within 1.0
  of the simulator's FP32 reference (28.82); measure preprocess/infer/postprocess
  separately
- Do not use OpenCV resize as a substitute for PIL without re-validation
```

---

## 16. Change log

| Date | Change |
|---|---|
| 2026-11-07 | CPU callback passes 548-image board AP gate. Quoted attributes and mislabeled channel-last grid covered by host regression; optional callback timing added. GPU routing and bounded Gather compilation remain diagnostic. |
| 2026-11-06 | Distinguish compiled dtype from UINT8 caller; add GridSample rewrite and strict CPU fallback; clarify simulator/board evidence and subset comparisons. See RK3588_REPAIR.md. |
| 2026-10 | Letterbox → stretch; PIL bilinear; LQE re-enabled; auto_hybrid 0.9999; non-finite calibration filter; final AP 28.69 (archived build — see the 2026-10-27 row for the current artifact) |
| 2026-10-27 | Non-finite filter narrowed to a single gather probe → probed the clean one; one inf image survived calibration and INT8 collapsed to AP 0.00. Filter now probes every decoder `GatherElements`; rebuild measures AP 28.71 and the gate passes. |
| 2026-10-28 | §9 gate rewritten to the enforced single-sided `--fail-drop` rule (the old `drop <= 0.3` is retired); §10.1 adds the C++ toolchain; new §10.4 documents the RKNPU2 C API sequence and the `n_elems` output rule; §11.3 compares against `gate.reference_AP`; §14 gains `deploy/cpp/`. |
