#!/usr/bin/env bash
# =============================================================================
# End-to-end ECDet-S RK3588 INT8 pipeline.
#
#   1. export_onnx.py  -> static opset-17 ONNX
#   2. make_calib.py   -> stratified 300-image calibration set
#   3. preprocess.py   -> reference tensor for the board self-check
#   4. convert.py      -> rk3588 INT8 .rknn
#   5. sim_eval.py     -> full val COCOeval + PyTorch comparison
#
# Any step failure stops the pipeline (set -e + pipefail).  Logs are kept per
# step under deploy/artifacts/logs/ so a paper run can be audited afterwards.
#
# Environment names.
#
# CALIB_ENV deliberately runs in the RKNN environment, not the training one:
# calibration PNGs are produced by deploy/preprocess.py's PIL resampler, and the
# simulator that scores the result runs on the rknn env's Pillow.  Splitting the
# two across environments means the calibration distribution and the inference
# distribution are resampled by different Pillow builds -- a silent mismatch if
# their versions ever diverge (they do not for VisDrone sizes today, but
# `pillow = "*"` in pixi.toml leaves that to chance).  One environment removes
# the failure mode.  Override CALIB_ENV=default to use the training env.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ART="$ROOT/deploy/artifacts"
LOG_DIR="$ART/logs"
SUMMARY="$ART/sim_eval_summary.json"

TRAIN_ENV="${TRAIN_ENV:-default}"
RKNN_ENV="${RKNN_ENV:-rknn}"
CALIB_ENV="${CALIB_ENV:-$RKNN_ENV}"
CONFIG="$ROOT/ecdetseg/configs/ecdet/ecdet_s_p150u.yml"
WEIGHTS="$ROOT/ecdetseg/outputs/p150u/e3_s42/best_ema.pth"
if [[ ! -f "$WEIGHTS" && -f "$ROOT/ecdetseg/outputs/p150u/e3_s42/best.pth" ]]; then
  WEIGHTS="$ROOT/ecdetseg/outputs/p150u/e3_s42/best.pth"
fi
DATA_ROOT="${VISDRONE_ROOT:-}"
NUM_CALIB=300
LIMIT=0
CURRENT_STEP="initialization"

usage() {
  cat <<EOF
Usage: $0 [options]
  --config PATH        ECDet yaml (default: $CONFIG)
  --weights PATH       checkpoint (default: best_ema.pth, falls back to best.pth)
  --data-root PATH     VisDrone root (default: auto-detect)
  --num-calib N        calibration images (default: 300)
  --limit N            simulator debug limit (default: 0 = all 548)
  --train-env NAME     pixi env for torch/ONNX export (default: $TRAIN_ENV)
  --calib-env NAME     pixi env for the calibration PNGs (default: $CALIB_ENV)
  --rknn-env NAME      pixi RKNN env (default: $RKNN_ENV)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --weights) WEIGHTS="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --num-calib) NUM_CALIB="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --train-env) TRAIN_ENV="$2"; shift 2 ;;
    --calib-env) CALIB_ENV="$2"; shift 2 ;;
    --rknn-env) RKNN_ENV="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$ART" "$LOG_DIR"
# Never print a stale conclusion if an earlier step fails.
rm -f "$SUMMARY"

PIXI=(pixi run --manifest-path "$ROOT/pixi.toml")
TRAIN_PY=("${PIXI[@]}" -e "$TRAIN_ENV" python)
CALIB_PY=("${PIXI[@]}" -e "$CALIB_ENV" python)
RKNN_PY=("${PIXI[@]}" -e "$RKNN_ENV" python)

# Everything that resamples an image must agree on one Pillow build: the
# calibration PNGs, the board-reference tensor and the simulator all feed the
# same INT8 model, and Pillow changed its pass-order rule between 12 and 13
# (200x900 and 1000x3000 resize differently).  VisDrone sizes happen to be
# stable across that change, which is exactly why a silent split would go
# unnoticed until a different camera resolution showed up.
PILLOW_CALIB="$("${CALIB_PY[@]}" -c 'import PIL; print(PIL.__version__)' 2>/dev/null || echo unknown)"
PILLOW_RKNN="$("${RKNN_PY[@]}" -c 'import PIL; print(PIL.__version__)' 2>/dev/null || echo unknown)"
if [[ "$PILLOW_CALIB" != "$PILLOW_RKNN" ]]; then
  echo "[run_all] FATAL: calibration env ($CALIB_ENV) has Pillow $PILLOW_CALIB but" >&2
  echo "[run_all]        the simulator env ($RKNN_ENV) has Pillow $PILLOW_RKNN." >&2
  echo "[run_all]        Their bilinear resizers differ, so calibration images and" >&2
  echo "[run_all]        eval images would be resampled differently.  Point" >&2
  echo "[run_all]        CALIB_ENV at $RKNN_ENV (the default) or pin Pillow." >&2
  exit 1
fi
echo "[run_all] Pillow: $PILLOW_RKNN (calib env '$CALIB_ENV', simulator env '$RKNN_ENV')"

DATA_ARGS=()
if [[ -n "$DATA_ROOT" ]]; then
  DATA_ARGS=(--data-root "$DATA_ROOT")
fi

print_summary() {
  if [[ -f "$SUMMARY" ]]; then
    python3 - "$SUMMARY" <<'PY'
import json
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
ref = d["pytorch_reference"]["AP"]
sim = d["rknn_sim_int8"]["AP"]
gate = d.get("gate") or {}
# Same gate reference sim_eval.py used, so this line cannot disagree with the
# GATE line below it.  Printing PyTorch-based drop next to an FP32-based gate
# was how a passing run could look like a regression.
base = gate.get("reference_AP")
delta = (sim - base) if base is not None else (sim - ref)
print("")
print("RESULT: PyTorch(stretch) mAP=%.2f | RKNN INT8(stretch) mAP=%.2f | INT8-FP32=%+.2f"
      % (ref, sim, delta))
if gate:
    print("GATE  : %s -- %s"
          % ("PASSED" if gate.get("passed") else "FAILED", gate.get("reason", "")))
sim_info = d.get("simulator") or {}
if sim_info.get("source") == "rebuild":
    print("NOTE  : INT8 mAP came from an in-process rebuild, not from load_rknn of the shipped .rknn.")
PY
  fi
}

on_error() {
  local rc=$?
  echo "" >&2
  echo "[run_all] FAILED at step: $CURRENT_STEP (exit=$rc)" >&2
  echo "[run_all] per-step logs: $LOG_DIR" >&2
  print_summary
  case "$CURRENT_STEP" in
    export)  echo "[run_all] hint: fix ONNX export/parity before quantising." >&2 ;;
    calib)   echo "[run_all] hint: check --data-root and annotation file names." >&2 ;;
    preprocess) echo "[run_all] hint: the reference frame must be larger than 640x640 (see --require-resize); check the val image directory and OpenCV codec support." >&2 ;;
    convert) echo "[run_all] hint: see deploy/README.md error table; ensure rknn env is active." >&2 ;;
    sim_eval) echo "[run_all] hint: re-run accuracy_analysis.py; see summary JSON." >&2 ;;
  esac
  exit "$rc"
}
trap on_error ERR

run_step() {
  local name="$1"; shift
  CURRENT_STEP="$name"
  local log="$LOG_DIR/${name}.log"
  echo ""
  echo "==============================================================================="
  echo "[run_all] step: $name"
  echo "[run_all] log : $log"
  echo "==============================================================================="
  "$@" 2>&1 | tee "$log"
}

# 1. ONNX export -------------------------------------------------------------
run_step export \
  "${TRAIN_PY[@]}" "$ROOT/deploy/export_onnx.py" \
  --repo-root "$ROOT" \
  --config "$CONFIG" \
  --weights "$WEIGHTS" \
  --output "$ART/ecdet_s_p150u_640.onnx" \
  --meta "$ART/ecdet_s_p150u_640_meta.json" \
  --keep-lqe

# 2. Stratified calibration set ---------------------------------------------
# Runs in CALIB_ENV (= the rknn env by default) so the calibration PNGs and the
# simulated inference share one PIL build.  make_calib only needs cv2/numpy.
run_step calib \
  "${CALIB_PY[@]}" "$ROOT/deploy/calib.py" \
  --repo-root "$ROOT" \
  "${DATA_ARGS[@]}" \
  --out-dir "$ART" \
  --num "$NUM_CALIB" \
  --seed 42

# 3. Preprocess reference tensor for the board self-check --------------------
# This MUST use a full-resolution frame.  The earlier version fed the first
# calibration PNG, which is already 640x640 -- PIL then returns a byte-identical
# copy and the resize path is never exercised, so the check passed no matter
# what the board did.  --require-resize makes that mistake a hard failure.
PRE_SRC="$("${CALIB_PY[@]}" - "$ROOT" ${DATA_ROOT:+"$DATA_ROOT"} <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from deploy import INPUT_SIZE, resolve_data_root

repo_root = Path(sys.argv[1])
explicit = sys.argv[2] if len(sys.argv) > 2 else None
val_dir = resolve_data_root(repo_root, explicit) / "val"
if not val_dir.is_dir():
    sys.exit(f"val image directory not found: {val_dir}")
for path in sorted(val_dir.iterdir()):
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        # Prefer a frame that actually needs downscaling.
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None and (img.shape[0] > INPUT_SIZE or img.shape[1] > INPUT_SIZE):
            print(path)
            sys.exit(0)
sys.exit("no val frame larger than %dx%d found" % (INPUT_SIZE, INPUT_SIZE))
PY
)"
echo "[run_all] preprocess reference source: $PRE_SRC"
run_step preprocess \
  "${CALIB_PY[@]}" "$ROOT/deploy/preprocess.py" \
  --image "$PRE_SRC" \
  --require-resize \
  --output-npy "$ART/preprocess_reference.npy" \
  --output-meta "$ART/preprocess_reference.json"

# 4. RKNN INT8 conversion ----------------------------------------------------
run_step convert \
  "${RKNN_PY[@]}" "$ROOT/deploy/convert.py" \
  --repo-root "$ROOT" \
  --onnx "$ART/ecdet_s_p150u_640.onnx" \
  --calib "$ART/calib.txt" \
  --output "$ART/ecdet_s_p150u_640_rk3588_int8.rknn" \
  --meta "$ART/ecdet_s_p150u_640_meta.json" \
  --target rk3588 \
  --hybrid-cos-thresh 0.9999

# 5. Full validation simulation ---------------------------------------------
SIM_ARGS=(
  --repo-root "$ROOT"
  --rknn "$ART/ecdet_s_p150u_640_rk3588_int8.rknn"
  --meta "$ART/ecdet_s_p150u_640_meta.json"
  --out-detections "$ART/detections_sim.json"
  --summary "$SUMMARY"
  --fp32-baseline
  --hybrid-cos-thresh 0.9999
)
if [[ -n "$DATA_ROOT" ]]; then
  SIM_ARGS+=(--data-root "$DATA_ROOT")
fi
if [[ "$LIMIT" -gt 0 ]]; then
  SIM_ARGS+=(--limit "$LIMIT")
fi

run_step sim_eval \
  "${RKNN_PY[@]}" "$ROOT/deploy/sim_eval.py" "${SIM_ARGS[@]}"

# Final one-line conclusion --------------------------------------------------
CURRENT_STEP="summary"
if [[ -f "$SUMMARY" ]]; then
  print_summary
else
  echo "[run_all] warning: summary not found: $SUMMARY" >&2
fi

echo "[run_all] all steps completed."
