#!/usr/bin/env bash
# ============================================================
# Throughput benchmark: total_batch_size 8 vs 10 (VisDrone, ECDet-S).
# Runs BENCH_EPOCHS epochs for each batch size, records it/s and peak XPU mem,
# and prints a verdict (adopt bs10 if images/s ratio >= 1.15 and no OOM).
#
#   REQUIRES A FREE XPU (single Arc). Refuses to run if train.py is active.
#   VISDRONE_ROOT=/.../VisDrone bash scripts/bench_throughput.sh
#
# Env: BENCH_CONFIG (default configs/ecdet/ecdet_s_p150u.yml),
#      BENCH_EPOCHS (default 2), BATCH_SIZES (default "8 10")
# Results: outputs/_bench/bs{8,10}/train.log + outputs/_bench/summary.txt
# ============================================================
set -u -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$HERE/.." && pwd)"
MANIFEST="$(cd "$PROJECT_DIR/.." && pwd)/pixi.toml"

if pgrep -af "train.py" | grep -v pgrep >/dev/null; then
  echo "ABORT: a train.py is running (single GPU). Stop it first." >&2
  pgrep -af train.py >&2; exit 1
fi
: "${VISDRONE_ROOT:?bench: set VISDRONE_ROOT}"

CFG="${BENCH_CONFIG:-configs/ecdet/ecdet_s_p150u.yml}"
EPOCHS="${BENCH_EPOCHS:-2}"
SIZES="${BATCH_SIZES:-8 10}"
OUT="$PROJECT_DIR/outputs/_bench"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.txt"; : > "$SUMMARY"

data_args=(
  "seed=42" "epochs=$EPOCHS" "stop_epoch=$EPOCHS" "flat_epoch=1"
  "train_dataloader.dataset.img_folder=$VISDRONE_ROOT/train"
  "train_dataloader.dataset.ann_file=$VISDRONE_ROOT/annotations_VisDrone_train.json"
  "val_dataloader.dataset.img_folder=$VISDRONE_ROOT/val"
  "val_dataloader.dataset.ann_file=$VISDRONE_ROOT/annotations_VisDrone_val.json"
)

declare -A ITPS MEM
for bs in $SIZES; do
  d="$OUT/bs$bs"; rm -rf "$d"; mkdir -p "$d"
  echo "== running bs=$bs, epochs=$EPOCHS =="
  ( cd "$PROJECT_DIR" && LOG_XPU_MEM=1 pixi run --manifest-path "$MANIFEST" python train.py \
      -c "$CFG" -d xpu -u "output_dir=$d" "total_batch_size=$bs" "${data_args[@]}" ) >"$d/train.log" 2>&1
  rc=$?
  # per-iteration time is printed by the metric logger as "Total time: ... (x.xxxx s / it)"
  sit="$(grep -oE '\([0-9.]+ s / it\)' "$d/train.log" | tail -1 | grep -oE '[0-9.]+' || true)"
  mem="$(grep -oE 'peak_alloc_gb=[0-9.]+' "$d/train.log" | tail -1 | cut -d= -f2 || true)"
  if [ -z "$sit" ]; then
    echo "  bs=$bs: NO s/it found (rc=$rc) -> likely OOM/crash; see $d/train.log"
    ITPS["$bs"]=""; MEM["$bs"]="${mem:-NA}"
    continue
  fi
  itps="$(awk -v s="$sit" 'BEGIN{printf "%.3f", 1.0/s}')"
  ITPS["$bs"]="$itps"; MEM["$bs"]="${mem:-NA}"
  echo "  bs=$bs: s/it=$sit  it/s=$itps  imgs/s=$(awk -v a="$itps" -v b="$bs" 'BEGIN{printf "%.3f", a*b}')  peak_gb=${mem:-NA}"
done

{
  echo "config=$CFG epochs=$EPOCHS"
  for bs in $SIZES; do
    echo "bs=$bs it/s=${ITPS[$bs]:-NA} peak_gb=${MEM[$bs]:-NA}"
  done
} >> "$SUMMARY"

r8="${ITPS[8]:-}"; r10="${ITPS[10]:-}"
if [ -n "$r8" ] && [ -n "$r10" ]; then
  ratio="$(awk -v a="$r10" -v b="$r8" 'BEGIN{printf "%.4f", (a*10)/(b*8)}')"
  echo "throughput ratio (bs10 vs bs8, imgs/s) = $ratio" | tee -a "$SUMMARY"
  if awk -v r="$ratio" 'BEGIN{exit !(r>=1.15)}'; then
    echo "VERDICT: adopt total_batch_size=10 (ratio >= 1.15)" | tee -a "$SUMMARY"
  else
    echo "VERDICT: keep total_batch_size=8 (ratio < 1.15) -> edit protocol/p150u.yml accordingly" | tee -a "$SUMMARY"
  fi
else
  echo "VERDICT: inconclusive (missing data) -> keep bs=8 until a clean bench exists" | tee -a "$SUMMARY"
fi
echo "summary: $SUMMARY"
