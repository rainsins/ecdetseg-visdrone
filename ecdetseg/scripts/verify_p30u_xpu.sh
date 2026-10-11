#!/usr/bin/env bash
# ============================================================
# XPU smoke verification for P30u — RUN ONLY AFTER the live training ends.
# Refuses to run if any train.py is active (single 16G Arc → would OOM).
#
# For E1/E2/E3 it runs 2 epochs on a 64-image VisDrone subset and checks:
#   * weight loading differs (E1: ViT backbone msg; E2/E3: "Load model.state_dict, ...")
#   * scheduler line (flat_epochs / no_aug_epochs / warmup_iter) matches intent
#   * per-iter time from the log
#   * `-r` resume works on the produced last.pth
# Peak XPU memory: TODO needs a sampling hook (torch.xpu.max_memory_allocated)
#   or `xpu-smi` polling; not wired yet to avoid touching training code.
# ============================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/.." && pwd)"

if pgrep -af train.py >/dev/null 2>&1; then
  echo "ABORT: a train.py is running; do not touch XPU now." >&2
  pgrep -af train.py >&2
  exit 1
fi

: "${VISDRONE_ROOT:?verify: set VISDRONE_ROOT}"
export P30U_OUT_ROOT="${P30U_OUT_ROOT:-$PROJECT/outputs/p30u_verify}"
export RETRY_WAIT="${RETRY_WAIT:-5}"
export NO_PROGRESS_LIMIT="${NO_PROGRESS_LIMIT:-2}"
export SKIP_FINAL_EVAL=1

SUBSET="$P30U_OUT_ROOT/_subset"
mkdir -p "$SUBSET"
python "$HERE/make_subset.py" --ann "$VISDRONE_ROOT/annotations_VisDrone_train.json" \
       --out "$SUBSET/train64.json" --n 64
python "$HERE/make_subset.py" --ann "$VISDRONE_ROOT/annotations_VisDrone_val.json" \
       --out "$SUBSET/val64.json" --n 64

VOPTS=( "epochs=2" "stop_epoch=2" "flat_epoch=1"
        "train_dataloader.dataset.ann_file=$SUBSET/train64.json"
        "val_dataloader.dataset.ann_file=$SUBSET/val64.json" )

for init in e1 e2 e3; do
  echo "==================== verify $init ===================="
  rm -rf "$P30U_OUT_ROOT/${init}_s0"
  bash "$HERE/run_exp.sh" "$init" 0 -u "${VOPTS[@]}"
  log="$P30U_OUT_ROOT/${init}_s0/train.log"
  echo "-- scheduler:      $(grep -m1 'FlatCosineLRScheduler with' "$log" || echo NA)"
  echo "-- ViT backbone:   $(grep -m1 'Pretrained ViT weights loaded' "$log" || echo '(none)')"
  echo "-- tuned load:     $(grep -m1 'Load model.state_dict,' "$log" || echo '(none)')"
  echo "-- per-iter time:  $(grep -m1 's / it' "$log" || echo NA)"

  echo "-- resume check (delete DONE, rerun -> expect -r last.pth) --"
  rm -f "$P30U_OUT_ROOT/${init}_s0/DONE"
  bash "$HERE/run_exp.sh" "$init" 0 -u "${VOPTS[@]}" >/dev/null 2>&1 || true
  grep -m1 -E "attempt #[0-9]+: resume from|Resume checkpoint from" "$P30U_OUT_ROOT/${init}_s0/train.log" \
    || echo "   (resume line not found)"
done

echo "==================== done ===================="
echo "Inspect logs under $P30U_OUT_ROOT for E1-vs-E2/E3 tuning-load differences."
