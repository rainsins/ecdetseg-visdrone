#!/usr/bin/env bash
# ============================================================
# P30u experiment launcher / guardian  (EdgeCrafter ecdetseg)
#
#   run_exp.sh <init: e1|e2|e3> <seed> [extra -u overrides...]
#
# Responsibilities:
#   * one place for the E1/E2/E3 init table
#   * fixed output dir  outputs/p30u/<init>_s<seed>/
#   * start snapshot (config dump, git hash/diff, pixi.lock hash, cmd, time)
#   * crash-resilient resume  (last.pth -> newest good checkpoint -> cold)
#   * progress-aware retry    (stop after N attempts with no new checkpoint)
#   * SIGINT/SIGTERM forwarding (Ctrl-C / `pm2 stop`) with no retry
#   * merged train.log + crash tail dump
#   * final EMA evaluation on val -> eval.json (NO test-dev)
#
# Env (required/optional):
#   VISDRONE_ROOT   required; dataset root (paths are injected via -u)
#   WEIGHTS_DIR     optional; dir holding ecdet_s.pth / ecdet_s_o3652coco.pth (default: ecdetseg/)
#   P30U_RES        optional; input resolution (default 640)
#   NO_PROGRESS_LIMIT, RETRY_WAIT  optional tunables
#   PYTHON_BIN      optional test hook: replace `pixi run python train.py`
#   FAKE_CKPT_OK, FAKE_*  test hooks (see scripts/fake_train.sh)
# ============================================================
set -u -o pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run_exp.sh <e1|e2|e3> <seed> [extra -u key=value ...]
  e1 : cold start, no tuning checkpoint (backbone from ViTAdapter.weights_path)
  e2 : -t $WEIGHTS_DIR/ecdet_s.pth
  e3 : -t $WEIGHTS_DIR/ecdet_s_o3652coco.pth
Env: VISDRONE_ROOT (required), WEIGHTS_DIR, P30U_RES, NO_PROGRESS_LIMIT, RETRY_WAIT
EOF
}

[ $# -ge 2 ] || { usage; exit 2; }
INIT="$1"; SEED="$2"; shift 2
EXTRA=( "$@" )

case "$INIT" in e1|e2|e3) : ;; *) echo "run_exp: bad init '$INIT'" >&2; usage; exit 2 ;; esac
case "$SEED" in ''|*[!0-9]*) echo "run_exp: seed must be a non-negative integer" >&2; exit 2 ;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"            # ecdetseg/
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"             # pixi.toml lives here
MANIFEST="$REPO_ROOT/pixi.toml"
GIT_ROOT="$(cd "$REPO_ROOT" && git rev-parse --show-toplevel 2>/dev/null || echo "$REPO_ROOT")"

: "${VISDRONE_ROOT:?run_exp: set VISDRONE_ROOT to the VisDrone dataset dir}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$PROJECT_DIR}"
CONFIG="${P30U_CONFIG:-configs/ecdet/ecdet_s_p30u.yml}"
NO_PROGRESS_LIMIT="${NO_PROGRESS_LIMIT:-3}"
RETRY_WAIT="${RETRY_WAIT:-30}"

# ----- init table (single source of truth) -----
case "$INIT" in
  e1) TUNING="" ;;
  e2) TUNING="$WEIGHTS_DIR/ecdet_s.pth" ;;
  e3) TUNING="$WEIGHTS_DIR/ecdet_s_o3652coco.pth" ;;
esac

OUT_ROOT="${P30U_OUT_ROOT:-$PROJECT_DIR/outputs/p30u}"
OUT_DIR="$OUT_ROOT/${INIT}_s${SEED}"
export OUT_DIR                       # visible to train hook / fake_train.sh
TRAIN_LOG="$OUT_DIR/train.log"
DONE_FILE="$OUT_DIR/DONE"
LAST_CKPT="$OUT_DIR/last.pth"
mkdir -p "$OUT_DIR" "$OUT_DIR/snapshot"

# ----- XPU env -----
export PYTORCH_SYCL_ALLOC_CONF="${PYTORCH_SYCL_ALLOC_CONF:-expandable_segments:True}"
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS="${SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS:-0}"
# NOTE (audit.md Q11): on torch 2.9.1+xpu PYTORCH_SYCL_ALLOC_CONF is NOT referenced by
#   torch (only PYTORCH_CUDA_ALLOC_CONF exists) -> currently a no-op, kept for
#   forward-compat; SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS IS recognised
#   (consumed by libur_adapter_level_zero.so).

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$TRAIN_LOG"; }

# ----- overrides passed to train.py (-u) -----
OVERRIDES=()
OVERRIDES+=( "output_dir=$OUT_DIR" "seed=$SEED" )
OVERRIDES+=( "train_dataloader.dataset.img_folder=$VISDRONE_ROOT/train" )
OVERRIDES+=( "train_dataloader.dataset.ann_file=$VISDRONE_ROOT/annotations_VisDrone_train.json" )
OVERRIDES+=( "val_dataloader.dataset.img_folder=$VISDRONE_ROOT/val" )
OVERRIDES+=( "val_dataloader.dataset.ann_file=$VISDRONE_ROOT/annotations_VisDrone_val.json" )
[ "${P30U_RES:-640}" != "640" ] && OVERRIDES+=( "eval_spatial_size=[${P30U_RES}, ${P30U_RES}]" )
[ ${#EXTRA[@]} -gt 0 ] && OVERRIDES+=( "${EXTRA[@]}" )

# ----- one-time snapshot -----
snapshot() {
  local snap="$OUT_DIR/snapshot"
  [ -f "$snap/command.txt" ] && return 0            # keep the original snapshot
  ( cd "$PROJECT_DIR" && python scripts/dump_config.py -c "$CONFIG" -u "${OVERRIDES[@]}" ) \
      > "$snap/config.yml" 2> "$snap/config.dump.err"
  ( cd "$GIT_ROOT" && git rev-parse HEAD )            > "$snap/git_commit.txt" 2>&1
  ( cd "$GIT_ROOT" && git status --porcelain )        > "$snap/git_status.txt" 2>&1
  ( cd "$GIT_ROOT" && git diff )                      > "$snap/git_diff.patch" 2>&1
  if command -v sha256sum >/dev/null; then
    sha256sum "$REPO_ROOT/pixi.lock" | awk '{print $1}' > "$snap/pixi.lock.sha256" 2>/dev/null
  fi
  date '+%F %T%z'                                     > "$snap/start_time.txt"
  { printf 'init=%s seed=%s\n' "$INIT" "$SEED"
    printf 'config=%s\n' "$CONFIG"
    printf 'tuning=%s\n' "${TUNING:-<none>}"
    printf 'output_dir=%s\n' "$OUT_DIR"
    printf 'cmd='; printf '%q ' "$SCRIPT_DIR/run_exp.sh" "$INIT" "$SEED" "${EXTRA[@]:-}"; printf '\n'
    printf 'env: P30U_RES=%s PYTORCH_SYCL_ALLOC_CONF=%s SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=%s\n' \
      "${P30U_RES:-640}" "$PYTORCH_SYCL_ALLOC_CONF" "$SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"
  } > "$snap/command.txt"
}

# ----- checkpoint helpers -----
ckpt_loadable() {
  [ -f "$1" ] || return 1
  [ "${FAKE_CKPT_OK:-0}" = "1" ] && return 0
  ( cd "$PROJECT_DIR" && \
    pixi run --manifest-path "$MANIFEST" python - "$1" <<'PY'
import sys, torch
try:
    torch.load(sys.argv[1], map_location="cpu", weights_only=True)
except Exception as e:                       # noqa: BLE001
    print("ckpt load failed: %s" % e, file=sys.stderr)
    sys.exit(1)
PY
  )
}

find_resume_ckpt() {
  local c
  local -a cands=()
  [ -f "$LAST_CKPT" ] && cands+=( "$LAST_CKPT" )
  while IFS= read -r c; do cands+=( "$c" ); done \
    < <(ls -1 "$OUT_DIR"/checkpoint*.pth 2>/dev/null | sort -r)
  for c in "${cands[@]:-}"; do
    [ -z "$c" ] && continue
    if ckpt_loadable "$c"; then printf '%s\n' "$c"; return 0; fi
    log "resume: unusable checkpoint, skipping: $c"
  done
  return 1
}

progress_count() {   # number of *.pth newer than marker file $1
  [ -f "$1" ] || { echo 0; return; }
  find "$OUT_DIR" -maxdepth 1 -name '*.pth' -newer "$1" 2>/dev/null | wc -l
}

crash_tail() {
  local ts; ts="$(date '+%m%d_%H%M%S')"
  [ -f "$TRAIN_LOG" ] && tail -n 200 "$TRAIN_LOG" > "$OUT_DIR/crash_${ts}.log"
  log "crash tail saved: crash_${ts}.log"
}

# ----- signal handling -----
CHILD_PID=""
SIGNALED=0
forward_signal() {
  SIGNALED=1
  log "signal received; forwarding to training group '${CHILD_PID:-<none>}' (no retry)"
  if [ -n "$CHILD_PID" ] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill -INT "-$CHILD_PID" 2>/dev/null || kill -INT "$CHILD_PID" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$CHILD_PID" 2>/dev/null || break; sleep 1; done
    kill -TERM "-$CHILD_PID" 2>/dev/null || true
    wait "$CHILD_PID" 2>/dev/null || true
  fi
  exit 130
}
trap forward_signal INT TERM

build_cmd() {   # $@ = mode args (-r ckpt | -t ckpt | nothing)
  TRAIN_ARGS=( -c "$CONFIG" -d xpu "$@" -u "${OVERRIDES[@]}" )
  if [ -n "${PYTHON_BIN:-}" ]; then
    CMD=( "$PYTHON_BIN" "${TRAIN_ARGS[@]}" )
  else
    CMD=( pixi run --manifest-path "$MANIFEST" python train.py "${TRAIN_ARGS[@]}" )
  fi
}

run_training() {   # returns train exit code; forwards logs to train.log
  ( cd "$PROJECT_DIR" && exec setsid "${CMD[@]}" ) \
      > >(tee -a "$TRAIN_LOG") 2>&1 &
  CHILD_PID=$!
  wait "$CHILD_PID"; local rc=$?
  CHILD_PID=""
  return "$rc"
}

# ============================================================
snapshot
log "=== P30u guardian start | init=$INIT seed=$SEED out=$OUT_DIR ==="
log "config=$CONFIG tuning=${TUNING:-<none>} res=${P30U_RES:-640}"

if [ -f "$DONE_FILE" ]; then
  log "DONE marker already present -> experiment complete; refusing to resume. (delete $DONE_FILE to rerun)"
  exit 0
fi

no_prog=0
attempt=0
while :; do
  attempt=$((attempt + 1))
  MODE_ARGS=()
  if RESUME_CKPT="$(find_resume_ckpt)"; then
    MODE_ARGS=( -r "$RESUME_CKPT" )
    log "attempt #$attempt: resume from $RESUME_CKPT"
  elif [ -n "$TUNING" ] && [ -f "$TUNING" ]; then
    MODE_ARGS=( -t "$TUNING" )
    log "attempt #$attempt: cold start with -t $TUNING"
  else
    MODE_ARGS=()
    log "attempt #$attempt: cold start (${INIT}, no -t)"
  fi

  MARKER="$OUT_DIR/.attempt_marker"; : > "$MARKER"
  build_cmd "${MODE_ARGS[@]}"
  log "launch: ${CMD[*]}"
  run_training; rc=$?

  if [ "$SIGNALED" = "1" ]; then
    log "training interrupted by signal; exiting."
    exit 130
  fi

  new_ckpt="$(progress_count "$MARKER")"
  if [ "$new_ckpt" -gt 0 ]; then
    no_prog=0
    log "attempt #$attempt exit=$rc; progress detected ($new_ckpt new checkpoint file(s))."
  else
    no_prog=$((no_prog + 1))
    log "attempt #$attempt exit=$rc; NO new checkpoint (no_prog=$no_prog/$NO_PROGRESS_LIMIT)."
  fi

  if [ "$rc" -eq 0 ]; then
    if grep -q "Training time" "$TRAIN_LOG"; then
      log "training finished normally (exit 0, 'Training time' observed)."
    else
      log "training exited 0 (no explicit 'Training time' marker found; treating as complete)."
    fi
    : > "$DONE_FILE"
    log "wrote DONE marker."
    break
  fi

  crash_tail
  if [ "$no_prog" -ge "$NO_PROGRESS_LIMIT" ]; then
    log "ERROR: $NO_PROGRESS_LIMIT consecutive attempts with no progress -> stopping."
    log "       Likely OOM / deterministic crash. Inspect crash_*.log; lower batch size or P30U_RES."
    exit 3
  fi
  sleep "$RETRY_WAIT"
done

# ----- final evaluation (EMA on val; NO test-dev) -----
if [ -f "$DONE_FILE" ] && [ "${SKIP_FINAL_EVAL:-0}" != "1" ]; then
  log "=== final evaluation (val, EMA) ==="
  run_eval() {   # $1=ckpt $2=label $3=note
    local ckpt="$1" label="$2" note="$3" out="$OUT_DIR/eval_${label}.json"
    [ -f "$ckpt" ] || { log "eval[$label]: missing $ckpt"; return 0; }
    ( cd "$PROJECT_DIR" && \
      pixi run --manifest-path "$MANIFEST" python scripts/eval_final.py \
        -c "$CONFIG" -d xpu -r "$ckpt" -o "$out" \
        -u output_dir="$OUT_DIR" "${OVERRIDES[@]}" ) >> "$TRAIN_LOG" 2>&1
    if [ -f "$out" ]; then
      python - "$out" "$label" "$note" <<'PY' >> "$OUT_DIR/eval.jsonl"
import json, sys
d = json.load(open(sys.argv[1])); d["label"] = sys.argv[2]; d["note"] = sys.argv[3]
print(json.dumps(d))
PY
      log "eval[$label] -> $(basename "$out")"
    else
      log "eval[$label]: failed (see train.log)"
    fi
  }
  run_eval "$OUT_DIR/best.pth"  "best_ema" "EMA@best; best was SELECTED ON VAL -> reference only"
  run_eval "$OUT_DIR/last.pth"  "last_ema" "EMA@last; NOTE last.pth freezes at stop_epoch (see audit.md Q6)"
  log "final eval summary: $OUT_DIR/eval.jsonl"
fi

log "=== P30u guardian done ==="
