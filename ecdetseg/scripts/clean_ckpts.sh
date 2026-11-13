#!/usr/bin/env bash
# ============================================================
# Checkpoint pruner for long schedules (P150u: 150 epochs, checkpoint_freq=1).
# Keeps, per run dir:
#   * the newest K `checkpoint*.pth`   (default K=3, rolling snapshots)
#   * ALL of: last.pth, final.pth, best.pth, best.pth.info
#   * snapshot/, log.txt, train.log, eval*, *.json, *.png  (untouched)
# Deletes older `checkpoint*.pth` only.
#
#   bash scripts/clean_ckpts.sh outputs/p150u            # all run dirs under it
#   bash scripts/clean_ckpts.sh outputs/p150u/e1_s42 3   # one dir, keep 3
#   DRY=1 bash scripts/clean_ckpts.sh outputs/p150u      # print only
# ============================================================
set -u -o pipefail

TARGET="${1:?usage: clean_ckpts.sh <run_dir_or_root> [keep=3]}"
K="${2:-3}"
DRY="${DRY:-0}"

clean_one() {
  local d="$1"
  [ -d "$d" ] || return 0
  local -a cks=()
  while IFS= read -r f; do cks+=( "$f" ); done < <(ls -1 "$d"/checkpoint*.pth 2>/dev/null | sort)
  local n=${#cks[@]}
  [ "$n" -le "$K" ] && return 0
  local del=$((n - K)) i
  for ((i = 0; i < del; i++)); do
    if [ "$DRY" = "1" ]; then echo "would rm ${cks[$i]}"; else rm -f "${cks[$i]}"; echo "rm ${cks[$i]}"; fi
  done
}

# A "run dir" has log.txt or DONE; otherwise treat TARGET as a root of run dirs.
if [ -f "$TARGET/log.txt" ] || [ -f "$TARGET/DONE" ]; then
  clean_one "$TARGET"
else
  for d in "$TARGET"/*/; do clean_one "${d%/}"; done
fi
