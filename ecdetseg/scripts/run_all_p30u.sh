#!/usr/bin/env bash
# ============================================================
# Sequential P30u batch runner (for pm2 / nohup).
# Single Arc GPU -> ONE experiment at a time. Order: seeds x inits.
#   SEEDS="42 43 44" INITS="e1 e2 e3" [CONTINUE_ON_FAIL=1] bash scripts/run_all_p30u.sh
# Forwards SIGTERM to the in-flight run_exp.sh (which forwards to training).
# ============================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_EXP="${RUN_EXP:-$HERE/run_exp.sh}"
SEEDS="${SEEDS:-42 43 44}"
INITS="${INITS:-e1 e2 e3}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-0}"

CURRENT=""
forward() {
  echo "[run_all] signal -> forwarding to in-flight run_exp (pid=${CURRENT:-none})"
  if [ -n "$CURRENT" ] && kill -0 "$CURRENT" 2>/dev/null; then
    kill -TERM "$CURRENT" 2>/dev/null || true   # TERM: SIGINT is ignored by async bg jobs
    wait "$CURRENT" 2>/dev/null || true
  fi
  exit 130
}
trap forward INT TERM

for seed in $SEEDS; do
  for init in $INITS; do
    echo "====== run_all: init=$init seed=$seed ======"
    bash "$RUN_EXP" "$init" "$seed" &
    CURRENT=$!
    rc=0; wait "$CURRENT" || rc=$?
    CURRENT=""
    echo "====== run_all: init=$init seed=$seed rc=$rc ======"
    if [ "$rc" -ne 0 ]; then
      if [ "$CONTINUE_ON_FAIL" != "1" ]; then
        echo "[run_all] stopping on first failure (rc=$rc); CONTINUE_ON_FAIL=1 to keep going"
        exit "$rc"
      fi
    fi
  done
done
echo "[run_all] all runs finished"
