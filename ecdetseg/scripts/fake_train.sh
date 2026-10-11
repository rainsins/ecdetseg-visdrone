#!/usr/bin/env bash
# ============================================================
# Fake "train.py" used by scripts/test_run_exp.sh to exercise the
# guardian (retry / progress / DONE / signals) WITHOUT torch or XPU.
#
# It is invoked by run_exp.sh via PYTHON_BIN and receives the same
# arguments train.py would. Behaviour is driven by env vars:
#   FAKE_CODES        comma list of exit codes per attempt (default 1,1,0)
#   FAKE_NO_PROGRESS  "1" -> do not create a checkpoint (simulate stuck/OOM)
#   FAKE_SLEEP        seconds to sleep (simulate a long run; for signal tests)
# ============================================================
set -u

echo "FAKE train.py invoked with: $*"

# a real config dump is not needed; just satisfy any reader
echo "FAKE: pretend config parsed"

if [ -n "${FAKE_SLEEP:-}" ]; then
  echo "FAKE: sleeping ${FAKE_SLEEP}s (signal test)"
  sleep "$FAKE_SLEEP"
  echo "FAKE: woke up"
  exit 0
fi

: "${OUT_DIR:?fake_train: OUT_DIR must be exported by run_exp.sh}"
state="$OUT_DIR/.fake_attempts"
n=0; [ -f "$state" ] && n="$(cat "$state")"
n=$((n + 1)); echo "$n" > "$state"

if [ "${FAKE_NO_PROGRESS:-0}" != "1" ]; then
  mkdir -p "$OUT_DIR"
  : > "$OUT_DIR/checkpoint$(printf '%04d' "$n").pth"    # progress signal
  echo "FAKE: wrote checkpoint$(printf '%04d' "$n").pth"
fi

codes="${FAKE_CODES:-1,1,0}"
IFS=',' read -r -a arr <<< "$codes"
idx=$((n - 1)); [ "$idx" -ge "${#arr[@]}" ] && idx=$(( ${#arr[@]} - 1 ))
code="${arr[$idx]}"
echo "FAKE: attempt $n -> exit $code"
if [ "$code" = "0" ]; then
  echo "Training time 0:00:00.00"
fi
exit "$code"
