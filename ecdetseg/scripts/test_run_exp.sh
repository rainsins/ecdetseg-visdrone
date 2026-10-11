#!/usr/bin/env bash
# ============================================================
# Static tests for scripts/run_exp.sh — NO torch, NO XPU, NO dataset.
# Drives the guardian with scripts/fake_train.sh (PYTHON_BIN) and checks:
#   T1 retry -> progress -> DONE
#   T2 no-progress -> stop (exit 3)
#   T3 DONE marker -> short-circuit (no launch)
#   T4 SIGTERM -> forwarded to child, no retry, exit 130
# ============================================================
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="$HERE/run_exp.sh"
FAKE="$HERE/fake_train.sh"
OUTROOT="${P30U_TEST_OUTROOT:-/tmp/opencode/p30u_test}"

export VISDRONE_ROOT="${VISDRONE_ROOT:-/tmp/fake_visdrone}"
export P30U_OUT_ROOT="$OUTROOT"
export PYTHON_BIN="$FAKE"
export RETRY_WAIT=0
export FAKE_CKPT_OK=1
export SKIP_FINAL_EVAL=1          # never hit pixi/torch in tests

pass=0; fail=0
ok()   { echo "  PASS: $1"; pass=$((pass+1)); }
bad()  { echo "  FAIL: $1"; fail=$((fail+1)); }

reset() { rm -rf "$OUTROOT"; unset FAKE_CODES FAKE_NO_PROGRESS FAKE_SLEEP; }

echo "== T1: retry -> progress -> DONE =="
reset; export FAKE_CODES=1,1,0
bash "$RUN" e1 42 >/tmp/opencode/t1.out 2>&1; rc=$?
[ "$rc" -eq 0 ] && ok "exit 0" || bad "exit=$rc (want 0)"
[ -f "$OUTROOT/e1_s42/DONE" ] && ok "DONE written" || bad "DONE missing"
[ "$(grep -c 'FAKE train.py invoked' "$OUTROOT/e1_s42/train.log")" -eq 3 ] \
  && ok "3 attempts logged" || bad "unexpected attempt count"
[ -f "$OUTROOT/e1_s42/snapshot/config.yml" ] && ok "snapshot config" || bad "snapshot missing"
[ -f "$OUTROOT/e1_s42/snapshot/git_commit.txt" ] && ok "snapshot git commit" || bad "git snapshot missing"

echo "== T2: no-progress -> stop =="
reset; export FAKE_CODES=1,1,1,1; export FAKE_NO_PROGRESS=1
bash "$RUN" e1 43 >/tmp/opencode/t2.out 2>&1; rc=$?
[ "$rc" -eq 3 ] && ok "exit 3" || bad "exit=$rc (want 3)"
[ ! -f "$OUTROOT/e1_s43/DONE" ] && ok "no DONE" || bad "DONE should not exist"
[ "$(grep -c 'FAKE train.py invoked' "$OUTROOT/e1_s43/train.log")" -eq 3 ] \
  && ok "stopped after 3 attempts" || bad "attempt count != 3"

echo "== T3: DONE short-circuit =="
reset; mkdir -p "$OUTROOT/e1_s44"; : > "$OUTROOT/e1_s44/DONE"
bash "$RUN" e1 44 >/tmp/opencode/t3.out 2>&1; rc=$?
[ "$rc" -eq 0 ] && ok "exit 0" || bad "exit=$rc (want 0)"
[ ! -f "$OUTROOT/e1_s44/.fake_attempts" ] && ok "trainer not launched" || bad "trainer was launched"

echo "== T4: SIGTERM forwarding, no retry =="
reset; export FAKE_SLEEP=30
bash "$RUN" e1 45 >/tmp/opencode/t4.out 2>&1 &
pid=$!
sleep 3
kill -TERM "$pid" 2>/dev/null
wait "$pid"; rc=$?
[ "$rc" -eq 130 ] && ok "exit 130" || bad "exit=$rc (want 130)"
[ ! -f "$OUTROOT/e1_s45/DONE" ] && ok "no DONE" || bad "DONE should not exist"
[ "$(grep -c 'FAKE train.py invoked' "$OUTROOT/e1_s45/train.log")" -eq 1 ] \
  && ok "no retry (1 launch)" || bad "retried after signal"

echo
echo "== summary: $pass passed, $fail failed =="
[ "$fail" -eq 0 ]
