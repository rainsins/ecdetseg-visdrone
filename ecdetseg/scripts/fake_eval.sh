#!/usr/bin/env bash
# Fake eval_final.py for scripts/test_run_exp.sh — writes a dummy JSON to -o.
set -u
out=""; weights="ema"
while [ $# -gt 0 ]; do
  case "$1" in
    -o)        out="$2"; shift 2 ;;
    --weights) weights="$2"; shift 2 ;;
    *)         shift ;;
  esac
done
[ -n "$out" ] && printf '{"AP":0.1,"AP50":0.2,"AP75":0.05,"APS":0.01,"APM":0.1,"APL":0.2,"params":1,"epoch":3,"weights_type":"%s","ckpt":"fake"}\n' "$weights" > "$out"
exit 0
