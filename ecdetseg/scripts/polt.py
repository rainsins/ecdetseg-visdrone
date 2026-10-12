#!/usr/bin/env python3
"""Plot training curves from a P30u run's log.txt.

Usage:
    python scripts/polt.py [RUN_DIR]        # default: outputs/p30u/e1_s42
    python scripts/polt.py outputs/p30u/e2_s42
    python scripts/polt.py /abs/path/to/run

RUN_DIR may be absolute, or relative to the ecdetseg project root (resolved from
this script's location), so the result no longer depends on the current directory.
Writes <RUN_DIR>/dfine_curve.png.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                 # headless-safe
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent      # ecdetseg/


def main():
    ap = argparse.ArgumentParser(description="Plot curves from a run's log.txt")
    ap.add_argument("run_dir", nargs="?", default="outputs/p30u/e1_s42",
                    help="run output dir (absolute, or relative to ecdetseg/)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir

    log_path = run_dir / "log.txt"
    if not log_path.is_file():
        raise SystemExit(f"log.txt not found: {log_path}")

    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):      # skip any non-JSON header
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        raise SystemExit(f"no JSON records found in {log_path}")

    ep = [d["epoch"] for d in records]
    lr = [d.get("train_lr") for d in records]
    loss = [d.get("train_loss") for d in records]
    coco = [d["test_coco_eval_bbox"] for d in records]
    AP = [c[0] for c in coco]             # mAP@0.5:0.95
    AP50 = [c[1] for c in coco]
    APs = [c[3] for c in coco]            # small-object AP

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(ep, AP, label="mAP@0.5:0.95")
    ax1.plot(ep, AP50, label="mAP@0.5")
    ax1.plot(ep, APs, label="AP-small")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("AP"); ax1.legend(); ax1.grid(alpha=0.3)
    ax1.set_title("Validation AP")

    ax2.plot(ep, loss, color="tab:red", label="train_loss")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("train_loss", color="tab:red")
    ax3 = ax2.twinx()
    ax3.plot(ep, lr, color="tab:blue", label="lr")
    ax3.set_ylabel("lr", color="tab:blue")
    ax2.set_title("Loss & LR schedule")

    out = run_dir / "dfine_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"read : {log_path}")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
