#!/usr/bin/env python3
"""
Evaluate one checkpoint on the val set and dump a small JSON.
Thin wrapper around the existing engine; NO test-dev, NO training.

    python scripts/eval_final.py -c configs/ecdet/ecdet_s_p30u.yml -d xpu \
        -r outputs/p30u/e1_s42/final.pth -o /tmp/eval.json --weights ema

--weights ema : evaluate the EMA weights (`ckpt['ema'].module`)
--weights raw : evaluate the raw model weights (`ckpt['model']`)
--weights both: emit a JSON list with both (manual use)

Output object keys: weights_type, epoch, ckpt, AP, AP50, AP75, APS, APM, APL, params.
"""
import argparse
import json
import os
import sys


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    project = os.path.abspath(os.path.join(here, ".."))            # ecdetseg/
    if project not in sys.path:
        sys.path.insert(0, project)

    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-r", "--resume", required=True)
    ap.add_argument("-d", "--device", default="xpu")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--weights", choices=["ema", "raw", "both"], default="ema")
    ap.add_argument("-u", "--update", nargs="+", default=[])
    args = ap.parse_args()

    from engine.core import YAMLConfig, yaml_utils
    from engine.misc import dist_utils
    from engine.solver import TASKS
    from engine.solver.ec_engine import evaluate

    update = yaml_utils.parse_cli(args.update)
    cfg = YAMLConfig(args.config, resume=args.resume, device=args.device, **update)
    # resume supplies the backbone; skip the standalone ViT weights (same as train.py)
    if "ViTAdapter" in cfg.yaml_cfg:
        cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()
    epoch = getattr(solver, "last_epoch", None)

    def _evaluate(module):
        stats, _ = evaluate(module, solver.criterion, solver.postprocessor,
                            solver.val_dataloader, solver.evaluator, solver.device)
        s = stats.get("coco_eval_bbox") or stats.get("coco_eval_mask")
        if s is None:
            raise RuntimeError("no coco_eval stats produced")
        return s

    which = ["ema", "raw"] if args.weights == "both" else [args.weights]
    results = []
    for w in which:
        if w == "ema":
            module = solver.ema.module if solver.ema else solver.model
        else:
            module = dist_utils.de_parallel(solver.model)
        s = _evaluate(module)
        results.append({
            "weights_type": w,
            "epoch": epoch,
            "AP": s[0], "AP50": s[1], "AP75": s[2],
            "APS": s[3], "APM": s[4], "APL": s[5],
            "params": int(sum(p.numel() for p in module.parameters())),
            "ckpt": os.path.abspath(args.resume),
        })

    with open(args.out, "w") as f:
        json.dump(results[0] if len(results) == 1 else results, f, indent=2)
    print("[eval_final] weights=%s -> %s" % (args.weights, args.out))
    for r in results:
        print("    %-3s AP=%.4f AP50=%.4f AP75=%.4f (epoch=%s)" %
              (r["weights_type"], r["AP"], r["AP50"], r["AP75"], r["epoch"]))


if __name__ == "__main__":
    main()
