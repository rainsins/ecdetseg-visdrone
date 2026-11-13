#!/usr/bin/env python3
"""PyTorch reference evaluation with the *deployment* preprocessing.

Why this script exists
----------------------
This is the anchor for the quantisation-loss claim.  It must run the checkpoint
through exactly the same ``preprocess.py`` + ``postprocess.py`` as the
ONNX/RKNN path, otherwise the difference between the two numbers mixes geometry
with quantisation.

Historically the pipeline used letterbox here and stretch in training, which
produced a misleadingly large "quantisation" loss (23.76 vs 28.82).  Letterbox
is gone; both sides now use a direct stretch resize.

Both sides must also use the *same resampler*.  ``--preprocess stretch`` used to
call ``cv2.resize(INTER_LINEAR)`` while the deployment path used PIL bilinear;
on a 1360x765 VisDrone frame those differ on 87.7% of pixels (max 95/255), so
the flag silently produced a non-comparable number.  There is now one
implementation (``preprocess.py``, PIL) and ``stretch-cv2`` is kept only as an
explicitly-labelled diagnostic of that divergence.

It uses the repository's ECDet model but the deployment numpy postprocessor, so
the output is directly comparable with ``sim_eval.py``.

Examples
--------
    pixi run -e default python deploy/eval_pytorch.py \
        --config ecdetseg/configs/ecdet/ecdet_s_p150u.yml \
        --weights ecdetseg/outputs/p150u/e3_s42/best.pth

    # diagnostic: quantify how much cv2.INTER_LINEAR costs vs PIL bilinear
    pixi run -e default python deploy/eval_pytorch.py --preprocess stretch-cv2

The checkpoint is probed automatically (EMA / model / raw), exactly like
``export_onnx.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deploy import (  # noqa: E402
    INPUT_SIZE,
    MEAN_RGB_01,
    NUM_CLASSES,
    STD_RGB_01,
    Timer,
    add_repo_to_sys_path,
    default_repo_root,
    format_seconds,
    log_environment,
    resolve_data_root,
    resolve_ecdet_root,
    setup_logging,
    write_json,
)
from deploy.export_onnx import disable_lqe, probe_state_dict  # noqa: E402
from deploy.postprocess import postprocess, to_coco_detections  # noqa: E402
from deploy.preprocess import PreprocessMeta, preprocess_path  # noqa: E402
from deploy.sim_eval import evaluate_coco  # noqa: E402


def _pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


PREPROCESS_MODES = ("stretch", "stretch-cv2")


def _stretch_tensor_cv2(path: Path, size: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Diagnostic only: cv2.INTER_LINEAR stretch.

    NOT the deployment path and NOT comparable with the 28.82 anchor -- PIL
    bilinear differs on ~88% of pixels for a 1360x765 downscale.  Kept so the
    cost of the wrong resampler can be measured instead of guessed.
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cv2.imread returned None: {path}")
    h, w = bgr.shape[:2]
    resized = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.uint8)
    meta = PreprocessMeta(
        orig_w=w,
        orig_h=h,
        resized_w=size,
        resized_h=size,
        input_w=size,
        input_h=size,
        pad_x=0,
        pad_y=0,
        scale_x=size / float(w),
        scale_y=size / float(h),
        normalized=False,
    ).to_dict()
    return chw, meta


def _resolve_with_fallbacks(value: Path, bases: Sequence[Path]) -> Path:
    value = Path(value).expanduser()
    if value.is_absolute():
        return value.resolve()
    cwd_candidate = (Path.cwd() / value).resolve()
    if cwd_candidate.is_file() or cwd_candidate.is_dir():
        return cwd_candidate
    for base in bases:
        candidate = (Path(base) / value).resolve()
        if candidate.is_file() or candidate.is_dir():
            return candidate
    return cwd_candidate


def _load_model(config_path: Path, weights_path: Path, weights_key: str, ecdet_root: Path, keep_lqe: bool, logger: logging.Logger):
    add_repo_to_sys_path(ecdet_root)
    from engine.core import YAMLConfig  # noqa: E402

    cfg = YAMLConfig(str(config_path))
    if "ViTAdapter" in cfg.yaml_cfg:
        cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True
    model = cfg.model
    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    state, chosen, info = probe_state_dict(ckpt, preferred=weights_key)
    logger.info("checkpoint container=%s tensors=%d epoch=%s", chosen, len(state), info.get("epoch", "?"))
    model.load_state_dict(state, strict=True)
    model = model.deploy().eval()
    if keep_lqe:
        logger.warning("keeping LQE; this does NOT match the exported ONNX/RKNN graph")
    else:
        # Match export_onnx.py: deployment graphs use the raw score head.
        n_lqe = disable_lqe(model)
        if n_lqe:
            logger.info("disabled %d LQE layer(s) to match the deployment graph", n_lqe)
    return model, cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_repo_root())
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--weights-key", choices=["auto", "ema", "model", "raw"], default="auto")
    parser.add_argument(
        "--preprocess",
        choices=PREPROCESS_MODES,
        default="stretch",
        help="stretch = deployment path (PIL bilinear); stretch-cv2 = diagnostic only",
    )
    parser.add_argument("--ann", type=Path, default=None)
    parser.add_argument("--images", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--device", default="auto", help="auto | xpu | cuda | cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--score-thresh", type=float, default=0.0)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--keep-lqe", action="store_true", help="evaluate the original LQE branch")
    args = parser.parse_args()

    logger = setup_logging(name="pytorch_eval")
    log_environment(logger)

    repo_root = args.repo_root.expanduser().resolve()
    ecdet_root = resolve_ecdet_root(repo_root)
    config_path = (
        _resolve_with_fallbacks(args.config, [ecdet_root, ecdet_root / "configs" / "ecdet", repo_root])
        if args.config else (ecdet_root / "configs" / "ecdet" / "ecdet_s_p150u.yml").resolve()
    )
    weights_path = (
        _resolve_with_fallbacks(
            args.weights,
            [ecdet_root / "outputs" / "p150u" / "e3_s42", ecdet_root / "outputs", ecdet_root, repo_root],
        )
        if args.weights else (ecdet_root / "outputs" / "p150u" / "e3_s42" / "best.pth").resolve()
    )
    if args.weights is None and not weights_path.is_file():
        fallback = weights_path.with_name("best_ema.pth")
        if fallback.is_file():
            weights_path = fallback
    data_root = resolve_data_root(repo_root, args.data_root)
    ann_path = (
        _resolve_with_fallbacks(args.ann, [data_root, repo_root, Path.cwd()])
        if args.ann else (data_root / "annotations_VisDrone_val.json").resolve()
    )
    images_dir = (
        _resolve_with_fallbacks(args.images, [data_root, repo_root, Path.cwd()])
        if args.images else (data_root / "val").resolve()
    )
    out_path = (args.out or (repo_root / "deploy" / "artifacts" / f"eval_pytorch_{args.preprocess}.json")).expanduser().resolve()

    for p in (config_path, weights_path, ann_path):
        if not p.is_file():
            logger.error("missing required path: %s", p)
            return 2
    if not images_dir.is_dir():
        logger.error("val image directory not found: %s", images_dir)
        return 2

    device = _pick_device(args.device)
    logger.info("config=%s", config_path)
    logger.info("weights=%s", weights_path)
    logger.info("ann=%s", ann_path)
    logger.info("images=%s", images_dir)
    logger.info("preprocess=%s device=%s", args.preprocess, device)

    try:
        model, cfg = _load_model(
            config_path, weights_path, args.weights_key, ecdet_root, args.keep_lqe, logger
        )
        model = model.to(device)
    except Exception:
        logger.error("model build/load failed")
        traceback.print_exc()
        return 1

    eval_size = int(cfg.yaml_cfg.get("eval_spatial_size", [INPUT_SIZE, INPUT_SIZE])[0])
    if eval_size != INPUT_SIZE:
        logger.warning("config eval_spatial_size=%s; using %s", eval_size, eval_size)

    coco = json.load(open(ann_path, "r", encoding="utf-8"))
    images = sorted(coco["images"], key=lambda x: int(x["id"]))
    if args.limit and args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise ValueError("no val images selected")

    mean = torch.tensor(MEAN_RGB_01, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD_RGB_01, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    detections: List[Dict[str, Any]] = []
    image_ids: List[int] = []
    start = time.perf_counter()

    with torch.inference_mode():
        for idx, im in enumerate(images, start=1):
            path = images_dir / im["file_name"]
            if args.preprocess == "stretch-cv2":
                if idx == 1:
                    logger.warning(
                        "--preprocess stretch-cv2 uses cv2.INTER_LINEAR, which is NOT the "
                        "deployment resampler; this AP is a diagnostic, not a reference."
                    )
                chw, meta = _stretch_tensor_cv2(path, eval_size)
            else:
                pre = preprocess_path(path, size=eval_size)
                chw, meta = pre.chw, pre.meta_dict()

            x = torch.from_numpy(chw).to(device=device, dtype=torch.float32).unsqueeze(0) / 255.0
            x = (x - mean) / std
            out = model(x)
            logits = out["pred_logits"].detach().float().cpu().numpy()
            boxes = out["pred_boxes"].detach().float().cpu().numpy()
            det = postprocess(
                logits,
                boxes,
                meta,
                num_classes=NUM_CLASSES,
                score_thresh=args.score_thresh,
                max_det=args.max_det,
            )
            image_id = int(im["id"])
            detections.extend(to_coco_detections(det, image_id, score_thresh=args.score_thresh))
            image_ids.append(image_id)
            if idx % max(1, args.log_every) == 0 or idx == len(images):
                elapsed = time.perf_counter() - start
                rate = idx / elapsed if elapsed > 0 else 0.0
                eta = (len(images) - idx) / rate if rate > 0 else float("inf")
                logger.info(
                    "progress %d/%d (%.1f%%) dets=%d avg=%.1f ms/img eta=%s",
                    idx,
                    len(images),
                    100.0 * idx / len(images),
                    len(detections),
                    1000.0 * elapsed / idx,
                    format_seconds(eta),
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(detections, f, ensure_ascii=False)
        f.write("\n")

    with Timer("COCOeval (PyTorch)", logger):
        metrics = evaluate_coco(
            ann_path=ann_path,
            detections=detections,
            max_det=args.max_det,
            image_ids=image_ids,
            logger=logger,
        )

    summary = {
        "framework": "pytorch",
        "preprocess": args.preprocess,
        "device": str(device),
        "weights": str(weights_path),
        "ann": str(ann_path),
        "images": str(images_dir),
        "metrics": metrics,
        "detections": str(out_path),
    }
    summary_path = out_path.with_suffix(".summary.json")
    write_json(summary_path, summary)
    logger.info(
        "RESULT: PyTorch[%s] AP=%.2f AP50=%.2f AP75=%.2f APs=%.2f APm=%.2f APl=%.2f",
        args.preprocess,
        metrics["AP"],
        metrics["AP50"],
        metrics["AP75"],
        metrics["APs"],
        metrics["APm"],
        metrics["APl"],
    )
    logger.info("wrote %s and %s", out_path, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
