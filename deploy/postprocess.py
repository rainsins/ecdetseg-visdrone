#!/usr/bin/env python3
"""Numpy postprocessing for ECDet-S, shared by simulator and board.

The ONNX/RKNN graph deliberately ends at the raw decoder outputs:

    pred_logits : float32 [1, 300, 10]   (pre-sigmoid class logits)
    pred_boxes  : float32 [1, 300, 4]    (normalised cxcywh)

Why keep topk/sort/gather *outside* the graph?
  * RKNN's INT8 compiler handles dynamic TopK/Sort poorly, and the exported
    graph would produce variable-length outputs that many board runtimes cannot
    consume directly.
  * COCO evaluation needs 300 ranked detections per image, but a board product
    usually needs a score threshold and NMS-free top-k.  Keeping the selection in
    numpy makes that policy tunable without re-converting the model.
  * Most importantly, the graph stays a pure function of the image, so its
    quantisation error can be diagnosed layer by layer (accuracy_analysis.py).

Box geometry is mapped back to the source with the PyTorch-compatible stretch
rule: multiply normalised coordinates by ``orig_w``/``orig_h``.  The metadata's
pad/scale fields are neutral and intentionally unused.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deploy import NUM_CLASSES, NUM_QUERIES, setup_logging  # noqa: E402


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid (avoids overflow for large negative logits)."""
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def _squeeze_batch(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim >= 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"{name}: expected [1,N,D] or [N,D], got shape={arr.shape}")


def split_outputs(
    outputs: Sequence[np.ndarray],
    num_classes: int = NUM_CLASSES,
    output_names: Sequence[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Identify logits and boxes from RKNN's output list.

    RKNN does not promise the same output ordering as ONNX.  We therefore match
    by the last dimension (num_classes vs 4).  ``output_names`` is accepted for
    logging/forward compatibility but is intentionally not required.
    """
    arrays = [np.asarray(o) for o in outputs]
    if len(arrays) < 2:
        raise ValueError(f"RKNN returned {len(arrays)} outputs, expected >=2: {[a.shape for a in arrays]}")

    # num_classes == 4 makes the "last dim is 4" test match both tensors, so the
    # shape heuristic cannot decide.  Refuse instead of picking arbitrarily:
    # a silently swapped logits/boxes pair still produces plausible-looking but
    # wrong boxes, which is far worse than a loud failure.
    if num_classes == 4:
        raise ValueError(
            "split_outputs cannot disambiguate logits from boxes when num_classes == 4 "
            f"(shapes={[a.shape for a in arrays]}). Pass the tensors explicitly or extend "
            "this function with a structural discriminator."
        )

    logits_candidates = [a for a in arrays if a.ndim >= 2 and a.shape[-1] == num_classes]
    boxes_candidates = [a for a in arrays if a.ndim >= 2 and a.shape[-1] == 4]

    if len(logits_candidates) == 1 and len(boxes_candidates) == 1:
        return logits_candidates[0], boxes_candidates[0]

    if len(logits_candidates) == 1 and len(boxes_candidates) > 1:
        raise ValueError(
            f"multiple tensors look like boxes (last dim 4): {[a.shape for a in boxes_candidates]}. "
            "Cannot pick one unambiguously."
        )

    # Fallback: trust the exporter order (logits, boxes) but say so loudly.
    if len(arrays) == 2:
        logging.getLogger("postprocess").warning(
            "Could not identify outputs by shape %s; falling back to exporter order (logits, boxes).",
            [a.shape for a in arrays],
        )
        return arrays[0], arrays[1]

    raise ValueError(
        "Cannot identify logits/boxes. Shapes="
        + str([a.shape for a in arrays])
        + f", num_classes={num_classes}, names={list(output_names) if output_names else None}"
    )


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32)
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return np.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis=-1)


def invert_letterbox_boxes(
    boxes_xyxy_norm: np.ndarray,
    meta: Mapping[str, Any],
    clip: bool = True,
) -> np.ndarray:
    """Map normalised xyxy boxes from the 640 stretch canvas to the source.

    Stretch preprocessing means there is no padding and no per-axis letterbox
    scale: ``x_orig = x_norm * orig_w`` and ``y_orig = y_norm * orig_h``.  This
    is exactly what the PyTorch reference postprocessor does (multiply by
    ``orig_target_sizes``).

    The function name is kept for API compatibility with existing callers; the
    metadata's ``pad_x``/``pad_y``/``scale_x``/``scale_y`` fields are now
    neutral and intentionally ignored.

    ``clip`` is on by default because pixels outside the source image are not
    observable; pass ``--no-clip`` to reproduce the raw PyTorch postprocessor.
    """
    boxes = np.asarray(boxes_xyxy_norm, dtype=np.float32).copy()
    orig_w = float(meta["orig_w"])
    orig_h = float(meta["orig_h"])

    boxes[..., [0, 2]] *= orig_w
    boxes[..., [1, 3]] *= orig_h

    if clip:
        boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], 0.0, orig_w)
        boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], 0.0, orig_h)

    # Guard against numerically inverted boxes after clipping.
    x1 = np.minimum(boxes[..., 0], boxes[..., 2])
    y1 = np.minimum(boxes[..., 1], boxes[..., 3])
    x2 = np.maximum(boxes[..., 0], boxes[..., 2])
    y2 = np.maximum(boxes[..., 1], boxes[..., 3])
    boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] = x1, y1, x2, y2
    return boxes


def postprocess(
    logits: np.ndarray,
    boxes: np.ndarray,
    meta: Mapping[str, Any],
    num_classes: int = NUM_CLASSES,
    num_queries: int = NUM_QUERIES,
    score_thresh: float = 0.0,
    max_det: int = NUM_QUERIES,
    clip: bool = True,
) -> Dict[str, np.ndarray]:
    """Sigmoid + flattened top-k + cxcywh->xyxy + stretch inverse.

    The ranking matches the official ECDet/D-FINE ``PostProcessor``: flatten
    ``[queries, classes]``, sigmoid, take the global top-k, then decode
    ``label = index % num_classes`` and ``query = index // num_classes``.
    This means one query may appear under several classes, exactly as in the
    PyTorch reference evaluation.
    """
    logits = _squeeze_batch(logits, "pred_logits")
    boxes = _squeeze_batch(boxes, "pred_boxes")

    # The simulator and rknn_lite both dequantize outputs to float32.  An int8
    # output here would mean a quantized-output deployment, which needs scale/
    # zero-point handling that this pipeline intentionally does not implement.
    if not np.issubdtype(logits.dtype, np.floating) or not np.issubdtype(boxes.dtype, np.floating):
        raise ValueError(
            f"decoder outputs must be float32, got logits={logits.dtype}, boxes={boxes.dtype}. "
            "Disable quantized outputs or dequantize with the RKNN output scale/zero_point first."
        )

    if logits.shape[-1] != num_classes:
        raise ValueError(f"pred_logits last dim {logits.shape[-1]} != num_classes {num_classes}")
    if boxes.shape[-1] != 4:
        raise ValueError(f"pred_boxes last dim {boxes.shape[-1]} != 4")
    if logits.shape[0] != boxes.shape[0]:
        raise ValueError(f"query mismatch: logits {logits.shape}, boxes {boxes.shape}")
    if not np.isfinite(logits).all() or not np.isfinite(boxes).all():
        n_bad = int((~np.isfinite(logits)).sum() + (~np.isfinite(boxes)).sum())
        raise ValueError(f"non-finite values in decoder outputs ({n_bad} elements); check quantisation")

    scores_flat = sigmoid(logits).reshape(-1)
    k = int(min(max_det, scores_flat.size))
    # Stable top-k: descending score, ties broken by flat index for determinism.
    order = np.argsort(-scores_flat, kind="stable")[:k]
    scores = scores_flat[order]
    labels = (order % num_classes).astype(np.int64)
    query_idx = (order // num_classes).astype(np.int64)

    if score_thresh > 0:
        keep = scores >= float(score_thresh)
        scores = scores[keep]
        labels = labels[keep]
        query_idx = query_idx[keep]

    selected = boxes[query_idx].astype(np.float32)
    boxes_xyxy_norm = cxcywh_to_xyxy(selected)
    boxes_orig = invert_letterbox_boxes(boxes_xyxy_norm, meta, clip=clip)

    # Drop degenerate boxes *after* inversion; COCO's loadRes rejects negative
    # or zero-area boxes in some pycocotools versions.
    w = boxes_orig[..., 2] - boxes_orig[..., 0]
    h = boxes_orig[..., 3] - boxes_orig[..., 1]
    valid = (w > 1e-6) & (h > 1e-6)
    return {
        "boxes_xyxy": boxes_orig[valid],
        "scores": scores[valid].astype(np.float32),
        "labels": labels[valid].astype(np.int64),
    }


def to_coco_detections(
    detections: Mapping[str, np.ndarray],
    image_id: int,
    score_thresh: float = 0.0,
) -> List[Dict[str, Any]]:
    """Convert decoded xyxy detections into COCO result records (xywh)."""
    boxes = np.asarray(detections["boxes_xyxy"], dtype=np.float32)
    scores = np.asarray(detections["scores"], dtype=np.float32)
    labels = np.asarray(detections["labels"], dtype=np.int64)

    records: List[Dict[str, Any]] = []
    for box, score, label in zip(boxes, scores, labels):
        if float(score) < score_thresh:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        w, h = x2 - x1, y2 - y1
        if w <= 1e-6 or h <= 1e-6:
            continue
        records.append(
            {
                "image_id": int(image_id),
                "category_id": int(label),
                "bbox": [x1, y1, w, h],
                "score": float(score),
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone numpy postprocess for ECDet-S outputs")
    parser.add_argument("--logits", required=True, help=".npy file, [1,300,10] or [300,10]")
    parser.add_argument("--boxes", required=True, help=".npy file, [1,300,4] or [300,4]")
    parser.add_argument("--meta", required=True, help="preprocess metadata json")
    parser.add_argument("--image-id", type=int, default=0)
    parser.add_argument("--output", required=True, help="output COCO detection json (a list)")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--score-thresh", type=float, default=0.0)
    parser.add_argument("--no-clip", action="store_true", help="do not clip boxes to the source image")
    args = parser.parse_args()

    logger = setup_logging(name="postprocess")
    logits = np.load(args.logits)
    boxes = np.load(args.boxes)
    with open(args.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    dets = postprocess(
        logits,
        boxes,
        meta,
        num_classes=args.num_classes,
        score_thresh=args.score_thresh,
        clip=not args.no_clip,
    )
    records = to_coco_detections(dets, args.image_id, score_thresh=args.score_thresh)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")
    logger.info(
        "wrote %d detections to %s (score range %.4f..%.4f)",
        len(records),
        out,
        float(dets["scores"].min()) if len(dets["scores"]) else -1,
        float(dets["scores"].max()) if len(dets["scores"]) else -1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
