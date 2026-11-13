#!/usr/bin/env python3
"""Export ECDet-S (D-FINE decoder) to a static ONNX graph for RKNN.

What is exported
----------------
Only the raw decoder outputs::

    images (1,3,640,640) float32
        -> pred_logits (1,300,10) float32   # pre-sigmoid
        -> pred_boxes  (1,300,4)  float32   # normalised cxcywh

The final ``sigmoid / topk / box_convert / gather`` block is *not* exported.
That block is numpy in ``postprocess.py`` (see the rationale there).  Note that
D-FINE's internal encoder query-selection TopK is part of the decoder semantics
and therefore remains in the graph; it is not the final detection TopK.

Checkpoint probing
------------------
``best_ema.pth`` contains ``{"ema": {"module": state_dict, "updates": ...}}`` in
this repo, but different training revisions may save ``{"model": ...}`` or a bare
state dict.  ``probe_state_dict`` handles all three and prints the actual top
level keys when it cannot decide, instead of failing with a cryptic KeyError.

This script runs in the **train** environment (torch + onnx + onnxruntime).
"""

from __future__ import annotations

import argparse
import inspect
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deploy import (  # noqa: E402
    DEFAULT_META_NAME,
    DEFAULT_ONNX_NAME,
    INPUT_SIZE,
    MEAN_RGB_255,
    NUM_CLASSES,
    NUM_QUERIES,
    PYTORCH_REFERENCE,
    STD_RGB_255,
    VISDRONE_CLASSES,
    Timer,
    add_repo_to_sys_path,
    default_repo_root,
    ensure_files,
    log_environment,
    resolve_ecdet_root,
    setup_logging,
    write_json,
)

_KNOWN_PREFIXES = (
    "module.",
    "model.",
    "ema.",
    "ema.module.",
    "model.module.",
    "state_dict.",
)


def _strip_known_prefix(state: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
    """Remove a uniform wrapper prefix (DataParallel/EMA/model) from all keys."""
    keys = list(state.keys())
    for prefix in _KNOWN_PREFIXES:
        if keys and all(k.startswith(prefix) for k in keys):
            return {k[len(prefix) :]: v for k, v in state.items()}
    return state


def _looks_like_state_dict(obj: Any) -> bool:
    return isinstance(obj, dict) and len(obj) > 0 and all(torch.is_tensor(v) for v in obj.values())


def _state_from_object(obj: Any) -> Mapping[str, torch.Tensor]:
    if _looks_like_state_dict(obj):
        return obj
    if hasattr(obj, "state_dict") and callable(obj.state_dict):
        candidate = obj.state_dict()
        if _looks_like_state_dict(candidate):
            return candidate
    if isinstance(obj, dict):
        for key in ("module", "state_dict", "model"):
            if key in obj:
                return _state_from_object(obj[key])
    raise TypeError(f"cannot extract a state_dict from object of type {type(obj)!r}")


def probe_state_dict(
    checkpoint: Any,
    preferred: str = "auto",
) -> Tuple[Mapping[str, torch.Tensor], str, Dict[str, Any]]:
    """Return ``(state_dict, chosen_key, info)`` from a training checkpoint.

    ``preferred`` is one of ``auto``, ``ema``, ``model``, ``raw``.  ``auto``
    prefers EMA because the filename and the reported reference numbers are for
    best_ema.pth.
    """
    if not isinstance(checkpoint, dict):
        return _strip_known_prefix(_state_from_object(checkpoint)), "object", {}

    top_keys = list(checkpoint.keys())
    info: Dict[str, Any] = {"top_level_keys": top_keys}

    if _looks_like_state_dict(checkpoint):
        return _strip_known_prefix(checkpoint), "raw", info

    order: Sequence[str]
    if preferred == "auto":
        order = ("ema", "model", "state_dict")
    elif preferred == "raw":
        order = ("state_dict",)
    else:
        order = (preferred,)

    errors = []
    for key in order:
        if key not in checkpoint:
            errors.append(f"{key}: not present")
            continue
        try:
            state = _strip_known_prefix(_state_from_object(checkpoint[key]))
            info["chosen_container"] = key
            if isinstance(checkpoint[key], dict):
                info["chosen_container_keys"] = list(checkpoint[key].keys())
            if "epoch" in checkpoint:
                info["epoch"] = int(checkpoint["epoch"])
            return state, key, info
        except Exception as exc:  # noqa: BLE001 - report all probes
            errors.append(f"{key}: {exc}")

    # Last resort: a bare dict nested under an unexpected key.
    for key, value in checkpoint.items():
        try:
            if _looks_like_state_dict(value):
                return _strip_known_prefix(value), key, info
        except Exception:
            pass

    raise KeyError(
        "Cannot find a model state_dict in the checkpoint. "
        f"Top-level keys={top_keys}. Probe results={errors}. "
        "Use --weights-key ema|model|raw to select explicitly."
    )


class _NoLQE(nn.Module):
    """Drop-in replacement for D-FINE's LQE module during deployment.

    LQE adds a learned confidence correction computed from a TopK over the
    33 regression bins.  RKNN 2.5 quantises that Softmax/TopK path very badly
    (lqe_layers.3 cosine ~0.05 in accuracy_analysis), which collapses every
    score to ~0.01.  The upstream code itself notes "Lqe does not affect the
    performance here", so we export without it and keep the raw score head.
    This module preserves the two-argument call signature used by the decoder.
    """

    def forward(self, scores: torch.Tensor, pred_corners: torch.Tensor) -> torch.Tensor:
        return scores


def disable_lqe(model: nn.Module) -> int:
    """Replace every decoder LQE layer with :class:`_NoLQE`; return count."""
    replaced = 0
    decoder = getattr(model, "decoder", None)
    inner = getattr(decoder, "decoder", None)
    lqe_layers = getattr(inner, "lqe_layers", None)
    if lqe_layers is None:
        return 0
    for idx, layer in enumerate(lqe_layers):
        if not isinstance(layer, _NoLQE):
            lqe_layers[idx] = _NoLQE()
            replaced += 1
    return replaced


class _ClampScoreHead(nn.Module):
    """Clamp the encoder score head before the query-selection TopK.

    VisDrone contains frames whose FP32 encoder logits reach about -310.  If
    RKNN calibrates the score-head output range on those frames, the rare
    positive peaks (~+2.6) are clipped to 0 and every query-selection score
    collapses.  Clamping only the far background tail keeps the top-k ordering
    intact while giving the quantizer a bounded, deployment-friendly range.
    """

    def __init__(self, head: nn.Module, limit: float, input_limit: float = 0.0):
        super().__init__()
        self.head = head
        self.limit = float(limit)
        self.input_limit = float(input_limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_limit > 0:
            x = torch.clamp(x, min=-self.input_limit, max=self.input_limit)
        return torch.clamp(self.head(x), min=-self.limit, max=self.limit)


def wrap_score_head(model: nn.Module, limit: float, input_limit: float) -> int:
    if limit <= 0 and input_limit <= 0:
        return 0
    decoder = getattr(model, "decoder", None)
    head = getattr(decoder, "enc_score_head", None)
    if head is None or isinstance(head, _ClampScoreHead):
        return 0
    decoder.enc_score_head = _ClampScoreHead(head, limit, input_limit)
    return 1


class DetectorExport(nn.Module):
    """Thin tuple-output wrapper so ONNX has stable named outputs."""

    def __init__(self, model: nn.Module, score_clamp: float = 0.0):
        super().__init__()
        self.model = model
        self.score_clamp = float(score_clamp)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(images)
        if not isinstance(outputs, dict) or "pred_logits" not in outputs or "pred_boxes" not in outputs:
            raise RuntimeError(
                "ECDET deploy model must return a dict with pred_logits/pred_boxes; "
                f"got {type(outputs)} keys={list(outputs.keys()) if isinstance(outputs, dict) else None}"
            )
        logits = outputs["pred_logits"]
        if self.score_clamp > 0:
            logits = torch.clamp(logits, min=-self.score_clamp, max=self.score_clamp)
        return logits, outputs["pred_boxes"]


def _shape_of(value_info) -> list:
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return dims


def inspect_onnx(path: Path, logger: logging.Logger) -> Dict[str, Any]:
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    logger.info("onnx.checker: OK")

    inferred = onnx.shape_inference.infer_shapes(model)
    inputs = [{"name": vi.name, "shape": _shape_of(vi)} for vi in inferred.graph.input]
    outputs = [{"name": vi.name, "shape": _shape_of(vi)} for vi in inferred.graph.output]
    for item in inputs:
        logger.info("ONNX input : %-16s %s", item["name"], item["shape"])
    for item in outputs:
        logger.info("ONNX output: %-16s %s", item["name"], item["shape"])

    if len(outputs) != 2:
        raise RuntimeError(f"expected exactly 2 ONNX outputs (logits, boxes), got {len(outputs)}")

    import collections

    op_counts = collections.Counter(node.op_type for node in model.graph.node)
    logger.info("ONNX nodes=%d opset=%d", len(model.graph.node), int(model.opset_import[0].version))
    selected = {k: v for k, v in op_counts.items() if k in ("TopK", "Sort", "Gather", "NonMaxSuppression")}
    if selected:
        logger.info(
            "detection-related ops present: %s (D-FINE internal query-selection TopK is part of "
            "the decoder; the final detection TopK is not exported)",
            selected,
        )
    return {
        "inputs": inputs,
        "outputs": outputs,
        "opset": int(model.opset_import[0].version),
        "num_nodes": len(model.graph.node),
        "selected_op_counts": selected,
    }


def check_onnxruntime_parity(
    onnx_path: Path,
    torch_outputs: Sequence[np.ndarray],
    dummy: np.ndarray,
    logger: logging.Logger,
    atol: float,
) -> Dict[str, Any]:
    """Compare ONNX Runtime against the PyTorch export on deterministic input.

    ONNX checker only proves the graph is syntactically valid; a silent operator
    or layout mistake would survive it.  This parity check catches that before a
    slow RKNN conversion.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        logger.warning("onnxruntime unavailable, skipping parity check: %s", exc)
        return {"skipped": "onnxruntime unavailable"}

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_outputs = session.run(None, {session.get_inputs()[0].name: dummy})

    report: Dict[str, Any] = {}
    for idx, (name, ref, got) in enumerate(zip(session.get_outputs(), torch_outputs, ort_outputs)):
        name = name.name if hasattr(name, "name") else str(idx)
        max_abs = float(np.max(np.abs(ref.astype(np.float32) - got.astype(np.float32))))
        report[name] = {"max_abs_diff": max_abs}
        logger.info("parity %-16s max_abs_diff=%.3e", name, max_abs)
        if max_abs > atol:
            raise RuntimeError(
                f"ONNX Runtime parity failed for {name}: max_abs_diff={max_abs:.3e} > atol={atol:.3e}. "
                "The exported graph does not reproduce PyTorch; do not proceed to RKNN conversion."
            )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_repo_root())
    parser.add_argument("--config", type=Path, default=None,
                        help="default: <ecdet>/configs/ecdet/ecdet_s_p150u.yml")
    parser.add_argument("--weights", type=Path, default=None,
                        help="default: <ecdet>/outputs/p150u/e3_s42/best_ema.pth")
    parser.add_argument("--weights-key", choices=["auto", "ema", "model", "raw"], default="auto")
    parser.add_argument("--output", type=Path, default=None,
                        help=f"default: <repo>/deploy/artifacts/{DEFAULT_ONNX_NAME}")
    parser.add_argument("--meta", type=Path, default=None,
                        help=f"default: <repo>/deploy/artifacts/{DEFAULT_META_NAME}")
    parser.add_argument("--input-size", type=int, default=None,
                        help=f"override square input size (project default {INPUT_SIZE})")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--atol", type=float, default=2e-3, help="ONNX Runtime parity tolerance")
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--allow-partial-load", action="store_true",
                        help="use strict=False when loading the checkpoint")
    parser.add_argument("--keep-lqe", action="store_true",
                        help="export the LQE confidence branch (default: disabled for RKNN INT8)")
    parser.add_argument("--score-clamp", type=float, default=30.0,
                        help="clamp encoder/final logits to [-v, v] for stable INT8 calibration (0=off)")
    parser.add_argument("--score-input-clamp", type=float, default=10.0,
                        help="clamp the encoder memory before the score head (0=off)")
    args = parser.parse_args()

    logger = setup_logging(name="export")
    log_environment(logger)
    try:
        import onnx  # noqa: F401
    except ImportError:
        logger.error(
            "onnx is not installed in the training environment. Install onnx==1.16.1 "
            "(or add it to the train feature) before exporting."
        )
        return 2

    repo_root = args.repo_root.expanduser().resolve()
    ecdet_root = resolve_ecdet_root(repo_root)
    config_path = (args.config or (ecdet_root / "configs" / "ecdet" / "ecdet_s_p150u.yml")).expanduser().resolve()
    weights_path = (args.weights or (ecdet_root / "outputs" / "p150u" / "e3_s42" / "best_ema.pth")).expanduser().resolve()
    # The stated checkpoint name is best_ema.pth.  Some training revisions save
    # the same selected EMA weights as best.pth; use it only when the caller did
    # not explicitly ask for a path, and say so loudly.
    if args.weights is None and not weights_path.is_file():
        fallback = weights_path.with_name("best.pth")
        if fallback.is_file():
            logger.warning("best_ema.pth not found; falling back to %s (selected EMA checkpoint)", fallback)
            weights_path = fallback.resolve()
    artifacts = (repo_root / "deploy" / "artifacts").resolve()
    output_path = (args.output or (artifacts / DEFAULT_ONNX_NAME)).expanduser().resolve()
    meta_path = (args.meta or (artifacts / DEFAULT_META_NAME)).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    ensure_files([config_path, weights_path], "config/checkpoint")
    logger.info("repo_root=%s", repo_root)
    logger.info("ecdet_root=%s", ecdet_root)
    logger.info("config=%s", config_path)
    logger.info("weights=%s", weights_path)

    add_repo_to_sys_path(ecdet_root)
    from engine.core import YAMLConfig  # noqa: E402

    with Timer("build config + model in deploy mode", logger):
        cfg = YAMLConfig(str(config_path))
        # The checkpoint already contains the full backbone.  Prevent ViTAdapter
        # from downloading/loading its pretrained file during construction.
        if "ViTAdapter" in cfg.yaml_cfg:
            cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True
        model = cfg.model
        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        state, chosen_key, ckpt_info = probe_state_dict(ckpt, preferred=args.weights_key)
        logger.info(
            "checkpoint container=%s epoch=%s tensors=%d",
            chosen_key,
            ckpt_info.get("epoch", "?"),
            len(state),
        )
        try:
            missing, unexpected = model.load_state_dict(state, strict=not args.allow_partial_load)
        except RuntimeError as exc:
            # Re-run non-strict to produce a precise diff for the diagnostic.
            missing, unexpected = model.load_state_dict(state, strict=False)
            logger.error("strict load failed: %s", exc)
            logger.error("missing keys (%d): %s", len(missing), list(missing)[:20])
            logger.error("unexpected keys (%d): %s", len(unexpected), list(unexpected)[:20])
            logger.error(
                "Likely causes: wrong --weights-key, a segmentation checkpoint, or an architecture/config mismatch."
            )
            return 2
        if missing or unexpected:
            logger.warning("non-strict load: missing=%d unexpected=%d", len(missing), len(unexpected))
            if missing:
                logger.warning("  missing sample: %s", list(missing)[:10])
            if unexpected:
                logger.warning("  unexpected sample: %s", list(unexpected)[:10])

        model = model.deploy()  # eval + fuse Conv/BN + strip unused decoder heads
        model.eval()
        if args.keep_lqe:
            logger.warning("keeping LQE in the ONNX graph; RKNN INT8 accuracy may collapse")
            lqe_disabled = False
        else:
            n_lqe = disable_lqe(model)
            lqe_disabled = n_lqe > 0
            logger.info("disabled %d LQE layer(s) for INT8-friendly export", n_lqe)
        n_clamped = wrap_score_head(model, args.score_clamp, args.score_input_clamp)
        logger.info(
            "clamped %d encoder score head(s): input +/-%.1f, logits +/-%.1f",
            n_clamped,
            args.score_input_clamp,
            args.score_clamp,
        )

    eval_spatial = cfg.yaml_cfg.get("eval_spatial_size", [INPUT_SIZE, INPUT_SIZE])
    if isinstance(eval_spatial, int):
        eval_spatial = [eval_spatial, eval_spatial]
    if args.input_size is not None:
        input_size = int(args.input_size)
        if input_size != int(eval_spatial[0]):
            logger.warning(
                "input size %d overrides config eval_spatial_size=%s; decoder anchors may mismatch!",
                input_size,
                eval_spatial,
            )
    else:
        input_size = int(eval_spatial[0])
    if int(eval_spatial[0]) != int(eval_spatial[1]):
        raise ValueError(f"eval_spatial_size must be square for this pipeline, got {eval_spatial}")
    if input_size != INPUT_SIZE:
        logger.warning("project fact is %sx%s, exporting %sx%s", INPUT_SIZE, INPUT_SIZE, input_size, input_size)

    num_classes = int(cfg.yaml_cfg.get("num_classes", NUM_CLASSES))
    if num_classes != NUM_CLASSES:
        logger.warning("config num_classes=%d differs from frozen project fact %d", num_classes, NUM_CLASSES)

    # One deterministic input is used for both the torch reference and the ORT
    # parity check; otherwise we would compare two different tensors.
    dummy_np = np.random.RandomState(42).randn(1, 3, input_size, input_size).astype(np.float32)
    dummy = torch.from_numpy(dummy_np)
    export_model = DetectorExport(model, score_clamp=args.score_clamp).eval()
    with torch.no_grad():
        ref_logits, ref_boxes = export_model(dummy)
    logger.info("torch output pred_logits=%s pred_boxes=%s", tuple(ref_logits.shape), tuple(ref_boxes.shape))
    expected = (1, NUM_QUERIES, num_classes)
    if tuple(ref_logits.shape) != expected:
        raise RuntimeError(f"pred_logits shape {tuple(ref_logits.shape)} != expected {expected}")
    if tuple(ref_boxes.shape) != (1, NUM_QUERIES, 4):
        raise RuntimeError(f"pred_boxes shape {tuple(ref_boxes.shape)} != (1,{NUM_QUERIES},4)")

    export_kwargs: Dict[str, Any] = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        # The legacy TorchScript exporter produces the static, op-compatible
        # graph that rknn-toolkit2 2.3 understands best.
        export_kwargs["dynamo"] = False

    with Timer(f"torch.onnx.export (opset={args.opset}, static {input_size})", logger):
        torch.onnx.export(
            export_model,
            (dummy,),
            str(output_path),
            input_names=["images"],
            output_names=["pred_logits", "pred_boxes"],
            opset_version=int(args.opset),
            do_constant_folding=True,
            verbose=False,
            **export_kwargs,
        )
    logger.info("wrote %s (%.1f MiB)", output_path, output_path.stat().st_size / (1024 ** 2))

    with Timer("onnx.checker + shape inference", logger):
        graph_info = inspect_onnx(output_path, logger)

    parity: Dict[str, Any] = {"skipped": True}
    if not args.skip_parity:
        with Timer("ONNX Runtime vs PyTorch parity", logger):
            try:
                parity = check_onnxruntime_parity(
                    output_path,
                    [ref_logits.detach().cpu().numpy(), ref_boxes.detach().cpu().numpy()],
                    dummy_np,
                    logger,
                    atol=args.atol,
                )
            except Exception as exc:  # noqa: BLE001 - keep the ONNX artifact for inspection
                logger.error("parity check failed: %s", exc)
                logger.error("The ONNX file was written but should be treated as invalid.")
                raise

    metadata = {
        "model": "ECDet-S / D-FINE (ViTAdapter + HybridEncoder + ECTransformer)",
        "config": str(config_path),
        "checkpoint": str(weights_path),
        "checkpoint_container": chosen_key,
        "checkpoint_info": {k: v for k, v in ckpt_info.items() if k != "top_level_keys"},
        "opset": int(args.opset),
        "input_name": "images",
        "input_shape": [1, 3, input_size, input_size],
        "input_dtype": "float32 (ONNX); uint8 RGB NCHW at RKNN runtime",
        "output_names": ["pred_logits", "pred_boxes"],
        "output_shapes": [[1, NUM_QUERIES, num_classes], [1, NUM_QUERIES, 4]],
        "num_classes": num_classes,
        "num_queries": NUM_QUERIES,
        "class_names": list(VISDRONE_CLASSES),
        "preprocess": {
            "resize": "stretch",
            "resize_backend": "PIL bilinear (torchvision Resize-compatible)",
            "letterbox": False,
            "input_size": input_size,
            "pad_value": 0,
            "color_order": "RGB",
            "layout": "NCHW",
            "normalized_in_python": False,
        },
        "rknn_normalization": {
            "mean_values_rgb_255": list(MEAN_RGB_255),
            "std_values_rgb_255": list(STD_RGB_255),
            "quant_img_RGB2BGR": False,
        },
        "postprocess_location": "numpy (deploy/postprocess.py); final topk excluded from ONNX",
        "lqe_disabled": bool(lqe_disabled),
        "score_clamp": float(args.score_clamp),
        "score_input_clamp": float(args.score_input_clamp),
        "lqe_note": (
            "LQE's regression-bin TopK/Softmax is quantised incorrectly by RKNN 2.5; "
            "the deployment graph uses the raw decoder score head."
        ),
        "pytorch_reference_percent": dict(PYTORCH_REFERENCE),
        "onnx_graph_info": graph_info,
        "onnxruntime_parity": parity,
    }
    write_json(meta_path, metadata)
    logger.info("wrote metadata %s", meta_path)
    logger.info("next: pixi run -e rknn python deploy/convert.py --onnx %s", output_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        # Full traceback is essential here: export failures are usually shape or
        # custom-op issues that cannot be summarised in one line.
        traceback.print_exc()
        raise SystemExit(1)
