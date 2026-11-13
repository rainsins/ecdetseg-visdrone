#!/usr/bin/env python3
"""Evaluate the INT8 RKNN model in the x86 simulator on the full VisDrone val set.

Pipeline per image:
    preprocess.py (stretch resize, RGB, CHW, uint8)
        -> rknn.inference(data_format="nchw")
        -> postprocess.py (sigmoid, topk300, cxcywh->xyxy, stretch inverse)
        -> COCO detection record

At the end: pycocotools COCOeval with ``maxDets=[1,10,100,300]``.  The custom
summary below mirrors the repository's ``coco_summarize`` (which reports the
trailing maxDets slot, 300) so the simulator number is comparable with the
PyTorch reference 28.82 / 48.54 / 28.63.  Stock pycocotools hard-codes 100 for
the AP line and would silently produce a different number.

Exit code is non-zero when |sim AP - reference AP| > --fail-drop (4 points by
default), with instructions to run accuracy_analysis.py.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deploy import (  # noqa: E402
    DEFAULT_META_NAME,
    DEFAULT_ONNX_NAME,
    DEFAULT_RKNN_NAME,
    INPUT_SIZE,
    MEAN_RGB_01,
    MEAN_RGB_255,
    NUM_CLASSES,
    NUM_QUERIES,
    PYTORCH_REFERENCE,
    STD_RGB_01,
    STD_RGB_255,
    VISDRONE_CLASSES,
    Timer,
    format_seconds,
    log_environment,
    read_json,
    resolve_data_root,
    setup_logging,
    write_json,
)
from deploy.convert import _call_supported, _pushd, _validate_calib  # noqa: E402
from deploy.postprocess import postprocess, split_outputs, to_coco_detections  # noqa: E402
from deploy.preprocess import preprocess_path  # noqa: E402


def _import_rknn():
    try:
        from rknn.api import RKNN  # type: ignore
        return RKNN
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "Cannot import rknn.api.RKNN. sim_eval.py must run in the rknn env:\n"
            "  pixi run -e rknn python deploy/sim_eval.py ...\n"
            f"Original error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# COCO evaluation
# ---------------------------------------------------------------------------
def _summarize_dets(coco_eval, max_det: int) -> np.ndarray:
    """Same 12 statistics as the repo's custom coco_summarize.

    Kept as a local copy so the deployment directory has no runtime dependency
    on the training package (and so the metric definition is auditable here).
    """
    p = coco_eval.params
    if not coco_eval.eval:
        raise RuntimeError("COCOeval.accumulate() must run before summarize")

    def _summarize(ap: int = 1, iou_thr=None, area_rng: str = "all") -> float:
        aind = [i for i, a in enumerate(p.areaRngLbl) if a == area_rng]
        mind = [i for i, m in enumerate(p.maxDets) if m == max_det]
        if ap == 1:
            s = coco_eval.eval["precision"]
            if iou_thr is not None:
                t = np.where(iou_thr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
        else:
            s = coco_eval.eval["recall"]
            if iou_thr is not None:
                t = np.where(iou_thr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, aind, mind]
        if len(s[s > -1]) == 0:
            return -1.0
        return float(np.mean(s[s > -1]))

    stats = np.zeros((12,), dtype=np.float64)
    stats[0] = _summarize(1)
    stats[1] = _summarize(1, iou_thr=0.5)
    stats[2] = _summarize(1, iou_thr=0.75)
    stats[3] = _summarize(1, area_rng="small")
    stats[4] = _summarize(1, area_rng="medium")
    stats[5] = _summarize(1, area_rng="large")
    stats[6] = _summarize(0, iou_thr=0.5, area_rng="all")
    stats[7] = _summarize(0, iou_thr=0.75, area_rng="all")
    stats[8] = _summarize(0)
    stats[9] = _summarize(0, area_rng="small")
    stats[10] = _summarize(0, area_rng="medium")
    stats[11] = _summarize(0, area_rng="large")
    coco_eval.stats = stats
    return stats


def _per_class_ap(coco_eval, max_det: int) -> Dict[str, float]:
    """AP@[.5:.95] per category at the trailing maxDets slot (diagnostic only)."""
    p = coco_eval.params
    precision = coco_eval.eval["precision"]  # T,R,K,A,M
    aind = [i for i, a in enumerate(p.areaRngLbl) if a == "all"][0]
    mind = [i for i, m in enumerate(p.maxDets) if m == max_det][0]
    cat_ids = [int(c) for c in p.catIds]
    out: Dict[str, float] = {}
    for k, cat_id in enumerate(cat_ids):
        s = precision[:, :, k, aind, mind]
        out[VISDRONE_CLASSES[cat_id] if 0 <= cat_id < len(VISDRONE_CLASSES) else str(cat_id)] = (
            float(np.mean(s[s > -1])) * 100.0 if len(s[s > -1]) else -1.0
        )
    return out


def evaluate_coco(
    ann_path: Path,
    detections: Sequence[Mapping[str, Any]],
    max_det: int,
    image_ids: Sequence[int],
    logger: logging.Logger,
) -> Dict[str, Any]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not detections:
        raise RuntimeError(
            "No detections were produced. Every box was empty/degenerate or did not pass --score-thresh."
        )
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(ann_path))
        coco_dt = coco_gt.loadRes(list(detections))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.maxDets = [1, 10, 100, int(max_det)]
    coco_eval.params.imgIds = sorted(int(x) for x in image_ids)
    logger.info(
        "COCOeval: %d images, %d detections, maxDets=%s",
        len(coco_eval.params.imgIds),
        len(detections),
        coco_eval.params.maxDets,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.evaluate()
        coco_eval.accumulate()
    stats = _summarize_dets(coco_eval, max_det=max_det)
    per_class = _per_class_ap(coco_eval, max_det=max_det)
    return {
        "AP": float(stats[0] * 100),
        "AP50": float(stats[1] * 100),
        "AP75": float(stats[2] * 100),
        "APs": float(stats[3] * 100),
        "APm": float(stats[4] * 100),
        "APl": float(stats[5] * 100),
        "AR300": float(stats[8] * 100),
        "per_class_AP": per_class,
        "num_detections": len(detections),
        "num_images": len(coco_eval.params.imgIds),
        "maxDets": list(coco_eval.params.maxDets),
    }


# ---------------------------------------------------------------------------
# Simulator construction
# ---------------------------------------------------------------------------
def _crosscheck_conversion_report(
    rknn_path: Path,
    target: str,
    mean_values: Sequence[float],
    std_values: Sequence[float],
    quantized_algorithm: str,
    optimization_level: int,
    auto_hybrid: bool,
    hybrid_cos_thresh: float,
    logger: logging.Logger,
) -> None:
    """Compare the simulator's rebuild config against the shipped .rknn report.

    ``rknn-toolkit2`` rejects ``load_rknn`` + simulator, so the mAP reported here
    is measured on an *in-process rebuild*, not on the delivered artifact.  That
    is only meaningful if the rebuild reproduces the artifact's config exactly.
    ``_call_supported`` silently drops keyword arguments the toolkit does not
    recognise, so a drift would otherwise be invisible.
    """
    report_path = rknn_path.with_suffix(rknn_path.suffix + ".json")
    if not report_path.is_file():
        logger.warning("no conversion report at %s; cannot verify the rebuild matches the "
                       "shipped artifact", report_path)
        return
    try:
        report = read_json(report_path)
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        logger.warning("could not read conversion report %s: %s", report_path, exc)
        return

    expected = {
        "target_platform": target,
        "quantized_algorithm": quantized_algorithm,
        "optimization_level": int(optimization_level),
        "auto_hybrid": bool(auto_hybrid),
    }
    mismatches = []
    if report.get("op_target"):
        mismatches.append(f"op_target: rebuild=None artifact={report['op_target']!r}; "
                          "explicit device-routing experiments require board validation")
    for key, want in expected.items():
        if key not in report:
            continue
        got = report[key]
        if isinstance(want, float):
            same = abs(float(got) - want) < 1e-6
        else:
            same = got == want
        if not same:
            mismatches.append(f"{key}: rebuild={want!r} artifact={got!r}")

    reported_thresh = report.get("auto_hybrid_cos_thresh")
    if reported_thresh is not None and abs(float(reported_thresh) - float(hybrid_cos_thresh)) > 1e-6:
        mismatches.append(
            f"auto_hybrid_cos_thresh: rebuild={hybrid_cos_thresh!r} artifact={reported_thresh!r}"
        )

    for key, ours in (("mean_values_rgb_255", mean_values), ("std_values_rgb_255", std_values)):
        theirs = report.get(key)
        if theirs is None:
            continue
        if len(theirs) != len(ours) or any(abs(float(a) - float(b)) > 1e-3 for a, b in zip(theirs, ours)):
            mismatches.append(f"{key}: rebuild={list(ours)} artifact={list(theirs)}")

    if mismatches:
        raise RuntimeError(
            "simulator rebuild config does not match the shipped .rknn conversion report:\n  "
            + "\n  ".join(mismatches)
            + f"\n(report: {report_path})\n"
            "The simulator mAP would describe a different model than the delivered artifact. "
            "Pass --no-config-check to override."
        )
    logger.info("rebuild config matches the shipped artifact report (%s)", report_path.name)
    sha = report.get("rknn_sha256")
    if sha and rknn_path.is_file():
        import hashlib

        digest = hashlib.sha256(rknn_path.read_bytes()).hexdigest()
        if digest != sha:
            logger.warning(
                "artifact sha256 %s differs from the report %s; the .rknn was rebuilt after "
                "the report was written", digest[:16], str(sha)[:16]
            )
        else:
            logger.info("artifact sha256 matches the report (%s...)", sha[:16])


def _init_runtime_simulator(rknn) -> bool:
    """init_runtime(target=None); tolerate API differences across versions."""
    try:
        ret = rknn.init_runtime(target=None)
    except TypeError:
        ret = rknn.init_runtime()
    return ret in (None, 0)


def _build_simulator(
    RKNN,
    onnx_path: Path,
    calib_txt: Path,
    input_shape: Sequence[int],
    target: str,
    mean_values: Sequence[float],
    std_values: Sequence[float],
    quantized_algorithm: str,
    optimization_level: int,
    auto_hybrid: bool,
    hybrid_cos_thresh: float,
    logger: logging.Logger,
):
    """Build the quantized model in-process and attach the simulator runtime.

    rknn-toolkit2 2.5 explicitly rejects ``load_rknn`` + simulator:
      "RKNN model that loaded by 'load_rknn' not support inference on the
       simulator, please set 'target' first! ... use 'load_xxx' & 'build'".
    The .rknn artifact is still the board deliverable; this rebuild is only for
    the x86 simulator.
    """
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model required for simulator rebuild not found: {onnx_path}")
    _validate_calib(calib_txt)
    rknn = RKNN(verbose=False)
    try:
        with Timer("rknn.config (simulator)", logger):
            ret = rknn.config(
                **_call_supported(
                    rknn.config,
                    mean_values=[list(mean_values)],
                    std_values=[list(std_values)],
                    target_platform=target,
                    quantized_dtype="w8a8",
                    quantized_algorithm=quantized_algorithm,
                    quantized_method="channel",
                    optimization_level=optimization_level,
                    auto_hybrid_cos_thresh=hybrid_cos_thresh,
                    quant_img_RGB2BGR=False,
                )
            )
            if ret not in (None, 0):
                raise RuntimeError(f"rknn.config returned {ret}")

        with Timer("rknn.load_onnx (simulator)", logger):
            ret = rknn.load_onnx(
                **_call_supported(
                    rknn.load_onnx,
                    model=str(onnx_path),
                    inputs=["images"],
                    input_size_list=[list(input_shape)],
                )
            )
            if ret not in (None, 0):
                raise RuntimeError(f"rknn.load_onnx returned {ret}")

        with Timer("rknn.build (INT8, simulator)", logger):
            with _pushd(calib_txt.parent):
                ret = rknn.build(
                    **_call_supported(
                        rknn.build,
                        do_quantization=True,
                        dataset=calib_txt.name,
                        rknn_batch_size=1,
                        auto_hybrid=auto_hybrid,
                    )
                )
            if ret not in (None, 0):
                raise RuntimeError(f"rknn.build returned {ret}")

        with Timer("rknn.init_runtime (simulator)", logger):
            if not _init_runtime_simulator(rknn):
                raise RuntimeError("rknn.init_runtime(target=None) failed after build")
        logger.info("simulator ready (in-process INT8 build)")
        return rknn
    except Exception:
        try:
            rknn.release()
        except Exception:  # noqa: BLE001
            pass
        raise


def create_simulator(
    rknn_path: Path,
    onnx_path: Path,
    calib_txt: Path,
    meta: Mapping[str, Any],
    target: str,
    mean_values: Sequence[float],
    std_values: Sequence[float],
    quantized_algorithm: str,
    optimization_level: int,
    auto_hybrid: bool,
    hybrid_cos_thresh: float,
    allow_rebuild: bool,
    logger: logging.Logger,
) -> tuple[Any, Dict[str, Any]]:
    """Load the .rknn for board parity, but fall back to an in-process build.

    Returns ``(rknn, info)`` where ``info["source"]`` is ``"load_rknn"`` or
    ``"rebuild"``.  This matters for how the resulting mAP may be described: on
    ``rebuild`` the number describes a freshly built model, not the delivered
    ``.rknn`` file, and the paper must say so.
    """
    RKNN = _import_rknn()
    input_shape = list(meta.get("input_shape", [1, 3, INPUT_SIZE, INPUT_SIZE]))

    if rknn_path.is_file():
        logger.info("trying load_rknn + simulator: %s", rknn_path)
        rknn = RKNN(verbose=False)
        ret = rknn.load_rknn(str(rknn_path))
        if ret in (None, 0) and _init_runtime_simulator(rknn):
            logger.info("simulator ready from .rknn")
            return rknn, {
                "source": "load_rknn",
                "artifact": str(rknn_path),
                "describes_shipped_artifact": True,
            }
        logger.warning(
            "load_rknn cannot be used with the simulator in this toolkit version. "
            "This is a toolkit limitation, not a model error."
        )
        try:
            rknn.release()
        except Exception:  # noqa: BLE001
            pass
        if not allow_rebuild:
            raise RuntimeError(
                "Simulator requires load_onnx + build; re-run without --no-rebuild "
                "or pass --onnx/--calib."
            )
    elif not allow_rebuild:
        raise FileNotFoundError(f"RKNN model not found: {rknn_path}")

    logger.warning(
        "the mAP below is measured on an IN-PROCESS REBUILD from ONNX, not on %s. "
        "Report it as such: the rebuild is equivalent only because the config was "
        "cross-checked against the artifact's conversion report.",
        rknn_path.name,
    )
    logger.info("rebuilding INT8 model in-process for the simulator (this can take 10-15 min)")
    rknn = _build_simulator(
        RKNN=RKNN,
        onnx_path=onnx_path,
        calib_txt=calib_txt,
        input_shape=input_shape,
        target=target,
        mean_values=mean_values,
        std_values=std_values,
        quantized_algorithm=quantized_algorithm,
        optimization_level=optimization_level,
        auto_hybrid=auto_hybrid,
        hybrid_cos_thresh=hybrid_cos_thresh,
        logger=logger,
    )
    return rknn, {
        "source": "rebuild",
        "artifact": str(rknn_path),
        "describes_shipped_artifact": False,
        "rebuilt_from": str(onnx_path),
        "note": "rknn-toolkit2 rejects load_rknn + simulator; config was cross-checked "
                "against the artifact's .rknn.json report",
    }


# ---------------------------------------------------------------------------
# FP32 ONNX stretch baseline (same preprocess as the deployment graph)
# ---------------------------------------------------------------------------
def run_fp32_baseline(
    onnx_path: Path,
    val_images: Path,
    ann_path: Path,
    meta: Mapping[str, Any],
    limit: int,
    score_thresh: float,
    max_det: int,
    clip: bool,
    log_every: int,
    logger: logging.Logger,
) -> tuple[List[Dict[str, Any]], List[int], Dict[str, Any]]:
    """Run the same stretch pipeline through FP32 ONNX Runtime.

    The PyTorch reference (28.82) and this pipeline now both use a direct
    640x640 stretch resize.  The FP32 ONNX pass is still useful as a
    same-runtime sanity check and keeps the gate independent of checkpoint
    loading details.
    """
    import onnxruntime as ort

    coco = read_json(ann_path)
    images = sorted(coco["images"], key=lambda x: int(x["id"]))
    if limit and limit > 0:
        images = images[:limit]
    if not images:
        raise ValueError("no validation images selected")
    input_size = int(meta.get("input_shape", [1, 3, INPUT_SIZE, INPUT_SIZE])[2])
    num_classes = int(meta.get("num_classes", NUM_CLASSES))
    output_names = list(meta.get("output_names", ["pred_logits", "pred_boxes"]))
    mean = np.asarray(MEAN_RGB_01, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(STD_RGB_01, dtype=np.float32).reshape(1, 3, 1, 1)

    logger.info("loading FP32 ONNX baseline: %s", onnx_path)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    detections: List[Dict[str, Any]] = []
    image_ids: List[int] = []
    start = time.perf_counter()
    for idx, im in enumerate(images, start=1):
        image_id = int(im["id"])
        pre = preprocess_path(val_images / im["file_name"], size=input_size)
        x = (pre.as_batch().astype(np.float32) / 255.0 - mean) / std
        outputs = sess.run(None, {input_name: x.astype(np.float32)})
        logits, boxes = split_outputs(outputs, num_classes=num_classes, output_names=output_names)
        det = postprocess(
            logits,
            boxes,
            pre.meta_dict(),
            num_classes=num_classes,
            score_thresh=score_thresh,
            max_det=max_det,
            clip=clip,
        )
        detections.extend(to_coco_detections(det, image_id, score_thresh=score_thresh))
        image_ids.append(image_id)
        if idx % max(1, log_every) == 0 or idx == len(images):
            elapsed = time.perf_counter() - start
            logger.info(
                "FP32 baseline progress %d/%d (%.1f%%) dets=%d avg=%.1f ms/img",
                idx,
                len(images),
                100.0 * idx / len(images),
                len(detections),
                1000.0 * elapsed / idx,
            )
    total = time.perf_counter() - start
    return detections, image_ids, {"num_images": len(images), "total_seconds": total}


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------
def run_simulation(
    rknn,
    val_images: Path,
    ann_path: Path,
    meta: Mapping[str, Any],
    limit: int,
    score_thresh: float,
    max_det: int,
    clip: bool,
    warmup: int,
    log_every: int,
    logger: logging.Logger,
) -> tuple[List[Dict[str, Any]], List[int], Dict[str, Any]]:
    coco = read_json(ann_path)
    images = sorted(coco["images"], key=lambda x: int(x["id"]))
    if limit and limit > 0:
        images = images[:limit]
    if not images:
        raise ValueError("no validation images selected")
    first_path = val_images / images[0]["file_name"]
    if not first_path.is_file():
        raise FileNotFoundError(f"first validation image not found: {first_path}")

    num_classes = int(meta.get("num_classes", NUM_CLASSES))
    output_names = list(meta.get("output_names", ["pred_logits", "pred_boxes"]))
    input_size = int(meta.get("input_shape", [1, 3, INPUT_SIZE, INPUT_SIZE])[2])

    # Warm-up is important: the first simulator inference includes graph setup
    # and should not be counted in throughput.
    if warmup > 0:
        warm_result = preprocess_path(first_path, size=input_size)
        with Timer(f"warmup ({warmup} runs)", logger):
            for _ in range(warmup):
                _ = rknn.inference(inputs=[warm_result.as_batch()], data_format="nchw")

    detections: List[Dict[str, Any]] = []
    image_ids: List[int] = []
    preprocess_seconds = 0.0
    inference_seconds = 0.0
    postprocess_seconds = 0.0
    loop_start = time.perf_counter()

    for idx, im in enumerate(images, start=1):
        image_id = int(im["id"])
        path = val_images / im["file_name"]
        if not path.is_file():
            raise FileNotFoundError(f"validation image missing: {path}")

        t0 = time.perf_counter()
        pre = preprocess_path(path, size=input_size)
        t1 = time.perf_counter()
        try:
            outputs = rknn.inference(inputs=[pre.as_batch()], data_format="nchw")
        except Exception:
            logger.error(
                "RKNN inference failed at image %d/%d id=%d file=%s; input shape=%s dtype=%s",
                idx,
                len(images),
                image_id,
                path,
                pre.as_batch().shape,
                pre.as_batch().dtype,
            )
            logger.error(
                "Check: data_format must match the model (this script uses nchw); "
                "simulator target must be None; the calibrated model must be rk3588; "
                "input dtype must be uint8 RGB 0..255."
            )
            raise
        t2 = time.perf_counter()

        if isinstance(outputs, np.ndarray):
            outputs = [outputs]
        logits, boxes = split_outputs(outputs, num_classes=num_classes, output_names=output_names)
        det = postprocess(
            logits,
            boxes,
            pre.meta_dict(),
            num_classes=num_classes,
            score_thresh=score_thresh,
            max_det=max_det,
            clip=clip,
        )
        records = to_coco_detections(det, image_id, score_thresh=score_thresh)
        detections.extend(records)
        image_ids.append(image_id)
        t3 = time.perf_counter()

        preprocess_seconds += t1 - t0
        inference_seconds += t2 - t1
        postprocess_seconds += t3 - t2

        if idx % max(1, log_every) == 0 or idx == len(images):
            elapsed = time.perf_counter() - loop_start
            rate = idx / elapsed if elapsed > 0 else 0.0
            eta = (len(images) - idx) / rate if rate > 0 else math.inf
            logger.info(
                "progress %d/%d (%.1f%%) dets=%d avg=%.1f ms/img eta=%s",
                idx,
                len(images),
                100.0 * idx / len(images),
                len(detections),
                1000.0 * elapsed / idx,
                format_seconds(eta),
            )

    total = time.perf_counter() - loop_start
    timing = {
        "num_images": len(images),
        "total_seconds": total,
        "ms_per_image": 1000.0 * total / len(images),
        "images_per_second": len(images) / total if total > 0 else float("inf"),
        "preprocess_seconds": preprocess_seconds,
        "inference_seconds": inference_seconds,
        "postprocess_seconds": postprocess_seconds,
        "preprocess_ms_per_image": 1000.0 * preprocess_seconds / len(images),
        "inference_ms_per_image": 1000.0 * inference_seconds / len(images),
        "postprocess_ms_per_image": 1000.0 * postprocess_seconds / len(images),
    }
    try:
        rknn.release()
    except Exception:  # noqa: BLE001
        pass
    return detections, image_ids, timing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--rknn", type=Path, default=None)
    parser.add_argument("--onnx", type=Path, default=None,
                        help="needed for the in-process INT8 simulator rebuild")
    parser.add_argument("--calib", type=Path, default=None,
                        help="calib.txt needed for the in-process simulator rebuild")
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--val-images", type=Path, default=None)
    parser.add_argument("--ann", type=Path, default=None)
    parser.add_argument("--out-detections", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="debug: only the first N val images (0=all 548)")
    parser.add_argument("--score-thresh", type=float, default=0.0)
    parser.add_argument("--max-det", type=int, default=NUM_QUERIES)
    parser.add_argument("--no-clip", action="store_true", help="match raw PyTorch postprocessor (no image clipping)")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--fail-drop", type=float, default=1.0,
                        help="exit non-zero if INT8 AP is WORSE than the FP32 baseline by more than "
                             "this many points (one-sided; a better INT8 result is not a failure)")
    parser.add_argument("--ref-ap", type=float, default=PYTORCH_REFERENCE["AP"])
    parser.add_argument("--ref-ap50", type=float, default=PYTORCH_REFERENCE["AP50"])
    parser.add_argument("--ref-ap75", type=float, default=PYTORCH_REFERENCE["AP75"])
    parser.add_argument("--target", default="rk3588")
    parser.add_argument("--mean", type=float, nargs=3, default=list(MEAN_RGB_255))
    parser.add_argument("--std", type=float, nargs=3, default=list(STD_RGB_255))
    parser.add_argument("--quantized-algorithm", default="normal", choices=["normal", "mmse", "kl_divergence"])
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--no-auto-hybrid", action="store_true",
                        help="disable FP16 fallback for sensitive layers (not recommended)")
    parser.add_argument("--hybrid-cos-thresh", type=float, default=0.9999,
                        help="auto_hybrid cosine threshold; must match the shipped .rknn "
                             "(convert.py default is 0.999, the frozen artifact uses 0.9999)")
    parser.add_argument("--no-config-check", action="store_true",
                        help="skip the cross-check against the .rknn.json conversion report")
    parser.add_argument("--no-rebuild", action="store_true",
                        help="fail instead of rebuilding from ONNX when load_rknn cannot attach the simulator")
    parser.add_argument("--fp32-baseline", action="store_true",
                        help="also evaluate FP32 ONNX with the same stretch preprocessing; gate quantisation against it")
    parser.add_argument("--fp32-onnx", type=Path, default=None,
                        help="FP32 ONNX used for --fp32-baseline (default: artifacts ONNX)")
    args = parser.parse_args()

    logger = setup_logging(name="sim_eval")
    log_environment(logger)

    repo_root = args.repo_root.expanduser().resolve()
    artifacts = repo_root / "deploy" / "artifacts"
    rknn_path = (args.rknn or (artifacts / DEFAULT_RKNN_NAME)).expanduser().resolve()
    onnx_path = (args.onnx or (artifacts / DEFAULT_ONNX_NAME)).expanduser().resolve()
    calib_txt = (args.calib or (artifacts / "calib.txt")).expanduser().resolve()
    meta_path = (args.meta or (artifacts / DEFAULT_META_NAME)).expanduser().resolve()
    fp32_onnx = (args.fp32_onnx or onnx_path).expanduser().resolve()
    out_det = (args.out_detections or (artifacts / "detections_sim.json")).expanduser().resolve()
    summary_path = (args.summary or (artifacts / "sim_eval_summary.json")).expanduser().resolve()

    data_root = resolve_data_root(repo_root, args.data_root)
    val_images = (args.val_images or (data_root / "val")).expanduser().resolve()
    ann_path = (args.ann or (data_root / "annotations_VisDrone_val.json")).expanduser().resolve()

    if not rknn_path.is_file() and args.no_rebuild:
        logger.error("RKNN model not found: %s", rknn_path)
        logger.error("Run convert.py first, or omit --no-rebuild to let sim_eval build in-process.")
        return 2
    if not ann_path.is_file():
        logger.error("val annotation not found: %s", ann_path)
        logger.error("Pass --data-root /path/to/VisDrone or --ann.")
        return 2
    if not val_images.is_dir():
        logger.error("val image directory not found: %s", val_images)
        return 2

    meta: Dict[str, Any] = {}
    if meta_path.is_file():
        meta = read_json(meta_path)
        logger.info("metadata=%s", meta_path)
    else:
        logger.warning("metadata not found (%s); using frozen project constants", meta_path)
        meta = {
            "num_classes": NUM_CLASSES,
            "input_shape": [1, 3, INPUT_SIZE, INPUT_SIZE],
            "output_names": ["pred_logits", "pred_boxes"],
            "class_names": list(VISDRONE_CLASSES),
        }
    if int(meta.get("num_classes", NUM_CLASSES)) != NUM_CLASSES:
        logger.error("metadata num_classes=%s != %d", meta.get("num_classes"), NUM_CLASSES)
        return 2

    mean_values = list(meta.get("rknn_normalization", {}).get("mean_values_rgb_255", args.mean))
    std_values = list(meta.get("rknn_normalization", {}).get("std_values_rgb_255", args.std))
    logger.info("rknn=%s", rknn_path)
    logger.info("onnx=%s", onnx_path)
    logger.info("calib=%s", calib_txt)
    logger.info("val_images=%s", val_images)
    logger.info("ann=%s", ann_path)
    logger.info("mean=%s std=%s", mean_values, std_values)
    logger.info("limit=%s score_thresh=%.4f max_det=%d clip=%s", args.limit or "all", args.score_thresh, args.max_det, not args.no_clip)

    try:
        if not args.no_config_check:
            _crosscheck_conversion_report(
                rknn_path=rknn_path,
                target=args.target,
                mean_values=mean_values,
                std_values=std_values,
                quantized_algorithm=args.quantized_algorithm,
                optimization_level=args.optimization_level,
                auto_hybrid=not args.no_auto_hybrid,
                hybrid_cos_thresh=args.hybrid_cos_thresh,
                logger=logger,
            )
        rknn, simulator_info = create_simulator(
            rknn_path=rknn_path,
            onnx_path=onnx_path,
            calib_txt=calib_txt,
            meta=meta,
            target=args.target,
            mean_values=mean_values,
            std_values=std_values,
            quantized_algorithm=args.quantized_algorithm,
            optimization_level=args.optimization_level,
            auto_hybrid=not args.no_auto_hybrid,
            hybrid_cos_thresh=args.hybrid_cos_thresh,
            allow_rebuild=not args.no_rebuild,
            logger=logger,
        )
        detections, image_ids, timing = run_simulation(
            rknn=rknn,
            val_images=val_images,
            ann_path=ann_path,
            meta=meta,
            limit=args.limit,
            score_thresh=args.score_thresh,
            max_det=args.max_det,
            clip=not args.no_clip,
            warmup=args.warmup,
            log_every=args.log_every,
            logger=logger,
        )
    except Exception:
        logger.error("simulation aborted; full traceback follows")
        traceback.print_exc()
        return 1

    out_det.parent.mkdir(parents=True, exist_ok=True)
    with open(out_det, "w", encoding="utf-8") as f:
        json.dump(detections, f, ensure_ascii=False)
        f.write("\n")
    logger.info("wrote %d detections to %s", len(detections), out_det)

    with Timer("COCOeval", logger):
        metrics = evaluate_coco(
            ann_path=ann_path,
            detections=detections,
            max_det=args.max_det,
            image_ids=image_ids,
            logger=logger,
        )

    reference = {"AP": args.ref_ap, "AP50": args.ref_ap50, "AP75": args.ref_ap75}
    drops = {k: reference[k] - metrics[k] for k in reference}

    baseline_metrics = None
    baseline_drops = None
    if args.fp32_baseline:
        try:
            base_dets, base_ids, base_timing = run_fp32_baseline(
                onnx_path=fp32_onnx,
                val_images=val_images,
                ann_path=ann_path,
                meta=meta,
                limit=args.limit,
                score_thresh=args.score_thresh,
                max_det=args.max_det,
                clip=not args.no_clip,
                log_every=args.log_every,
                logger=logger,
            )
            with Timer("COCOeval (FP32 stretch baseline)", logger):
                baseline_metrics = evaluate_coco(
                    ann_path=ann_path,
                    detections=base_dets,
                    max_det=args.max_det,
                    image_ids=base_ids,
                    logger=logger,
                )
            baseline_drops = {k: baseline_metrics[k] - metrics[k] for k in ("AP", "AP50", "AP75")}
        except Exception:
            logger.error("FP32 stretch baseline failed; falling back to the PyTorch reference gate")
            traceback.print_exc()
            baseline_metrics = None

    logger.info("")
    if baseline_metrics is not None:
        logger.info(
            "%-8s %16s %16s %16s %16s",
            "metric", "PyTorch(stretch)", "FP32(stretch)", "RKNN INT8", "FP32-INT8",
        )
        for key in ("AP", "AP50", "AP75"):
            logger.info(
                "%-8s %16.2f %16.2f %16.2f %+16.2f",
                key,
                reference[key],
                baseline_metrics[key],
                metrics[key],
                baseline_drops[key],
            )
    else:
        logger.info("%-8s %10s %10s %10s", "metric", "PyTorch", "RKNN INT8", "drop")
        for key in ("AP", "AP50", "AP75"):
            logger.info("%-8s %10.2f %10.2f %+10.2f", key, reference[key], metrics[key], drops[key])
    logger.info(
        "APs=%.2f APm=%.2f APl=%.2f | %.1f ms/img (%.2f img/s)",
        metrics["APs"],
        metrics["APm"],
        metrics["APl"],
        timing["ms_per_image"],
        timing["images_per_second"],
    )
    logger.info("per-class AP@[.5:.95]:")
    for name, ap in metrics["per_class_AP"].items():
        logger.info("  %-16s %6.2f", name, ap)

    # ------------------------------------------------------------------
    # Acceptance gate.  Computed BEFORE the summary is written so the
    # verdict is part of the artifact, not just a log line.
    # ------------------------------------------------------------------
    gate_label = "FP32 stretch baseline"
    gate_drop = drops["AP"]
    gate_ref = reference["AP"]
    if baseline_metrics is not None:
        gate_drop = baseline_drops["AP"]
        gate_ref = baseline_metrics["AP"]

    if metrics["AP"] < 0:
        gate_passed = False
        gate_reason = "AP is -1: COCOeval found no valid detections"
    elif gate_drop > args.fail_drop:
        gate_passed = False
        # One-sided: only a regression matters.  The previous version used
        # abs(gate_drop), so an INT8 model that beat the FP32 baseline failed.
        gate_reason = (
            f"INT8 AP {metrics['AP']:.2f} is {gate_drop:.2f} points worse than "
            f"{gate_label} {gate_ref:.2f} (> {args.fail_drop:.2f})"
        )
    else:
        gate_passed = True
        # "drop" is reference - INT8, so a positive value means INT8 is lower.
        # Printing it as "+0.11" reads like an improvement, so state the
        # direction in words instead.
        if gate_drop > 0:
            gate_reason = (
                f"INT8 AP {metrics['AP']:.2f} is {gate_drop:.2f} points below "
                f"{gate_label} {gate_ref:.2f} (within limit {args.fail_drop:.2f})"
            )
        elif gate_drop < 0:
            gate_reason = (
                f"INT8 AP {metrics['AP']:.2f} is {-gate_drop:.2f} points above "
                f"{gate_label} {gate_ref:.2f}"
            )
        else:
            gate_reason = f"INT8 AP matches {gate_label} {gate_ref:.2f} exactly"

    summary = {
        "pytorch_reference": reference,
        # Historical key names are kept for downstream parsers, but the value is
        # a *stretch* FP32 ONNX baseline, not letterbox.  See README 4.5.
        "fp32_letterbox_baseline": baseline_metrics,
        "drop_vs_fp32_letterbox": baseline_drops,
        "rknn_sim_int8": metrics,
        "drop_points": drops,
        "gate": {
            "passed": gate_passed,
            "against": gate_label if baseline_metrics is not None else "PyTorch reference",
            "reference_AP": gate_ref,
            "drop": gate_drop,
            "fail_drop": args.fail_drop,
            "one_sided": True,
            "reason": gate_reason,
        },
        "timing": timing,
        "simulator": simulator_info,
        "config": {
            "rknn": str(rknn_path),
            "ann": str(ann_path),
            "val_images": str(val_images),
            "limit": args.limit,
            "score_thresh": args.score_thresh,
            "max_det": args.max_det,
            "clip": not args.no_clip,
            "data_format": "nchw",
            "fp32_baseline": bool(args.fp32_baseline and baseline_metrics is not None),
            "fp32_onnx": str(fp32_onnx),
            "hybrid_cos_thresh": args.hybrid_cos_thresh,
            "auto_hybrid": not args.no_auto_hybrid,
            "optimization_level": args.optimization_level,
            "quantized_algorithm": args.quantized_algorithm,
        },
    }
    write_json(summary_path, summary)
    logger.info("wrote summary %s", summary_path)

    if not gate_passed:
        logger.error("")
        logger.error("ACCEPTANCE GATE FAILED: %s", gate_reason)
        if metrics["AP"] >= 0:
            logger.error("Run the layer-wise diagnostic before changing the calibration set:")
            logger.error("  pixi run -e rknn python deploy/accuracy_analysis.py")
            logger.error(
                "Also check: RGB/BGR order, mean/std, calibration coverage, and "
                "non-finite calibration images."
            )
        return 1

    logger.info("")
    logger.info("ACCEPTANCE GATE PASSED: %s", gate_reason)

    logger.info(
        "RESULT: PyTorch(stretch) mAP=%.2f | RKNN INT8(stretch) mAP=%.2f | INT8-FP32=%+.2f",
        reference["AP"],
        metrics["AP"],
        metrics["AP"] - gate_ref,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
