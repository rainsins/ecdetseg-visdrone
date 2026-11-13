#!/usr/bin/env python3
"""The single preprocessing implementation used by simulator and board.

Contract (must not be duplicated elsewhere):
  * input : BGR uint8 image as returned by ``cv2.imread``
  * resize: direct stretch to 640x640 (no aspect preservation, no padding)
  * colour: BGR -> RGB
  * layout: HWC -> CHW
  * scale : *no* mean/std and *no* /255 here

This matches the PyTorch training/validation transform (``Resize(size=[640,
640])`` applied to a PIL image), so the deployment domain is identical to the
reported 28.82 mAP.  A letterbox geometry shift was measured at 23.76 mAP for
the same weights -- easy to mistake for quantisation loss.

Why PIL and not OpenCV?
  ``Image.resize(..., BILINEAR)`` uses a triangle kernel widened by the scale
  factor (``support * filterscale``), which is what makes it antialiased when
  downscaling.  Measured on a VisDrone 1360x765 frame against PIL:

      cv2.INTER_LINEAR : max=95/255, mean=6.78, 87.7% of pixels differ
      cv2.INTER_AREA   : max=34/255, mean=2.20, 75.1% of pixels differ

  Neither is equivalent, so neither may be used on the board either.

Why no mean/std here?
  RKNN's ``config(mean_values=..., std_values=...)`` is applied inside the
  quantized graph.  If Python also normalised, RKNN would normalise twice and
  the INT8 calibration statistics would be wrong.  Keeping the tensor as uint8
  also lets the board feed the camera buffer directly.  Note this applies to the
  RKNN path only: ``export_onnx.py`` exports a graph with *no* normalisation
  baked in, so the ONNX/PyTorch reference paths do divide by 255 and subtract
  mean/std themselves (see ``MEAN_RGB_01``).

Why keep scale/pad in the metadata?
  They are now neutral (pad=0, scale=1).  ``postprocess.py`` maps normalised
  boxes back to the source by multiplying by ``orig_w``/``orig_h`` and does not
  use these fields.  Keeping the keys preserves the JSON contract and avoids
  breaking board-side callers.

Self-check (the board port's only acceptance gate)
--------------------------------------------------
``--compare-candidate`` diffs a board-side dump against the reference produced
here.  It is only meaningful when the *source* image is larger than 640x640;
resizing a 640x640 image is a no-op and would make the check pass vacuously.
That mistake was present in an earlier version of ``run_all.sh`` (it fed a
640x640 calibration PNG), so it is now an error, not a warning.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image

# Allow `python deploy/preprocess.py` as well as `from deploy.preprocess import ...`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deploy import INPUT_SIZE, setup_logging  # noqa: E402


@dataclass(frozen=True)
class PreprocessMeta:
    """Metadata for the stretch preprocessing.

    ``pad_x``/``pad_y`` are always 0 and ``scale_x``/``scale_y`` are always 1.0
    for stretch.  They remain in the schema for compatibility with older
    artefacts; postprocessing uses ``orig_w``/``orig_h`` directly.
    """

    orig_w: int
    orig_h: int
    resized_w: int
    resized_h: int
    input_w: int
    input_h: int
    pad_x: int
    pad_y: int
    scale_x: float
    scale_y: float
    pad_value: int = 0
    color_order: str = "RGB"
    layout: str = "CHW"
    normalized: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreprocessResult:
    # uint8, shape (3, H, W); add a batch axis with as_batch().
    chw: np.ndarray
    meta: PreprocessMeta

    def as_batch(self) -> np.ndarray:
        return self.chw[None, ...]

    def meta_dict(self) -> Dict[str, Any]:
        return self.meta.to_dict()


def stretch_bgr(
    image_bgr: np.ndarray,
    size: int = INPUT_SIZE,
    pad_value: int = 0,  # accepted for API compatibility; unused
) -> PreprocessResult:
    """Stretch one BGR image to ``size x size`` and convert to RGB CHW.

    The resize is exactly ``PIL.Image.resize((size, size), BILINEAR)`` on the
    RGB view, which is what ``torchvision.transforms.v2.Resize`` does on the
    training/val path.  Resampling is per-band, so converting BGR->RGB before
    or after the resize is equivalent.
    """
    if image_bgr is None:
        raise ValueError("image is None (cv2.imread failed)")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 BGR image, got shape={image_bgr.shape}")

    orig_h, orig_w = int(image_bgr.shape[0]), int(image_bgr.shape[1])
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"invalid image size: {orig_w}x{orig_h}")

    # Match torchvision's PIL resize path used by the training/val transform.
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize((size, size), resample=Image.BILINEAR)
    chw = np.ascontiguousarray(np.asarray(pil, dtype=np.uint8).transpose(2, 0, 1))

    meta = PreprocessMeta(
        orig_w=orig_w,
        orig_h=orig_h,
        resized_w=size,
        resized_h=size,
        input_w=size,
        input_h=size,
        pad_x=0,
        pad_y=0,
        # Neutral stretch metadata.  Postprocessing uses orig_w/orig_h.
        scale_x=1.0,
        scale_y=1.0,
        pad_value=0,
    )
    return PreprocessResult(chw=chw, meta=meta)


# Backward-compatible alias for callers/imports written before the stretch
# switch (e.g. older board snippets).  Behaviour is stretch, not letterbox.
letterbox_bgr = stretch_bgr


def preprocess_path(path: str | Path, size: int = INPUT_SIZE, pad_value: int = 0) -> PreprocessResult:
    """Read a path with OpenCV and apply :func:`stretch_bgr`."""
    path = Path(path)
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(
            f"cv2.imread returned None for {path}. "
            "Check that the file exists, is a readable image, and is not a zero-byte file."
        )
    return stretch_bgr(image_bgr, size=size, pad_value=pad_value)


def save_meta(meta: PreprocessMeta | Dict[str, Any], path: str | Path) -> None:
    payload = meta.to_dict() if isinstance(meta, PreprocessMeta) else dict(meta)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_meta(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_tensors(reference: np.ndarray, candidate: np.ndarray) -> Dict[str, Any]:
    """Diff a board dump against the reference CHW tensor.

    Returns a report and raises nothing; ``main`` decides the exit code.
    """
    ref = np.asarray(reference)
    cand = np.asarray(candidate)
    report: Dict[str, Any] = {
        "reference_shape": list(ref.shape),
        "candidate_shape": list(cand.shape),
        "reference_dtype": str(ref.dtype),
        "candidate_dtype": str(cand.dtype),
        "shape_match": list(ref.shape) == list(cand.shape),
    }
    if not report["shape_match"]:
        report.update({"max_abs_diff": None, "mismatched_elements": None, "exact": False})
        return report

    # Cast to int16 so a uint8 wrap-around cannot hide a difference.
    diff = np.abs(ref.astype(np.int16) - cand.astype(np.int16))
    report.update(
        {
            "max_abs_diff": int(diff.max()) if diff.size else 0,
            "mismatched_elements": int((diff != 0).sum()),
            "mismatched_fraction": float((diff != 0).mean()) if diff.size else 0.0,
            "exact": bool((diff == 0).all()),
        }
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", required=True, help="source image (must be larger than --size to test resize)")
    parser.add_argument("--output-npy", default=None, help="write the reference uint8 CHW tensor here")
    parser.add_argument("--output-meta", default=None, help="write preprocessing metadata here")
    parser.add_argument("--size", type=int, default=INPUT_SIZE)
    parser.add_argument("--pad-value", type=int, default=0, help="deprecated; stretch has no padding")
    parser.add_argument(
        "--compare-candidate",
        default=None,
        help="board-side .npy dump to diff against the reference; exits 1 when max_abs_diff != 0",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="write the comparison report as JSON (useful as a CI/board artifact)",
    )
    parser.add_argument(
        "--allow-no-resize",
        action="store_true",
        help="permit an image whose size already equals --size (the self-check is then vacuous)",
    )
    parser.add_argument(
        "--require-resize",
        action="store_true",
        help="exit 2 if the source is already --size x --size; use this in CI/pipeline runs so a "
             "vacuous self-check fails instead of warning",
    )
    args = parser.parse_args()

    logger = setup_logging(name="preprocess")
    result = preprocess_path(args.image, size=args.size, pad_value=args.pad_value)
    logger.info(
        "image=%s orig=%dx%d -> resized=%dx%d pad=(%d,%d) scale=(%.6f,%.6f) tensor=%s %s",
        args.image,
        result.meta.orig_w,
        result.meta.orig_h,
        result.meta.resized_w,
        result.meta.resized_h,
        result.meta.pad_x,
        result.meta.pad_y,
        result.meta.scale_x,
        result.meta.scale_y,
        result.chw.shape,
        result.chw.dtype,
    )

    if result.meta.orig_w == args.size and result.meta.orig_h == args.size:
        message = (
            f"source image is already {args.size}x{args.size}: "
            "PIL returns a copy and the resize path is never exercised. "
            "Feed a full-resolution frame (VisDrone val is 1360x765 / 960x540 / 1920x1080)."
        )
        if args.require_resize:
            logger.error(message)
            return 2
        if args.compare_candidate and not args.allow_no_resize:
            logger.error(message)
            return 2
        logger.warning(message)

    if args.output_npy:
        out = Path(args.output_npy)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, result.chw)
        logger.info("wrote %s (%d bytes)", out, out.stat().st_size)
    if args.output_meta:
        save_meta(result.meta, args.output_meta)
        logger.info("wrote %s", args.output_meta)

    status = 0
    if args.compare_candidate:
        cand_path = Path(args.compare_candidate)
        if not cand_path.is_file():
            logger.error("candidate dump not found: %s", cand_path)
            return 2
        report = compare_tensors(result.chw, np.load(cand_path))
        logger.info(
            "compare vs %s: shape=%s dtype=%s max_abs_diff=%s mismatched=%s exact=%s",
            cand_path,
            report["candidate_shape"],
            report["candidate_dtype"],
            report["max_abs_diff"],
            report["mismatched_elements"],
            report["exact"],
        )
        if args.report:
            out = Path(args.report)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
                f.write("\n")
            logger.info("wrote %s", out)
        if not report["exact"]:
            logger.error(
                "preprocess mismatch: the board resize is not bit-identical to PIL bilinear. "
                "Check PRECISION_BITS=22, the 1<<21 accumulator seed, clip8 saturation, "
                "and the horizontal_first branch."
            )
            status = 1
        else:
            logger.info("preprocess self-check PASSED: max_abs_diff == 0")

    logger.info(
        "RKNN input contract: uint8 RGB, NCHW, 0..255; normalisation must be "
        "configured in rknn.config(mean_values/std_values), not here."
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
