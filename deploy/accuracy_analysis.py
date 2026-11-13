#!/usr/bin/env python3
"""Layer-wise INT8 error analysis for one representative VisDrone image.

Why rebuild instead of loading the .rknn?
  rknn-toolkit2 documents that ``accuracy_analysis`` may only be called after
  ``build()`` (or ``hybrid_quantization_step2``).  ``load_rknn`` alone is not
  sufficient, so this script deterministically rebuilds from the ONNX and the
  calibration list using exactly the same config as convert.py.

The representative input is written twice:
  * ``input.npy``: exact uint8 (1,3,640,640) tensor fed to the simulator
  * ``input.png``: the same tensor re-encoded as a stretched RGB image, fallback
    for toolkit builds that only accept image paths

Output: ``deploy/snapshot/`` contains ``error_analysis.txt`` (per-layer cosine /
Euclidean error, version dependent) and any auxiliary dumps produced by RKNN.
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deploy import (  # noqa: E402
    DEFAULT_META_NAME,
    DEFAULT_ONNX_NAME,
    INPUT_SIZE,
    MEAN_RGB_255,
    STD_RGB_255,
    Timer,
    log_environment,
    read_json,
    setup_logging,
)
from deploy.convert import _call_supported, _pushd, _validate_calib  # noqa: E402
from deploy.preprocess import preprocess_path  # noqa: E402


def _import_rknn():
    try:
        from rknn.api import RKNN  # type: ignore
        return RKNN
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "Cannot import rknn.api.RKNN. accuracy_analysis.py must run in the rknn env:\n"
            "  pixi run -e rknn python deploy/accuracy_analysis.py ...\n"
            f"Original error: {exc}"
        ) from exc


def _require_ok(ret, what: str) -> None:
    """RKNN C-backed APIs often return -1 instead of raising."""
    if ret not in (None, 0):
        raise RuntimeError(f"{what} returned {ret} (0=success, -1=failure)")


def _prepare_input(image_path: Path, snapshot: Path, size: int) -> tuple[Path, Path, Dict[str, Any]]:
    result = preprocess_path(image_path, size=size)
    snapshot.mkdir(parents=True, exist_ok=True)
    npy_path = snapshot / "input.npy"
    np.save(npy_path, result.as_batch())
    rgb = np.transpose(result.chw, (1, 2, 0))
    png_path = snapshot / "input.png"
    if not cv2.imwrite(str(png_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise IOError(f"failed to write {png_path}")
    return npy_path, png_path, result.meta_dict()


def _report_snapshot(snapshot: Path, logger: logging.Logger, print_lines: int) -> List[str]:
    files = sorted(p for p in snapshot.rglob("*") if p.is_file())
    logger.info("snapshot contains %d files under %s", len(files), snapshot)
    for p in files:
        logger.info("  %-60s %8.1f KiB", p.relative_to(snapshot), p.stat().st_size / 1024.0)

    interesting = [p for p in files if "error" in p.name.lower() or p.suffix in (".txt", ".csv")]
    for p in interesting:
        logger.info("---- %s (first %d lines) ----", p, print_lines)
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f):
                    if i >= print_lines:
                        logger.info("... (truncated)")
                        break
                    logger.info("%s", line.rstrip("\n"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read %s: %s", p, exc)
    return [str(p) for p in files]


def _call_accuracy_analysis(rknn, candidates: Sequence[Any], snapshot: Path, logger: logging.Logger) -> None:
    """Try npy first, then png, then an in-memory batch.

    The accepted input type changed across toolkit releases; trying in order is
    more useful than failing because of one API nuance.
    """
    errors = []
    for label, inputs in candidates:
        try:
            logger.info("accuracy_analysis inputs=%s", label)
            kwargs = _call_supported(
                rknn.accuracy_analysis,
                inputs=inputs,
                output_dir=str(snapshot),
                target=None,
            )
            ret = rknn.accuracy_analysis(**kwargs)
            if ret not in (None, 0):
                raise RuntimeError(f"accuracy_analysis returned {ret} (0=success, -1=failure)")
            return
        except Exception as exc:  # noqa: BLE001 - fallback path
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
            logger.warning("accuracy_analysis with %s failed: %s", label, exc)
    raise RuntimeError("accuracy_analysis failed for every input form:\n  " + "\n  ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--calib", type=Path, default=None)
    parser.add_argument("--image", type=Path, default=None,
                        help="representative image; default = first line of calib.txt")
    parser.add_argument("--snapshot", type=Path, default=None, help="default: <repo>/deploy/snapshot")
    parser.add_argument("--target", default="rk3588")
    parser.add_argument("--mean", type=float, nargs=3, default=list(MEAN_RGB_255))
    parser.add_argument("--std", type=float, nargs=3, default=list(STD_RGB_255))
    parser.add_argument("--quantized-algorithm", default="normal", choices=["normal", "mmse", "kl_divergence"])
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--no-auto-hybrid", action="store_true")
    parser.add_argument("--hybrid-cos-thresh", type=float, default=0.999)
    parser.add_argument("--print-lines", type=int, default=80)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logger = setup_logging(name="accuracy")
    log_environment(logger)
    RKNN = _import_rknn()

    repo_root = args.repo_root.expanduser().resolve()
    artifacts = repo_root / "deploy" / "artifacts"
    onnx_path = (args.onnx or (artifacts / DEFAULT_ONNX_NAME)).expanduser().resolve()
    meta_path = (args.meta or (artifacts / DEFAULT_META_NAME)).expanduser().resolve()
    calib_txt = (args.calib or (artifacts / "calib.txt")).expanduser().resolve()
    snapshot = (args.snapshot or (repo_root / "deploy" / "snapshot")).expanduser().resolve()
    snapshot.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, Any] = {}
    if meta_path.is_file():
        meta = read_json(meta_path)
    input_size = int(meta.get("input_shape", [1, 3, INPUT_SIZE, INPUT_SIZE])[2])

    image_path = args.image
    if image_path is None:
        if not calib_txt.is_file():
            logger.error("no --image and calib.txt not found: %s", calib_txt)
            return 2
        first_rel = calib_txt.read_text(encoding="utf-8").splitlines()[0].strip()
        image_path = (calib_txt.parent / first_rel).resolve()
    image_path = image_path.expanduser().resolve()
    if not image_path.is_file():
        logger.error("representative image not found: %s", image_path)
        return 2

    logger.info("representative image=%s", image_path)
    npy_path, png_path, pre_meta = _prepare_input(image_path, snapshot, size=input_size)
    logger.info("input npy=%s png=%s preprocess=%s", npy_path, png_path, pre_meta)

    for p in (onnx_path, calib_txt):
        if not p.is_file():
            logger.error("required file missing: %s", p)
            return 2

    # rknn-toolkit2 documents that accuracy_analysis can only be called after
    # build() (or hybrid_quantization_step2); load_rknn alone is not sufficient.
    rknn = RKNN(verbose=args.verbose)
    try:
        with Timer("load_onnx + build (same config as convert.py)", logger):
            ret = rknn.config(
                **_call_supported(
                    rknn.config,
                    mean_values=[list(args.mean)],
                    std_values=[list(args.std)],
                    target_platform=args.target,
                    quantized_dtype="w8a8",
                    quantized_algorithm=args.quantized_algorithm,
                    quantized_method="channel",
                    optimization_level=args.optimization_level,
                    auto_hybrid_cos_thresh=args.hybrid_cos_thresh,
                    quant_img_RGB2BGR=False,
                )
            )
            _require_ok(ret, "rknn.config")
            ret = rknn.load_onnx(
                **_call_supported(
                    rknn.load_onnx,
                    model=str(onnx_path),
                    inputs=["images"],
                    input_size_list=[[1, 3, input_size, input_size]],
                )
            )
            _require_ok(ret, "rknn.load_onnx")
            _validate_calib(calib_txt)
            with _pushd(calib_txt.parent):
                ret = rknn.build(
                    **_call_supported(
                        rknn.build,
                        do_quantization=True,
                        dataset=calib_txt.name,
                        rknn_batch_size=1,
                        auto_hybrid=not args.no_auto_hybrid,
                    )
                )
            _require_ok(ret, "rknn.build")

        # accuracy_analysis wants a list of input paths; .npy carries the exact
        # uint8 RGB NCHW tensor used by the simulator.
        candidates = [
            ("input.npy", [str(npy_path)]),
            ("input.png", [str(png_path)]),
            ("in-memory uint8 batch", [np.load(npy_path)]),
            ("raw source image", [str(image_path)]),
        ]
        with Timer("rknn.accuracy_analysis", logger):
            _call_accuracy_analysis(rknn, candidates, snapshot, logger)
    except Exception:
        logger.error("accuracy analysis failed; full traceback follows")
        traceback.print_exc()
        logger.error(
            "Typical causes: (1) toolkit rejects the .npy shape -> the script already retries input.png; "
            "(2) simulator target must be None; (3) model/calibration mismatch or unsupported operator."
        )
        return 1
    finally:
        try:
            rknn.release()
        except Exception:  # noqa: BLE001
            pass

    files = _report_snapshot(snapshot, logger, args.print_lines)
    logger.info(
        "next: rank layers by error in the txt dump. A large error at a specific "
        "block usually means that block should use hybrid quantisation or stay FP16."
    )
    if not files:
        logger.warning("snapshot is empty; the toolkit version may write to a different directory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
