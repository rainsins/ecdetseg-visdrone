#!/usr/bin/env python3
"""ONNX -> RKNN INT8 conversion for RK3588.

Critical configuration notes
-----------------------------
* ``mean_values``/``std_values`` are the ONLY place normalization happens.
  Preprocess feeds uint8 RGB 0..255, so the values are the ImageNet constants
  multiplied by 255 (RGB order).
* ``quant_img_RGB2BGR=False`` keeps the channel order consistent with the RGB
  ONNX graph.
* ``target_platform="rk3588"`` must match the board and is also required for the
  x86 simulator to load the model.
* Calibration images are already stretched 640x640 PNGs produced by
  ``preprocess.py`` (PIL bilinear), so RKNN's reader does not apply a second,
  different resize.  Any other resampler here would shift the calibration
  distribution away from the inference distribution.

On failure the script prints the full RKNN traceback and the three causes that
account for almost every failed conversion in this pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import logging
import os
import re
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deploy import (  # noqa: E402
    DEFAULT_META_NAME,
    DEFAULT_ONNX_NAME,
    DEFAULT_RKNN_NAME,
    INPUT_SIZE,
    MEAN_RGB_255,
    STD_RGB_255,
    Timer,
    default_repo_root,
    ensure_files,
    log_environment,
    read_json,
    setup_logging,
    write_json,
)

MIN_RKNN_VERSION = (2, 3, 0)


def _require_ok(ret, what: str) -> None:
    """Most RKNN builder methods return 0/-1 rather than raising."""
    if ret not in (None, 0):
        raise RuntimeError(f"{what} returned {ret} (0=success, -1=failure)")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_version(text: str) -> tuple:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", text or "")
    if not match:
        return (0, 0, 0)
    return tuple(int(x) for x in match.groups())


def _call_supported(func, **kwargs) -> Dict[str, Any]:
    """Drop kwargs not present in this RKNN version's signature.

    Rockchip has renamed/deprecated config keys between 2.2 and 2.3.  Silently
    filtering is safer than failing on an otherwise valid conversion; every drop
    is logged.
    """
    logger = logging.getLogger("convert")
    params = inspect.signature(func).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    kept = {}
    dropped = []
    for key, value in kwargs.items():
        if key in params:
            kept[key] = value
        else:
            dropped.append(key)
    if dropped:
        logger.warning("rknn API does not support %s; dropped", dropped)
    return kept


@contextmanager
def _pushd(path: Path) -> Iterator[None]:
    """RKNN resolves dataset entries relative to the current working directory."""
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _validate_calib(calib_txt: Path) -> List[Path]:
    if not calib_txt.is_file():
        raise FileNotFoundError(
            f"calibration list not found: {calib_txt}\n"
            "Run: pixi run -e train python deploy/make_calib.py"
        )
    lines = [ln.strip() for ln in calib_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"calibration list is empty: {calib_txt}")
    if any(Path(ln).is_absolute() for ln in lines):
        raise ValueError("calib.txt must contain relative paths, but at least one line is absolute")

    resolved = [(calib_txt.parent / ln).resolve() for ln in lines]
    missing = [p for p in resolved if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} calibration image(s) listed in {calib_txt} do not exist.\n"
            f"First missing: {missing[0]}\n"
            "Rebuild with make_calib.py or fix --calib."
        )
    return resolved


def _import_rknn():
    try:
        from rknn.api import RKNN  # type: ignore
        return RKNN
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "Cannot import rknn.api.RKNN. The converter does NOT run in the training env.\n"
            "  pixi run -e rknn python -c 'from rknn.api import RKNN; print(RKNN)'\n"
            "If the environment is missing, merge deploy/pixi_feature.toml into pixi.toml "
            "and run `pixi install -e rknn`.\n"
            "If the wheel is unavailable on PyPI, install the Rockchip manylinux wheel manually.\n"
            f"Original error: {exc}"
        ) from exc


def convert(
    onnx_path: Path,
    calib_txt: Path,
    output_path: Path,
    target: str,
    mean_values: Sequence[float],
    std_values: Sequence[float],
    quantized_algorithm: str,
    optimization_level: int,
    input_shape: Sequence[int],
    auto_hybrid: bool,
    hybrid_cos_thresh: float,
    verbose: bool,
    grid_sample_target: str = None,
) -> Dict[str, Any]:
    logger = logging.getLogger("convert")
    if grid_sample_target not in (None, "gpu", "cpu", "auto"):
        raise ValueError("invalid GridSample target")
    if grid_sample_target and output_path.exists():
        raise ValueError("routing experiments require a new output path")
    RKNN = _import_rknn()

    try:
        version = importlib.metadata.version("rknn-toolkit2")
    except importlib.metadata.PackageNotFoundError:
        version = getattr(__import__("rknn"), "__version__", "unknown")
    logger.info("rknn-toolkit2 version=%s", version)
    if _parse_version(str(version)) < MIN_RKNN_VERSION:
        raise RuntimeError(f"rknn-toolkit2 {version} is too old; require >= {'.'.join(map(str, MIN_RKNN_VERSION))}")

    calib_images = _validate_calib(calib_txt)
    logger.info("calibration images: %d (first=%s)", len(calib_images), calib_images[0].name)
    logger.info("auto_hybrid=%s cos_thresh=%.4f", auto_hybrid, hybrid_cos_thresh)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        logger.warning("removing stale output before conversion: %s", output_path)
        output_path.unlink()

    rknn = RKNN(verbose=verbose)
    try:
        config_kwargs = dict(
            mean_values=[list(mean_values)],
            std_values=[list(std_values)],
            target_platform=target,
            quantized_dtype="w8a8",
            quantized_algorithm=quantized_algorithm,
            quantized_method="channel",
            optimization_level=optimization_level,
            # DETR logits are extremely sparse; 8-bit activation quantisation
            # clips the rare positive peaks and destroys query selection.  Keep
            # layers below the cosine threshold in FP16 (RK3588 supports FP16).
            auto_hybrid_cos_thresh=hybrid_cos_thresh,
            # Keep RGB: the ONNX graph was trained with RGB and preprocess.py
            # already performed BGR->RGB.
            quant_img_RGB2BGR=False,
        )
        if grid_sample_target:
            config_kwargs["op_target"] = {"GridSample": grid_sample_target}
        config_kwargs = _call_supported(rknn.config, **config_kwargs)
        if grid_sample_target and "op_target" not in config_kwargs:
            raise RuntimeError("installed API cannot configure op_target; refusing silent fallback")
        logger.info("rknn.config %s", config_kwargs)
        with Timer("rknn.config", logger):
            _require_ok(rknn.config(**config_kwargs), "rknn.config")

        load_kwargs = _call_supported(
            rknn.load_onnx,
            model=str(onnx_path),
            inputs=["images"],
            input_size_list=[list(input_shape)],
        )
        logger.info("rknn.load_onnx %s", load_kwargs)
        with Timer("rknn.load_onnx", logger):
            _require_ok(rknn.load_onnx(**load_kwargs), "rknn.load_onnx")

        build_kwargs = _call_supported(
            rknn.build,
            do_quantization=True,
            dataset=calib_txt.name,  # relative: _pushd() sets cwd to calib.txt parent
            rknn_batch_size=1,
            auto_hybrid=auto_hybrid,
        )
        logger.info("rknn.build %s (cwd=%s)", build_kwargs, calib_txt.parent)
        with Timer("rknn.build (INT8 quantisation)", logger):
            with _pushd(calib_txt.parent):
                _require_ok(rknn.build(**build_kwargs), "rknn.build")

        with Timer("rknn.export_rknn", logger):
            _require_ok(rknn.export_rknn(str(output_path)), "rknn.export_rknn")
    finally:
        try:
            rknn.release()
        except Exception:  # noqa: BLE001
            pass

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"RKNN export produced no file: {output_path}")

    logger.info("wrote %s (%.1f MiB)", output_path, output_path.stat().st_size / (1024 ** 2))
    return {
        "rknn_toolkit2_version": str(version),
        "target_platform": target,
        "quantized_dtype": "w8a8",
        "quantized_algorithm": quantized_algorithm,
        "quantized_method": "channel",
        "optimization_level": optimization_level,
        "op_target": {"GridSample": grid_sample_target} if grid_sample_target else None,
        "auto_hybrid": bool(auto_hybrid),
        "auto_hybrid_cos_thresh": float(hybrid_cos_thresh),
        "mean_values_rgb_255": list(mean_values),
        "std_values_rgb_255": list(std_values),
        "quant_img_RGB2BGR": False,
        "onnx": str(onnx_path),
        "input_shape": list(input_shape),
        "onnx_sha256": _sha256(onnx_path),
        "calib_txt": str(calib_txt),
        "calib_images": len(calib_images),
        "rknn": str(output_path),
        "rknn_sha256": _sha256(output_path),
        "rknn_size_bytes": output_path.stat().st_size,
    }


def _print_failure_hints(logger: logging.Logger) -> None:
    logger.error("")
    logger.error("========== RKNN conversion failed ==========")
    logger.error("Three causes account for almost every failure in this pipeline:")
    logger.error(
        "1) Unsupported ONNX op/shape: rknn-toolkit2 2.3 does not implement every opset-17 op. "
        "Re-run export with --opset 17, inspect the operator list "
        "(`python -c \"import onnx; print({n.op_type for n in onnx.load('...').graph.node})\"`), "
        "and replace dynamic/unsupported ops in the export wrapper."
    )
    logger.error(
        "2) Calibration path/dataset problem: calib.txt must contain paths relative to the file itself; "
        "the script chdirs to that directory. Verify all 300 images exist and are RGB 640x640 PNGs."
    )
    logger.error(
        "3) Configure mismatch: mean/std channel order (RGB, 0..255), target_platform=rk3588, "
        "numpy<2 in the rknn env, and a wheel matching the host python (cp310/cp311)."
    )
    logger.error("Do NOT continue to sim_eval.py until conversion succeeds.")
    logger.error("============================================")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_repo_root())
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--calib", type=Path, default=None, help="default: <artifacts>/calib.txt")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--meta", type=Path, default=None, help="export metadata (optional cross-check)")
    parser.add_argument("--target", default="rk3588")
    parser.add_argument("--mean", type=float, nargs=3, default=list(MEAN_RGB_255),
                        help="RGB mean in 0..255 (default: ImageNet mean x255)")
    parser.add_argument("--std", type=float, nargs=3, default=list(STD_RGB_255),
                        help="RGB std in 0..255 (default: ImageNet std x255)")
    parser.add_argument("--quantized-algorithm", default="normal", choices=["normal", "mmse", "kl_divergence"])
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--no-auto-hybrid", action="store_true",
                        help="disable FP16 fallback for sensitive layers (not recommended)")
    parser.add_argument("--hybrid-cos-thresh", type=float, default=0.9999,
                        help="auto_hybrid cosine threshold; the shipped artifact was built with "
                             "0.9999, and sim_eval's config cross-check compares against the "
                             ".rknn.json report, so changing this silently invalidates the "
                             "simulated mAP")
    parser.add_argument("--verbose", action="store_true", help="RKNN verbose logs")
    parser.add_argument("--grid-sample-target", choices=["gpu", "cpu", "auto"],
                        help="experimental conversion-time op_target; use a new --output")
    args = parser.parse_args()
    if args.grid_sample_target and args.output is None:
        parser.error("--grid-sample-target requires an explicit new --output")

    logger = setup_logging(name="convert")
    log_environment(logger)

    repo_root = args.repo_root.expanduser().resolve()
    artifacts = repo_root / "deploy" / "artifacts"
    onnx_path = (args.onnx or (artifacts / DEFAULT_ONNX_NAME)).expanduser().resolve()
    calib_txt = (args.calib or (artifacts / "calib.txt")).expanduser().resolve()
    output_path = (args.output or (artifacts / DEFAULT_RKNN_NAME)).expanduser().resolve()
    meta_path = (args.meta or (artifacts / DEFAULT_META_NAME)).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ensure_files([onnx_path], "ONNX model")
    logger.info("onnx=%s", onnx_path)
    logger.info("calib=%s", calib_txt)
    logger.info("output=%s", output_path)

    input_shape = [1, 3, INPUT_SIZE, INPUT_SIZE]
    if meta_path.is_file():
        meta = read_json(meta_path)
        input_shape = list(meta.get("input_shape", input_shape))
        if input_shape != [1, 3, INPUT_SIZE, INPUT_SIZE]:
            logger.warning("metadata input_shape=%s differs from frozen project fact %s", input_shape, [1, 3, INPUT_SIZE, INPUT_SIZE])
        logger.info(
            "metadata: config=%s, checkpoint=%s, mean=%s, std=%s",
            meta.get("config"),
            meta.get("checkpoint"),
            meta.get("rknn_normalization", {}).get("mean_values_rgb_255"),
            meta.get("rknn_normalization", {}).get("std_values_rgb_255"),
        )
    else:
        logger.warning("metadata not found at %s; continuing with CLI values", meta_path)

    try:
        report = convert(
            onnx_path=onnx_path,
            calib_txt=calib_txt,
            output_path=output_path,
            target=args.target,
            mean_values=args.mean,
            std_values=args.std,
            quantized_algorithm=args.quantized_algorithm,
            optimization_level=args.optimization_level,
            input_shape=input_shape,
            auto_hybrid=not args.no_auto_hybrid,
            hybrid_cos_thresh=args.hybrid_cos_thresh,
            verbose=args.verbose,
            grid_sample_target=args.grid_sample_target,
        )
    except Exception:  # noqa: BLE001 - requirement: print the complete RKNN error
        traceback.print_exc()
        _print_failure_hints(logger)
        return 1

    report.update(
        {
            "repo_root": str(repo_root),
            "export_metadata": str(meta_path) if meta_path.is_file() else None,
        }
    )
    report_path = output_path.with_suffix(output_path.suffix + ".json")
    write_json(report_path, report)
    logger.info("wrote conversion report %s", report_path)
    logger.info(
        "next: pixi run -e rknn python deploy/sim_eval.py --rknn %s",
        output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
