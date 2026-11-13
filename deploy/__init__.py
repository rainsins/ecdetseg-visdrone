"""RK3588 deployment pipeline for ECDet-S / D-FINE on VisDrone.

This module intentionally contains only *project facts* and small, dependency-free
helpers.  Heavy imports (torch / rknn / cv2) are kept inside the individual
scripts so that importing :mod:`deploy` never initialises an XPU context or
requires the RKNN wheel.

Path policy
-----------
No absolute path is baked into the code.  Every script resolves defaults from
``Path(__file__).resolve().parents[1]`` (the EdgeCrafter workspace root) and
every relevant path can be overridden on the command line.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

# ---------------------------------------------------------------------------
# Frozen facts for this paper experiment.  Keeping them in one place prevents
# export / calibration / simulator from silently drifting apart.
# ---------------------------------------------------------------------------
NUM_CLASSES = 10                 # VisDrone 0..9, no category remap
NUM_QUERIES = 300                # DETR decoder queries
INPUT_SIZE = 640                 # static square input
SEED = 42

# ImageNet mean/std in the 0..1 convention used by the PyTorch config.
MEAN_RGB_01 = (0.485, 0.456, 0.406)
STD_RGB_01 = (0.229, 0.224, 0.225)
# RKNN config mean/std operate on the *file* domain (uint8 0..255).  Preprocess
# deliberately does not divide by 255, so RKNN must see 0..255 values.
MEAN_RGB_255 = tuple(round(v * 255.0, 3) for v in MEAN_RGB_01)
STD_RGB_255 = tuple(round(v * 255.0, 3) for v in STD_RGB_01)

VISDRONE_CLASSES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)

# Reference PyTorch EMA numbers (percent).  These are the values reported by
# outputs/p150u/e3_s42/eval_best_ema.json for the checkpoint used here.
PYTORCH_REFERENCE = {"AP": 28.82, "AP50": 48.54, "AP75": 28.63}

DEFAULT_ONNX_NAME = "ecdet_s_p150u_640.onnx"
DEFAULT_RKNN_NAME = "ecdet_s_p150u_640_rk3588_int8.rknn"
DEFAULT_META_NAME = "ecdet_s_p150u_640_meta.json"

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"


# ---------------------------------------------------------------------------
# Logging / timing
# ---------------------------------------------------------------------------
def setup_logging(level: int = logging.INFO, name: Optional[str] = None) -> logging.Logger:
    """Configure a single INFO-level console logger.

    ``force=True`` makes repeated calls from tests idempotent.  Timestamps are
    local time because the interesting duration is wall-clock on the build host.
    """
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        force=True,
    )
    return logging.getLogger(name or "deploy")


class Timer:
    """Context manager that logs elapsed wall-clock time for one pipeline step."""

    def __init__(self, label: str, logger: Optional[logging.Logger] = None):
        self.label = label
        self.logger = logger or logging.getLogger("deploy")
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        self.logger.info("%s ...", self.label)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.elapsed = time.perf_counter() - self.start
        if exc_type is None:
            self.logger.info("%s done in %.2fs", self.label, self.elapsed)
        else:
            self.logger.error("%s failed after %.2fs", self.label, self.elapsed)
        return False


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{sec:02d}s"
    return f"{minutes:d}m{sec:02d}s"


# ---------------------------------------------------------------------------
# Repo / data path discovery
# ---------------------------------------------------------------------------
def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_ecdet_root(repo_root: Path) -> Path:
    """Return the directory that contains ``engine/`` and ``configs/``.

    This repository keeps ECDet in ``ecdetseg/`` while the deployment directory
    lives at the workspace root.  Supporting both layouts keeps the scripts
    usable if the repo is flattened later.
    """
    repo_root = Path(repo_root).resolve()
    candidates = [repo_root / "ecdetseg", repo_root]
    for cand in candidates:
        if (cand / "engine" / "core").is_dir() and (cand / "configs").is_dir():
            return cand
    raise FileNotFoundError(
        "Cannot locate the ECDet code root. Looked for engine/core and configs in: "
        + ", ".join(str(c) for c in candidates)
    )


def resolve_data_root(repo_root: Path, explicit: Optional[os.PathLike | str] = None) -> Path:
    """Locate ``VisDrone`` without hard-coding an absolute path.

    Preferred order:
      1. ``--data-root`` from the command line
      2. ``<workspace>/dataset/VisDrone``
      3. ``<workspace>/../dataset/VisDrone`` (current layout on this machine)
      4. ``<cwd>/dataset/VisDrone``
    The first existing candidate wins; if none exists the first candidate is
    returned so the caller can emit a precise "missing annotation" error.
    """
    if explicit is not None:
        return Path(explicit).expanduser().resolve()

    repo_root = Path(repo_root).resolve()
    candidates = [
        repo_root / "dataset" / "VisDrone",
        repo_root.parent / "dataset" / "VisDrone",
        repo_root / "ecdetseg" / "dataset" / "VisDrone",
        Path.cwd() / "dataset" / "VisDrone",
    ]
    for cand in candidates:
        if (cand / "annotations_VisDrone_val.json").is_file():
            return cand.resolve()
    return candidates[0].resolve()


def resolve_user_path(value: os.PathLike | str, base: Path) -> Path:
    """Resolve a CLI path: absolute paths stay put, relative paths join ``base``."""
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = Path(base) / p
    return p.resolve()


def add_repo_to_sys_path(ecdet_root: Path) -> None:
    """Make ``import engine`` work exactly like the training scripts do."""
    root = str(Path(ecdet_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


# ---------------------------------------------------------------------------
# JSON helpers and environment diagnostics
# ---------------------------------------------------------------------------
def read_json(path: os.PathLike | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: os.PathLike | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def log_environment(logger: logging.Logger) -> None:
    """Log versions that affect ONNX/RKNN reproducibility."""
    logger.info("python=%s platform=%s", platform.python_version(), platform.platform())
    for module_name in ("numpy", "cv2", "onnx", "onnxruntime", "torch", "rknn", "pycocotools"):
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            logger.info("  %-12s %s", module_name, version)
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            logger.info("  %-12s unavailable (%s)", module_name, exc)


def ensure_files(paths: Sequence[Path], what: str = "required file") -> None:
    missing = [str(p) for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {what}:\n  " + "\n  ".join(missing))


def iter_coco_images(coco: dict) -> Iterable[dict]:
    return sorted(coco.get("images", []), key=lambda x: int(x["id"]))
