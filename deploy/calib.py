#!/usr/bin/env python3
"""Compatibility entry point for the calibration builder.

The implementation lives in ``deploy/make_calib.py`` (kept for backward
compatibility with the original pipeline).  Both names run the same code:

    python deploy/calib.py
    python deploy/make_calib.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deploy.make_calib import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
