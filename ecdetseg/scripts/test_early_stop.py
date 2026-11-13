#!/usr/bin/env python3
"""Minimal unit test for engine/early_stop.py (no torch, no XPU).

    python scripts/test_early_stop.py
"""
import importlib.util
import os

# Load the module by file path to avoid importing the engine package (which pulls torch).
_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine", "early_stop.py")
_spec = importlib.util.spec_from_file_location("early_stop", _p)
_es_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_es_mod)
EarlyStopper = _es_mod.EarlyStopper


def main():
    # disabled by default -> never stops, never counts
    es = EarlyStopper()
    assert es.enabled is False
    assert es.step(200, 0.5) is False and es.num_bad == 0 and es.best is None

    # enabled but epoch < start_epoch -> NO counting
    es = EarlyStopper(enabled=True, start_epoch=100, patience=3, min_delta=0.001)
    for e in range(100):
        assert es.step(e, 0.0) is False, e
    assert es.num_bad == 0 and es.best is None, (es.num_bad, es.best)

    # first counted epoch establishes best
    assert es.step(100, 0.30) is False
    assert es.best == 0.30 and es.best_epoch == 100 and es.num_bad == 0

    # improvements smaller than min_delta count as "no improvement"
    assert es.step(101, 0.3005) is False and es.num_bad == 1
    assert es.step(102, 0.3000) is False and es.num_bad == 2
    # third consecutive bad -> stop
    assert es.step(103, 0.2900) is True and es.stopped is True

    # a real improvement (>= min_delta) resets the counter
    es = EarlyStopper(enabled=True, start_epoch=0, patience=3, min_delta=0.001)
    es.step(0, 0.30); es.step(1, 0.30); es.step(2, 0.30)      # 2 bad
    assert es.num_bad == 2
    assert es.step(3, 0.40) is False and es.num_bad == 0 and es.best == 0.40

    print("early_stop tests OK")


if __name__ == "__main__":
    main()
