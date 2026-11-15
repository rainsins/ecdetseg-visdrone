#!/usr/bin/env python3
"""Minimal unit test for engine/early_stop.py (no torch, no XPU).

    python scripts/test_early_stop.py
"""
import importlib.util
import os
import json
from pathlib import Path
import tempfile
import unittest

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


class RecoveryTests(unittest.TestCase):
    def stopper(self):
        return EarlyStopper(enabled=True, start_epoch=2, patience=3, min_delta=0.001)

    def test_every_interruption_keeps_stop_epoch(self):
        metrics = [.1, .2, .3, .301, .4, .4, .4, .4, .9]
        full = self.stopper()
        expected = next(e for e, m in enumerate(metrics) if full.step(e, m))
        self.assertEqual(expected, 7)
        for split in range(expected + 1):
            first = self.stopper()
            for e in range(split + 1):
                first.step(e, metrics[e])
            resumed = self.stopper()
            resumed.load_state_dict(json.loads(json.dumps(first.state_dict())))
            actual = split if resumed.stopped else next(
                e for e in range(split + 1, len(metrics)) if resumed.step(e, metrics[e]))
            self.assertEqual(actual, expected)
            self.assertEqual(resumed.state_dict(), full.state_dict())

    def test_legacy_replay_missing_last_eval_and_duplicate_epochs(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as tmp:
            path = Path(tmp) / 'log.txt'
            rows = [dict(epoch=2, test_coco_eval_bbox=[.3]),
                    dict(epoch=3, test_coco_eval_bbox=[.9]),
                    dict(epoch=3, test_coco_eval_bbox=[.3]),
                    dict(epoch=8, test_coco_eval_bbox=[.9])]
            path.write_text('\n'.join(map(json.dumps, rows)) + '\n{"epoch":')
            es = self.stopper()
            es.restore_from_log(path, 4, .3)
            self.assertEqual(es.num_bad, 2)
            self.assertTrue(es.step(5, .3))
            missing = self.stopper()
            with self.assertRaisesRegex(ValueError, 'missing epoch 4'):
                missing.restore_from_log(path, 5, .3)

    def test_legacy_before_threshold_and_latched_stop(self):
        es = self.stopper()
        es.restore_from_log('/nonexistent/history', 1)
        self.assertIsNone(es.best)
        for e in range(2, 6):
            es.step(e, .3)
        state = es.state_dict()
        self.assertTrue(es.step(6, .99))
        self.assertEqual(state, es.state_dict())

    def test_reject_changed_config_duplicate_and_nan(self):
        es = self.stopper()
        es.step(2, .3)
        with self.assertRaisesRegex(ValueError, 'already observed'):
            es.step(2, .3)
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            es.step(3, float('nan'))
        changed = self.stopper()
        changed.patience = 4
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            changed.load_state_dict(es.state_dict())


if __name__ == '__main__':
    main()
    unittest.main()
