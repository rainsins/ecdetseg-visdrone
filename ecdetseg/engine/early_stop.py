"""
Early-stop fuse for EdgeCrafter (D-FINE) training.

Monitors the EMA COCO `mAP@0.5:0.95` already produced by the per-epoch validation
(`test_coco_eval_bbox[0]`, same metric as scripts/eval_final.py). It is a purely
additive safety net: it never touches best.pth / final.pth / EMA / optimizer.

Config (yaml, top-level, optional):
    early_stop:
      enabled: true        # default False -> no behaviour change for old protocols
      start_epoch: 100     # do not count anything before this epoch
      patience: 20         # consecutive non-improving epochs -> stop
      min_delta: 0.001     # an "improvement" must exceed best + min_delta

Semantics:
    * only epochs >= start_epoch are counted;
    * the first counted epoch establishes `best` (counts as improvement);
    * a counted epoch with metric <= best + min_delta increments the bad counter;
    * when the bad counter reaches `patience`, the stop decision stays latched.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


class EarlyStopper:
    def __init__(self, enabled=False, start_epoch=100, patience=20, min_delta=0.001, **kwargs):
        self.enabled = bool(enabled)
        self.start_epoch = int(start_epoch)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = None
        self.best_epoch = None
        self.num_bad = 0
        self.stopped = False
        self.last_epoch = -1
        self.restored = False

    def state_dict(self):
        return dict(version=1, enabled=self.enabled, start_epoch=self.start_epoch,
                    patience=self.patience, min_delta=self.min_delta, best=self.best,
                    best_epoch=self.best_epoch, num_bad=self.num_bad,
                    stopped=self.stopped, last_epoch=self.last_epoch)

    def load_state_dict(self, state):
        if state.get('version') != 1:
            raise ValueError('Unsupported early-stop checkpoint version')
        for key in ('enabled', 'start_epoch', 'patience', 'min_delta'):
            if state[key] != getattr(self, key):
                raise ValueError(f'Early-stop configuration changed: {key}')
        for key in ('best', 'best_epoch', 'num_bad', 'stopped', 'last_epoch'):
            setattr(self, key, state[key])
        self.restored = True

    def restore_from_log(self, path, last_epoch, current_metric=None):
        """Replay legacy validation history, never invent missing patience state.

        Legacy last.pth was saved before validation. Only its final epoch may
        use a freshly evaluated metric; earlier counted epochs must be logged.
        Last occurrence wins for epochs replayed after an interrupted save.
        """
        history = {}
        path = Path(path)
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a partial final line cannot supply a metric
                metric = row.get('test_coco_eval_bbox') or row.get('test_coco_eval_mask')
                if metric and isinstance(row.get('epoch'), int):
                    history[row['epoch']] = metric[0]
        if current_metric is not None:
            history.setdefault(last_epoch, current_metric)
        replay = type(self)(enabled=self.enabled, start_epoch=self.start_epoch,
                            patience=self.patience, min_delta=self.min_delta)
        for epoch in range(self.start_epoch, last_epoch + 1):
            if epoch not in history:
                raise ValueError(f'Cannot recover early-stop history: missing epoch {epoch} in {path}')
            if replay.stopped:
                raise ValueError('Legacy history continues past the recovered early-stop decision')
            replay.step(epoch, history[epoch])
        replay.last_epoch = last_epoch
        self.load_state_dict(replay.state_dict())
        print(f'[early-stop] recovered legacy history through epoch {last_epoch}; '
              f'bad={self.num_bad}, stopped={self.stopped}')

    def step(self, epoch: int, metric: float) -> bool:
        """Feed one epoch's EMA mAP; return True when training should stop."""
        if self.stopped:
            return True
        if epoch <= self.last_epoch:
            raise ValueError(f'Early-stop epoch already observed: {epoch}')
        if not math.isfinite(metric):
            raise ValueError(f'Non-finite early-stop metric at epoch {epoch}')
        self.last_epoch = epoch
        if not self.enabled or epoch < self.start_epoch:
            return False

        if self.best is None or metric > self.best + self.min_delta:
            self.best = metric if self.best is None else max(self.best, metric)
            self.best_epoch = epoch
            self.num_bad = 0
        else:
            self.num_bad += 1

        if self.num_bad >= self.patience:
            print(f"EARLY STOP at epoch {epoch}, best={self.best:.4f} (epoch {self.best_epoch})")
            self.stopped = True
            return True
        return False

    def __repr__(self):
        return (f"EarlyStopper(enabled={self.enabled}, start_epoch={self.start_epoch}, "
                f"patience={self.patience}, min_delta={self.min_delta})")
