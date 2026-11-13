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
    * when the bad counter reaches `patience`, `step()` returns True once.
"""
from __future__ import annotations


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

    def step(self, epoch: int, metric: float) -> bool:
        """Feed one epoch's EMA mAP; return True when training should stop."""
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
