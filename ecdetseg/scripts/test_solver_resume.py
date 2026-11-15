"""Execute the real ECSolver.fit body with deterministic training/eval doubles.

No accelerator required: verifies checkpoint sequencing, augmentation rewind,
legacy recovery, and a stopped checkpoint's no-additional-training semantics.
"""
import ast
import copy
import datetime
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace as NS
import unittest

from test_early_stop import EarlyStopper


class SolverResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/tmp/opencode')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.saved = {}
        self.trained = []
        self.crash_epoch = None
        self.metrics = [.1, .2, .3, .4, .4, .4, .4, .9]
        owner = self

        class Base:
            def train(self):
                self.output_dir = owner.root
                self.last_epoch = -1
                self.train_dataloader = NS(dataset=NS(_transforms=NS(stop_epoch=4)),
                                           set_epoch=lambda e: None)
                self.val_dataloader = self.evaluator = self.device = None
                self.model = self.criterion = self.postprocessor = None
                self.ema = self.scaler = self.writer = self.optimizer = None
                self.lr_warmup_scheduler = None
                self.lr_scheduler = NS(step=lambda: None)
                self.iou_type = 'bbox'
                if self.cfg.resume:
                    self.load_resume_state(self.cfg.resume)

            def state_dict(self):
                return dict(last_epoch=self.last_epoch,
                            early_stopper=self.early_stopper.state_dict())

            def load_resume_state(self, path):
                state = copy.deepcopy(owner.saved[Path(path).name])
                self.last_epoch = state['last_epoch']
                if self.early_stopper is not None and 'early_stopper' in state:
                    self.early_stopper.load_state_dict(state['early_stopper'])

        def train(*args, **kwargs):
            epoch = args[7]
            if epoch == owner.crash_epoch:
                raise InterruptedError('simulated interruption')
            owner.trained.append(epoch)
            return dict(loss=1.)

        def evaluate(*args):
            return dict(coco_eval_bbox=[owner.solver.last_epoch >= 0 and
                        owner.metrics[owner.solver.last_epoch] or .1]), None

        def save(state, path):
            owner.saved[Path(path).name] = copy.deepcopy(state)

        source = Path(__file__).resolve().parents[1] / 'engine/solver/ec_solver.py'
        tree = ast.parse(source.read_text())
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ECSolver')
        namespace = dict(BaseSolver=Base, EarlyStopper=EarlyStopper,
                         datetime=datetime, time=time, json=json, torch=NS(),
                         train_one_epoch=train, evaluate=evaluate, stats=lambda cfg: (1, 'test'),
                         dist_utils=NS(is_dist_available_and_initialized=lambda: False,
                                       is_main_process=lambda: True, save_on_master=save,
                                       current_device=lambda: 'cpu'))
        exec(compile(ast.Module(body=[klass], type_ignores=[]), str(source), 'exec'), namespace)
        self.cls = namespace['ECSolver']

    def run_solver(self, resume=None):
        self.solver = self.cls()
        self.solver.cfg = NS(yaml_cfg=dict(early_stop=dict(enabled=True, start_epoch=2,
                                    patience=3, min_delta=.001)), resume=resume,
                             lrsheduler=None, epochs=8, checkpoint_freq=1,
                             clip_max_norm=1, print_freq=1)
        self.solver.fit()

    def test_checkpoint_order_resume_and_augmentation_rewind(self):
        self.crash_epoch = 5
        with self.assertRaises(InterruptedError):
            self.run_solver()
        self.assertEqual(self.saved['last.pth']['last_epoch'], 4)
        self.assertEqual(self.saved['last.pth']['early_stopper']['last_epoch'], 4)
        self.assertEqual(self.saved['last.pth']['early_stopper']['num_bad'], 1)
        for name, state in self.saved.items():
            self.assertEqual(state['last_epoch'], state['early_stopper']['last_epoch'], name)
        self.crash_epoch = None
        self.run_solver('last.pth')
        self.assertEqual(self.trained, list(range(7)))
        self.assertTrue(self.saved['final.pth']['early_stopper']['stopped'])
        self.assertEqual(self.saved['final.pth'], self.saved['last.pth'])
        before = self.trained.copy()
        self.run_solver('last.pth')
        self.assertEqual(self.trained, before)

    def test_legacy_resume_and_best_threshold(self):
        self.saved['last.pth'] = dict(last_epoch=4)
        self.saved['best.pth'] = dict(last_epoch=3)
        (self.root / 'log.txt').write_text('\n'.join(json.dumps(dict(
            epoch=e, test_coco_eval_bbox=[self.metrics[e]])) for e in range(4)) + '\n')
        self.run_solver('last.pth')
        self.assertEqual(self.trained, [5, 6])
        self.assertEqual(self.saved['final.pth']['early_stopper']['num_bad'], 3)
        self.assertEqual(self.saved['best.pth']['last_epoch'], 3)

    def test_state_epoch_mismatch_rejected(self):
        es = EarlyStopper(enabled=True, start_epoch=2, patience=3, min_delta=.001)
        es.step(2, .3)
        self.saved['last.pth'] = dict(last_epoch=4, early_stopper=es.state_dict())
        with self.assertRaisesRegex(ValueError, 'out of sync'):
            self.run_solver('last.pth')
        self.assertFalse(self.trained)

    def test_augmentation_rewind_keeps_patience_after_older_best(self):
        self.metrics = [.1, .6, .4, .4, .4, .4, .4, .9]
        self.run_solver()
        self.assertEqual(self.trained, list(range(6)))
        self.assertEqual(self.saved['final.pth']['last_epoch'], 5)
        self.assertEqual(self.saved['final.pth']['early_stopper']['best_epoch'], 2)
        self.assertEqual(self.saved['best.pth']['last_epoch'], 1)


if __name__ == '__main__':
    unittest.main()
