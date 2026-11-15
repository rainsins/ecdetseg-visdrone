"""Runtime policy regressions; GPU-selection mocks do not execute GPU kernels."""
from contextlib import ExitStack, nullcontext
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from engine import device as runtime
from engine.core import BaseConfig, YAMLConfig
from engine.misc import dist_utils
from engine.optim.amp import GradScaler
from engine.optim.ema import ModelEMA
from engine.solver._solver import BaseSolver
from engine.solver.ec_engine import train_one_epoch
from runtime_preflight import check_existing_runtime, inspect_runtime

CONFIG = Path(__file__).resolve().parents[1] / 'configs/ecdet/ecdet_s_p150u.yml'


class DeviceRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {}, clear=True))
        self.stack.enter_context(patch.object(runtime, '_active_device', None))
        self.tmp = self.stack.enter_context(tempfile.TemporaryDirectory(dir='/tmp/opencode'))
        self.root = Path(self.tmp)

    def hardware(self, cuda, xpu):
        for name, available in [('cuda', cuda), ('xpu', xpu)]:
            module = getattr(torch, name)
            for key, value in [('is_available', available), ('device_count', 4), ('current_device', 0),
                               ('set_device', None), ('is_bf16_supported', True)]:
                self.stack.enter_context(patch.object(module, key, return_value=value))
            self.stack.enter_context(patch.object(module, 'device', side_effect=lambda *a: nullcontext()))

    def test_auto_priority_and_explicit_cpu_on_mixed_host(self):
        for cuda, xpu, expected in [(False, False, 'cpu'), (False, True, 'xpu'),
                                     (True, False, 'cuda'), (True, True, 'cuda')]:
            with self.subTest(cuda=cuda, xpu=xpu), ExitStack() as scope:
                for name, available in [('cuda', cuda), ('xpu', xpu)]:
                    scope.enter_context(patch.object(getattr(torch, name), 'is_available', return_value=available))
                    scope.enter_context(patch.object(getattr(torch, name), 'current_device', return_value=0))
                    scope.enter_context(patch.object(getattr(torch, name), 'device_count', return_value=1))
                self.assertEqual(runtime.resolve_device().type, expected)
        self.hardware(True, True)
        runtime.activate_device('cpu')
        scaler = GradScaler()
        self.assertEqual(scaler._device, 'cpu')
        self.assertEqual(dist_utils.current_device(), 'cpu')
        self.assertIsNone(dist_utils._accelerator_module())
        self.assertFalse(scaler.is_enabled())

    def test_unavailable_invalid_and_local_rank(self):
        self.hardware(False, True)
        with self.assertRaisesRegex(RuntimeError, 'unavailable'):
            runtime.resolve_device('cuda:0')
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            runtime.resolve_device('xpu:4')
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            runtime.resolve_device('meta')
        with patch.dict(os.environ, RANK='5', LOCAL_RANK='1', WORLD_SIZE='8'):
            self.assertEqual(str(runtime.resolve_device('xpu')), 'xpu:1')
            with self.assertRaisesRegex(ValueError, 'LOCAL_RANK'):
                runtime.resolve_device('xpu:0')

    def test_precision_and_scaler_matrix(self):
        self.hardware(True, True)
        for name, default in [('cpu', 'fp32'), ('cuda', 'fp16'), ('xpu', 'bf16')]:
            self.assertEqual(runtime.precision_policy(name)[0], default)
            for precision in ('fp32', 'fp16', 'bf16'):
                with self.subTest(device=name, precision=precision):
                    scaler = GradScaler(device=name, precision=precision)
                    self.assertEqual(scaler._device, name)
                    self.assertEqual(scaler.is_enabled(), precision == 'fp16')
                    self.assertEqual(scaler.amp_enabled, precision != 'fp32')
                    self.assertEqual(GradScaler(device=name, precision=precision, enabled=False).is_enabled(), False)
            self.assertEqual(runtime.precision_policy(name, use_amp=False)[0], 'fp32')
        with self.assertRaisesRegex(ValueError, 'FP32'):
            GradScaler(device='cpu', precision='fp32', enabled=True)
        with self.assertRaisesRegex(ValueError, 'use_amp'):
            runtime.precision_policy('cpu', 'bf16', False)
        with patch.object(torch.cuda, 'is_bf16_supported', return_value=False):
            with self.assertRaisesRegex(ValueError, 'BF16'):
                runtime.precision_policy('cuda:0', 'bf16')

    def test_real_config_scaler_binding_and_conflicts(self):
        self.hardware(True, True)
        for name, precision in [('cpu', 'fp32'), ('cuda:1', 'fp16'), ('xpu:2', 'bf16')]:
            cfg = YAMLConfig(str(CONFIG), device=name)
            self.assertEqual(cfg.scaler._device, name.split(':')[0])
            self.assertEqual(cfg.scaler.precision, precision)
        for kwargs in [dict(scaler={'device': 'xpu'}), dict(scaler={'precision': 'fp16'}),
                       dict(use_amp=False, precision='bf16')]:
            with self.assertRaises(ValueError):
                YAMLConfig(str(CONFIG), device='cpu', **kwargs).scaler
        cfg = YAMLConfig(str(CONFIG), device='cpu')
        cfg.yaml_cfg.pop('scaler')
        self.assertEqual(cfg.scaler.precision, 'fp32')
        cfg = BaseConfig()
        cfg.device, cfg.use_amp, cfg.precision = 'cpu', True, 'fp32'
        self.assertEqual(cfg.scaler._device, 'cpu')

    def test_distributed_failure_propagates_and_uses_local_index(self):
        self.hardware(True, True)
        with patch.dict(os.environ, RANK='5', LOCAL_RANK='1', WORLD_SIZE='8'):
            with patch.object(torch.distributed, 'init_process_group', side_effect=RuntimeError('init failure')) as init:
                with self.assertRaisesRegex(RuntimeError, 'init failure'):
                    dist_utils.setup_distributed(device='cuda')
                init.assert_called_once_with(backend='nccl', init_method='env://')
            with patch.object(dist_utils, 'is_dist_available_and_initialized', return_value=True), patch.object(dist_utils, 'DDP') as ddp:
                model = torch.nn.Linear(2, 2)
                dist_utils.warp_model(model)
                self.assertEqual(ddp.call_args.kwargs['device_ids'], [1])
        with patch.dict(os.environ, RANK='0'):
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                dist_utils.setup_distributed(device='cpu')

    def test_distributed_backend_matrix(self):
        self.hardware(True, True)
        for device, backend in [('cpu', 'gloo'), ('cuda', 'nccl'), ('xpu', 'xccl')]:
            with patch.dict(os.environ, RANK='0', LOCAL_RANK='0', WORLD_SIZE='1'), \
                 patch.object(torch.distributed, 'init_process_group') as init, \
                 patch.object(dist_utils, 'setup_print'):
                self.assertTrue(dist_utils.setup_distributed(device=device))
                init.assert_called_once_with(backend=backend, init_method='env://')

    def test_sync_only_uses_selected_accelerator(self):
        self.hardware(True, True)
        with patch.object(torch.cuda, 'synchronize') as cuda, patch.object(torch.xpu, 'synchronize') as xpu:
            runtime.activate_device('cpu')
            dist_utils.sync_time()
            cuda.assert_not_called()
            xpu.assert_not_called()
            runtime.activate_device('xpu:2')
            dist_utils.sync_time()
            xpu.assert_called_once_with(torch.device('xpu:2'))
            cuda.assert_not_called()

    def solver(self, precision='fp32'):
        solver = BaseSolver.__new__(BaseSolver)
        solver.output_dir = self.root
        solver.last_epoch = 0
        solver.runtime_policy = dict(device_type='cpu', precision=precision, scaler_enabled=precision == 'fp16')
        solver.model = torch.nn.Linear(3, 2)
        solver.optimizer = torch.optim.AdamW(solver.model.parameters())
        solver.scaler = GradScaler(device='cpu', precision=precision)
        solver.ema = ModelEMA(solver.model)
        return solver

    def test_checkpoint_roundtrip_and_migration_refused_before_mutation(self):
        solver = self.solver()
        solver.model(torch.ones(2, 3)).sum().backward()
        solver.optimizer.step()
        solver.ema.update(solver.model)
        state = copy.deepcopy(solver.state_dict())
        target = self.solver()
        target.load_state_dict(state)
        for key, value in solver.model.state_dict().items():
            torch.testing.assert_close(value, target.model.state_dict()[key])
        self.assertTrue(target.optimizer.state)
        self.assertEqual(target.ema.updates, 1)
        for field, value in [('device_type', 'cuda'), ('precision', 'bf16'), ('scaler_enabled', True)]:
            changed = copy.deepcopy(state)
            changed['runtime_policy'][field] = value
            with self.assertRaisesRegex(ValueError, 'Resume backend/precision'):
                target.load_state_dict(changed)
        del state['runtime_policy']
        with self.assertRaisesRegex(ValueError, 'Legacy resume'):
            target.load_state_dict(state)
        (self.root / 'runtime.json').write_text(json.dumps(dict(target.runtime_policy, legacy_policy_verified=True)))
        target.load_state_dict(state)
        fp16 = self.solver('fp16')
        state = fp16.state_dict()
        state['scaler'] = {}
        with self.assertRaisesRegex(ValueError, 'empty scaler'):
            fp16.load_state_dict(state)

    def test_runtime_metadata_guard_and_legacy_evidence(self):
        # FP16 destination evaluation can load a BF16 checkpoint's empty scaler.
        evaluator = self.solver('fp16')
        del evaluator.optimizer
        state = evaluator.state_dict()
        state['scaler'] = {}
        state['runtime_policy'] = dict(device_type='xpu', precision='bf16', scaler_enabled=False)
        evaluator.load_state_dict(state)
        current = inspect_runtime('cpu')
        self.assertEqual(current['precision'], 'fp32')
        self.assertFalse(check_existing_runtime(self.root, current))
        (self.root / 'runtime.json').write_text(json.dumps(current))
        self.assertFalse(check_existing_runtime(self.root, current))
        with self.assertRaisesRegex(ValueError, 'Runtime changed'):
            check_existing_runtime(self.root, dict(current, precision='bf16'))
        (self.root / 'runtime.json').unlink()
        (self.root / 'last.pth').touch()
        (self.root / 'snapshot').mkdir()
        (self.root / 'snapshot/config_merged.yml').write_text('use_amp: true\n')
        with self.assertRaisesRegex(ValueError, 'Legacy device evidence'):
            check_existing_runtime(self.root, current)
        (self.root / 'train.log').write_text('launch: python train.py -d cpu -c cfg.yml\n')
        self.assertTrue(check_existing_runtime(self.root, current))

    def test_cpu_fp32_real_training_loop_no_autocast_warning(self):
        runtime.activate_device('cpu')
        class Model(torch.nn.Linear):
            def forward(self, x, targets=None):
                return {'pred_boxes': super().forward(x)}
        class Loss(torch.nn.Module):
            def forward(self, outputs, targets, **kwargs):
                return {'loss_box': outputs['pred_boxes'].square().mean()}
        model = Model(3, 4)
        before = model.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            stats = train_one_epoch(False, None, model, Loss(), [(torch.ones(2, 3), [{}, {}])],
                                    optimizer, torch.device('cpu'), 0, scaler=GradScaler(device='cpu'))
        self.assertFalse(any('autocast' in str(w.message).lower() for w in caught))
        self.assertTrue(torch.isfinite(torch.tensor(stats['loss'])))
        self.assertFalse(torch.equal(before, model.weight))


if __name__ == '__main__':
    unittest.main()
