#!/usr/bin/env python3
"""Exercise benchmark management without torch, accelerator or real datasets."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import yaml

import benchmark as bench


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='ecdet-benchmark-', dir='/tmp/opencode')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / 'data'
        for split in ('train', 'val'):
            (self.data / split).mkdir(parents=True)
            (self.data / f'annotations_VisDrone_{split}.json').write_text('{}')
        self.weights = self.root / 'weights'
        self.weights.mkdir()
        (self.weights / 'ecdet_s.pth').write_bytes(b'fake weights; never loaded')
        self.output = self.root / 'runs'
        self.env = patch.dict(os.environ, {
            'PYTHON_BIN': str(bench.PROJECT / 'scripts/fake_train.sh'),
            'EVAL_BIN': str(bench.PROJECT / 'scripts/fake_eval.sh'),
            'FAKE_CODES': '0', 'FAKE_CKPT_OK': '1', 'RETRY_WAIT': '0',
            'SKIP_FINAL_EVAL': '1', 'SKIP_PLOT': '1',
            'FAKE_SLEEP': '', 'FAKE_NO_PROGRESS': '0',
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    def arguments(self, command='plan', experiment='B0', extra=()):
        return [command, experiment, '--data-root', str(self.data),
                '--weights-dir', str(self.weights), '--out-root', str(self.output), *extra]

    def cli(self, command='plan', experiment='B0', extra=()):
        return subprocess.run([os.sys.executable, str(bench.PROJECT / 'scripts/benchmark.py'),
                               *self.arguments(command, experiment, extra)],
                              cwd=bench.PROJECT, capture_output=True, text=True, timeout=40)

    def plan(self, extra=()):
        result = self.cli(extra=extra)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def start(self, extra=()):
        plan = self.plan(extra)
        result = self.cli('run', extra=extra)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return Path(plan['output_dir']), plan

    def test_preview_and_resolution_expansion(self):
        plan = self.plan(('--res', '960'))
        self.assertFalse(self.output.exists())
        cfg = plan['config']
        self.assertEqual(cfg['epochs'], 150)
        self.assertEqual(cfg['eval_spatial_size'], [960, 960])
        for split in ('train', 'val'):
            for op in cfg[f'{split}_dataloader']['dataset']['transforms']['ops']:
                if op['type'] == 'Resize':
                    self.assertEqual(op['size'], [960, 960])
                if op['type'] == 'Mosaic':
                    self.assertEqual(op['output_size'], 480)
        self.assertNotIn('__include__', cfg)

    def test_protocol_resolution_seed_variant_isolation(self):
        variants = [(), ('--allow-archived', '--protocol', 'p74u'), ('--allow-archived', '--protocol', 'p30u'),
                    ('--res', '960'), ('--seed', '43'), ('--init', 'e3'),
                    ('--variant', 'lr-study')]
        outputs = {self.plan(extra)['output_dir'] for extra in variants}
        self.assertEqual(len(outputs), len(variants))

    def test_default_protocol_and_archive_gate(self):
        plan = self.plan()
        self.assertEqual(plan['identity']['protocol'], 'p150u')
        self.assertEqual(plan['config']['epochs'], 150)
        self.assertEqual(plan['config']['train_dataloader']['total_batch_size'], 10)
        self.assertEqual(plan['config']['early_stop'],
                         dict(enabled=True, start_epoch=100, patience=20, min_delta=0.001))
        for protocol in ('p30u', 'p74u'):
            result = self.cli(extra=('--protocol', protocol))
            self.assertEqual(result.returncode, 2)
            self.assertIn('archived', result.stderr)

    def test_planned_methods_refused_before_writing(self):
        for method in ('B1', 'B2', 'B3', 'B4', 'B5'):
            result = self.cli('run', method)
            self.assertEqual(result.returncode, 2)
            self.assertIn('not verified', result.stderr)
        self.assertFalse(self.output.exists())

    def test_reserved_overrides_refused(self):
        for value in ('output_dir=/tmp/elsewhere', 'seed=99', '__include__=[other.yml]',
                      'device=cpu', 'train_dataloader.dataset.ann_file=other.json'):
            result = self.cli(extra=('-u', value))
            self.assertEqual(result.returncode, 2, result.stdout)
            self.assertIn('managed', result.stderr)

    def test_missing_tuning_weights_never_falls_back_to_cold(self):
        (self.weights / 'ecdet_s.pth').unlink()
        result = self.cli('run')
        self.assertEqual(result.returncode, 2)
        self.assertIn('missing initial_weights', result.stderr)
        self.assertFalse(self.output.exists())

    def test_launcher_snapshot_done_and_device(self):
        out, plan = self.start(('--device', 'cuda:0'))
        self.assertTrue((out / 'DONE').is_file())
        merged = yaml.safe_load((out / 'snapshot/config_merged.yml').read_text())
        self.assertEqual(bench.digest(merged), plan['config_sha256'])
        self.assertIn('-d cuda:0', (out / 'train.log').read_text())
        with tarfile.open(out / 'snapshot/source.tar.gz') as archive:
            self.assertIn('ecdetseg/scripts/benchmark.py', archive.getnames())
        result = self.cli('run', extra=('--device', 'cuda:0'))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((out / '.fake_attempts').read_text().strip(), '1')

    def test_changed_optimizer_refused_even_after_done(self):
        out, _ = self.start()
        result = self.cli('run', extra=('-u', 'optimizer.lr=0.123'))
        self.assertEqual(result.returncode, 2)
        self.assertIn('RESUME_MISMATCH: config_sha256', result.stderr)
        self.assertEqual((out / '.fake_attempts').read_text().strip(), '1')

    def test_checkpoint_resume_uses_existing_guardian(self):
        out, _ = self.start()
        (out / 'DONE').unlink()
        result = self.cli('run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('resume from', result.stdout)
        self.assertEqual((out / '.fake_attempts').read_text().strip(), '2')

    def test_snapshot_source_and_inputs_guard(self):
        out, plan = self.start()
        sources = bench.source_files()
        sources['ecdetseg/engine/new_module.py'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'source/environment'):
            bench.check_existing(plan, sources)
        path = out / 'snapshot/benchmark_config.yml'
        config = yaml.safe_load(path.read_text())
        config['optimizer']['lr'] = 0.123
        path.write_text(yaml.safe_dump(config))
        result = self.cli('run')
        self.assertEqual(result.returncode, 2)
        self.assertIn('snapshot/benchmark_config.yml', result.stderr)
        path.write_text(yaml.safe_dump(plan['config']))
        (self.data / 'annotations_VisDrone_val.json').write_text('{"changed":true}')
        result = self.cli('run')
        self.assertEqual(result.returncode, 2)
        self.assertIn('annotations or initialization weights changed', result.stderr)

    def test_concurrent_run_refused(self):
        import fcntl
        out, _ = self.start()
        with (out / '.benchmark.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.cli('run')
        self.assertEqual(result.returncode, 2)
        self.assertIn('already running', result.stderr)

    def test_summary_keeps_weight_labels_and_protocols_separate(self):
        for extra in ((), ('--allow-archived', '--protocol', 'p74u')):
            out, _ = self.start(extra)
            (out / 'last.pth').touch()
            (out / 'best.pth').touch()
            with patch.dict(os.environ, {'SKIP_FINAL_EVAL': '0'}):
                result = self.cli('run', extra=extra)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            count = bench.summarize(self.output, None)
        import csv
        rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
        self.assertEqual(count, 6)
        self.assertEqual({r['label'] for r in rows}, {'final_ema', 'final_raw', 'best_ema'})
        self.assertEqual({r['protocol'] for r in rows}, {'p150u', 'p74u'})
        self.assertTrue(all(r['config_sha256'] and r['source_sha256'] for r in rows))


if __name__ == '__main__':
    unittest.main()
