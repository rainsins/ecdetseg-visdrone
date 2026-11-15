import csv
import json
from pathlib import Path
import tempfile
import unittest

import yaml
import benchmark as bench
from historical_results import historical_rows, write_baseline_summary, METRICS
from dump_config import _load_yaml_utils_torch_free, _reset_cfg


class HistoricalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/tmp/opencode')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def make_run(self, seed=42, init='e2', complete=True, ap=.3, res=640):
        d = self.root / f'{init}_s{seed}'
        (d / 'snapshot').mkdir(parents=True)
        yu = _load_yaml_utils_torch_free()
        cfg = yu.load_config(str(bench.PROJECT / 'configs/ecdet/ecdet_s_p150u.yml'), cfg={})
        cfg.update(seed=seed, output_dir=str(d), eval_spatial_size=[res, res])
        cfg = _reset_cfg(cfg)
        (d / 'snapshot/config_merged.yml').write_text(yaml.safe_dump(cfg))
        weights = {'e1': '<none>', 'e2': '/weights/ecdet_s.pth', 'e3': '/weights/ecdet_s_o3652coco.pth'}[init]
        (d / 'snapshot/command.txt').write_text(f'init={init} seed={seed}\ntuning={weights}\n')
        (d / 'train.log').write_text('launch: python train.py -d xpu -c config\n')
        (d / 'log.txt').write_text(json.dumps(dict(epoch=0, test_coco_eval_bbox=[ap])) + '\n')
        if complete:
            for name in ('final.pth', 'best.pth', 'DONE'):
                (d / name).touch()
            results = []
            for label, kind in (('final_ema', 'ema'), ('final_raw', 'raw'), ('best_ema', 'ema')):
                results.append(dict(label=label, weights_type=kind, epoch=0, **dict.fromkeys(METRICS, ap)))
            (d / 'eval.jsonl').write_text('\n'.join(map(json.dumps, results)))
        return d

    def rows(self):
        return historical_rows(self.root, bench.PROJECT, bench.digest)

    def summary(self, rows):
        return write_baseline_summary(rows, self.root / 'summary.json')['groups']

    def test_pending_seed_and_no_invented_source_hash(self):
        d = self.make_run()
        original = (d / 'snapshot/config_merged.yml').read_bytes()
        self.make_run(43, complete=False)
        self.make_run(init='e3', ap=.9)
        rows = self.rows()
        self.assertEqual(len(rows), 7)
        self.assertTrue(all(not r['source_sha256'] for r in rows))
        self.assertTrue(all(r['config_status'] == 'matches_p150u' for r in rows))
        group, = self.summary(rows)
        self.assertEqual(group['completed_seeds'], [42])
        self.assertEqual(group['missing_seeds'], [43, 44])
        self.assertEqual(group['statistics']['AP']['mean'], .3)
        self.assertIsNone(group['statistics']['AP']['sample_std'])
        self.assertEqual((d / 'snapshot/config_merged.yml').read_bytes(), original)

    def test_three_seeds_sample_std_and_weight_isolation(self):
        for seed, ap in ((42, .2), (43, .3), (44, .4)):
            self.make_run(seed, ap=ap)
        rows = self.rows()
        for r in rows:
            if r['label'] != 'final_ema':
                r['AP'] = .99
        group, = self.summary(rows)
        self.assertTrue(group['complete'])
        self.assertAlmostEqual(group['statistics']['AP']['mean'], .3)
        self.assertAlmostEqual(group['statistics']['AP']['sample_std'], .1)

    def test_resolution_and_config_mismatch_never_pool(self):
        self.make_run(42)
        self.make_run(43, res=960)
        d = self.make_run(44)
        path = d / 'snapshot/config_merged.yml'
        cfg = yaml.safe_load(path.read_text())
        cfg['early_stop']['patience'] += 1
        path.write_text(yaml.safe_dump(cfg))
        rows = self.rows()
        self.assertEqual(len(self.summary(rows)), 3)
        self.assertTrue(all(r['config_status'] == 'differs_from_p150u' for r in rows if r['seed'] == 44))
        self.assertFalse(any(g['complete'] for g in self.summary(rows)))

    def test_incomplete_evaluation_and_duplicate_labels(self):
        d = self.make_run()
        (d / 'final.pth').unlink()
        self.assertEqual(self.rows()[0]['completion'], 'evaluation_incomplete')
        p = d / 'eval.jsonl'
        p.write_text(p.read_text() + '\n' + p.read_text().splitlines()[0])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.rows()

    def test_identity_mismatch_rejected(self):
        d = self.make_run()
        (d / 'snapshot/command.txt').write_text('init=e3 seed=42\ntuning=/weights/ecdet_s.pth\n')
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            self.rows()

    def test_cli_exports_history_without_managed_manifest(self):
        self.make_run()
        out = self.root / 'results.csv'
        result = bench.main(['summarize', '--out-root', str(self.root / 'managed'),
                             '--historical-root', str(self.root), '-o', str(out),
                             '--baseline-summary', str(self.root / 'summary.json')])
        self.assertEqual(result, 0)
        with out.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r['source_kind'] == 'historical' for r in rows))


if __name__ == '__main__':
    unittest.main()
