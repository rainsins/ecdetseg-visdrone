"""Read legacy P150u artifacts without modifying or inventing their provenance.

CSV metrics remain fractions (0..1); report tables use percentage points.
Historical source hashes are deliberately empty: current hashes are not past evidence.
"""
import copy
import csv
import json
import math
from pathlib import Path
import re
import statistics

import yaml

from dump_config import _load_yaml_utils_torch_free, _reset_cfg

METRICS = ('AP', 'AP50', 'AP75', 'APS', 'APM', 'APL')
LABELS = {'final_ema': 'ema', 'final_raw': 'raw', 'best_ema': 'ema'}


def comparable_config(config):
    result = copy.deepcopy(config)
    # Runtime locations and seed are not protocol differences. Dataset and
    # initialization identities are retained separately, never guessed.
    for key in ('__include__', 'seed', 'output_dir', 'summary_dir', 'device',
                'resume', 'tuning', 'config'):
        result.pop(key, None)
    return result


def read_evaluations(path):
    rows = {}
    if not path.exists():
        return rows
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            label = row['label']
            if label not in LABELS or row.get('weights_type') != LABELS[label]:
                raise ValueError('unknown/mismatched weight label')
            if label in rows:
                raise ValueError(f'duplicate evaluation label {label}')
            for key in METRICS:
                if not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]):
                    raise ValueError(f'invalid {key}')
            if not isinstance(row.get('epoch'), int):
                raise ValueError('missing evaluation epoch')
            rows[label] = row
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f'{path}:{number}: {exc}') from exc
    return rows


def historical_rows(root, project, digest):
    rows = []
    yu = _load_yaml_utils_torch_free()
    for directory in sorted(Path(root).expanduser().resolve().glob('*')):
        match = re.fullmatch(r'(e[123])_s(\d+)', directory.name)
        if not directory.is_dir() or not match:
            continue
        if (directory / 'benchmark.json').exists():
            continue  # managed runs are handled by the existing exporter
        init, seed_text = match.groups()
        seed = int(seed_text)
        snapshot = directory / 'snapshot/config_merged.yml'
        if not snapshot.exists():
            snapshot = directory / 'snapshot/config.yml'
        if not snapshot.exists():
            raise ValueError(f'missing historical config snapshot: {directory}')
        cfg = yaml.safe_load(snapshot.read_text())
        command_path = directory / 'snapshot/command.txt'
        command = command_path.read_text() if command_path.exists() else ''
        tuning = re.search(r'^tuning=(.*)$', command, re.M)
        command_identity = re.search(r'^init=(e[123]) seed=(\d+)$', command, re.M)
        if not command_identity or command_identity.groups() != (init, seed_text) or cfg.get('seed') != seed:
            raise ValueError(f'historical init/seed identity mismatch: {directory}')
        expected_weights = {'e1': '<none>', 'e2': 'ecdet_s.pth', 'e3': 'ecdet_s_o3652coco.pth'}[init]
        if not tuning or Path(tuning[1]).name != expected_weights:
            raise ValueError(f'historical initialization is not evidenced: {directory}')
        resolution = cfg.get('eval_spatial_size', [])
        if len(resolution) != 2 or resolution[0] != resolution[1]:
            raise ValueError(f'invalid historical resolution: {directory}')
        expected = yu.load_config(str(project / 'configs/ecdet/ecdet_s_p150u.yml'), cfg={})
        expected['eval_spatial_size'] = resolution
        for split in ('train', 'val'):
            for key in ('img_folder', 'ann_file'):
                expected[f'{split}_dataloader']['dataset'][key] = cfg[f'{split}_dataloader']['dataset'][key]
        expected = yaml.safe_load(yaml.safe_dump(_reset_cfg(expected)))
        normalized = comparable_config(cfg)
        consistent = normalized == comparable_config(expected)
        logs = {}
        log = directory / 'log.txt'
        if log.exists():
            for line in log.read_text().splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # active writer may have a partial final record
                if isinstance(record.get('epoch'), int):
                    logs[record['epoch']] = record
        last_epoch = max(logs, default=-1)
        evaluations = read_evaluations(directory / 'eval.jsonl')
        done = (directory / 'DONE').exists()
        complete = (done and (directory / 'final.pth').is_file() and
                    (directory / 'best.pth').is_file() and set(evaluations) == set(LABELS) and
                    last_epoch >= 0 and all(e in logs for e in range(last_epoch + 1)) and
                    all(evaluations[k]['epoch'] == last_epoch for k in ('final_ema', 'final_raw')) and
                    0 <= evaluations['best_ema']['epoch'] <= last_epoch)
        # Completion means artifacts complete, not necessarily all 150 epochs.
        status = 'complete' if complete else ('evaluation_incomplete' if done else 'incomplete')
        device_match = re.findall(r'(?:^|\s)-d\s+(\S+)', command)
        if not device_match and (directory / 'train.log').exists():
            device_match = re.findall(r'launch:.*? -d (\S+)', (directory / 'train.log').read_text(errors='replace'))
        devices = sorted(set(device_match))
        device = devices[0] if len(devices) == 1 else 'unknown_or_mixed'
        metadata = dict(experiment='B0', revision='legacy', protocol='p150u',
                        resolution=resolution[0], init=init, seed=seed, variant='historical',
                        device=device, source_kind='historical', completion=status,
                        config_status='matches_p150u' if consistent else 'differs_from_p150u',
                        config_sha256=digest(cfg), comparison_sha256=digest(normalized),
                        source_sha256='', provenance='original_snapshot_only',
                        snapshot=str(snapshot), initialization=tuning[1],
                        resume_records=len(list((directory / 'resume_history').glob('*/manifest.json'))),
                        logged_last_epoch=last_epoch, output_dir=str(directory))
        if evaluations:
            rows.extend({**result, **metadata} for result in evaluations.values())
        else:
            rows.append({**metadata, 'label': 'pending'})
    return rows


def write_baseline_summary(rows, path, seeds=(42, 43, 44)):
    groups = {}
    for row in rows:
        if row['init'] != 'e2' or row['protocol'] != 'p150u':
            continue
        # Never pool resolutions, devices, variants, configurations or sources.
        key = tuple(row.get(k, '') for k in ('experiment', 'revision', 'protocol',
                    'resolution', 'init', 'variant', 'device', 'source_kind', 'comparison_sha256', 'source_sha256'))
        groups.setdefault(key, []).append(row)
    output = []
    for key, group in groups.items():
        eligible = [r for r in group if r.get('completion') == 'complete' and
                    r.get('config_status') == 'matches_p150u' and r['label'] == 'final_ema' and
                    int(r['seed']) in seeds]
        by_seed = {}
        for row in eligible:
            seed = int(row['seed'])
            if seed in by_seed:
                raise ValueError(f'duplicate seed {seed} in baseline comparison group')
            by_seed[seed] = row
        stats = {metric: dict(mean=statistics.mean(r[metric] for r in by_seed.values()) if by_seed else None,
                              sample_std=statistics.stdev(r[metric] for r in by_seed.values()) if len(by_seed) > 1 else None)
                 for metric in METRICS}
        output.append(dict(group=dict(zip(('experiment', 'revision', 'protocol', 'resolution',
                           'init', 'variant', 'device', 'source_kind', 'comparison_sha256', 'source_sha256'), key)),
                           primary_result='final_ema', metric_units='fraction', expected_seeds=list(seeds),
                           completed_seeds=sorted(by_seed), missing_seeds=sorted(set(seeds)-set(by_seed)),
                           complete=len(by_seed) == len(seeds), n=len(by_seed), statistics=stats,
                           provenance_note='Historical source equivalence is unverified; see original snapshots and resume_history.'))
    payload = dict(schema_version=1, groups=output)
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    return payload
