#!/usr/bin/env python3
"""Torch-free experiment registry, immutable run identities and result export."""
import argparse
import copy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

import yaml

from dump_config import _load_yaml_utils_torch_free, _reset_cfg

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
REGISTRY = PROJECT / 'configs/ecdet/experiments/registry.yml'
# Add an ID only after its actual model/loss wiring and tests exist. Editing
# registry.status alone cannot accidentally turn an unimplemented method on.
IMPLEMENTED = {'B0': 'v1'}
RESERVED = {'__include__', 'output_dir', 'summary_dir', 'seed',
            'eval_spatial_size', 'device', 'resume', 'tuning', 'config'}
DATA_KEYS = ('img_folder', 'ann_file')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def token(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', value):
        raise ValueError(f'unsafe identity component: {value!r}')
    return value


def registry():
    data = yaml.safe_load(REGISTRY.read_text())
    if data['schema_version'] != 1:
        raise ValueError('unsupported registry schema')
    return data


def source_files():
    # Include untracked implementation files too; exclude datasets/weights/outputs.
    paths = set((PROJECT / 'engine').rglob('*.py'))
    for directory in ('configs',):
        for suffix in ('*.yml', '*.yaml'):
            paths.update((PROJECT / directory).rglob(suffix))
    paths.update(PROJECT / p for p in ('train.py', 'scripts/benchmark.py',
                 'scripts/runtime_preflight.py', 'requirements/runtime-common.txt',
                 'requirements/runtime-torch-constraints.txt', 'scripts/smoke_runtime.py',
                 'scripts/run_exp.sh', 'scripts/run_all.sh', 'scripts/record_execution.py',
                 'scripts/historical_results.py', 'scripts/dump_config.py', 'scripts/eval_final.py'))
    paths.update(REPO / p for p in ('pixi.toml', 'pixi.lock'))
    return {str(p.relative_to(REPO)): file_hash(p) for p in sorted(paths) if p.is_file()}


def make_plan(args):
    reg = registry()
    protocols = dict(reg['protocols'])
    if args.allow_archived:
        protocols.update(reg.get('archived_protocols', {}))
    if args.protocol not in protocols:
        raise ValueError(f'{args.protocol} is archived; use P150u for new experiments, or --allow-archived for historical work')
    method = reg['experiments'][args.experiment]
    if method['status'] != 'ready' or IMPLEMENTED.get(args.experiment) != method['revision']:
        raise ValueError(f"{args.experiment} is {method['status']}: model/loss wiring is not verified")
    if args.seed < 0 or args.res <= 0 or args.res % 32:
        raise ValueError('seed must be non-negative; resolution must be a positive multiple of 32')
    identity = dict(experiment=args.experiment, revision=token(method['revision']),
                    protocol=args.protocol, resolution=args.res, init=args.init,
                    seed=args.seed, variant=token(args.variant), device=args.device)
    if args.runtime_python:
        executable = Path(args.runtime_python).expanduser().absolute()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError('runtime Python must be an executable file')
        identity['runtime_python'] = str(executable)
    out = (Path(args.out_root).expanduser().resolve() /
           f"{args.experiment}-{identity['revision']}" / args.protocol /
           f'r{args.res}' / identity['variant'] / f'{args.init}_s{args.seed}')
    previous = out / 'benchmark.json'
    old = json.loads(previous.read_text()) if previous.exists() else None
    root = args.data_root or (old or {}).get('data_root')
    if not root:
        raise ValueError('set --data-root or VISDRONE_ROOT (no training is started by plan)')
    root = str(Path(root).expanduser().resolve())
    yu = _load_yaml_utils_torch_free()
    cfg = yu.load_config(str(PROJECT / protocols[args.protocol]), cfg={})
    cfg.pop('__include__', None)
    cfg = yu.merge_dict(cfg, copy.deepcopy(method['overrides']))
    for update in args.update:
        key = update.split('=', 1)[0].strip()
        if key.split('.')[0] in RESERVED or any(
                key == f'{split}_dataloader.dataset' or
                key.startswith(f'{split}_dataloader.dataset.{field}')
                for split in ('train', 'val') for field in DATA_KEYS):
            raise ValueError(f'{key} is managed; use dedicated options')
    cfg = yu.merge_dict(cfg, yu.parse_cli(args.update))
    if args.precision is not None:
        cfg['precision'] = args.precision
    cfg.update(output_dir=str(out), seed=args.seed, device=args.device,
               eval_spatial_size=[args.res, args.res])
    for split in ('train', 'val'):
        cfg[f'{split}_dataloader']['dataset'].update(
            img_folder=f'{root}/{split}', ann_file=f'{root}/annotations_VisDrone_{split}.json')
    cfg = _reset_cfg(cfg)
    # YAML normalization makes tuples/dates obey the same serialization as the launcher.
    cfg = yaml.safe_load(yaml.safe_dump(cfg))
    weights_dir = str(Path(args.weights_dir).expanduser().resolve())
    identity['weights_dir'] = weights_dir
    return dict(schema_version=1, identity=identity, method=method, data_root=root,
                output_dir=str(out), config=cfg, config_sha256=digest(cfg))


def check_existing(plan, sources):
    out = Path(plan['output_dir'])
    path = out / 'benchmark.json'
    if not path.exists():
        if out.exists() and any(p.name != '.benchmark.lock' for p in out.iterdir()):
            raise ValueError(f'refusing unmanaged/nonempty directory: {out}')
        return False
    old = json.loads(path.read_text())
    for field in ('schema_version', 'identity', 'method', 'config_sha256'):
        if old.get(field) != plan[field]:
            raise ValueError(f'RESUME_MISMATCH: {field}; use a new --variant or method revision')
    if old.get('source_sha256') != digest(sources):
        raise ValueError('RESUME_MISMATCH: source/environment files changed; use a new --variant')
    for name in ('benchmark_config.yml', 'config_merged.yml'):
        path = out / 'snapshot' / name
        if name == 'benchmark_config.yml' and not path.exists():
            raise ValueError('missing immutable benchmark_config.yml')
        if path.exists() and digest(yaml.safe_load(path.read_text())) != plan['config_sha256']:
            raise ValueError(f'RESUME_MISMATCH: snapshot/{name}')
    archive = out / 'snapshot/source.tar.gz'
    if not archive.is_file() or file_hash(archive) != old.get('archive_sha256'):
        raise ValueError('RESUME_MISMATCH: source archive missing or changed')
    return True


def check_inputs(plan):
    cfg = plan['config']
    files = {}
    for split in ('train', 'val'):
        ds = cfg[f'{split}_dataloader']['dataset']
        if not Path(ds['img_folder']).is_dir():
            raise ValueError(f"missing image directory: {ds['img_folder']}")
        files[f'{split}_annotations'] = Path(ds['ann_file'])
    identity = plan['identity']
    if identity['init'] == 'e1':
        adapter = cfg.get('ViTAdapter', {})
        if adapter.get('weights_path') and not adapter.get('skip_load_backbone', False):
            files['initial_weights'] = PROJECT / adapter['weights_path']
    else:
        # Same E1/E2/E3 table as run_exp.sh; fail instead of its legacy cold fallback.
        name = {'e2': 'ecdet_s.pth', 'e3': 'ecdet_s_o3652coco.pth'}[identity['init']]
        files['initial_weights'] = Path(identity['weights_dir']) / name
    result = {}
    for name, path in files.items():
        if not path.is_file():
            raise ValueError(f'missing {name}: {path}')
        result[name] = dict(path=str(path.resolve()), sha256=file_hash(path))
    return result


def run(plan):
    out = Path(plan['output_dir'])
    # Check before creating directories, then repeat under an exclusive run lock.
    inputs = check_inputs(plan)
    sources = source_files()
    check_existing(plan, sources)
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.benchmark.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f'experiment is already running: {out}') from exc
        exists = check_existing(plan, sources)
        if exists:
            old = json.loads((out / 'benchmark.json').read_text())
            if old.get('inputs') != inputs:
                raise ValueError('RESUME_MISMATCH: annotations or initialization weights changed')
        else:
            snap = out / 'snapshot'
            snap.mkdir()
            (snap / 'benchmark_config.yml').write_text(yaml.safe_dump(plan['config'], sort_keys=False))
            (snap / 'source_files.json').write_text(json.dumps(sources, indent=2) + '\n')
            with tarfile.open(snap / 'source.tar.gz', 'w:gz') as archive:
                for name in sources:
                    archive.add(REPO / name, arcname=name, recursive=False)
            manifest = {k: v for k, v in plan.items() if k != 'config'}
            manifest.update(inputs=inputs, source_sha256=digest(sources),
                            archive_sha256=file_hash(snap / 'source.tar.gz'),
                            created_at=datetime.now(timezone.utc).isoformat())
            (out / 'benchmark.json').write_text(json.dumps(manifest, indent=2) + '\n')
        identity = plan['identity']
        env = os.environ.copy()
        env.pop('EXP_PRECISION', None)  # policy comes from the immutable config
        env['RUNTIME_PYTHON'] = identity.get('runtime_python', '')
        env.update(EXP_CONFIG=str(out / 'snapshot/benchmark_config.yml'),
                   EXP_OUT_ROOT=str(out.parent), EXP_RES=str(identity['resolution']),
                   P30U_CONFIG=str(out / 'snapshot/benchmark_config.yml'),
                   P30U_OUT_ROOT=str(out.parent), P30U_RES=str(identity['resolution']),
                   P74U_RES=str(identity['resolution']), VISDRONE_ROOT=plan['data_root'],
                   WEIGHTS_DIR=identity['weights_dir'], EXP_DEVICE=identity['device'])
        command = ['bash', str(PROJECT / 'scripts/run_exp.sh'), identity['init'], str(identity['seed'])]
        # Keep the existing guardian responsible for child process-group shutdown.
        child = subprocess.Popen(command, cwd=PROJECT, env=env, pass_fds=(lock.fileno(),))
        previous = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, lambda number, frame: child.send_signal(number))
        try:
            return child.wait()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def summarize(root, output, historical_root=None, baseline_summary=None):
    fields = ['experiment', 'revision', 'protocol', 'resolution', 'init', 'seed',
              'variant', 'device', 'config_sha256', 'source_sha256', 'label',
              'weights_type', 'epoch', 'AP', 'AP50', 'AP75', 'APS', 'APM', 'APL',
              'params', 'ckpt', 'note', 'output_dir', 'source_kind', 'completion',
              'config_status', 'comparison_sha256', 'provenance', 'snapshot',
              'initialization', 'resume_records', 'logged_last_epoch']
    rows = []
    for path in sorted(Path(root).expanduser().resolve().rglob('benchmark.json')):
        manifest = json.loads(path.read_text())
        results = path.parent / 'eval.jsonl'
        if not results.exists():
            continue
        for line_no, line in enumerate(results.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
                if result['label'] not in ('final_ema', 'final_raw', 'best_ema'):
                    raise ValueError('unknown evaluation label')
                row = {**manifest['identity'], **result,
                       'config_sha256': manifest['config_sha256'],
                       'source_sha256': manifest['source_sha256'], 'output_dir': str(path.parent),
                       'source_kind': 'managed', 'provenance': 'benchmark_manifest'}
                rows.append({key: row.get(key, '') for key in fields})
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f'{results}:{line_no}: {exc}') from exc
    if historical_root is not None:
        from historical_results import historical_rows
        rows.extend(historical_rows(historical_root, PROJECT, digest))
    if baseline_summary:
        from historical_results import write_baseline_summary
        write_baseline_summary(rows, baseline_summary, tuple(registry()['baseline_seeds']))
    stream = open(output, 'w', newline='') if output else sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, '') for key in fields} for row in rows)
    finally:
        if output:
            stream.close()
    return len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list', help='show implemented versus planned experiments')
    for name in ('plan', 'run'):
        p = commands.add_parser(name, help='preview only' if name == 'plan' else 'explicitly launch/resume')
        p.add_argument('experiment', choices=registry()['experiments'])
        p.add_argument('--protocol', choices=list(registry()['protocols']) + list(registry().get('archived_protocols', {})),
                       default=registry()['default_protocol'], help='default: p150u; old protocols require --allow-archived')
        p.add_argument('--allow-archived', action='store_true', help='explicit opt-in to historical protocols')
        p.add_argument('--init', choices=('e1', 'e2', 'e3'), default=registry()['default_init'])
        p.add_argument('--seed', type=int, default=42)
        p.add_argument('--res', type=int, default=640)
        p.add_argument('--variant', default='default')
        p.add_argument('--device', default='xpu')
        p.add_argument('--precision', choices=('auto', 'fp32', 'fp16', 'bf16'), default=None)
        p.add_argument('--runtime-python', default=os.environ.get('RUNTIME_PYTHON'),
                       help='absolute Python executable for ALL training/eval/checkpoint commands')
        p.add_argument('--data-root', default=os.environ.get('VISDRONE_ROOT'))
        p.add_argument('--weights-dir', default=os.environ.get('WEIGHTS_DIR', str(PROJECT)))
        p.add_argument('--out-root', default=str(PROJECT / 'outputs/benchmark'))
        p.add_argument('-u', '--update', nargs='+', default=[])
    p = commands.add_parser('summarize', help='export per-run/per-weight rows; no mixed-protocol averaging')
    p.add_argument('--out-root', default=str(PROJECT / 'outputs/benchmark'))
    p.add_argument('-o', '--output')
    p.add_argument('--historical-root', help='include legacy P150u runs with original snapshots')
    p.add_argument('--baseline-summary', help='write isolated E2 seed completion/mean/sample-std JSON')
    args = parser.parse_args(argv)
    try:
        if args.command == 'list':
            for key, method in registry()['experiments'].items():
                ready = method['status'] == 'ready' and IMPLEMENTED.get(key) == method['revision']
                print(f"{key} {method['revision']:4} {'ready' if ready else 'planned':8} {method['name']}")
            return 0
        if args.command == 'summarize':
            summarize(args.out_root, args.output, args.historical_root, args.baseline_summary)
            return 0
        plan = make_plan(args)
        if args.command == 'plan':
            check_existing(plan, source_files())
            print(json.dumps(plan, indent=2))
            return 0
        return run(plan)
    except (ValueError, KeyError, OSError, yaml.YAMLError) as exc:
        print(f'benchmark: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
