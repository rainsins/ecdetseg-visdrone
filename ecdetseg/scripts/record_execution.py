#!/usr/bin/env python3
"""Append present-day execution evidence; never rewrite original snapshots."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import tarfile

import yaml
from benchmark import REPO, digest, file_hash, source_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--device', required=True)
    parser.add_argument('--init', required=True)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--weights')
    parser.add_argument('--reason', default='launcher invocation')
    args = parser.parse_args()
    out = Path(args.out).resolve()
    now = datetime.now(timezone.utc)
    history = out / 'resume_history' / now.strftime('%Y%m%dT%H%M%S%fZ')
    history.mkdir(parents=True, exist_ok=False)
    sources = source_files()
    with tarfile.open(history / 'source.tar.gz', 'w:gz') as archive:
        for name in sources:
            archive.add(REPO / name, arcname=name, recursive=False)
    (history / 'source_files.json').write_text(json.dumps(sources, indent=2))
    cfg = yaml.safe_load((out / 'snapshot/config_merged.yml').read_text())
    inputs = {}
    for split in ('train', 'val'):
        path = Path(cfg[f'{split}_dataloader']['dataset']['ann_file'])
        inputs[split + '_annotations'] = dict(path=str(path), sha256=file_hash(path))
    if args.weights:
        inputs['initial_weights'] = dict(path=args.weights, sha256=file_hash(args.weights))
    if args.checkpoint:
        inputs['resume_checkpoint'] = dict(path=args.checkpoint, sha256=file_hash(args.checkpoint))
    manifest = dict(schema_version=1, recorded_at=now.isoformat(), reason=args.reason,
                    mode='resume' if args.checkpoint else 'cold', init=args.init, seed=args.seed,
                    device=args.device, source_sha256=digest(sources),
                    archive_sha256=file_hash(history / 'source.tar.gz'), inputs=inputs,
                    original_snapshot_sha256={p.name: file_hash(p) for p in (out / 'snapshot').iterdir()
                                              if p.is_file() and not p.name.startswith('.')},
                    historical_source_equivalence='not asserted')
    if (out / 'runtime.json').exists():
        manifest['runtime'] = json.loads((out / 'runtime.json').read_text())
    diff = subprocess.run(['git', 'diff', '--binary'], cwd=REPO, capture_output=True, check=True)
    (history / 'current_git_diff.patch').write_bytes(diff.stdout)
    (history / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'[execution-record] {history}')


if __name__ == '__main__':
    main()
