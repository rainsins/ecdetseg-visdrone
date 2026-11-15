#!/usr/bin/env python3
"""Validate the actual interpreter/device before training, resume or evaluation."""
import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import platform
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torchvision
import yaml
from engine.device import activate_device, precision_policy
from engine.optim.amp import GradScaler


def inspect_runtime(device, precision='auto', use_amp=True, scaler_config=None):
    selected = activate_device(device)
    requested_precision = precision
    precision, dtype = precision_policy(selected, precision, use_amp)
    scaler_config = dict(scaler_config or {})
    scaler_config.pop('type', None)
    configured_device = scaler_config.pop('device', None)
    if configured_device is not None:
        from engine.device import resolve_device
        if resolve_device(configured_device) != selected:
            raise ValueError('scaler.device conflicts with selected device')
    configured_precision = scaler_config.pop('precision', None)
    if use_amp and configured_precision not in (None, requested_precision):
        raise ValueError('scaler.precision conflicts with selected precision')
    scaler = GradScaler(device=selected, precision=precision, **scaler_config) if use_amp else None
    # Exercise the requested device, not just an availability flag.
    x = torch.ones((8, 8), device=selected)
    with torch.autocast(selected.type, dtype=dtype, enabled=precision != 'fp32'):
        y = x @ x
    if not torch.isfinite(y).all().item():
        raise RuntimeError('Non-finite preflight result')
    packages = {name: version(name) for name in ('numpy', 'scipy', 'Pillow', 'pycocotools', 'PyYAML', 'transformers', 'calflops')}
    accelerator = getattr(torch, selected.type) if selected.type != 'cpu' else None
    return dict(device_type=selected.type, device=str(selected), precision=precision,
                scaler_enabled=scaler.is_enabled() if scaler is not None else False,
                packages=packages, recorded_at=datetime.now(timezone.utc).isoformat(),
                device_memory_bytes=accelerator.get_device_properties(selected).total_memory if accelerator else None,
                torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                python=platform.python_version(), executable=sys.executable,
                cuda_runtime=torch.version.cuda,
                device_name=getattr(torch, selected.type).get_device_name(selected) if selected.type != 'cpu' else platform.processor())


def check_existing_runtime(out, current):
    path = out / 'runtime.json'
    if path.exists():
        old = json.loads(path.read_text())
        keys = ('device_type', 'precision', 'scaler_enabled', 'torch', 'torchvision', 'python', 'cuda_runtime', 'device_name', 'packages')
        changes = {k: (old.get(k), current.get(k)) for k in keys if old.get(k) != current.get(k)}
        if changes:
            raise ValueError(f'Runtime changed: {changes}; use a new run directory')
        return old.get('legacy_policy_verified', False)
    if not any(out.glob('*.pth')):
        return False
    # Old checkpoints have no runtime policy. Do not fabricate past evidence.
    devices = set()
    for manifest in (out / 'resume_history').glob('*/manifest.json'):
        value = json.loads(manifest.read_text()).get('device')
        if value:
            devices.add(value.split(':')[0])
    log = out / 'train.log'
    if log.exists():
        devices.update(v.split(':')[0] for v in re.findall(r'launch:.*? -d (\S+)', log.read_text(errors='replace')))
    cfg = yaml.safe_load((out / 'snapshot/config_merged.yml').read_text())
    if devices != {current['device_type']}:
        raise ValueError(f'Legacy device evidence {sorted(devices)} does not establish {current["device_type"]}; use tuning in a new directory')
    legacy_precision = cfg.get('precision', 'auto')
    if legacy_precision == 'auto':
        legacy_precision = {'cpu': 'fp32', 'cuda': 'fp16', 'xpu': 'bf16'}[current['device_type']] if cfg.get('use_amp') else 'fp32'
    if legacy_precision != current['precision']:
        raise ValueError('Legacy precision changed; use a new run directory')
    legacy_scaler = cfg.get('scaler', {}).get('enabled')
    if legacy_scaler is None:
        legacy_scaler = bool(cfg.get('use_amp')) and current['device_type'] == 'cuda'
    if bool(legacy_scaler) != current['scaler_enabled']:
        raise ValueError('Legacy gradient scaling changed; use a new run directory')
    print('[runtime] legacy device/policy evidence checked; historical software equivalence remains unverified')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--precision', choices=('auto', 'fp32', 'fp16', 'bf16'), default=None)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load((args.out / 'snapshot/config_merged.yml').read_text()) if args.out else {}
    info = inspect_runtime(args.device, args.precision or cfg.get('precision', 'auto'), cfg.get('use_amp', True), cfg.get('scaler'))
    if args.out:
        info['legacy_policy_verified'] = check_existing_runtime(args.out, info)
        if not (args.out / 'runtime.json').exists():
            pending = args.out / '.runtime.json.pending'
            pending.write_text(json.dumps(info, indent=2) + '\n')
            pending.replace(args.out / 'runtime.json')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
