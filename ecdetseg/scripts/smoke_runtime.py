#!/usr/bin/env python3
"""Bounded real-data ECDet smoke: train, checkpoint/resume, train, EMA eval.

Diagnostic settings (default 320px/batch2) are NOT the P150u protocol.
Writes only to a newly created output directory. No full epoch is run.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from engine.core import YAMLConfig
from engine.misc import dist_utils
from engine.solver._solver import BaseSolver
from engine.solver.ec_engine import train_one_epoch, evaluate
from runtime_preflight import inspect_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', required=True)
    parser.add_argument('--precision', default='auto', choices=('auto', 'fp32', 'fp16', 'bf16'))
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--res', type=int, default=320)
    parser.add_argument('--batch', type=int, default=2)
    args = parser.parse_args()
    if args.res < 64 or args.res % 32 or not 2 <= args.batch <= 4:
        parser.error('smoke requires resolution >=64 divisible by32 and batch2..4')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    dist_utils.setup_distributed(device=args.device, seed=42)
    info = inspect_runtime(args.device, args.precision)
    project = Path(__file__).resolve().parents[1]
    overrides = dict(device=args.device, precision=args.precision,
                     output_dir=str(args.out), tuning=str(args.weights.resolve()),
                     eval_spatial_size=[args.res, args.res],
                     ViTAdapter=dict(skip_load_backbone=True))
    for split in ('train', 'val'):
        overrides[split + '_dataloader'] = dict(total_batch_size=args.batch, num_workers=0,
            dataset=dict(img_folder=str(args.data_root / split),
                         ann_file=str(args.data_root / f'annotations_VisDrone_{split}.json')))
    cfg = YAMLConfig(str(project / 'configs/ecdet/ecdet_s_p150u.yml'), **overrides)
    solver = BaseSolver(cfg)
    start = time.monotonic()
    solver.train()
    solver.train_dataloader.set_epoch(0)
    batches = list(itertools.islice(solver.train_dataloader, 1))
    if len(batches) != 1:
        raise RuntimeError('No full training batch available')
    parameter = next(p for p in solver.model.parameters() if p.requires_grad)
    before = parameter.detach().clone()
    def step(epoch):
        return train_one_epoch(False, None, solver.model, solver.criterion, batches,
            solver.optimizer, solver.device, epoch, cfg.clip_max_norm,
            scaler=solver.scaler, ema=solver.ema, print_freq=1)
    first = step(0)
    if torch.equal(before, parameter):
        raise RuntimeError('Optimizer did not change the tracked parameter')
    solver.last_epoch = 0
    checkpoint = args.out / 'smoke.pth'
    torch.save(solver.state_dict(), checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    saved_parameter = parameter.detach().clone()
    with torch.no_grad():
        parameter.add_(1)
    solver.load_state_dict(state)
    torch.testing.assert_close(parameter, saved_parameter, rtol=0, atol=0)
    del state
    second = step(1)
    val = list(itertools.islice(solver.val_dataloader, 1))
    if not val:
        raise RuntimeError('No validation batch available')
    stats, _ = evaluate(solver.ema.module, solver.criterion, solver.postprocessor,
                        val, solver.evaluator, solver.device)
    result = dict(runtime=info, resolution=args.res, batch=args.batch, train_steps=2,
                  validation_batches=1, checkpoint_roundtrip=True, optimizer_changed=True,
                  loss_before_resume=first['loss'], loss_after_resume=second['loss'],
                  elapsed_seconds=time.monotonic()-start,
                  subset_metrics_not_benchmark=stats)
    (args.out / 'smoke.json').write_text(json.dumps(result, indent=2) + '\n')
    if solver.writer:
        solver.writer.close()
    print(json.dumps(result, indent=2))
    dist_utils.cleanup()


if __name__ == '__main__':
    main()
