"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

import datetime
import json
import time

import torch

from ..misc import dist_utils, stats
from ..optim.lr_scheduler import FlatCosineLRScheduler
from ..early_stop import EarlyStopper
from ._solver import BaseSolver
from .ec_engine import evaluate, train_one_epoch


class ECSolver(BaseSolver):

    def fit(self, ):
        args = self.cfg

        # early-stop fuse (config-gated; absent/disabled -> None => no behaviour change)
        _es = (getattr(self.cfg, 'yaml_cfg', None) or {}).get('early_stop') or {}
        self.early_stopper = EarlyStopper(**_es) if _es.get('enabled') else None
        # Must exist before BaseSolver.train() loads a resume checkpoint.
        self.train()

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-"*42 + "Start training" + "-"*43)
        
        stop_aug_epoch = self.train_dataloader.dataset._transforms.stop_epoch  # epoch to stop augmentation
        if args.lrsheduler is not None:
            no_aug_epochs = args.epochs - stop_aug_epoch
            flat_epochs = self.train_dataloader.dataset._transforms.mosaic_epoch if args.flat_epoch is None else args.flat_epoch
            iter_per_epoch = len(self.train_dataloader)
            warmup_iter = min(args.warmup_iter, 3 * iter_per_epoch)  
            
            print(f'FlatCosineLRScheduler with flat_epochs: {flat_epochs}, no_aug_epochs: {no_aug_epochs}, warmup_iter: {args.warmup_iter}')
            self.lr_scheduler = FlatCosineLRScheduler(
                self.optimizer,
                args.lr_gamma,
                iter_per_epoch,
                total_epochs=args.epochs,
                warmup_iter=warmup_iter,
                flat_epochs=flat_epochs,
                no_aug_epochs=no_aug_epochs,
                base_lrs=self.optimizer_base_lrs,
            )
            self.self_lr_scheduler = True
        else:
            self.self_lr_scheduler = False

        top1 = 0
        best_stat = {'epoch': -1, }
        # evaluate again before resume training
        if self.last_epoch >= 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )
            for k in test_stats:
                best_stat['epoch'] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]
                print(f'best_stat: {best_stat}')
            if self.early_stopper is not None:
                if not self.early_stopper.restored:
                    metric = test_stats.get('coco_eval_bbox') or test_stats.get('coco_eval_mask')
                    self.early_stopper.restore_from_log(
                        self.output_dir / 'log.txt', self.last_epoch,
                        metric[0] if metric else None)
                elif self.early_stopper.last_epoch != self.last_epoch:
                    raise ValueError('Checkpoint epoch and early-stop state are out of sync')
            # Resume must retain the historical VAL-selected best threshold.
            log_path = self.output_dir / 'log.txt'
            if log_path.exists():
                history = {}
                for line in log_path.read_text().splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row.get('epoch'), int) and row['epoch'] <= self.last_epoch:
                        history[row['epoch']] = row
                for previous_epoch, row in sorted(history.items()):
                    for k in test_stats:
                        metric = row.get(f'test_{k}')
                        if metric and metric[0] > best_stat[k]:
                            best_stat[k] = metric[0]
                            best_stat['epoch'] = previous_epoch
                            top1 = max(top1, metric[0])

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epochs):
            if self.early_stopper is not None and self.early_stopper.stopped:
                print('[early-stop] restored stop decision; no additional training epoch')
                break

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
                
            if epoch == stop_aug_epoch:
                if dist_utils.is_dist_available_and_initialized():
                    torch.distributed.barrier()
                # Rewind model/optimizer/EMA as before, not the patience clock.
                stopper = self.early_stopper
                self.early_stopper = None
                try:
                    self.load_resume_state(str(self.output_dir / 'best.pth'))
                finally:
                    self.early_stopper = stopper
                self.last_epoch = epoch - 1

            train_stats = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer, 
                self.device, 
                epoch, 
                max_norm=args.clip_max_norm, 
                print_freq=args.print_freq, 
                ema=self.ema, 
                scaler=self.scaler, 
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer
            )

            if not self.self_lr_scheduler:  # update by epoch 
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch += 1

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )

            # Commit only evaluated epochs, with this epoch's patience state.
            should_stop = False
            if self.early_stopper is not None:
                metric = test_stats.get('coco_eval_bbox') or test_stats.get('coco_eval_mask')
                if metric is None:
                    raise ValueError('Early stopping requires a validation AP metric')
                should_stop = self.early_stopper.step(epoch, metric[0])

            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat[k] > top1:
                    best_stat_print['epoch'] = epoch
                    top1 = best_stat[k]
                    if self.output_dir:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best.pth')
                        print(f'[best] saved best.pth at epoch {epoch} (NOTE: selected on VAL)')

                best_stat_print[k] = max(best_stat[k], top1)
                print(f'best_stat: {best_stat_print}')  # global best

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'last.pth']
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if self.iou_type in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval[self.iou_type].eval,
                                    self.output_dir / "eval" / name)
            if should_stop:
                break

            accelerator = getattr(torch, dist_utils.current_device(), None)
            if accelerator is not None:  # Just for clearing up GPU memory. You can remove it if you have enough GPU memory.
                accelerator.empty_cache()

        # Always persist the LAST-epoch weights, independent of the val-selected 'best'.
        if self.output_dir:
            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'final.pth')
            if dist_utils.is_main_process():
                (self.output_dir / 'best.pth.info').write_text(
                    "best.pth is SELECTED ON THE VAL SPLIT (best val AP); it is not a "
                    "held-out selection. Report test-dev for final numbers. final.pth "
                    "always holds the last epoch.\n")
            print(f'[final] saved final.pth (last epoch = {self.last_epoch})')

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval[self.iou_type].eval, self.output_dir / "eval.pth")

        return
