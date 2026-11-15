"""
reference
- https://github.com/pytorch/vision/blob/main/references/detection/utils.py
- https://github.com/facebookresearch/detr/blob/master/util/misc.py#L406

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import atexit
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn
import torch.distributed
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

# from torch.utils.data.dataloader import DataLoader
from ..data import DataLoader
from ..device import active_device, activate_device


def _accelerator_module():
    device = active_device()
    return getattr(torch, device.type) if device.type != 'cpu' else None


def current_device() -> str:
    """Return the device type string ('xpu', 'cuda' or 'cpu') for the active accelerator."""
    return active_device().type


def setup_distributed(print_rank: int=0, print_method: str='builtin', seed: int=None, device=None):
    """
    env setup
    args:
        print_rank,
        print_method, (builtin, rich)
        seed,
    """
    selected = activate_device(device)
    keys = ('RANK', 'LOCAL_RANK', 'WORLD_SIZE')
    enabled_dist = any(k in os.environ for k in keys)
    if enabled_dist:
        if not all(k in os.environ for k in keys):
            raise ValueError('Incomplete distributed environment: require RANK, LOCAL_RANK, WORLD_SIZE')
        backend = {'cpu': 'gloo', 'cuda': 'nccl', 'xpu': 'xccl'}[selected.type]
        torch.distributed.init_process_group(backend=backend, init_method='env://')
        print(f'Initialized distributed mode: {backend} device={selected}')
    else:
        print('Not init distributed mode.')

    setup_print(get_rank() == print_rank, method=print_method)
    if seed is not None:
        setup_seed(seed)

    return enabled_dist


def setup_print(is_main, method='builtin'):
    """This function disables printing when not in master process
    """
    import builtins as __builtin__

    if method == 'builtin':
        builtin_print = __builtin__.print

    elif method == 'rich':
        import rich
        builtin_print = rich.print

    else:
        raise AttributeError('')

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_available_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


@atexit.register
def cleanup():
    """cleanup distributed environment
    """
    if is_dist_available_and_initialized():
        torch.distributed.destroy_process_group()


def get_rank():
    if not is_dist_available_and_initialized():
        return 0
    return torch.distributed.get_rank()


def get_world_size():
    if not is_dist_available_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)



def warp_model(
    model: torch.nn.Module,
    sync_bn: bool=False,
    dist_mode: str='ddp',
    find_unused_parameters: bool=False,
    compile: bool=False,
    compile_mode: str='reduce-overhead',
    **kwargs
):
    if is_dist_available_and_initialized():
        local_index = active_device().index
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) if sync_bn else model
        if dist_mode == 'dp':
            if current_device() != 'cuda':
                raise RuntimeError('DataParallel is only supported for CUDA; use DDP for XPU.')
            model = DP(model, device_ids=[local_index], output_device=local_index)
        elif dist_mode == 'ddp':
            if current_device() != 'cpu':
                model = DDP(model, device_ids=[local_index], output_device=local_index,
                            find_unused_parameters=find_unused_parameters)
            else:
                model = DDP(model, device_ids=None, find_unused_parameters=find_unused_parameters)
        else:
            raise AttributeError('')

    if compile:
        model = torch.compile(model, mode=compile_mode)

    return model

def de_model(model):
    return de_parallel(de_complie(model))


def warp_loader(loader, shuffle=False):
    if is_dist_available_and_initialized():
        sampler = DistributedSampler(loader.dataset, shuffle=shuffle)
        loader = DataLoader(loader.dataset,
                            loader.batch_size,
                            sampler=sampler,
                            drop_last=loader.drop_last,
                            collate_fn=loader.collate_fn,
                            pin_memory=loader.pin_memory,
                            num_workers=loader.num_workers)
    return loader



def is_parallel(model) -> bool:
    # Returns True if model is of type DP or DDP
    return type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)


def de_parallel(model) -> nn.Module:
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def reduce_dict(data, avg=True):
    """
    Args
        data dict: input, {k: v, ...}
        avg bool: true
    """
    world_size = get_world_size()
    if world_size < 2:
        return data

    with torch.no_grad():
        keys, values = [], []
        for k in sorted(data.keys()):
            keys.append(k)
            values.append(data[k])

        values = torch.stack(values, dim=0)
        torch.distributed.all_reduce(values)

        if avg is True:
            values /= world_size

        return {k: v for k, v in zip(keys, values)}


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    torch.distributed.all_gather_object(data_list, data)
    return data_list


def sync_time():
    """sync_time
    """
    accelerator = _accelerator_module()
    if accelerator is not None:
        accelerator.synchronize(active_device())

    return time.time()



def setup_seed(seed: int, deterministic=False):
    """setup_seed for reproducibility
    torch.manual_seed(3407) is all you need. https://arxiv.org/abs/2109.08203
    """
    seed = seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    accelerator = _accelerator_module()
    if accelerator is not None:
        accelerator.manual_seed_all(seed)

    # memory will be large when setting deterministic to True
    if current_device() == 'cuda' and torch.backends.cudnn.is_available() and deterministic:
        torch.backends.cudnn.deterministic = True


# for torch.compile
def check_compile():
    import warnings

    import torch
    gpu_ok = False
    if current_device() == 'cuda':
        device_cap = torch.cuda.get_device_capability(active_device())
        if device_cap in ((7, 0), (8, 0), (9, 0)):
            gpu_ok = True
    if not gpu_ok:
        warnings.warn(
            "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower "
            "than expected."
        )
    return gpu_ok

def is_compile(model):
    import torch._dynamo
    return type(model) in (torch._dynamo.OptimizedModule, )

def de_complie(model):
    return model._orig_mod if is_compile(model) else model
