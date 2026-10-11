"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""


import torch

from ..core import register

__all__ = ['GradScaler', 'get_amp_dtype']


def _default_amp_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch, 'xpu', None) is not None and torch.xpu.is_available():
        return 'xpu'
    return 'cpu'


def get_amp_dtype(device_type: str = None):
    dt = device_type or _default_amp_device()
    if dt == 'cuda':
        return torch.float16
    if dt == 'xpu':
        return torch.bfloat16
    return torch.float32


@register()
class GradScaler(torch.amp.GradScaler):
    def __init__(self, device: str = None, enabled: bool = None, **kwargs):
        device = device or _default_amp_device()
        if enabled is None:
            enabled = (device == 'cuda')
        super().__init__(device=device, enabled=enabled, **kwargs)
