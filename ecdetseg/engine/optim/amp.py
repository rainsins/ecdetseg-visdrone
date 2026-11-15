"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""


import torch

from ..core import register
from ..device import active_device, resolve_device, precision_policy

__all__ = ['GradScaler', 'get_amp_dtype']


def _default_amp_device() -> str:
    return active_device().type


def get_amp_dtype(device_type: str = None, precision='auto'):
    return precision_policy(device_type or active_device(), precision)[1]


@register()
class GradScaler(torch.amp.GradScaler):
    def __init__(self, device: str = None, enabled: bool = None, precision='auto', **kwargs):
        device = resolve_device(device) if device is not None else active_device()
        self.precision, self.amp_dtype = precision_policy(device, precision)
        self.amp_enabled = self.precision != 'fp32'
        if enabled is None:
            enabled = self.precision == 'fp16'
        if enabled and not self.amp_enabled:
            raise ValueError('Gradient scaling cannot be enabled for FP32 policy')
        super().__init__(device=device.type, enabled=enabled, **kwargs)
