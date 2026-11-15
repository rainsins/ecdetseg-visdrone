"""Runtime device/precision policy; independent of model and config registries."""
import os
import torch

_active_device = None


def resolve_device(requested=None):
    requested = str(requested or 'auto')
    if requested == 'auto':
        requested = next((name for name in ('cuda', 'xpu')
                          if getattr(torch, name, None) is not None and
                          getattr(torch, name).is_available()), 'cpu')
    device = torch.device(requested)
    if device.type not in ('cpu', 'cuda', 'xpu'):
        raise ValueError(f'Unsupported device: {device}')
    if device.type == 'cpu':
        return torch.device('cpu')
    accelerator = getattr(torch, device.type, None)
    if accelerator is None or not accelerator.is_available():
        raise RuntimeError(f'Requested {device} is unavailable in torch {torch.__version__}')
    local_rank = os.environ.get('LOCAL_RANK') if 'RANK' in os.environ else None
    index = device.index
    if local_rank is not None:
        if index is not None and index != int(local_rank):
            raise ValueError('Explicit device index conflicts with LOCAL_RANK; use an unindexed device under torchrun')
        index = int(local_rank)
    if index is None:
        index = accelerator.current_device()
    if not 0 <= index < accelerator.device_count():
        raise ValueError(f'Invalid {device.type} index {index}')
    return torch.device(device.type, index)


def activate_device(requested=None):
    global _active_device
    device = resolve_device(requested)
    if device.type != 'cpu':
        getattr(torch, device.type).set_device(device.index)
    _active_device = device
    return device


def active_device():
    return _active_device if _active_device is not None else resolve_device()


def precision_policy(device, precision='auto', use_amp=True):
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda', 'xpu'):
        raise ValueError(f'Unsupported device: {device}')
    if precision not in ('auto', 'fp32', 'fp16', 'bf16'):
        raise ValueError(f'Unsupported precision: {precision}')
    if not use_amp and precision not in ('auto', 'fp32'):
        raise ValueError('Explicit mixed precision conflicts with use_amp=False')
    selected = precision
    if selected == 'auto':
        selected = {'cpu': 'fp32', 'cuda': 'fp16', 'xpu': 'bf16'}[device.type] if use_amp else 'fp32'
    if selected == 'bf16' and device.type == 'cuda':
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError(f'BF16 is not supported on {device}')
    if selected == 'bf16' and device.type == 'xpu':
        with torch.xpu.device(device):
            if not torch.xpu.is_bf16_supported():
                raise ValueError(f'BF16 is not supported on {device}')
    if selected != 'fp32' and not torch.amp.autocast_mode.is_autocast_available(device.type):
        raise ValueError(f'Autocast is not supported on {device}')
    dtype = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[selected]
    return selected, dtype
