"""Standalone model construction and strict official checkpoint loading."""
from pathlib import Path
from types import MethodType
import torch
import torch.nn.functional as F
from . import models
from .quan_w import Conv1dLSQ, Conv2dLSQ, LinearLSQ

QUANTIZED_LAYERS = (Conv1dLSQ, Conv2dLSQ, LinearLSQ)


def build_model(num_classes=1000, time_steps=1,
                architecture='spikformer_8_15M_CAFormer', **kwargs):
    if time_steps < 1:
        raise ValueError('time_steps must be positive')
    factory = getattr(models, architecture, None)
    if not architecture.startswith('spikformer_') or not callable(factory):
        raise ValueError('Unknown architecture: ' + architecture)
    model = factory(num_classes=num_classes, **kwargs)
    model.T = time_steps
    return model


def signed_weight(module):
    step = module.legacy_signed_step.reshape(-1, *([1] * (module.weight.ndim - 1)))
    lo, hi = -2 ** (module.nbits - 1), 2 ** (module.nbits - 1) - 1
    return (module.weight / step).clamp(lo, hi).round() * step


def legacy_forward(module, x):
    weight = signed_weight(module)
    if isinstance(module, Conv2dLSQ):
        return F.conv2d(x, weight, module.bias, module.stride, module.padding, module.dilation, module.groups)
    if isinstance(module, Conv1dLSQ):
        return F.conv1d(x, weight, module.bias, module.stride, module.padding, module.dilation, module.groups)
    return F.linear(x, weight, module.bias)


def load_frozen_checkpoint(path, device='cpu', num_classes=1000, time_steps=1,
                           architecture='spikformer_8_15M_CAFormer', quantization='auto'):
    """Load state dicts only; preserve signed steps of released legacy checkpoints.

    Modern positive-scale checkpoints retain their native quantizer. Legacy
    signed forward is for frozen inference, not a weight-training STE.
    """
    if quantization not in ('auto', 'legacy_signed', 'positive'):
        raise ValueError(quantization)
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    state = payload.get('model', payload.get('state_dict', payload))
    if not isinstance(state, dict):
        raise TypeError('Expected a tensor state dictionary')
    state = dict(state)
    if state and all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    markers = [k for k in state if k.endswith('_lsq_positive_scale_version')]
    mode = ('positive' if markers else 'legacy_signed') if quantization == 'auto' else quantization
    if mode == 'legacy_signed' and markers:
        raise ValueError('Positive-scale checkpoint cannot be interpreted as legacy signed alpha')
    model = build_model(num_classes, time_steps, architecture)
    expected = model.state_dict()
    # Explicitly unused field in the released legacy classifier, not a relaxed load.
    if 'top_conv.topconv.weight' in state and 'top_conv.topconv.weight' not in expected:
        state.pop('top_conv.topconv.weight')
    missing = set(expected) - set(state)
    allowed = {k for k in missing if '.act.' in k or k.endswith('_lsq_positive_scale_version')}
    if missing - allowed:
        raise RuntimeError('Missing model fields: ' + repr(sorted(missing - allowed)))
    for k in missing:
        if '.act.' in k:
            state[k] = expected[k]
    steps = {}
    if mode == 'legacy_signed':
        for name, module in model.named_modules():
            if isinstance(module, QUANTIZED_LAYERS):
                step = state[name + '.alpha'].detach().clone()
                if not torch.isfinite(step).all() or (step == 0).any():
                    raise ValueError('Invalid signed quantization step: ' + name)
                steps[name] = step
    model.load_state_dict(state, strict=True)
    if mode == 'legacy_signed':
        for name, module in model.named_modules():
            if name in steps:
                module.register_buffer('legacy_signed_step', steps[name])
                module.forward = MethodType(legacy_forward, module)
    model.to(device).eval().requires_grad_(False)
    return model
