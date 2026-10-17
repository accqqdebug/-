"""Continuous-time event SNN model construction, without experiment infrastructure."""
import torch.nn as nn
from spikingjelly.activation_based import surrogate
from yolox.models import YOLOPAFPN, SpikingYOLOXHead, SpikingYOLOX
from yolox.utils.utils_snn import convert_to_spiking
from ct_frontend import EventTraceEmbedding


def build_model(num_classes=2, height=128, width=160, steps=3,
                depth=0.33, width_multiplier=0.5, tau_ms=(5., 100.)):
    channels = [256, 512, 1024]
    backbone = YOLOPAFPN(depth, width_multiplier, in_channels=channels, in_dim=2, act='silu')
    backbone = convert_to_spiking(backbone, spike_fn=surrogate.ATan(2.))
    head = SpikingYOLOXHead(num_classes, width_multiplier, in_channels=channels,
                           act='silu', spike_fn=surrogate.ATan(2.), full_spike=False)
    frontend = EventTraceEmbedding(height, width, tau_ms=tau_ms)
    model = SpikingYOLOX(backbone, head, frontend, T=steps)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eps = 1e-3
            module.momentum = 0.03
    model.head.initialize_biases(1e-2)
    return model
