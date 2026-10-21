"""MSF-YOLOX adapters; inherited neck/head are NOT new paper contributions."""
import math
import torch
from torch import nn
from .MHSANet import mem_update_MSF, mem_update
from .paper_components import PaperPLIF, fcbr, kcl
from .max_spikeformer import MaxSpikeFormerBlock
from .yolo_pafpn_fully_snn import YOLOPAFPN
from .yolo_head_snn import YOLOXHead
from .yolox import YOLOX


class LearnedMSF(nn.Module):
    """Preserve upstream MSF firing/reset, learn decay: an adaptation, not binary PLIF."""
    def __init__(self, original):
        super().__init__()
        self.decay_w = nn.Parameter(torch.tensor(math.log(original.decay / (1 - original.decay))))
        self.init_thre, self.D = original.init_thre, original.D
        self.activation = original.act_fun_dict[original.surro_gate]

    def forward(self, x):
        mem, spike, values = torch.zeros_like(x[0]), torch.zeros_like(x[0]), []
        for value in x:
            mem = mem * self.decay_w.sigmoid() * (~(spike > 0)).to(x.dtype) + value
            spike = self.activation(mem, self.init_thre, self.D)
            values.append(spike)
        return torch.stack(values)


def replace_decay(module, strict_binary=False):
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, mem_update_MSF):
            replacement = PaperPLIF(decay=child.decay, thresh=child.init_thre) if strict_binary else LearnedMSF(child)
            setattr(module, name, replacement)
            count += 1
        elif isinstance(child, mem_update):
            setattr(module, name, PaperPLIF(decay=child.decay, thresh=child.thresh))
            count += 1
        else:
            count += replace_decay(child, strict_binary)
    return count


class PaperTransfer(YOLOX):
    def __init__(self, dataset='gen1', variant='baseline', time_steps=5, in_channels=3):
        if dataset not in ('gen1', 'coco') or variant not in ('baseline', 'paper', 'paper_binary'):
            raise ValueError('unsupported dataset/variant')
        backbone = YOLOPAFPN()
        head = YOLOXHead(2 if dataset == 'gen1' else 80)
        super().__init__(backbone, head)
        self.dataset, self.variant, self.epoch = dataset, variant, 0
        self.backbone.backbone.T = time_steps
        stem = self.backbone.backbone.conv1[0][0]
        if in_channels != stem.in_channels:
            self.backbone.backbone.conv1[0][0] = nn.Conv2d(in_channels, stem.out_channels, stem.kernel_size, stem.stride, stem.padding, bias=False)
        self.replaced_neurons = 0
        if variant != 'baseline':
            self.replaced_neurons = replace_decay(self, strict_binary=variant == 'paper_binary')
            # Deepest feature remains on the live backbone -> neck -> detection path.
            self.backbone.backbone.layer4 = nn.Sequential(
                self.backbone.backbone.layer4,
                MaxSpikeFormerBlock(1024, 1024, mlp_ratio=4., attention_mode='spike'),
            )
        self.head.initialize_biases(.01)
        self._features = {}
        # Hooks only cache features in training. No FFT/prediction replay during eval.
        if variant != 'baseline':
            for kind, layers in [('reference', self.head.stems), ('cls', self.head.cls_convs), ('reg', self.head.reg_convs)]:
                for scale, layer in enumerate(layers):
                    layer.register_forward_hook(self._cache(kind, scale))

    def _cache(self, kind, scale):
        def hook(module, inputs, output):
            if self.training:
                self._features[kind, scale] = output
        return hook

    def forward(self, x, targets=None):
        self._features.clear()
        outputs = super().forward(x, targets)
        if not self.training or self.variant == 'baseline':
            return outputs
        spectral, predictions = [], []
        for scale in range(len(self.head.stems)):
            ref = self._features['reference', scale]
            cls, reg = self._features['cls', scale], self._features['reg', scale]
            spectral.append(fcbr(ref, cls, reg))
            if self.dataset == 'gen1':
                t, b, c, h, w = reg.shape
                parts = [layer(feat.flatten(0, 1)).reshape(t, b, -1, h, w)
                         for layer, feat in [(self.head.reg_preds[scale], reg),
                                             (self.head.obj_preds[scale], reg),
                                             (self.head.cls_preds[scale], cls)]]
                predictions.append(torch.cat(parts, dim=2))
        spectral = torch.stack(spectral).mean()
        temporal = kcl(predictions) if predictions else spectral * 0
        gain = (.005 if self.dataset == 'gen1' else .01) * min(max(self.epoch, 0) / 10., 1.)
        outputs['fcbr_loss'], outputs['kcl_loss'] = spectral, temporal
        outputs['total_loss'] = outputs['total_loss'] + gain * spectral + (.05 if self.dataset == 'gen1' else 0.) * temporal
        self._features.clear()
        return outputs
