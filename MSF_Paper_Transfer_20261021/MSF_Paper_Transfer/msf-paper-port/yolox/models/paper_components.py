"""Paper-derived training objectives and PLIF. Tensor contract: T,B,C,H,W."""
import math
import torch
from torch import nn


class RectSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold):
        ctx.save_for_backward(x)
        ctx.threshold = threshold
        return (x >= threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad * ((x - ctx.threshold).abs() < .5).to(grad.dtype), None


class PaperPLIF(nn.Module):
    def __init__(self, act=True, decay=.25, thresh=.5):
        super().__init__()
        self.decay_w = nn.Parameter(torch.tensor(math.log(decay / (1 - decay))))
        self.act, self.thresh = act, thresh

    def forward(self, x):
        mem, spike, out = torch.zeros_like(x[0]), torch.zeros_like(x[0]), []
        for value in x:
            reset = (1 - spike.detach()) if self.act else 1
            mem = mem * self.decay_w.sigmoid() * reset + value
            spike = RectSpike.apply(mem, self.thresh) if self.act else mem
            out.append(spike)
        return torch.stack(out)


def fcbr(reference, classification, regression, groups=8, cutoff=.30, order=4):
    """Reference detach stops its direct gradient, not shared-backbone branch gradients."""
    if reference.shape != classification.shape or reference.shape != regression.shape:
        raise ValueError('FCBR needs aligned reference, classification, regression features')
    _, b, c, h, w = reference.shape
    if c % groups:
        raise ValueError('channels must be divisible by groups')
    def spectrum(x):
        x = x.float().mean(0).reshape(b, groups, c // groups, h, w).mean(2)
        return torch.fft.rfft2(x, norm='ortho')
    ref = spectrum(reference.detach())
    cls, reg = spectrum(classification), spectrum(regression)
    fy = torch.fft.fftfreq(h, device=reference.device).view(-1, 1)
    fx = torch.fft.rfftfreq(w, device=reference.device).view(1, -1)
    radius = torch.sqrt(fx.square() + fy.square())
    low = 1 / (1 + (radius / (cutoff / 2)) ** (2 * order))
    high = 1 - low
    def discrepancy(a, mask):
        numerator = (mask * (a - ref).abs()).sum((-2, -1))
        denominator = (mask * (a.abs() + ref.abs())).sum((-2, -1)).clamp_min(1e-8)
        return (numerator / denominator).mean()
    a, z = (cls.abs() * high).flatten(-2), (reg.abs() * high).flatten(-2)
    cosine = torch.nn.functional.cosine_similarity(a, z, dim=-1, eps=1e-8).mean()
    return discrepancy(cls, low) + discrepancy(reg, high) + .05 * cosine


def kcl(predictions):
    """Equation 13: mean adjacent dense prediction squared differences, each scale equal."""
    terms = [(p[1:] - p[:-1]).float().square().mean() for p in predictions if p.shape[0] > 1]
    return torch.stack(terms).mean() if terms else predictions[0].sum() * 0
