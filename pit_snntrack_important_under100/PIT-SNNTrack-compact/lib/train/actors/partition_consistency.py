"""Numerically stable endpoint membrane agreement for partition-paired training."""
import torch
import torch.nn.functional as F


def endpoint_state_consistency(state_pairs):
    if not state_pairs:
        raise ValueError("state consistency requires nonempty endpoint states")
    losses = []
    for state_a, state_b in state_pairs:
        if state_a.shape != state_b.shape:
            raise ValueError("paired endpoint states have different shapes")
        # AMP states can exceed sqrt(float16.max). FP32 accumulation avoids
        # a silently infinite RMS denominator that suppresses the objective.
        a, b = state_a.float(), state_b.float()
        scale = torch.maximum(a.detach().square().mean().sqrt(),
                              b.detach().square().mean().sqrt()).clamp_min(1e-3)
        losses.append(F.smooth_l1_loss(a / scale, b / scale))
    return torch.stack(losses).mean()
