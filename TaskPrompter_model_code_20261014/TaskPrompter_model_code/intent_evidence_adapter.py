"""Opt-in evidence-integration candidates, not validated performance claims."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from intent_neuron_candidate import IntentNeuronCandidate


class EvidenceAdapter(IntentNeuronCandidate):
    def __init__(self, *args, anchored=False, evidence='constant',
                 readout_mode='legacy', oscillatory=False, reverse=False,
                 continuous=False, allow_state=False, membrane_floor=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        if evidence not in ('constant', 'pyramid') or readout_mode not in ('legacy', 'linear', 'sgdr', 'events', 'membrane'):
            raise ValueError('unknown evidence/readout mode')
        if (self.context_routing or self.task_expert_rank or self.adaptive_threshold
                or self.carry_readout_enabled or self.horizon_mix or self.modulate_gain
                or self.dynamics_type != 'lif' or not self.task_conditioned
                or self.readout != 'rate_membrane'):
            raise ValueError('candidate requires shared, threshold/decay-only, reset dual-readout adapter')
        self.evidence = evidence
        self.readout_mode = readout_mode
        self.reverse = reverse
        self.continuous = continuous
        self.anchored = anchored
        if not 0 <= membrane_floor <= 1:
            raise ValueError('membrane_floor must be in [0,1]')
        if membrane_floor and readout_mode != 'sgdr':
            raise ValueError('membrane floor is only defined for sgdr')
        self.membrane_floor = float(membrane_floor)
        self.allow_state = allow_state
        if anchored:
            nn.init.orthogonal_(self.dynamics_projector[-1].weight, gain=0.1)
        with torch.no_grad():
            anchor = torch.stack([self.dynamics_projector(e).detach().clone()
                                  for e in self.task_embedding.weight])
        self.register_buffer('controller_anchor', anchor)
        # Cartesian coefficients avoid zero-amplitude phase-gradient singularity.
        if oscillatory:
            self.phase_coefficients = nn.Parameter(torch.zeros(self.num_tasks, 2, self.bottleneck_channels))
        else:
            self.register_parameter('phase_coefficients', None)
        self.diagnostics = {}

    def _dynamics(self, task_index, feature):
        control = self.dynamics_projector(self.task_embedding.weight[task_index])
        if self.anchored:
            control = control - self.controller_anchor[task_index]
        delta_decay, delta_threshold = control.chunk(2, -1)
        if self.dynamics_mode == 'threshold':
            delta_decay = torch.zeros_like(delta_decay)
        elif self.dynamics_mode == 'decay':
            delta_threshold = torch.zeros_like(delta_threshold)
        decay = torch.sigmoid(self.base_decay_logit + self.decay_logit_limit * torch.tanh(delta_decay))
        base = F.softplus(self.base_threshold_raw) + 0.5
        threshold = base * torch.exp(self.threshold_log_limit * torch.tanh(delta_threshold))
        self._threshold_control = delta_threshold
        return decay[None, :, None, None], threshold[None, :, None, None], torch.ones_like(threshold)[None, :, None, None]

    def evidence_stream(self, current):
        if self.evidence == 'constant':
            return [current] * self.steps
        # Bands telescope to current; multiplying by T matches time-mean input.
        levels = []
        for k in range(self.steps):
            kernel = 2 * (self.steps - k) - 1
            levels.append(F.avg_pool2d(F.pad(current, (kernel//2,) * 4, mode='replicate'), kernel, stride=1)
                          if kernel > 1 else current)
        bands = [levels[0]] + [levels[k] - levels[k-1] for k in range(1, self.steps)]
        stream = [self.steps * band for band in bands]
        return list(reversed(stream)) if self.reverse else stream

    def forward(self, feature, task_index, membrane_state=None, return_state=False):
        if membrane_state is not None and not self.allow_state:
            raise ValueError('use reset here; state-transfer experiments are separate')
        current = self.normalization(self.down_project(feature))
        if self.input_activation == 'gelu':
            current = F.gelu(current)
        decay, threshold, _ = self._dynamics(task_index, feature)
        if membrane_state is not None and (membrane_state.shape != current.shape
                or membrane_state.device != current.device or membrane_state.dtype != current.dtype):
            raise ValueError('projected state must match the bottleneck')
        membrane = torch.zeros_like(current) if membrane_state is None else membrane_state
        membrane_sum = torch.zeros_like(current)
        event_sum = torch.zeros_like(current)
        weighted_sum = torch.zeros_like(current)
        normalizer = torch.zeros_like(decay)
        thresholds = []
        for k, evidence in enumerate(self.evidence_stream(current)):
            effective = threshold
            if self.phase_coefficients is not None:
                angle = 2 * math.pi * k / self.steps
                coeff = self.phase_coefficients[task_index]
                modulation = coeff[0] * math.cos(angle) + coeff[1] * math.sin(angle)
                base = F.softplus(self.base_threshold_raw) + 0.5
                effective = (base * torch.exp(self.threshold_log_limit * torch.tanh(
                    self._threshold_control + modulation)))[None, :, None, None]
            pre = decay * membrane + (1 - decay) * evidence
            event = (torch.sigmoid(self.surrogate_scale * (pre - effective))
                     if self.continuous else self._surrogate_spike(pre - effective))
            membrane = pre - event * effective
            event_sum = event_sum + event
            weighted_sum = decay * weighted_sum + event
            normalizer = decay * normalizer + 1
            membrane_sum = membrane_sum + membrane
            thresholds.append(effective.detach().mean())
        rate = event_sum / self.steps
        if self.readout_mode == 'legacy':
            event_feature = rate
            memory_feature = membrane_sum / (self.steps * threshold)
        else:
            event_feature = weighted_sum / normalizer
            memory_feature = membrane / threshold
            if self.readout_mode == 'sgdr':
                if self.continuous:
                    gate = 1 - torch.exp(-event_sum)
                else:
                    hard = (event_sum > 0).to(current.dtype)
                    soft = torch.sigmoid(4 * (event_sum - 0.5))
                    gate = hard.detach() - soft.detach() + soft
                gate = self.membrane_floor + (1 - self.membrane_floor) * gate
                memory_feature = gate * memory_feature
            elif self.readout_mode == 'events':
                memory_feature = torch.zeros_like(memory_feature)
            elif self.readout_mode == 'membrane':
                event_feature = torch.zeros_like(event_feature)
        readout = torch.cat([event_feature, memory_feature], 1)
        scale = (self.residual_scale_limit * self.scale_multiplier *
                 torch.tanh(self.residual_scale[task_index]))[None, :, None, None]
        output = feature + scale * self.up_project(readout)
        self.diagnostics = dict(event_fraction=rate.detach().mean(),
            active_fraction=(event_sum.detach() > 0).float().mean(),
            thresholds=torch.stack(thresholds),
            readout=readout.detach())
        return (output, membrane) if return_state else output


class TargetStateProjection(nn.Module):
    """Target orthogonal channel projection, with explicit reset bypass.

    Orthogonal channel basis does not establish semantic task compatibility.
    Gate starts small (sigmoid(-4)), not exactly zero; reset is explicit.
    """
    def __init__(self, num_tasks, channels, rank=8):
        super().__init__()
        if not 1 <= rank <= channels:
            raise ValueError('rank outside channel dimension')
        self.basis = nn.Parameter(torch.randn(num_tasks, channels, rank))
        self.gate_logits = nn.Parameter(torch.full((num_tasks, num_tasks), -4.0))

    def forward(self, state, source, target, reset=False):
        if reset:
            return torch.zeros_like(state)
        basis = torch.linalg.qr(self.basis[target], mode='reduced')[0]
        projected = torch.einsum('cr,dr,bdhw->bchw', basis, basis, state.detach())
        return torch.sigmoid(self.gate_logits[source, target]) * projected


def execute_star_features(adapter, task_features, projection, order=None, reset=False):
    """Two-pass same-image star interaction on precomputed task features.

    Immutable source states and sorted accumulation remove chain-order effects.
    Two passes cost extra compute; this is not a free order-invariance claim.
    Caller owns prediction heads, loss, and optimizer registration of projection.
    """
    canonical = sorted(task_features)
    order = canonical if order is None else list(order)
    if sorted(order) != canonical:
        raise ValueError('order must contain each task exactly once')
    initial = {t: adapter(task_features[t], t, return_state=True)[1].detach()
               for t in canonical}
    outputs = {}
    for target in order:
        sources = [s for s in canonical if s != target]
        state = (sum(projection(initial[s],s,target,reset=reset) for s in sources) / len(sources)
                 if sources else torch.zeros_like(initial[target]))
        outputs[target] = adapter(task_features[target], target, membrane_state=state)
    return outputs
