"""Task-routed spiking memory for causal multi-task dense prediction.

Unlike the legacy adapter, one call to this module represents one physical
frame.  State is passed explicitly and is therefore owned by the caller's
per-task state bank; task execution order is never treated as time.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskRoutedSpikingMemory(nn.Module):
    """Sparse expert memory with task-local state and spike/membrane readout."""

    physical_time_only = True

    def __init__(self, num_tasks, channels, bottleneck_channels=64,
                 num_experts=4, dynamics_type='lif', surrogate_scale=4.0,
                 decay_init=0.8, threshold_init=0.7,
                 residual_scale_init=0.1, residual_scale_limit=1.0,
                 router_temperature=1.0, target_spike_rate=0.15,
                 target_active_experts=1.0, record_stats=False):
        super().__init__()
        if num_tasks < 1 or channels < 1 or bottleneck_channels < 1:
            raise ValueError('task, channel, and bottleneck sizes must be positive')
        if num_experts < 1:
            raise ValueError('num_experts must be positive')
        if dynamics_type not in ('lif', 'ann'):
            raise ValueError("dynamics_type must be 'lif' or 'ann'")
        if not 0.0 < float(decay_init) < 1.0:
            raise ValueError('decay_init must be in (0, 1)')
        if float(threshold_init) < 0.5:
            raise ValueError('threshold_init must be >= 0.5')
        if not -0.99 < float(residual_scale_init) < 0.99:
            raise ValueError('residual_scale_init must be in (-0.99, 0.99)')
        if not 0.0 < float(residual_scale_limit) <= 1.0:
            raise ValueError('residual_scale_limit must be in (0, 1]')
        if router_temperature <= 0.0:
            raise ValueError('router_temperature must be positive')

        self.num_tasks = int(num_tasks)
        self.channels = int(channels)
        self.bottleneck_channels = int(bottleneck_channels)
        self.num_experts = int(num_experts)
        self.dynamics_type = dynamics_type
        self.surrogate_scale = float(surrogate_scale)
        self.residual_scale_limit = float(residual_scale_limit)
        self.router_temperature = float(router_temperature)
        self.target_spike_rate = float(target_spike_rate)
        self.target_active_experts = float(target_active_experts)
        self.record_stats = bool(record_stats)

        self.expert_down = nn.ModuleList([
            nn.Conv2d(channels, bottleneck_channels, 1, bias=False)
            for _ in range(num_experts)
        ])
        self.expert_norm = nn.ModuleList([
            nn.GroupNorm(1, bottleneck_channels)
            for _ in range(num_experts)
        ])
        # Dual spike/membrane readout protects continuous tasks from rate-code
        # quantisation while leaving the communication path binary.
        self.expert_up = nn.ModuleList([
            nn.Conv2d(2 * bottleneck_channels, channels, 1, bias=False)
            for _ in range(num_experts)
        ])
        # Exact warm-start identity while preserving a first-step gradient into
        # the output projections through the non-zero residual scale.
        for projection in self.expert_up:
            nn.init.zeros_(projection.weight)

        self.context_router = nn.Linear(channels, num_experts, bias=False)
        nn.init.zeros_(self.context_router.weight)
        self.task_route_logits = nn.Parameter(
            torch.full((num_tasks, num_experts), -1.0))
        with torch.no_grad():
            for task_index in range(num_tasks):
                self.task_route_logits[task_index, task_index % num_experts] = 1.0

        decay_logit = math.log(float(decay_init) / (1.0 - float(decay_init)))
        threshold_target = max(float(threshold_init) - 0.5, 1e-6)
        threshold_raw = max(math.log(math.expm1(threshold_target)), -10.0)
        self.decay_logit = nn.Parameter(torch.full(
            (num_experts, bottleneck_channels), decay_logit))
        self.threshold_raw = nn.Parameter(torch.full(
            (num_experts, bottleneck_channels), threshold_raw))

        self.residual_scale = nn.Parameter(torch.full(
            (num_tasks, channels), math.atanh(float(residual_scale_init))))
        self.scale_multiplier = 1.0
        self.last_spike_rate = {}
        self.last_route = {}
        self._last_spike_rate_tensor = None
        self._last_route_probability = None
        self._spike_rate_tensors = []
        self._route_probability_tensors = []

    def reset_regularization_statistics(self):
        """Start a new forward-level statistics window across all tasks."""
        self._last_spike_rate_tensor = None
        self._last_route_probability = None
        self._spike_rate_tensors = []
        self._route_probability_tensors = []

    def _surrogate_binary(self, value):
        soft = torch.sigmoid(self.surrogate_scale * value)
        hard = (value >= 0).to(soft.dtype)
        return hard.detach() - soft.detach() + soft

    def _route(self, feature, task_index):
        pooled = F.adaptive_avg_pool2d(feature, 1).flatten(1)
        logits = (self.task_route_logits[task_index].unsqueeze(0) +
                  self.context_router(pooled))
        probability = torch.sigmoid(logits / self.router_temperature)
        hard = (probability >= 0.5).to(probability.dtype)
        # Straight-through binary routing.  The cyclic task prior ensures that
        # every task has a live expert before optimisation and avoids the
        # identical-state collapse observed in the legacy controller.
        route = hard.detach() - probability.detach() + probability
        return route, probability

    def _validate_state(self, state, reference):
        if state is None:
            return torch.zeros_like(reference)
        if not isinstance(state, dict) or 'membrane' not in state:
            raise ValueError('state must be a dictionary containing membrane')
        membrane = state['membrane']
        if (not torch.is_tensor(membrane) or membrane.shape != reference.shape or
                membrane.device != reference.device or
                membrane.dtype != reference.dtype):
            raise ValueError('state membrane must match [B,E,C,H,W] drive')
        return membrane

    def forward(self, feature, task_index, membrane_state=None,
                return_state=False):
        if not 0 <= int(task_index) < self.num_tasks:
            raise ValueError('task_index out of range')
        route, route_probability = self._route(feature, int(task_index))
        drives = []
        for down, norm in zip(self.expert_down, self.expert_norm):
            drives.append(F.gelu(norm(down(feature))))
        drive = torch.stack(drives, dim=1)
        route_5d = route[:, :, None, None, None]
        drive = drive * route_5d
        membrane = self._validate_state(membrane_state, drive)

        decay = torch.sigmoid(self.decay_logit)[None, :, :, None, None]
        threshold = (F.softplus(self.threshold_raw) + 0.5)[
            None, :, :, None, None]
        # Discrete current-based LIF: decay applies to the previous membrane,
        # not to the incoming current. Scaling the current by (1 - decay)
        # makes a cold-start single-frame pass almost silent when decay is
        # large, which is especially harmful for static dense prediction.
        integrated = decay * membrane + drive
        if self.dynamics_type == 'lif':
            activity = self._surrogate_binary(integrated - threshold)
            next_membrane = integrated - activity * threshold
        else:
            # Parameter-matched analog temporal control: same experts, router,
            # recurrent decay and readout width, replacing only binary firing.
            activity = F.relu(integrated)
            next_membrane = integrated

        residual = torch.zeros_like(feature)
        active_count = route.sum(dim=1, keepdim=True).clamp_min(1.0)
        for expert_index, up in enumerate(self.expert_up):
            readout = torch.cat([
                activity[:, expert_index],
                torch.tanh(next_membrane[:, expert_index] /
                           threshold[:, expert_index]),
            ], dim=1)
            expert_weight = (route[:, expert_index:expert_index + 1] /
                             active_count).unsqueeze(-1).unsqueeze(-1)
            residual = residual + expert_weight * up(readout)

        scale = (self.residual_scale_limit * self.scale_multiplier *
                 torch.tanh(self.residual_scale[int(task_index)])).view(
                     1, -1, 1, 1)
        output = feature + scale * residual

        if self.dynamics_type == 'lif':
            rate = activity.mean()
        else:
            rate = (activity > 0).to(activity.dtype).mean()
        self._last_spike_rate_tensor = rate
        self._last_route_probability = route_probability
        self._spike_rate_tensors.append(rate)
        self._route_probability_tensors.append(route_probability)
        if self.record_stats:
            self.last_spike_rate[int(task_index)] = rate.detach()
            self.last_route[int(task_index)] = route.detach().mean(dim=0)

        state = {
            'membrane': next_membrane,
            'route': route.detach(),
        }
        if return_state:
            return output, state
        return output

    def regularization_loss(self):
        """Differentiable spike-budget and router-load regularizer."""
        if not self._spike_rate_tensors:
            return self.residual_scale.sum() * 0.0
        spike_rate = torch.stack(self._spike_rate_tensors).mean()
        spike_loss = (spike_rate - self.target_spike_rate).pow(2)
        route_probability = torch.cat(
            self._route_probability_tensors, dim=0)
        active = route_probability.sum(dim=1).mean()
        route_loss = (active - self.target_active_experts).pow(2)
        # Entropy across the batch discourages a context router that sends all
        # samples to the same expert while retaining sparse per-sample routes.
        mean_route = route_probability.mean(dim=0).clamp(1e-6, 1-1e-6)
        balance = (mean_route * mean_route.log() +
                   (1.0 - mean_route) * (1.0 - mean_route).log()).mean()
        return spike_loss + 0.1 * route_loss + 0.01 * balance
