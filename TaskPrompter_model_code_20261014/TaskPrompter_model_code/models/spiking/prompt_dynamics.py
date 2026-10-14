"""Layer-unrolled spiking dynamics for task-prompt communication.

The transformer depth is treated as an iterative inference axis, not physical
time.  State is reset for every image and shared weights are applied between
selected transformer blocks.  Binary events gate cross-task communication;
the continuous membrane remains available to regression-oriented tasks.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskRelationalSpikingPrompt(nn.Module):
    """Frequency-preserving, task-relational prompt dynamics.

    Args:
        task_names: Ordered task names matching TaskPrompter prompt layout.
        channels: Transformer embedding dimension.
        prompt_len: Number of prompt tokens per task.
        bottleneck_channels: Width of the recurrent membrane.
        dynamics_type: ``lif`` or a parameter-matched ``ann`` control.
    """

    def __init__(self, task_names, channels, prompt_len,
                 bottleneck_channels=96, dynamics_type='lif',
                 surrogate_scale=4.0, decay_init=0.8,
                 threshold_init=0.7, residual_scale_init=0.1,
                 residual_scale_limit=1.0, relation_temperature=1.0,
                 relation_strength=0.5, frequency_strength=0.5,
                 lateral_inhibition=0.1, target_spike_rate=0.15,
                 target_relation_degree=1.0, use_frequency=True,
                 use_relations=True, relation_topk=1, readout_mix=None,
                 record_stats=False, relation_source='summary',
                 relation_gate_mode='topk', relation_threshold_init=0.2,
                 relation_threshold_temperature=0.25,
                 adaptive_threshold=False, adaptive_threshold_scale=0.25,
                 adaptive_readout=False, activity_min_rate=0.05,
                 activity_max_rate=0.20, activity_regularizer='target',
                 relation_degree_regularizer='target',
                 hard_activity_cap=False):
        super().__init__()
        if not task_names:
            raise ValueError('task_names must not be empty')
        if channels < 1 or prompt_len < 1 or bottleneck_channels < 1:
            raise ValueError('channel, prompt, and bottleneck sizes must be positive')
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
        if relation_temperature <= 0.0:
            raise ValueError('relation_temperature must be positive')
        if relation_source not in ('summary', 'delta'):
            raise ValueError("relation_source must be 'summary' or 'delta'")
        if relation_gate_mode not in ('topk', 'conflict'):
            raise ValueError(
                "relation_gate_mode must be 'topk' or 'conflict'")
        if relation_threshold_temperature <= 0.0:
            raise ValueError('relation_threshold_temperature must be positive')
        if adaptive_threshold_scale < 0.0:
            raise ValueError('adaptive_threshold_scale must be non-negative')
        if not 0.0 <= activity_min_rate <= activity_max_rate <= 1.0:
            raise ValueError('activity rate bounds must satisfy 0 <= min <= max <= 1')
        if activity_regularizer not in ('target', 'range'):
            raise ValueError(
                "activity_regularizer must be 'target' or 'range'")
        if relation_degree_regularizer not in ('target', 'upper_bound'):
            raise ValueError(
                "relation_degree_regularizer must be 'target' or 'upper_bound'")
        if (len(task_names) > 1 and
                (int(relation_topk) < 1 or
                 int(relation_topk) >= len(task_names))):
            raise ValueError(
                'relation_topk must be in [1, num_tasks - 1]')

        self.task_names = tuple(task_names)
        self.num_tasks = len(self.task_names)
        self.channels = int(channels)
        self.prompt_len = int(prompt_len)
        self.bottleneck_channels = int(bottleneck_channels)
        self.dynamics_type = str(dynamics_type)
        self.surrogate_scale = float(surrogate_scale)
        self.residual_scale_limit = float(residual_scale_limit)
        self.relation_temperature = float(relation_temperature)
        self.target_spike_rate = float(target_spike_rate)
        self.target_relation_degree = float(target_relation_degree)
        self.use_frequency = bool(use_frequency)
        self.use_relations = bool(use_relations)
        self.relation_topk = int(relation_topk)
        self.record_stats = bool(record_stats)
        self.relation_source = str(relation_source)
        self.relation_gate_mode = str(relation_gate_mode)
        self.relation_threshold_temperature = float(
            relation_threshold_temperature)
        self.adaptive_threshold = bool(adaptive_threshold)
        self.adaptive_threshold_scale = float(adaptive_threshold_scale)
        self.adaptive_readout = bool(adaptive_readout)
        self.activity_min_rate = float(activity_min_rate)
        self.activity_max_rate = float(activity_max_rate)
        self.activity_regularizer = str(activity_regularizer)
        self.relation_degree_regularizer = str(
            relation_degree_regularizer)
        self.hard_activity_cap = bool(hard_activity_cap)

        self.self_projection = nn.Linear(channels, bottleneck_channels,
                                         bias=False)
        self.frequency_projection = nn.Linear(
            channels, bottleneck_channels, bias=False)
        self.relation_value = nn.Linear(channels, bottleneck_channels,
                                        bias=False)
        self.readout_projection = nn.Linear(
            bottleneck_channels, channels, bias=False)
        # An exact identity warm start preserves the checkpoint's predictions.
        # The first update trains this projection; subsequent updates propagate
        # gradients into the recurrent current generators.
        nn.init.zeros_(self.readout_projection.weight)

        self.task_relation_logits = nn.Parameter(torch.zeros(
            self.num_tasks, self.num_tasks))
        # A destination-specific conflict threshold lets a task reject all
        # incoming messages.  This is the key distinction from forced top-k
        # routing, which cannot represent negative transfer by abstention.
        self.relation_threshold = nn.Parameter(torch.full(
            (self.num_tasks,), float(relation_threshold_init)))
        self.task_frequency_gain = nn.Parameter(torch.zeros(
            self.num_tasks, bottleneck_channels))
        self.relation_strength = nn.Parameter(torch.tensor(
            math.atanh(max(min(float(relation_strength), 0.99), -0.99))))
        self.frequency_strength = nn.Parameter(torch.tensor(
            math.atanh(max(min(float(frequency_strength), 0.99), -0.99))))
        self.lateral_inhibition = nn.Parameter(torch.tensor(
            math.log(math.expm1(max(float(lateral_inhibition), 1e-6)))))

        decay_logit = math.log(float(decay_init) / (1.0 - float(decay_init)))
        threshold_target = max(float(threshold_init) - 0.5, 1e-6)
        threshold_raw = max(math.log(math.expm1(threshold_target)), -10.0)
        self.decay_logit = nn.Parameter(torch.full(
            (self.num_tasks, bottleneck_channels), decay_logit))
        self.threshold_raw = nn.Parameter(torch.full(
            (self.num_tasks, bottleneck_channels), threshold_raw))
        self.threshold_context = nn.Linear(
            bottleneck_channels, bottleneck_channels, bias=False)
        nn.init.zeros_(self.threshold_context.weight)
        self.residual_scale = nn.Parameter(torch.full(
            (self.num_tasks, channels),
            math.atanh(float(residual_scale_init))))

        if readout_mix is None:
            defaults = {
                'semseg': 0.75,
                'edge': 0.85,
                'depth': 0.20,
                'normals': 0.20,
                'human_parts': 0.65,
                'sal': 0.75,
            }
            readout_mix = [defaults.get(task, 0.5) for task in self.task_names]
        if isinstance(readout_mix, dict):
            readout_mix = [float(readout_mix.get(task, 0.5))
                           for task in self.task_names]
        if len(readout_mix) != self.num_tasks:
            raise ValueError('readout_mix must provide one value per task')
        readout_mix = torch.tensor(readout_mix, dtype=torch.float32)
        if torch.any((readout_mix < 0.0) | (readout_mix > 1.0)):
            raise ValueError('readout_mix values must be in [0, 1]')
        self.register_buffer('readout_mix', readout_mix)
        # Three task-adaptive choices: event communication, continuous
        # membrane detail, or an identity bypass.  A tiny initial bypass keeps
        # the warm start close to the prior dual-path implementation.
        bypass = torch.full_like(readout_mix, 1e-3)
        event = readout_mix.clamp(1e-4, 1.0 - 1e-4) * (1.0 - bypass)
        membrane_mix = (1.0 - readout_mix).clamp(
            1e-4, 1.0 - 1e-4) * (1.0 - bypass)
        mode_probabilities = torch.stack(
            (event, membrane_mix, bypass), dim=-1)
        self.readout_mode_logits = nn.Parameter(mode_probabilities.log())
        off_diagonal = 1.0 - torch.eye(self.num_tasks)
        self.register_buffer('off_diagonal_mask', off_diagonal)

        self.last_spike_rate = None
        self.last_relation_degree = None
        self.reset_statistics()

    def reset_statistics(self):
        """Start a new per-forward differentiable statistics window."""
        self._spike_rate_tensors = []
        self._relation_probability_tensors = []
        self._relation_degree_tensors = []
        self.last_spike_rate = None
        self.last_relation_degree = None

    def _surrogate_gate(self, value):
        soft = torch.sigmoid(self.surrogate_scale * value)
        hard = (value >= 0.0).to(soft.dtype)
        return hard.detach() - soft.detach() + soft

    def _signed_spike(self, membrane, threshold):
        magnitude_gate = self._surrogate_gate(membrane.abs() - threshold)
        # tanh supplies a smooth signed surrogate at zero while the forward
        # path remains in {-1, 0, +1}.
        soft_sign = torch.tanh(self.surrogate_scale * membrane)
        hard_sign = torch.sign(membrane)
        signed = hard_sign.detach() - soft_sign.detach() + soft_sign
        return magnitude_gate * signed

    def _high_frequency_summary(self, image_tokens, spatial_size=None):
        batch, token_count, channels = image_tokens.shape
        if spatial_size is None:
            side = int(round(math.sqrt(token_count)))
            spatial_size = (side, side) if side * side == token_count else None
        if (spatial_size is not None and
                int(spatial_size[0]) * int(spatial_size[1]) == token_count):
            height, width = int(spatial_size[0]), int(spatial_size[1])
            feature = image_tokens.transpose(1, 2).reshape(
                batch, channels, height, width)
            low = F.avg_pool2d(feature, kernel_size=3, stride=1, padding=1)
            high = feature - low
            return high.abs().mean(dim=(2, 3))
        centered = image_tokens - image_tokens.mean(dim=1, keepdim=True)
        return centered.abs().mean(dim=1)

    def _relation_gate(self, prompt_signal):
        normalized = F.normalize(prompt_signal, dim=-1, eps=1e-6)
        similarity = torch.bmm(normalized, normalized.transpose(1, 2))
        logits = (similarity / self.relation_temperature +
                  self.task_relation_logits.unsqueeze(0))
        mask = self.off_diagonal_mask.unsqueeze(0)
        # A zero-threshold sigmoid would activate every off-diagonal prompt
        # pair whenever pretrained prompts have positive cosine similarity.
        # Select exactly top-k *source* tasks per destination instead.  The
        # straight-through softmax preserves gradients but the forward path is
        # genuinely sparse for both LIF and parameter-matched ANN controls.
        masked_logits = logits.masked_fill(mask == 0, -1e4)
        indices = masked_logits.topk(self.relation_topk, dim=-1).indices
        selector = torch.zeros_like(masked_logits).scatter_(-1, indices, 1.0)
        if self.relation_gate_mode == 'topk':
            probability = F.softmax(
                masked_logits / self.relation_temperature, dim=-1) * mask
            hard = selector
        else:
            threshold = self.relation_threshold.view(1, -1, 1)
            probability = torch.sigmoid(
                (masked_logits - threshold) /
                self.relation_threshold_temperature) * mask
            hard = selector * (masked_logits >= threshold).to(
                probability.dtype) * mask
        gate = hard.detach() - probability.detach() + probability
        return gate, probability, hard

    def _validate_state(self, state, reference):
        if state is None:
            return torch.zeros_like(reference)
        if (not torch.is_tensor(state) or state.shape != reference.shape or
                state.device != reference.device or
                state.dtype != reference.dtype):
            raise ValueError('prompt membrane state must match [B,T,Cb] current')
        return state

    def forward(self, task_prompts, previous_prompts, image_tokens,
                membrane_state=None, spatial_size=None):
        expected_prompts = self.num_tasks * self.prompt_len
        if task_prompts.dim() != 3 or task_prompts.shape[1] != expected_prompts:
            raise ValueError('task_prompts has incompatible shape')
        if previous_prompts.shape != task_prompts.shape:
            raise ValueError('previous_prompts must match task_prompts')
        batch = task_prompts.shape[0]
        current = task_prompts.reshape(
            batch, self.num_tasks, self.prompt_len, self.channels)
        previous = previous_prompts.reshape_as(current)
        prompt_delta = (current - previous).mean(dim=2)
        prompt_summary = current.mean(dim=2)

        self_current = self.self_projection(prompt_delta)
        drive = self_current

        relation_probability = None
        relation_hard = None
        if self.use_relations and self.num_tasks > 1:
            relation_signal = (prompt_delta if self.relation_source == 'delta'
                               else prompt_summary)
            relation_gate, relation_probability, relation_hard = \
                self._relation_gate(relation_signal)
            normalization = relation_gate.sum(dim=-1, keepdim=True).clamp_min(1.0)
            cross_value = self.relation_value(prompt_delta)
            cross_current = torch.bmm(relation_gate, cross_value) / normalization
            inhibition = (F.softplus(self.lateral_inhibition) *
                          relation_gate.mean(dim=-1, keepdim=True) *
                          self_current)
            drive = drive + torch.tanh(self.relation_strength) * cross_current
            drive = drive - inhibition

        if self.use_frequency:
            high_frequency = self._high_frequency_summary(
                image_tokens, spatial_size=spatial_size)
            frequency_current = self.frequency_projection(high_frequency)
            frequency_current = frequency_current.unsqueeze(1) * (
                1.0 + torch.tanh(self.task_frequency_gain).unsqueeze(0))
            drive = drive + (torch.tanh(self.frequency_strength) *
                             frequency_current)

        membrane = self._validate_state(membrane_state, drive)
        decay = torch.sigmoid(self.decay_logit).unsqueeze(0)
        base_threshold = (F.softplus(self.threshold_raw) + 0.5).unsqueeze(0)
        if self.adaptive_threshold:
            threshold_delta = torch.tanh(
                self.threshold_context(drive.detach().abs()))
            threshold = base_threshold * (
                1.0 + self.adaptive_threshold_scale * threshold_delta)
        else:
            threshold = base_threshold
        integrated = decay * membrane + drive
        if self.dynamics_type == 'lif':
            activity = self._signed_spike(integrated, threshold)
            if self.hard_activity_cap:
                # A soft activity loss did not keep the learned model under
                # budget.  Compete across membrane channels instead: at most
                # floor(r_max * C) events per task and layer can be emitted.
                # Non-winning membrane is retained, so suppressed evidence
                # can win a later layer rather than being discarded.
                winners = int(math.floor(
                    self.activity_max_rate * self.bottleneck_channels))
                if winners <= 0:
                    activity = torch.zeros_like(activity)
                elif winners < self.bottleneck_channels:
                    indices = integrated.abs().topk(
                        winners, dim=-1).indices
                    winner_mask = torch.zeros_like(activity).scatter_(
                        -1, indices, 1.0)
                    activity = activity * winner_mask
            next_membrane = integrated - activity * threshold
            # The forward value is the hard event rate, while the surrogate
            # activity keeps the spike-budget objective differentiable.
            event_rate = activity.abs().mean()
        else:
            activity = torch.tanh(integrated / threshold)
            next_membrane = integrated
            event_rate = activity.abs().mean()

        membrane_readout = torch.tanh(next_membrane / threshold)
        if self.adaptive_readout:
            mode = F.softmax(self.readout_mode_logits, dim=-1)
            event_mix = mode[:, 0].view(1, self.num_tasks, 1)
            membrane_mix = mode[:, 1].view(1, self.num_tasks, 1)
            # The third probability is an explicit identity bypass: it
            # contributes no residual and can suppress harmful task updates.
            readout = (event_mix * activity +
                       membrane_mix * membrane_readout)
        else:
            mix = self.readout_mix.view(1, self.num_tasks, 1)
            readout = mix * activity + (1.0 - mix) * membrane_readout
        residual = self.readout_projection(readout)
        scale = (self.residual_scale_limit *
                 torch.tanh(self.residual_scale)).unsqueeze(0)
        residual = (scale * residual).unsqueeze(2)
        updated = current + residual

        self._spike_rate_tensors.append(event_rate)
        if relation_probability is not None:
            self._relation_probability_tensors.append(relation_probability)
            self._relation_degree_tensors.append(
                relation_hard.sum(dim=-1).mean())
        if self.record_stats:
            self.last_spike_rate = event_rate.detach()
            if relation_probability is not None:
                self.last_relation_degree = relation_hard.detach().sum(
                    dim=-1).mean()
        return updated.reshape_as(task_prompts), next_membrane

    def regularization_components(self):
        zero = self.residual_scale.sum() * 0.0
        if not self._spike_rate_tensors:
            return {'spike_rate': zero, 'relation_degree': zero,
                    'relation_balance': zero}
        spike_rate = torch.stack(self._spike_rate_tensors).mean()
        if self.activity_regularizer == 'range':
            spike_loss = (
                F.relu(self.activity_min_rate - spike_rate).pow(2) +
                F.relu(spike_rate - self.activity_max_rate).pow(2))
        else:
            spike_loss = (spike_rate - self.target_spike_rate).pow(2)
        if not self._relation_probability_tensors:
            return {'spike_rate': spike_loss, 'relation_degree': zero,
                    'relation_balance': zero}
        probability = torch.stack(self._relation_probability_tensors)
        degree = torch.stack(self._relation_degree_tensors).mean()
        if self.relation_degree_regularizer == 'upper_bound':
            degree_loss = F.relu(
                degree - self.target_relation_degree).pow(2)
        else:
            degree_loss = (degree - self.target_relation_degree).pow(2)
        task_load = probability.sum(dim=-2).mean(dim=(0, 1))
        balance_loss = task_load.var(unbiased=False)
        return {'spike_rate': spike_loss,
                'relation_degree': degree_loss,
                'relation_balance': balance_loss}

    def regularization_loss(self):
        components = self.regularization_components()
        return (components['spike_rate'] +
                0.1 * components['relation_degree'] +
                0.01 * components['relation_balance'])

    def statistics(self):
        """Detached forward statistics for TensorBoard and audit logs."""
        if not self._spike_rate_tensors:
            zero = self.residual_scale.sum().detach() * 0.0
            return {'event_rate': zero, 'relation_degree': zero}
        event_rate = torch.stack(self._spike_rate_tensors).mean().detach()
        if not self._relation_degree_tensors:
            relation_degree = event_rate * 0.0
        else:
            relation_degree = torch.stack(
                self._relation_degree_tensors).mean().detach()
        if self.adaptive_readout:
            mode = F.softmax(self.readout_mode_logits, dim=-1).detach()
            event_mix = mode[:, 0].mean()
            bypass_mix = mode[:, 2].mean()
        else:
            event_mix = self.readout_mix.mean().detach()
            bypass_mix = event_mix * 0.0
        return {'event_rate': event_rate,
                'relation_degree': relation_degree,
                'event_readout_mix': event_mix,
                'identity_bypass_mix': bypass_mix}
