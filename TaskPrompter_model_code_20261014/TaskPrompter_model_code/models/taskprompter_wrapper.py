# By Hanrong Ye
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.spiking import TaskRoutedSpikingMemory

INTERPOLATE_MODE = 'bilinear'


class TaskModulatedSpikingAdapter(nn.Module):
    """Residual PLIF bottleneck with optional task-conditioned dynamics.

    This first-stage hybrid adapter deliberately keeps the visual encoder and
    prediction heads analog.  It isolates whether task-dependent membrane
    decay and firing thresholds help beyond an otherwise identical spiking
    bottleneck.  A zero-initialized residual scale makes the module an exact
    identity at construction time, which permits auditable warm starts and
    safe ablations.
    """

    def __init__(self, num_tasks, channels, bottleneck_channels=128, steps=4,
                 task_conditioned=False, controller_dim=64,
                 surrogate_scale=4.0, record_stats=False,
                 context_routing=False, route_limit=0.25,
                 residual_scale_limit=1.0, task_expert_rank=0,
                 dynamics_mode='both', dynamics_type='lif',
                 readout='rate', adaptive_threshold=False,
                 threshold_init=None, input_activation='gelu',
                 residual_scale_init=0.0, modulate_gain=False,
                 decay_init=None, carry_readout=False,
                 carry_scale_init=0.0):
        super().__init__()
        if bottleneck_channels < 1:
            raise ValueError('bottleneck_channels must be positive')
        if steps < 1:
            raise ValueError('steps must be positive')
        if readout not in ('rate', 'rate_membrane'):
            raise ValueError(
                "readout must be 'rate' or 'rate_membrane', got {}".format(
                    readout))
        if threshold_init is not None and float(threshold_init) < 0.5:
            raise ValueError(
                'threshold_init must be >= 0.5 (softplus floor), got {}'.format(
                    threshold_init))
        if input_activation not in ('gelu', 'identity'):
            raise ValueError(
                "input_activation must be 'gelu' or 'identity', got {}".format(
                    input_activation))
        if not -0.99 < float(residual_scale_init) < 0.99:
            raise ValueError(
                'residual_scale_init must be in (-0.99, 0.99), got {}'.format(
                    residual_scale_init))
        if decay_init is not None and not 0.0 < float(decay_init) < 1.0:
            raise ValueError(
                'decay_init must be in (0, 1), got {}'.format(decay_init))
        if surrogate_scale <= 0:
            raise ValueError('surrogate_scale must be positive')
        if route_limit <= 0 or route_limit > 1:
            raise ValueError('route_limit must be in (0, 1]')
        if residual_scale_limit <= 0 or residual_scale_limit > 1:
            raise ValueError('residual_scale_limit must be in (0, 1]')
        if task_expert_rank < 0 or task_expert_rank > bottleneck_channels:
            raise ValueError('task_expert_rank must be in [0, bottleneck_channels]')

        self.num_tasks = int(num_tasks)
        self.steps = int(steps)
        self.task_conditioned = bool(task_conditioned)
        self.surrogate_scale = float(surrogate_scale)
        self.record_stats = bool(record_stats)
        self.context_routing = bool(context_routing)
        self.route_limit = float(route_limit)
        self.residual_scale_limit = float(residual_scale_limit)
        if dynamics_mode not in ('both', 'threshold', 'decay'):
            raise ValueError(
                'spike_dynamics_mode must be one of both/threshold/decay, '
                'got {}'.format(dynamics_mode))
        self.dynamics_mode = dynamics_mode
        if dynamics_type not in ('lif', 'ann'):
            raise ValueError(
                'spike_dynamics_type must be lif or ann, got {}'.format(
                    dynamics_type))
        self.dynamics_type = dynamics_type
        self.task_expert_rank = int(task_expert_rank)
        self.readout = readout
        self.adaptive_threshold = bool(adaptive_threshold)
        self.bottleneck_channels = int(bottleneck_channels)
        self.input_activation = input_activation
        # Dual readout concatenates the firing rate with the normalised
        # membrane trace, recovering the sub-threshold information that the
        # quantised spike count throws away.
        self.readout_channels = (2 * self.bottleneck_channels
                                 if readout == 'rate_membrane'
                                 else self.bottleneck_channels)
        self.down_project = nn.Conv2d(
            channels, bottleneck_channels, kernel_size=1, bias=False)
        self.normalization = nn.GroupNorm(1, bottleneck_channels)
        self.up_project = nn.Conv2d(
            self.readout_channels, channels, kernel_size=1, bias=False)

        # Common neuron dynamics are trainable in both TP-04S and TP-05N.
        # sigmoid(0)=0.5 and softplus(0)+0.5~=1.193 are conservative starts.
        self.base_decay_logit = nn.Parameter(torch.zeros(bottleneck_channels))
        self.base_threshold_raw = nn.Parameter(torch.zeros(bottleneck_channels))
        if decay_init is not None:
            # Slow membrane: a high decay (e.g. 0.9) lets the membrane state
            # survive the whole integration window *and* the switch between
            # tasks, which is the regime where cross-task state carry and the
            # learned retention gate can actually influence the output.
            logit = math.log(float(decay_init) / (1.0 - float(decay_init)))
            with torch.no_grad():
                self.base_decay_logit.fill_(logit)
        if threshold_init is not None:
            # Choose the raw parameter such that softplus(raw) + 0.5 equals the
            # requested threshold. The empirical firing rate of the trained
            # adapters was 2-7%, far below the 10-30% operating range of
            # published directly-trained SNNs, so the default 1.193 threshold
            # is too high for this feature scale.
            target = max(float(threshold_init) - 0.5, 1e-6)
            raw_init = max(math.log(math.expm1(target)), -10.0)
            with torch.no_grad():
                self.base_threshold_raw.fill_(raw_init)
        if self.adaptive_threshold:
            # ALIF: threshold trace adaptation. beta starts at zero, so the
            # dynamics are bitwise identical to the fixed-threshold neuron
            # before optimisation (auditable warm start). For the memoryless
            # ANN control the adaptation parameters are unused, so freeze them
            # to keep DDP's unused-parameter tracking quiet.
            self.adapt_beta = nn.Parameter(torch.zeros(bottleneck_channels))
            self.trace_decay_logit = nn.Parameter(torch.zeros(bottleneck_channels))
        else:
            self.adapt_beta = None
            self.trace_decay_logit = None
        if self.dynamics_type == 'ann':
            # The LIF parameters are unused by the memoryless ANN control;
            # freeze them so DDP does not flag them as unused parameters.
            self.base_decay_logit.requires_grad_(False)
            self.base_threshold_raw.requires_grad_(False)
            if self.adapt_beta is not None:
                self.adapt_beta.requires_grad_(False)
                self.trace_decay_logit.requires_grad_(False)

        if self.task_conditioned:
            self.task_embedding = nn.Embedding(num_tasks, controller_dim)
            self.modulate_gain = bool(modulate_gain)
            gain_factor = 3 if self.modulate_gain else 2
            self.dynamics_projector = nn.Sequential(
                nn.Linear(controller_dim, controller_dim),
                nn.GELU(),
                nn.Linear(controller_dim, bottleneck_channels * gain_factor),
            )
            nn.init.normal_(self.task_embedding.weight, std=0.02)
            nn.init.zeros_(self.dynamics_projector[-1].weight)
            nn.init.zeros_(self.dynamics_projector[-1].bias)
        else:
            self.modulate_gain = False
            self.task_embedding = None
            self.dynamics_projector = None

        # A lightweight task- and sample-dependent router.  The depthwise 3x3
        # context path follows TaskExpert's observation that local context is
        # useful for expert routing, while the shared bottleneck plus task
        # route retains MLoRE's generic/low-rank decomposition.  Both logits
        # are zero-initialized, so enabling routing is exactly neutral before
        # optimization.  Bounding the multiplicative route prevents a task
        # from overwhelming the shared low-rank residual.
        if self.context_routing:
            self.context_router = nn.Conv2d(
                bottleneck_channels, bottleneck_channels, kernel_size=3,
                padding=1, groups=bottleneck_channels, bias=False)
            self.task_router = nn.Embedding(num_tasks, bottleneck_channels)
            nn.init.zeros_(self.context_router.weight)
            nn.init.zeros_(self.task_router.weight)
        else:
            self.context_router = None
            self.task_router = None

        # The shared projection is a generic low-rank expert.  Add optional
        # task-specific low-rank output experts so a beneficial residual for
        # depth or normals does not have to perturb semseg and edges.  Their
        # output projections start at zero, preserving exact identity and
        # making this a clean extension of the shared adapter.
        if self.task_expert_rank:
            self.task_expert_down = nn.ModuleList([
                nn.Conv2d(self.readout_channels, self.task_expert_rank,
                          kernel_size=1, bias=False)
                for _ in range(num_tasks)
            ])
            self.task_expert_up = nn.ModuleList([
                nn.Conv2d(self.task_expert_rank, channels,
                          kernel_size=1, bias=False)
                for _ in range(num_tasks)
            ])
            for layer in self.task_expert_down:
                nn.init.normal_(layer.weight, std=0.02)
            for layer in self.task_expert_up:
                nn.init.zeros_(layer.weight)
        else:
            self.task_expert_down = None
            self.task_expert_up = None

        # Cross-task state readout: the carried membrane state of the previous
        # task is projected straight into the residual of the current task.
        # The projection is zero-initialised (bitwise neutral before training),
        # but unlike the membrane-init path it gives the retention gate a
        # strong gradient, because the carried state now influences every
        # output pixel instead of only the first integration step.
        self.carry_readout_enabled = bool(carry_readout)
        if self.carry_readout_enabled:
            self.carry_readout = nn.Conv2d(
                bottleneck_channels, channels, kernel_size=1, bias=False)
            nn.init.zeros_(self.carry_readout.weight)
            # The carry path needs its own gain: the shared residual scale is
            # zero-initialised (exact identity), which would otherwise block
            # every gradient into the carry projection for the first thousand
            # steps. A small non-zero init lets the path bootstrap; setting
            # either this gain or the retention gate to zero restores exact
            # identity, which keeps the ablation clean.
            self.carry_scale = nn.Parameter(
                torch.full((1,), float(carry_scale_init)))
        else:
            self.carry_readout = None
            self.carry_scale = None

        # Per-task residual strength allows exact identity at initialization.
        self.residual_scale = nn.Parameter(torch.zeros(num_tasks, channels))
        if float(residual_scale_init):
            # Opt-in non-identity start: the spiking residual contributes from
            # step zero, so its projections receive gradients immediately
            # instead of waiting for the scalar scale to grow out of zero.
            with torch.no_grad():
                self.residual_scale.fill_(
                    math.atanh(float(residual_scale_init)))
        # Inference-time ablation knob: multiplies the effective residual so
        # that a checkpoint whose scale stayed near identity can be amplified
        # without retraining. Defaults to 1.0 (numerically unchanged).
        self.scale_multiplier = 1.0
        self.last_spike_rate = {}

    def _dynamics(self, task_index, feature):
        delta_decay = 0.0
        delta_threshold = 0.0
        delta_gain = 0.0
        if self.task_conditioned:
            index = torch.tensor(
                task_index, device=feature.device, dtype=torch.long)
            dynamics = self.dynamics_projector(self.task_embedding(index))
            if self.modulate_gain:
                delta_decay, delta_threshold, delta_gain = dynamics.chunk(3, dim=-1)
            else:
                delta_decay, delta_threshold = dynamics.chunk(2, dim=-1)
            # Ablation switch: keep only one of the two conditioned terms.
            if self.dynamics_mode == 'threshold':
                delta_decay = torch.zeros_like(delta_decay)
            elif self.dynamics_mode == 'decay':
                delta_threshold = torch.zeros_like(delta_threshold)
        decay = torch.sigmoid(self.base_decay_logit + delta_decay)
        threshold = F.softplus(
            self.base_threshold_raw + delta_threshold) + 0.5
        # Task-conditioned input gain (neuromodulation-style excitability);
        # exactly 1.0 before optimisation because the projector is zero-init.
        if torch.is_tensor(delta_gain):
            gain = (1.0 + torch.tanh(delta_gain)).view(1, -1, 1, 1)
        else:
            gain = 1.0
        return (decay.view(1, -1, 1, 1),
                threshold.view(1, -1, 1, 1),
                gain)

    def _surrogate_spike(self, membrane_minus_threshold):
        # Binary values in the forward pass, sigmoid derivative in backward.
        soft = torch.sigmoid(
            self.surrogate_scale * membrane_minus_threshold)
        hard = (membrane_minus_threshold >= 0).to(soft.dtype)
        return hard.detach() - soft.detach() + soft

    def forward(self, feature, task_index, membrane_state=None,
                return_state=False):
        current = self.normalization(self.down_project(feature))
        if self.input_activation == 'gelu':
            current = F.gelu(current)
        if self.context_router is not None:
            context_logits = F.adaptive_avg_pool2d(
                self.context_router(current), 1).flatten(1)
            route_logits = context_logits + self.task_router.weight[task_index]
            route = 1.0 + self.route_limit * torch.tanh(route_logits)
            current = current * route.unsqueeze(-1).unsqueeze(-1)
        membrane = None
        spike_sum = None
        if self.dynamics_type == 'ann':
            # ANN control: identical modules and residual path, but the LIF
            # recurrence is replaced by a memoryless ReLU bottleneck, so the
            # comparison isolates the spiking dynamics themselves. When the
            # dual readout is requested the control gets two analog channels
            # (ReLU + tanh) of the same width, so the comparison isolates the
            # dynamics rather than the readout width.
            if self.readout == 'rate_membrane':
                spike_rate_feature = torch.cat(
                    [F.relu(current), torch.tanh(current)], dim=1)
            else:
                spike_rate_feature = F.relu(current)
        else:
            decay, threshold, gain = self._dynamics(task_index, feature)
            if self.modulate_gain:
                # Neuromodulation: the task-conditioned input gain scales the
                # excitability of the spiking population before integration.
                current = current * gain
            if membrane_state is None:
                membrane = torch.zeros_like(current)
            else:
                if (not torch.is_tensor(membrane_state) or
                        membrane_state.shape != current.shape or
                        membrane_state.device != current.device or
                        membrane_state.dtype != current.dtype):
                    raise ValueError(
                        'membrane_state must match the current bottleneck tensor')
                membrane = membrane_state
            spike_sum = torch.zeros_like(current)
            membrane_sum = (torch.zeros_like(current)
                            if self.readout == 'rate_membrane' else None)
            if self.adaptive_threshold:
                trace = torch.zeros_like(current)
                trace_decay = torch.sigmoid(
                    self.trace_decay_logit).view(1, -1, 1, 1)
                adapt_beta = self.adapt_beta.view(1, -1, 1, 1)
            for _ in range(self.steps):
                membrane = decay * membrane + (1.0 - decay) * current
                if self.adaptive_threshold:
                    threshold_eff = threshold + adapt_beta * trace
                else:
                    threshold_eff = threshold
                spike = self._surrogate_spike(membrane - threshold_eff)
                if self.adaptive_threshold:
                    trace = trace_decay * trace + spike
                membrane = membrane - spike * threshold_eff
                spike_sum = spike_sum + spike
                if membrane_sum is not None:
                    membrane_sum = membrane_sum + membrane
            spike_rate_feature = spike_sum / float(self.steps)
            if self.readout == 'rate_membrane':
                # Normalised mean membrane trace: the sub-threshold analogue
                # of the rate code (RMP-style readout).
                membrane_feature = membrane_sum / (
                    float(self.steps) * threshold)
                spike_rate_feature = torch.cat(
                    [spike_rate_feature, membrane_feature], dim=1)
        # Diagnostics are opt-in because synchronizing a scalar to CPU four
        # times per training step measurably slows distributed training.
        if self.record_stats:
            rate_only = (spike_sum / float(self.steps)
                         if spike_sum is not None else spike_rate_feature)
            self.last_spike_rate[int(task_index)] = (
                rate_only.detach().mean())
        residual = self.up_project(spike_rate_feature)
        if self.task_expert_up is not None:
            residual = residual + self.task_expert_up[task_index](
                self.task_expert_down[task_index](spike_rate_feature))
        scale = (self.residual_scale_limit * self.scale_multiplier *
                 torch.tanh(self.residual_scale[task_index])).view(1, -1, 1, 1)
        output = feature + scale * residual
        if self.carry_readout is not None and membrane_state is not None:
            # Selective cross-task state readout (see __init__).
            output = output + self.carry_scale * self.carry_readout(
                membrane_state)
        if return_state:
            if membrane is None:
                # Memoryless ANN control: keep the state interface uniform so
                # cross-task priming stays a well-defined no-op instead of a
                # None that downstream state handling would trip over.
                return output, torch.zeros_like(current)
            return output, membrane
        return output


class IndependentTaskSpikingAdapters(nn.Module):
    """One fixed-dynamics residual adapter per task, with no shared gradients."""

    def __init__(self, num_tasks, channels, bottleneck_channels=128, steps=4,
                 surrogate_scale=4.0, residual_scale_limit=1.0,
                 dynamics_type='lif', readout='rate',
                 adaptive_threshold=False, threshold_init=None,
                 input_activation='gelu', residual_scale_init=0.0,
                 decay_init=None, carry_readout=False,
                 carry_scale_init=0.0):
        super().__init__()
        self.adapters = nn.ModuleList([
            TaskModulatedSpikingAdapter(
                num_tasks=1, channels=channels,
                bottleneck_channels=bottleneck_channels, steps=steps,
                task_conditioned=False, surrogate_scale=surrogate_scale,
                residual_scale_limit=residual_scale_limit,
                dynamics_type=dynamics_type, readout=readout,
                adaptive_threshold=adaptive_threshold,
                threshold_init=threshold_init,
                input_activation=input_activation,
                residual_scale_init=residual_scale_init,
                decay_init=decay_init, carry_readout=carry_readout,
                carry_scale_init=carry_scale_init)
            for _ in range(num_tasks)
        ])
        self.last_spike_rate = {}

    @property
    def record_stats(self):
        return all(adapter.record_stats for adapter in self.adapters)

    @record_stats.setter
    def record_stats(self, enabled):
        for adapter in self.adapters:
            adapter.record_stats = bool(enabled)

    def forward(self, feature, task_index, membrane_state=None,
                return_state=False):
        result = self.adapters[task_index](
            feature, 0, membrane_state=membrane_state,
            return_state=return_state)
        if return_state:
            output, new_state = result
        else:
            output = result
        if self.adapters[task_index].record_stats:
            self.last_spike_rate[int(task_index)] = (
                self.adapters[task_index].last_spike_rate[0])
        if return_state:
            return output, new_state
        return output


class TaskStateRetentionGate(nn.Module):
    """Learned per-(previous, next) task retention of membrane state.

    Selective state reset: when the perceptual goal switches from task A to
    task B, the membrane trajectory of A is not thrown away blindly, nor kept
    blindly.  A gate r(A -> B) in (0, 1) decides channel-wise how much of the
    previous spiking state survives the switch.  The pair table is zero
    initialised, i.e. every switch starts at an even 50/50 split.
    """

    def __init__(self, num_tasks, channels):
        super().__init__()
        self.num_tasks = int(num_tasks)
        self.channels = int(channels)
        self.pair_embedding = nn.Embedding(self.num_tasks * self.num_tasks,
                                           self.channels)
        nn.init.zeros_(self.pair_embedding.weight)

    def forward(self, previous_index, next_index):
        pair = int(previous_index) * self.num_tasks + int(next_index)
        logits = self.pair_embedding(
            torch.tensor(pair, device=self.pair_embedding.weight.device,
                         dtype=torch.long))
        retention = torch.sigmoid(logits).view(1, -1, 1, 1)
        return retention


class TaskConditionedFeatureGate(nn.Module):
    """Shared task controller that produces channel-wise affine modulation.

    The last projection is zero-initialized, so enabling the controller is an
    exact identity transformation before training.  This makes the proposed
    model directly auditable against the baseline at initialization.
    """

    def __init__(self, num_tasks, channels, controller_dim=64):
        super().__init__()
        self.task_embedding = nn.Embedding(num_tasks, controller_dim)
        self.projector = nn.Sequential(
            nn.Linear(controller_dim, controller_dim),
            nn.GELU(),
            nn.Linear(controller_dim, channels * 2),
        )
        nn.init.normal_(self.task_embedding.weight, std=0.02)
        nn.init.zeros_(self.projector[-1].weight)
        nn.init.zeros_(self.projector[-1].bias)

    def forward(self, feature, task_index):
        index = torch.tensor(task_index, device=feature.device, dtype=torch.long)
        affine = self.projector(self.task_embedding(index))
        gamma, beta = affine.chunk(2, dim=-1)
        gamma = torch.tanh(gamma).view(1, -1, 1, 1)
        beta = beta.view(1, -1, 1, 1)
        return feature * (1.0 + gamma) + beta

class TaskPrompterWrapper(nn.Module):
    def __init__(self, p, backbone, heads):
        super(TaskPrompterWrapper, self).__init__()
        self.tasks = p.TASKS.NAMES

        self.backbone = backbone
        self.heads = heads 

        self.use_task_controller_gate = bool(p.get('use_task_controller_gate', False))
        self.task_to_index = {task: index for index, task in enumerate(self.tasks)}
        if self.use_task_controller_gate:
            controller_dim = int(p.get('task_controller_dim', 64))
            if controller_dim < 1:
                raise ValueError('task_controller_dim must be positive')
            self.task_controller_gate = TaskConditionedFeatureGate(
                len(self.tasks), int(p.final_embed_dim), controller_dim)
        else:
            self.task_controller_gate = None

        self.use_spiking_adapter = bool(p.get('use_spiking_adapter', False))
        self.spike_memory_type = str(p.get('spike_memory_type', 'legacy'))
        if self.spike_memory_type not in ('legacy', 'task_routed_temporal'):
            raise ValueError(
                'spike_memory_type must be legacy or task_routed_temporal')
        self.spike_independent_adapters = bool(
            p.get('spike_independent_adapters', False))
        configured_spike_tasks = p.get('spike_active_tasks', self.tasks)
        if isinstance(configured_spike_tasks, str):
            configured_spike_tasks = [configured_spike_tasks]
        self.spike_active_tasks = set(configured_spike_tasks)
        unknown_spike_tasks = self.spike_active_tasks.difference(self.tasks)
        if unknown_spike_tasks:
            raise ValueError('Unknown spike_active_tasks: {}'.format(
                sorted(unknown_spike_tasks)))
        if self.use_spiking_adapter:
            if not self.spike_active_tasks:
                raise ValueError(
                    'spike_active_tasks must not be empty when the adapter is enabled')
            if self.spike_memory_type == 'task_routed_temporal':
                if self.spike_independent_adapters:
                    raise ValueError(
                        'task-routed temporal memory already owns task-local '
                        'state; spike_independent_adapters must be false')
                self.spiking_adapter = TaskRoutedSpikingMemory(
                    num_tasks=len(self.tasks),
                    channels=int(p.final_embed_dim),
                    bottleneck_channels=int(
                        p.get('spike_bottleneck_dim', 64)),
                    num_experts=int(p.get('spike_num_experts', 4)),
                    dynamics_type=str(p.get('spike_dynamics_type', 'lif')),
                    surrogate_scale=float(
                        p.get('spike_surrogate_scale', 4.0)),
                    decay_init=float(p.get('spike_decay_init', 0.8)),
                    threshold_init=float(
                        p.get('spike_threshold_init', 0.7)),
                    residual_scale_init=float(
                        p.get('spike_residual_scale_init', 0.1)),
                    residual_scale_limit=float(
                        p.get('spike_residual_scale_limit', 1.0)),
                    router_temperature=float(
                        p.get('spike_router_temperature', 1.0)),
                    target_spike_rate=float(
                        p.get('spike_target_rate', 0.15)),
                    target_active_experts=float(
                        p.get('spike_target_active_experts', 1.0)),
                    record_stats=bool(p.get('spike_record_stats', False)),
                )
            elif self.spike_independent_adapters:
                if p.get('spike_context_routing', False):
                    raise ValueError(
                        'context routing is not supported by independent adapters')
                if int(p.get('spike_task_expert_rank', 0)):
                    raise ValueError(
                        'task experts are redundant with independent adapters')
                self.spiking_adapter = IndependentTaskSpikingAdapters(
                    num_tasks=len(self.tasks),
                    channels=int(p.final_embed_dim),
                    bottleneck_channels=int(p.get('spike_bottleneck_dim', 128)),
                    steps=int(p.get('spike_steps', 4)),
                    surrogate_scale=float(p.get('spike_surrogate_scale', 4.0)),
                    residual_scale_limit=float(
                        p.get('spike_residual_scale_limit', 1.0)),
                    dynamics_type=str(p.get('spike_dynamics_type', 'lif')),
                    readout=str(p.get('spike_readout', 'rate')),
                    adaptive_threshold=bool(
                        p.get('spike_adaptive_threshold', False)),
                    threshold_init=p.get('spike_threshold_init', None),
                    input_activation=str(
                        p.get('spike_input_activation', 'gelu')),
                    residual_scale_init=float(
                        p.get('spike_residual_scale_init', 0.0)),
                    decay_init=p.get('spike_decay_init', None),
                    carry_readout=bool(p.get('spike_cross_task_readout', False)),
                    carry_scale_init=float(
                        p.get('spike_carry_scale_init', 0.1)),
                )
            else:
                self.spiking_adapter = TaskModulatedSpikingAdapter(
                    num_tasks=len(self.tasks),
                    channels=int(p.final_embed_dim),
                    bottleneck_channels=int(p.get('spike_bottleneck_dim', 128)),
                    steps=int(p.get('spike_steps', 4)),
                    task_conditioned=bool(p.get('spike_task_conditioned', False)),
                    controller_dim=int(p.get('spike_controller_dim', 64)),
                    surrogate_scale=float(p.get('spike_surrogate_scale', 4.0)),
                    record_stats=bool(p.get('spike_record_stats', False)),
                    context_routing=bool(p.get('spike_context_routing', False)),
                    route_limit=float(p.get('spike_route_limit', 0.25)),
                    residual_scale_limit=float(
                        p.get('spike_residual_scale_limit', 1.0)),
                    task_expert_rank=int(p.get('spike_task_expert_rank', 0)),
                    dynamics_mode=str(p.get('spike_dynamics_mode', 'both')),
                    dynamics_type=str(p.get('spike_dynamics_type', 'lif')),
                    readout=str(p.get('spike_readout', 'rate')),
                    adaptive_threshold=bool(
                        p.get('spike_adaptive_threshold', False)),
                    threshold_init=p.get('spike_threshold_init', None),
                    input_activation=str(
                        p.get('spike_input_activation', 'gelu')),
                    residual_scale_init=float(
                        p.get('spike_residual_scale_init', 0.0)),
                    modulate_gain=bool(p.get('spike_modulate_gain', False)),
                    decay_init=p.get('spike_decay_init', None),
                    carry_readout=bool(p.get('spike_cross_task_readout', False)),
                    carry_scale_init=float(
                        p.get('spike_carry_scale_init', 0.1)),
                )
        else:
            self.spiking_adapter = None

        if self.spiking_adapter is not None:
            multiplier = float(p.get('spike_residual_scale_multiplier', 1.0))
            layers = (list(self.spiking_adapter.adapters)
                      if self.spike_independent_adapters
                      else [self.spiking_adapter])
            for layer in layers:
                layer.scale_multiplier = multiplier

        if 'dd_label_map_size' in p.keys():
            self.target_size = p.dd_label_map_size
        else:
            self.target_size = None

        # Selective-reset machinery: a learned per-task-pair gate that decides
        # how much of the previous task's membrane state survives a switch.  It
        # is attached to the spiking adapter module so the adapter-only
        # training prefix ('spiking_adapter.') covers it automatically.
        self.cross_task_state = bool(p.get('spike_cross_task_state', False))
        if self.cross_task_state:
            if self.spiking_adapter is None:
                raise ValueError(
                    'spike_cross_task_state needs use_spiking_adapter: true')
            if self.spike_memory_type == 'task_routed_temporal':
                raise ValueError(
                    'task_routed_temporal forbids cross-task membrane carry')
            self.spiking_adapter.state_retention = TaskStateRetentionGate(
                len(self.tasks), int(p.get('spike_bottleneck_dim', 128)))
        elif self.spiking_adapter is not None:
            self.spiking_adapter.state_retention = None

        # Streaming execution: when a policy is configured the wrapper runs a
        # task *sequence* on one encoded scene (used for stream training and
        # for switch-aware evaluation).
        self.stream_policy = p.get('spike_stream_policy', None)
        if self.stream_policy is not None:
            self.stream_policy = str(self.stream_policy)
        configured_stream = p.get('spike_stream_tasks', None)
        if configured_stream is None:
            self.stream_tasks = None
        else:
            if isinstance(configured_stream, str):
                configured_stream = [configured_stream]
            unknown = [t for t in configured_stream if t not in self.tasks]
            if unknown:
                raise ValueError(
                    'Unknown spike_stream_tasks: {}'.format(unknown))
            self.stream_tasks = list(configured_stream)
        self.stream_detach_state = bool(
            p.get('spike_stream_detach_state', False))
        # Persistent perception: when enabled, the spiking state is carried
        # across *frames* (forward calls), not only across tasks inside one
        # frame. Call reset_stream_state() at the start of a new stream.
        self.stream_persist_frames = bool(
            p.get('spike_stream_persist_frames', False))
        self.stream_persist_eval = bool(
            p.get('spike_stream_persist_eval', False))
        self._frame_states = {}
        # State-coupled iterative refinement: apply the spiking adapter to its
        # own corrected feature several times, carrying the membrane state
        # between passes. Defaults to one pass (numerically unchanged).
        self.refine_steps = int(p.get('spike_refine_steps', 1))
        if self.refine_steps < 1:
            raise ValueError('spike_refine_steps must be >= 1')

    def spiking_regularization_loss(self):
        """Return all differentiable SNN budget losses for this forward."""
        zero = next(self.parameters()).sum() * 0.0
        total = zero
        prompt_regularizer = getattr(
            self.backbone, 'spike_prompt_regularization_loss', None)
        if prompt_regularizer is not None:
            total = total + prompt_regularizer()
        adapter_regularizer = getattr(
            self.spiking_adapter, 'regularization_loss', None)
        if adapter_regularizer is not None:
            total = total + adapter_regularizer()
        return total

    def spike_prompt_statistics(self):
        dynamics = getattr(self.backbone, 'spike_prompt_dynamics', None)
        if dynamics is None:
            zero = next(self.parameters()).sum().detach() * 0.0
            return {'event_rate': zero, 'relation_degree': zero}
        return dynamics.statistics()

    def _normalize_requested_tasks(self, requested_tasks):
        if requested_tasks is None:
            requested_tasks = list(self.tasks)
        elif isinstance(requested_tasks, str):
            requested_tasks = [requested_tasks]
        else:
            requested_tasks = list(requested_tasks)

        unknown_tasks = [task for task in requested_tasks if task not in self.tasks]
        if unknown_tasks:
            raise ValueError('Unknown requested tasks: {}'.format(unknown_tasks))
        if len(requested_tasks) != len(set(requested_tasks)):
            raise ValueError('requested_tasks contains duplicates: {}'.format(requested_tasks))
        if not requested_tasks:
            raise ValueError('requested_tasks must contain at least one task')
        return requested_tasks

    def encode_visual_memory(self, x):
        """Cache the shared transformer states for repeated task execution."""
        return {
            'backbone_memory': self.backbone.encode_visual_memory(x),
            'img_size': x.size()[-2:],
        }

    @staticmethod
    def _detach_spike_state(state):
        if state is None:
            return None
        if torch.is_tensor(state):
            return state.detach()
        if isinstance(state, dict):
            return {
                key: (value.detach() if torch.is_tensor(value) else value)
                for key, value in state.items()
            }
        raise ValueError('spike state must be a tensor or dictionary')

    def execute_stream(self, memory, task_sequence=None,
                       state_policy='reset', detach_state=False,
                       initial_states=None):
        """Run a task *sequence* on one encoded scene, carrying spiking state.

        state_policy:
          'reset'       every task starts from a zero membrane (classic).
          'task_local'  a task resumes only its own previous state.
          'carry'       the next task starts from the previous task's state.
          'cross_prime' the next task starts from r(A->B) * state(A), with a
                        learned selective reset gate (default 0.5 at init).
        initial_states optionally seeds the per-task states (used for
        cross-frame persistence in stream inference).
        """
        if state_policy not in ('reset', 'task_local', 'carry', 'cross_prime'):
            raise ValueError('unknown state_policy {}'.format(state_policy))
        if (self.spike_memory_type == 'task_routed_temporal' and
                state_policy in ('carry', 'cross_prime')):
            raise ValueError(
                'task_routed_temporal state follows physical time per task; '
                'cross-task carry is not a valid temporal transition')
        if task_sequence is None:
            task_sequence = list(self.tasks)
        outputs = {}
        states = dict(initial_states) if initial_states else {}
        previous = None
        for task in task_sequence:
            state = None
            if state_policy == 'task_local':
                state = states.get(task)
                if detach_state and state is not None:
                    state = self._detach_spike_state(state)
            elif state_policy == 'carry':
                if previous is not None:
                    state = states[previous]
                    if detach_state and state is not None:
                        state = self._detach_spike_state(state)
            elif state_policy == 'cross_prime':
                state = states.get(task)
                if state is None and previous is not None:
                    retention_gate = getattr(
                        self.spiking_adapter, 'state_retention', None)
                    if retention_gate is None:
                        raise ValueError(
                            'cross_prime needs spike_cross_task_state: true')
                    retention = retention_gate(
                        self.task_to_index[previous],
                        self.task_to_index[task])
                    # Truncated BPTT: the source state may be detached, but the
                    # retention gate itself stays differentiable because the
                    # detach happens before the multiplication.
                    source = states[previous]
                    if source is None:
                        # No state to prime from (memoryless controls).
                        state = None
                    else:
                        if detach_state:
                            source = self._detach_spike_state(source)
                            if isinstance(source, dict):
                                source = source['membrane']
                        state = retention * source
            pod = {task: state} if state is not None else {}
            out, updated = self.execute_task(
                memory, [task], spike_states=pod, return_spike_states=True)
            outputs.update(out)
            states[task] = updated[task]
            previous = task
        return outputs, states

    def execute_task(self, memory, requested_tasks=None, spike_states=None,
                     return_spike_states=False):
        requested_tasks = self._normalize_requested_tasks(requested_tasks)
        if not isinstance(memory, dict) or 'backbone_memory' not in memory:
            raise ValueError('memory must be produced by encode_visual_memory')
        target_size = self.target_size if self.target_size is not None else memory['img_size']
        task_features, info = self.backbone.decode_visual_memory(
            memory['backbone_memory'], requested_tasks=requested_tasks)

        if spike_states is None:
            spike_states = {}
        elif not isinstance(spike_states, dict):
            raise ValueError('spike_states must be a task-to-tensor dictionary')
        out = {}
        updated_states = dict(spike_states)
        for t in requested_tasks:
            _task_fea = task_features[t]
            if self.task_controller_gate is not None:
                _task_fea = self.task_controller_gate(
                    _task_fea, self.task_to_index[t])
            if (self.spiking_adapter is not None and
                    t in self.spike_active_tasks):
                state = spike_states.get(t)
                for _ in range(self.refine_steps):
                    try:
                        if return_spike_states:
                            _task_fea, state = self.spiking_adapter(
                                _task_fea, self.task_to_index[t],
                                membrane_state=state, return_state=True)
                        else:
                            _task_fea = self.spiking_adapter(
                                _task_fea, self.task_to_index[t],
                                membrane_state=state)
                            state = None
                    except ValueError:
                        # Stale state from a different batch size or input
                        # geometry (e.g. a training stream leaking into
                        # validation): restart the membrane from zero.
                        state = None
                        if return_spike_states:
                            _task_fea, state = self.spiking_adapter(
                                _task_fea, self.task_to_index[t],
                                membrane_state=None, return_state=True)
                        else:
                            _task_fea = self.spiking_adapter(
                                _task_fea, self.task_to_index[t],
                                membrane_state=None)
                            state = None
                if return_spike_states:
                    updated_states[t] = state
            elif return_spike_states:
                # Streams may deliberately bypass SNN dynamics for a task
                # whose dense target is harmed by one-step quantisation. Keep
                # an explicit empty slot so execute_stream has the same state
                # dictionary contract for active and bypassed tasks.
                updated_states[t] = None
            if t != '3ddet':
                out[t] = F.interpolate(self.heads[t](_task_fea), target_size, mode=INTERPOLATE_MODE)
            else:
                out[t] = self.heads[t](_task_fea)
        if return_spike_states:
            return out, updated_states
        return out

    def reset_stream_state(self):
        """Forget the persistent cross-frame state (start of a new stream)."""
        self._frame_states = {}

    def _forward_single_frame(self, x, requested_tasks=None,
                              force_persist=False):
        reset_statistics = getattr(
            self.spiking_adapter, 'reset_regularization_statistics', None)
        if reset_statistics is not None:
            reset_statistics()
        memory = self.encode_visual_memory(x)
        if self.stream_policy is not None:
            # Persistence during evaluation is opt-in and requires an explicit
            # sequence boundary via reset_stream_state/sequence_start.  This
            # prevents validation samples from leaking state into one another.
            persist = (force_persist or
                       (self.stream_persist_frames and
                        (self.training or self.stream_persist_eval)))
            initial = self._frame_states if persist else None
            outputs, states = self.execute_stream(
                memory, task_sequence=self.stream_tasks,
                state_policy=self.stream_policy,
                detach_state=self.stream_detach_state,
                initial_states=initial)
            if persist:
                if self.stream_detach_state:
                    states = {
                        task: self._detach_spike_state(value)
                        for task, value in states.items()
                    }
                self._frame_states = states
            if requested_tasks is not None:
                requested = self._normalize_requested_tasks(requested_tasks)
                return {task: outputs[task] for task in requested}
            return outputs
        return self.execute_task(memory, requested_tasks=requested_tasks)

    def forward_sequence(self, x, requested_tasks=None, reset_state=True):
        """Run a real frame sequence shaped [B,T,C,H,W].

        One recurrent update is performed per physical frame.  This deliberately
        avoids manufacturing a temporal axis by repeating one static feature.
        """
        if not torch.is_tensor(x) or x.dim() != 5:
            raise ValueError('sequence input must have shape [B,T,C,H,W]')
        if self.stream_policy is None:
            raise ValueError('forward_sequence needs spike_stream_policy')
        if self.stream_policy != 'task_local':
            raise ValueError(
                'physical-time sequences require task_local state policy')
        if reset_state:
            self.reset_stream_state()
        frame_outputs = []
        for frame_index in range(x.shape[1]):
            frame_outputs.append(self._forward_single_frame(
                x[:, frame_index], requested_tasks=requested_tasks,
                force_persist=True))
        tasks = self._normalize_requested_tasks(requested_tasks)
        return {
            task: torch.stack([output[task] for output in frame_outputs], dim=1)
            for task in tasks
        }

    def forward(self, x, requested_tasks=None, sequence_start=False):
        if sequence_start:
            self.reset_stream_state()
        if torch.is_tensor(x) and x.dim() == 5:
            return self.forward_sequence(
                x, requested_tasks=requested_tasks, reset_state=True)
        return self._forward_single_frame(
            x, requested_tasks=requested_tasks, force_persist=False)
