"""Opt-in TaskPrompter prototype; not integrated or task-quality validated."""
import torch
import torch.nn.functional as F
from models.taskprompter_wrapper import TaskModulatedSpikingAdapter


class IntentNeuronCandidate(TaskModulatedSpikingAdapter):
    """Modulate effective threshold, with optional finite-horizon compensation.

    State dict keys match the legacy adapter. Zero controller output preserves
    legacy dynamics when horizon_mix=0. ANN still uses the legacy ANN branch;
    it is NOT the intent-conditioned analog comparator required for a paper.
    """
    def __init__(self, *args, threshold_log_limit=0.5,
                 decay_logit_limit=1.0, horizon_mix=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 <= horizon_mix <= 1:
            raise ValueError('horizon_mix must be in [0, 1]')
        if threshold_log_limit <= 0 or decay_logit_limit <= 0:
            raise ValueError('modulation limits must be positive')
        if horizon_mix and not self.modulate_gain:
            raise ValueError('horizon compensation needs modulate_gain=True')
        self.threshold_log_limit = float(threshold_log_limit)
        self.decay_logit_limit = float(decay_logit_limit)
        self.horizon_mix = float(horizon_mix)

    def _dynamics(self, task_index, feature):
        delta_decay = torch.zeros_like(self.base_decay_logit)
        delta_threshold = torch.zeros_like(self.base_threshold_raw)
        delta_gain = torch.zeros_like(self.base_decay_logit)
        if self.task_conditioned:
            index = torch.tensor(task_index, device=feature.device, dtype=torch.long)
            control = self.dynamics_projector(self.task_embedding(index))
            if self.modulate_gain:
                delta_decay, delta_threshold, delta_gain = control.chunk(3, -1)
            else:
                delta_decay, delta_threshold = control.chunk(2, -1)
            if self.dynamics_mode == 'threshold':
                delta_decay = torch.zeros_like(delta_decay)
            elif self.dynamics_mode == 'decay':
                delta_threshold = torch.zeros_like(delta_threshold)
        decay = torch.sigmoid(self.base_decay_logit +
                              self.decay_logit_limit * torch.tanh(delta_decay))
        threshold = (F.softplus(self.base_threshold_raw) + 0.5) * torch.exp(
            self.threshold_log_limit * torch.tanh(delta_threshold))
        gain = 1.0 + torch.tanh(delta_gain)
        if self.horizon_mix:
            compensation = 1.0 / (1.0 - decay.pow(self.steps)).clamp_min(0.1)
            gain = gain * (1.0 + self.horizon_mix * (compensation - 1.0))
        return tuple(value.view(1, -1, 1, 1) for value in (decay, threshold, gain))
