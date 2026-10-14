"""Research-only memoryless control; requires intent_neuron_candidate.py.

Uses all effective controller outputs as static excitability parameters.
This is one declared analog comparator, not an exact analog equivalent of LIF.
Optional routing, experts, adaptation, and state carry are excluded to isolate
the controller and bottleneck. Nothing is registered in the training model.
"""
from intent_neuron_candidate import IntentNeuronCandidate
import torch
import torch.nn.functional as F


class IntentAnalogControl(IntentNeuronCandidate):
    def __init__(self, *args, **kwargs):
        for key in ('context_routing', 'task_expert_rank', 'adaptive_threshold',
                    'carry_readout', 'horizon_mix'):
            if kwargs.get(key, False):
                raise ValueError('analog comparison excludes ' + key)
        if kwargs.get('dynamics_type', 'lif') != 'lif':
            raise ValueError('use lif construction to retain controller gradients')
        super().__init__(*args, **kwargs)

    def forward(self, feature, task_index, membrane_state=None,
                return_state=False):
        if membrane_state is not None:
            raise ValueError('memoryless control requires reset state')
        current = self.normalization(self.down_project(feature))
        if self.input_activation == 'gelu':
            current = F.gelu(current)
        decay, threshold, gain = self._dynamics(task_index, feature)
        # Same first-step drive as reset LIF, without recurrence or hard spikes.
        drive = (1.0 - decay) * current * gain
        rate = torch.sigmoid(self.surrogate_scale * (drive - threshold))
        if self.readout == 'rate_membrane':
            analog = torch.cat([rate, (drive - rate * threshold) / threshold], 1)
        else:
            analog = rate
        residual = self.up_project(analog)
        scale = (self.residual_scale_limit * self.scale_multiplier *
                 torch.tanh(self.residual_scale[task_index])).view(1, -1, 1, 1)
        output = feature + scale * residual
        # Analog activation must not be labeled as a measured spike rate.
        if return_state:
            return output, torch.zeros_like(current)
        return output
