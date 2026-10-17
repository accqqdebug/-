"""Exact event-time linear synaptic traces with optional surrogate spike readout.

Only the linear traces are analytically integrated. This does not claim an exact
parallel solution of nonlinear LIF reset or STP. Readouts are dense; the detector
remains the upstream EAS-SNN detector. State is explicit, never shared across clips.
"""
import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TraceState:
    bank: torch.Tensor
    time_us: int

    def detach(self):
        return TraceState(self.bank.detach(), self.time_us)


class EventTraceEmbedding(nn.Module):
    def __init__(self, height, width, tau_ms=(5., 100.), spikes=True):
        super().__init__()
        self.height, self.width = height, width
        self.spikes = spikes
        self.log_min, self.log_max = math.log(100.), math.log(1_000_000.)
        ratio = (torch.tensor(tau_ms).log() + math.log(1000.) - self.log_min) / (self.log_max - self.log_min)
        self.tau_raw = nn.Parameter(torch.logit(ratio))
        self.gate_logits = nn.Parameter(torch.zeros(len(tau_ms), 2))
        if spikes:
            from spikingjelly.activation_based import surrogate
            self.surrogate = surrogate.ATan()

    @property
    def tau_us(self):
        return (self.log_min + self.tau_raw.sigmoid() * (self.log_max - self.log_min)).exp()

    def encode(self, event_list, readouts, states=None):
        if len(event_list) != len(readouts):
            raise ValueError('One readout sequence is required per sample')
        states = states or [None] * len(event_list)
        if len(states) != len(event_list):
            raise ValueError('One state is required per sample')
        samples, final_states = [], []
        device = self.tau_raw.device
        tau = self.tau_us[:, None]
        pixels = 2 * self.height * self.width
        for events, times, previous in zip(event_list, readouts, states):
            events, times = events.to(device), times.to(device=device, dtype=torch.int64)
            if times.ndim != 1 or len(times) == 0 or torch.any(times[1:] < times[:-1]):
                raise ValueError('Readouts must be nonempty and nondecreasing')
            t = events[:, 2].to(torch.int64)
            if len(t) and (torch.any(t[1:] < t[:-1]) or t[-1] > times[-1]):
                raise ValueError('Unsorted or future events supplied')
            x, y = events[:, 0].long(), events[:, 1].long()
            if len(t) and (torch.any(x < 0) or torch.any(x >= self.width) or torch.any(y < 0) or torch.any(y >= self.height)):
                raise ValueError('Event outside sensor bounds')
            index = ((events[:, 3] > 0).long() * self.height + y) * self.width + x
            if previous is not None:
                if previous.time_us > times[0] or (len(t) and t[0] < previous.time_us):
                    raise ValueError('State or events violate chronological chunk ordering')
                if previous.bank.shape != (len(tau), 2, self.height, self.width):
                    raise ValueError('Incompatible trace state')
            frames = []
            for time_us in times:
                mask = t <= time_us
                dt = (time_us - t[mask]).to(dtype=tau.dtype)
                weights = (-dt[None] / tau).exp()
                bank = torch.zeros(len(tau), pixels, device=device, dtype=tau.dtype)
                bank = bank.scatter_add(1, index[mask][None].expand(len(tau), -1), weights)
                bank = bank.reshape(len(tau), 2, self.height, self.width)
                if previous is not None:
                    decay = (-(time_us - previous.time_us).to(tau.dtype) / tau[:, 0]).exp()
                    bank = bank + previous.bank.to(device) * decay[:, None, None, None]
                gate = self.gate_logits.softmax(dim=0)[:, :, None, None]
                voltage = torch.log1p((bank * gate).sum(dim=0))
                frames.append(self.surrogate(voltage - 1.) if self.spikes else voltage)
            samples.append(torch.stack(frames))
            final_states.append(TraceState(bank, int(times[-1])))
        return torch.stack(samples, dim=1), final_states  # T,B,2,H,W

    def forward(self, batch):
        return self.encode(batch['events'], batch['readouts'], batch.get('states'))[0]


class FixedWindowEmbedding(nn.Module):
    """A causal fixed-window count control with matched output geometry."""
    def __init__(self, height, width, window_us=50000, spikes=True):
        super().__init__()
        self.height, self.width, self.window_us, self.spikes = height, width, window_us, spikes
        self.register_buffer('anchor', torch.zeros(()))
        if spikes:
            from spikingjelly.activation_based import surrogate
            self.surrogate = surrogate.ATan()

    def forward(self, batch):
        samples = []
        for events, times in zip(batch['events'], batch['readouts']):
            events = events.to(self.anchor.device)
            t = events[:, 2].long()
            if len(t) and t[-1] > times[-1]:
                raise ValueError('Future events supplied')
            index = ((events[:, 3] > 0).long() * self.height + events[:, 1].long()) * self.width + events[:, 0].long()
            frames = []
            for time_us in times.to(t.device):
                mask = (t <= time_us) & (t > time_us - self.window_us)
                counts = torch.zeros(2 * self.height * self.width, device=t.device)
                counts = counts.scatter_add(0, index[mask], torch.ones(mask.sum(), device=t.device))
                voltage = torch.log1p(counts.reshape(2, self.height, self.width))
                frames.append(self.surrogate(voltage - 1.) if self.spikes else voltage)
            samples.append(torch.stack(frames))
        return torch.stack(samples, dim=1)


class UpstreamARSNNAdapter(nn.Module):
    """Keep the author's learned sampler; only adapt raw events to its count bins."""
    def __init__(self, embedding, height, width, micro_steps=8):
        super().__init__()
        self.embedding, self.height, self.width, self.micro_steps = embedding, height, width, micro_steps

    def forward(self, batch):
        device = next(self.embedding.parameters()).device
        samples = []
        for events, times in zip(batch['events'], batch['readouts']):
            events = events.to(device)
            end = int(times[-1])
            start = int(batch['start_us'])
            t = events[:, 2].long()
            if len(t) and (t[-1] > end or t[0] < start):
                raise ValueError('Events outside observation interval')
            time_bin = ((t - start) * self.micro_steps // max(1, end - start)).clamp(max=self.micro_steps - 1)
            index = ((time_bin * 2 + (events[:, 3] > 0).long()) * self.height + events[:, 1].long()) * self.width + events[:, 0].long()
            counts = torch.zeros(self.micro_steps * 2 * self.height * self.width, device=device)
            counts = counts.scatter_add(0, index, torch.ones(len(index), device=device))
            samples.append(counts.reshape(self.micro_steps, 2, self.height, self.width))
        count_input = torch.stack(samples, dim=0).unsqueeze(1)  # B,Tl,Tm,2,H,W
        result = self.embedding(count_input)
        return result[0] if result.ndim > 5 else result
