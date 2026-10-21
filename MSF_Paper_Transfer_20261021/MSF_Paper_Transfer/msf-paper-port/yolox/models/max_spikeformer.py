import torch
import torch.nn as nn

from .paper_components import PaperPLIF as mem_update


ATTENTION_MODES = ('spike', 'spike_spatial', 'spike_residual', 'dense', 'identity')


class SpikeMaxAttention(nn.Module):
    """Spatial multi-head attention with selectable spike or dense dynamics.

    ``spike`` uses binary Q/K/V and projection spikes without Softmax:
    ``Q(K^T V)``. ``dense`` reproduces a continuous membrane path with
    scaled dot-product Softmax attention for a controlled ablation.
    """

    def __init__(self, dim, num_heads=8, mode='spike'):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim ({dim}) must be divisible by num_heads ({num_heads})')
        if mode not in ('spike', 'spike_spatial', 'dense'):
            raise ValueError(f'unsupported attention mode: {mode!r}')

        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.dim = dim
        self.mode = mode

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

        use_spikes = mode != 'dense'
        self.qkv_mem = mem_update(act=use_spikes)
        self.proj_mem = mem_update(act=use_spikes)
        self.use_spatial_projection = mode == 'spike_spatial'
        if self.use_spatial_projection:
            # SDT-v2 uses local 3x3 projections around spike attention.  The
            # depthwise form keeps this detector ablation parameter-matched.
            self.qkv_local = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
            self.qkv_local_norm = nn.BatchNorm2d(dim)
            self.proj_local = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
            self.proj_local_norm = nn.BatchNorm2d(dim)

    @staticmethod
    def _apply_local_projection(x, conv, norm):
        t, b, h, w, c = x.shape
        spatial = x.permute(0, 1, 4, 2, 3).reshape(t * b, c, h, w)
        spatial = norm(conv(spatial))
        return spatial.reshape(t, b, c, h, w).permute(0, 1, 3, 4, 2).contiguous()

    def forward(self, x):
        # x: [T, B, H, W, C]
        t, b, h, w, c = x.shape
        tokens = h * w
        if self.use_spatial_projection:
            x = self._apply_local_projection(x, self.qkv_local, self.qkv_local_norm)
        x_flat = x.reshape(t, b, tokens, c)

        qkv = self.qkv_mem(self.qkv(x_flat))
        qkv = qkv.reshape(t, b, tokens, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.mode in ('spike', 'spike_spatial'):
            # Associativity avoids allocating a quadratic token-token matrix.
            kv = k.transpose(-2, -1) @ v
            x_out = (q @ kv) * self.scale
        else:
            attention = (q @ k.transpose(-2, -1)) * self.scale
            attention = attention.softmax(dim=-1)
            x_out = attention @ v

        x_out = x_out.transpose(2, 3).reshape(t, b, tokens, c)
        x_out = self.proj_mem(self.proj(x_out)).reshape(t, b, h, w, c)
        if self.use_spatial_projection:
            x_out = self._apply_local_projection(x_out, self.proj_local, self.proj_local_norm)
        return x_out


class MaxSpikeFormerBlock(nn.Module):
    """MaxSpikeFormer block used by the Gen1 mechanism ablations.

    Modes:
      - ``spike``: binary Q/K/V, Softmax-free attention, binary MLP.
      - ``spike_spatial``: spike attention with SDT-style local projections.
      - ``spike_residual``: spike attention with SDT-style pre-activation residuals.
      - ``dense``: continuous Q/K/V, Softmax attention, continuous MLP.
      - ``identity``: bypass the complete block for a no-MSF control.
    """

    def __init__(self, c1, c2, num_heads=8, mlp_ratio=4.0, attention_mode='spike'):
        super().__init__()
        if attention_mode not in ATTENTION_MODES:
            raise ValueError(
                f'attention_mode must be one of {ATTENTION_MODES}, got {attention_mode!r}'
            )
        if c1 != c2:
            raise ValueError(f'MaxSpikeFormer requires c1 == c2, got {c1} and {c2}')

        self.attention_mode = attention_mode
        if attention_mode == 'identity':
            return

        dim = c2
        use_spikes = attention_mode not in ('dense', 'identity')
        self.pre_activation_residual = attention_mode == 'spike_residual'

        self.norm1 = nn.LayerNorm(dim)
        attention_kernel = 'spike_spatial' if attention_mode == 'spike_spatial' else (
            'dense' if attention_mode == 'dense' else 'spike'
        )
        self.attn = SpikeMaxAttention(dim, num_heads=num_heads, mode=attention_kernel)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.mem1 = mem_update(act=use_spikes)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.mem2 = mem_update(act=use_spikes)
        if self.pre_activation_residual:
            self.residual1 = mem_update(act=True)
            self.residual2 = mem_update(act=True)

    def forward(self, x):
        if self.attention_mode == 'identity':
            return x

        # [T, B, C, H, W] -> [T, B, H, W, C]
        x_permuted = x.permute(0, 1, 3, 4, 2).contiguous()
        attention_output = self.attn(self.norm1(x_permuted))
        if self.pre_activation_residual:
            x_permuted = self.residual1(x_permuted + attention_output)
        else:
            x_permuted = x_permuted + attention_output

        mlp_input = self.norm2(x_permuted)
        mlp_output = self.mem1(self.fc1(mlp_input))
        mlp_output = self.mem2(self.fc2(mlp_output))
        if self.pre_activation_residual:
            x_permuted = self.residual2(x_permuted + mlp_output)
        else:
            x_permuted = x_permuted + mlp_output

        return x_permuted.permute(0, 1, 4, 2, 3).contiguous()
