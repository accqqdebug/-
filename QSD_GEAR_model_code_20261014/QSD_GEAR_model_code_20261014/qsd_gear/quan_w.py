"""
@inproceedings{
    esser2020learned,
    title={LEARNED STEP SIZE QUANTIZATION},
    author={Steven K. Esser and Jeffrey L. McKinstry and Deepika Bablani and Rathinakumar Appuswamy and Dharmendra S. Modha},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=rkgO66VKDS}
}
    https://quanoview.readthedocs.io/en/latest/_raw/LSQ.html
"""
import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
import math
import warnings
from torch.nn.parameter import Parameter
from ._quan_base_plus import _Conv2dQ, Qmodes, _LinearQ, _ActQ, _LinearQ_v2,_Conv1dQ

__all__ = [
    'Conv2dLSQ',
    'Conv1dLSQ',
    'LinearLSQ',
    'ActLSQ',
    'LinearMCN',
    'LinearLSQ_v2',
    'positive_lsq_scale',
    'summarize_lsq_scales',
]


LSQ_SCALE_EPS = 1e-8
LSQ_INIT_SCALE_MIN = 1e-5
LSQ_POSITIVE_SCALE_VERSION = 1
_LEGACY_SCALE_WARNING_EMITTED = False


def positive_lsq_scale(raw_scale):
    """Map an unconstrained LSQ parameter to a finite positive step size."""
    return F.softplus(raw_scale) + LSQ_SCALE_EPS


def _inverse_softplus(value):
    value = torch.as_tensor(value).clamp_min(LSQ_SCALE_EPS)
    return torch.where(value > 20.0, value, torch.log(torch.expm1(value)))


class _PositiveLSQScaleMixin:
    """Provide positive LSQ steps and migrate legacy signed-alpha checkpoints."""

    def _setup_positive_lsq_scales(self, *names):
        self._positive_lsq_scale_names = tuple(names)
        self.register_buffer(
            "_lsq_positive_scale_version",
            torch.tensor(LSQ_POSITIVE_SCALE_VERSION, dtype=torch.int16),
        )
        with torch.no_grad():
            for name in self._positive_lsq_scale_names:
                parameter = getattr(self, name, None)
                if parameter is not None:
                    parameter.copy_(_inverse_softplus(torch.ones_like(parameter)))

    def get_lsq_scale(self, name="alpha"):
        if name not in self._positive_lsq_scale_names:
            raise KeyError(f"unknown LSQ scale parameter: {name}")
        raw_scale = getattr(self, name, None)
        if raw_scale is None:
            return None
        return positive_lsq_scale(raw_scale)

    def set_lsq_scale(self, value, name="alpha"):
        if name not in self._positive_lsq_scale_names:
            raise KeyError(f"unknown LSQ scale parameter: {name}")
        parameter = getattr(self, name, None)
        if parameter is None:
            raise ValueError(f"LSQ scale {name} is disabled")
        value = torch.as_tensor(value, device=parameter.device, dtype=parameter.dtype)
        value = torch.nan_to_num(
            value.abs(),
            nan=LSQ_SCALE_EPS,
            posinf=1.0,
            neginf=LSQ_SCALE_EPS,
        ).clamp_min(LSQ_SCALE_EPS)
        with torch.no_grad():
            parameter.copy_(_inverse_softplus(value).expand_as(parameter))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        global _LEGACY_SCALE_WARNING_EMITTED
        version_key = prefix + "_lsq_positive_scale_version"
        legacy_checkpoint = version_key not in state_dict
        if legacy_checkpoint:
            repaired_values = 0
            for name in self._positive_lsq_scale_names:
                key = prefix + name
                if key not in state_dict:
                    continue
                legacy = state_dict[key]
                invalid = ~torch.isfinite(legacy) | (legacy <= 0)
                repaired_values += int(invalid.sum().item())
                effective = torch.nan_to_num(
                    legacy.abs(),
                    nan=LSQ_SCALE_EPS,
                    posinf=1.0,
                    neginf=LSQ_SCALE_EPS,
                ).clamp_min(LSQ_SCALE_EPS)
                state_dict[key] = _inverse_softplus(effective)
            state_dict[version_key] = self._lsq_positive_scale_version.detach().clone()
            if not _LEGACY_SCALE_WARNING_EMITTED:
                warnings.warn(
                    "Loading a legacy signed-alpha LSQ checkpoint. Weight steps are "
                    "migrated to positive softplus parameters; non-positive or invalid "
                    f"values are repaired (first migrated module repaired {repaired_values}).",
                    RuntimeWarning,
                )
                _LEGACY_SCALE_WARNING_EMITTED = True
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def summarize_lsq_scales(model):
    """Return auditable weight/activation LSQ scale health statistics."""
    weight_scales = []
    activation_scales = []
    activation_enabled_layers = 0
    activation_disabled_layers = 0
    for module in model.modules():
        if not isinstance(module, _PositiveLSQScaleMixin):
            continue
        if isinstance(module, ActLSQ):
            if not module.enabled:
                activation_disabled_layers += 1
                continue
            activation_enabled_layers += 1
            destination = activation_scales
        else:
            destination = weight_scales
        for name in module._positive_lsq_scale_names:
            scale = module.get_lsq_scale(name)
            if scale is not None:
                destination.append(scale.detach().reshape(-1).cpu())

    def summarize(values, prefix):
        if not values:
            return {
                f"{prefix}_values": 0,
                f"{prefix}_min": math.nan,
                f"{prefix}_max": math.nan,
                f"{prefix}_nonfinite": 0,
                f"{prefix}_nonpositive": 0,
            }
        values = torch.cat(values)
        finite = torch.isfinite(values)
        finite_values = values[finite]
        return {
            f"{prefix}_values": int(values.numel()),
            f"{prefix}_min": (
                float(finite_values.min().item()) if finite_values.numel() else math.nan
            ),
            f"{prefix}_max": (
                float(finite_values.max().item()) if finite_values.numel() else math.nan
            ),
            f"{prefix}_nonfinite": int((~finite).sum().item()),
            f"{prefix}_nonpositive": int((finite & (values <= 0)).sum().item()),
        }

    result = summarize(weight_scales, "lsq_weight_scale")
    result.update(summarize(activation_scales, "lsq_activation_scale"))
    result["activation_lsq_enabled_layers"] = activation_enabled_layers
    result["activation_lsq_disabled_layers"] = activation_disabled_layers
    return result


class FunLSQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp):
        assert alpha > 0, 'alpha = {}'.format(alpha)
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp
        q_w = (weight / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = weight / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        # indicate_middle = torch.ones(indicate_small.shape).to(indicate_small.device) - indicate_small - indicate_big
        indicate_middle = 1.0 - indicate_small - indicate_big  # Thanks to @haolibai
        grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                -q_w + q_w.round())) * grad_weight * g).sum().unsqueeze(dim=0)
        grad_weight = indicate_middle * grad_weight
        # The following operation can make sure that alpha is always greater than zero in any case and can also
        # suppress the update speed of alpha. (Personal understanding)
        # grad_alpha.clamp_(-alpha.item(), alpha.item())  # FYI
        return grad_weight, grad_alpha, None, None, None


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


class Conv2dLSQ(_PositiveLSQScaleMixin, _Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits_w=4,
                 mode=Qmodes.kernel_wise, activation_quantization="off", **kwargs):
        super(Conv2dLSQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits_w, mode=mode)
        self._setup_positive_lsq_scales("alpha")
        self.act = ActLSQ(
            in_features=in_channels,
            nbits_a=nbits_w,
            enabled=activation_quantization == "lsq",
            channel_axis=1,
        )

    def forward(self, x):
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        # w_reshape = self.weight.reshape([self.weight.shape[0], -1]).transpose(0, 1)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.set_lsq_scale(2 * self.weight.abs().mean() / math.sqrt(Qp))
            # self.alpha.data.copy_(self.weight.abs().max() * 2)
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        # Method1: 31GB GPU memory (AlexNet w4a4 bs 2048) 17min/epoch
        alpha = grad_scale(self.get_lsq_scale(), g)
        # print(alpha.shape)
        # print(self.weight.shape)
        alpha = alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha

        x = self.act(x)
        # w = w.clamp(Qn, Qp)
        # q_w = round_pass(w)
        # w_q = q_w * alpha

        # Method2: 25GB GPU memory (AlexNet w4a4 bs 2048) 32min/epoch
        # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        # wq = y.transpose(0, 1).reshape(self.weight.shape).detach() + self.weight - self.weight.detach()
        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class Conv1dLSQ(_PositiveLSQScaleMixin, _Conv1dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits_w=4,
                 mode=Qmodes.kernel_wise, activation_quantization="off", **kwargs):
        super(Conv1dLSQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits_w, mode=mode)
        self._setup_positive_lsq_scales("alpha")
        self.act = ActLSQ(
            in_features=in_channels,
            nbits_a=nbits_w,
            enabled=activation_quantization == "lsq",
            channel_axis=1,
        )

    def forward(self, x):
        if self.alpha is None:
            return F.conv1d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        # w_reshape = self.weight.reshape([self.weight.shape[0], -1]).transpose(0, 1)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.set_lsq_scale(2 * self.weight.abs().mean() / math.sqrt(Qp))
            # self.alpha.data.copy_(self.weight.abs().max() * 2)
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        # Method1: 31GB GPU memory (AlexNet w4a4 bs 2048) 17min/epoch
        alpha = grad_scale(self.get_lsq_scale(), g)

        # print(alpha.shape)
        alpha = alpha.unsqueeze(1).unsqueeze(2)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        x = self.act(x)
        # w = w.clamp(Qn, Qp)
        # q_w = round_pass(w)
        # w_q = q_w * alpha

        # Method2: 25GB GPU memory (AlexNet w4a4 bs 2048) 32min/epoch
        # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        # wq = y.transpose(0, 1).reshape(self.weight.shape).detach() + self.weight - self.weight.detach()
        return F.conv1d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
class LinearLSQ_v2(_PositiveLSQScaleMixin, _LinearQ_v2):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4,
                 activation_quantization="off", **kwargs):
        super(LinearLSQ_v2, self).__init__(in_features=in_features,
                                           out_features=out_features, bias=bias, nbits=nbits_w, mode=Qmodes.kernel_wise)
        self._setup_positive_lsq_scales("alpha", "beta")
        self.act = ActLSQ(
            in_features=in_features,
            nbits_a=nbits_w,
            enabled=activation_quantization == "lsq",
            channel_axis=-1,
        )

    def forward(self, x):
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            initial_scale = 2 * self.weight.abs().mean() / math.sqrt(Qp)
            self.set_lsq_scale(torch.sqrt(initial_scale), "alpha")
            self.set_lsq_scale(torch.sqrt(initial_scale), "beta")
            # self.alpha.data.copy_(self.weight.abs().max() / 2 ** (self.nbits - 1))
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        # Method1:
        alpha = grad_scale(self.get_lsq_scale("alpha"), g)
        beta = grad_scale(self.get_lsq_scale("beta"), g)
        alpha = alpha.unsqueeze(1)
        beta = beta.unsqueeze(0)
        scale = alpha @ beta  # ).transpose(0,1)
        # print(scale.shape)
        w_q = round_pass((self.weight / scale).clamp(Qn, Qp)) * scale

        x = self.act(x)
        # w = self.weight / alpha
        # w = w.clamp(Qn, Qp)
        # q_w = round_pass(w)
        # w_q = q_w * alpha

        # Method2:
        # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        return F.linear(x, w_q, self.bias)


class LinearLSQ(_PositiveLSQScaleMixin, _LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4,
                 activation_quantization="off", **kwargs):
        super(LinearLSQ, self).__init__(in_features=in_features,
                                        out_features=out_features, bias=bias, nbits=nbits_w, mode=Qmodes.kernel_wise)
        self._setup_positive_lsq_scales("alpha")
        self.act = ActLSQ(
            in_features=in_features,
            nbits_a=nbits_w,
            enabled=activation_quantization == "lsq",
            channel_axis=-1,
        )

    def qw(self, weight):
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.set_lsq_scale(2 * self.weight.abs().mean() / math.sqrt(Qp))
            # self.alpha.data.copy_(self.weight.abs().max() / 2 ** (self.nbits - 1))
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        # Method1:
        alpha = grad_scale(self.get_lsq_scale(), g)
        alpha = alpha.unsqueeze(1)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        return w_q

    def forward(self, x):
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)

        w_q = self.qw(self.weight)

        x = self.act(x)
        # w = self.weight / alpha
        # w = w.clamp(Qn, Qp)
        # q_w = round_pass(w)
        # w_q = q_w * alpha

        # Method2:
        # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        return F.linear(x, w_q, self.bias)


class LinearMCN(Linear):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4,
                 activation_quantization="off", **kwargs):
        super(LinearMCN, self).__init__(in_features=in_features,
                                        out_features=out_features, bias=bias)
        self.act = ActLSQ(
            in_features=in_features,
            nbits_a=nbits_w,
            enabled=activation_quantization == "lsq",
            channel_axis=-1,
        )
        self.MCF_Function = MCF_Function.apply
        self.nbits = nbits_w
        self.generate_MFilters()
        self.out_features = out_features

    # self.in_features = in_features

    def generate_MFilters(self):
        self.MFilters = Parameter(torch.randn(self.out_features, 1))

    # self.beta = Parameter(torch.randn(1, self.in_features))

    def forward(self, x):
        if self.MFilters is None:
            return F.linear(x, self.weight, self.bias)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        bin = 0.02
        self.MFilters.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
        # self.beta.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))

        x = self.act(x)
        # print(self.weight.shape)
        # alpha = (self.beta @ self.MFilters).transpose(0,1)
        weight_bin = (self.weight / self.MFilters).clamp(Qn, Qp).round()  # * bin
        w_q = self.MCF_Function(self.weight, self.MFilters, weight_bin)

        return F.linear(x, w_q, self.bias)


class MCF_Function(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, MScale, weight_bin):
        weight = weight_bin * MScale

        ctx.save_for_backward(weight, MScale, weight_bin)
        return weight

    @staticmethod
    def backward(ctx, gradOutput):
        weight, MScale, weight_bin = ctx.saved_tensors

        nChannel = MScale.size()[0]
        nOutputPlane = gradOutput.size()[0]
        nInputPlane = gradOutput.size()[1]
        para_loss = 0.0001

        target1 = para_loss * (weight - weight_bin * MScale)
        gradWeight = target1 + (gradOutput * MScale)

        target2 = (weight - weight_bin * MScale) * weight_bin
        grad_h2_sum = torch.sum(gradOutput * weight, keepdim=True, dim=1)

        grad_target2 = torch.sum(target2, keepdim=True, dim=1)

        gradMScale = grad_h2_sum - para_loss * grad_target2
        gradweight_bin = gradOutput * 0
        return gradWeight, gradMScale, gradweight_bin


class ActLSQ(_PositiveLSQScaleMixin, _ActQ):
    def __init__(self, in_features, nbits_a=4, mode=Qmodes.kernel_wise,
                 enabled=False, channel_axis=1, **kwargs):
        super(ActLSQ, self).__init__(in_features=in_features, nbits=nbits_a, mode=mode)
        self.enabled = bool(enabled)
        self.channel_axis = int(channel_axis)
        self._setup_positive_lsq_scales("alpha")
        if self.zero_point is not None:
            torch.nn.init.zeros_(self.zero_point)
            self.zero_point.requires_grad_(False)
        if not self.enabled:
            if self.alpha is not None:
                self.alpha.requires_grad_(False)
            self.init_state.fill_(0)
            self.signed.fill_(0)

    def forward(self, x):
        if not self.enabled or self.alpha is None:
            return x

        channel_axis = self.channel_axis % x.ndim
        if x.shape[channel_axis] != self.alpha.numel():
            raise ValueError(
                "activation LSQ channel dimension mismatch: "
                f"axis {channel_axis} has {x.shape[channel_axis]} values, "
                f"expected {self.alpha.numel()}"
            )

        if self.training and self.init_state == 0:
            self.signed.fill_(int(bool((x.detach().min() < -1e-5).item())))
            qp = 2 ** (self.nbits - 1) - 1 if self.signed.item() else 2 ** self.nbits - 1
            reduce_dims = tuple(i for i in range(x.ndim) if i != channel_axis)
            initial_scale = 2 * x.detach().abs().mean(dim=reduce_dims) / math.sqrt(qp)
            self.set_lsq_scale(initial_scale.clamp_min(LSQ_INIT_SCALE_MIN))
            self.init_state.fill_(1)
        elif self.init_state == 0:
            raise RuntimeError(
                "enabled activation LSQ must be initialized by a training forward "
                "or loaded from a v2 checkpoint before evaluation"
            )

        if self.signed.item():
            qn = -2 ** (self.nbits - 1)
            qp = 2 ** (self.nbits - 1) - 1
        else:
            qn = 0
            qp = 2 ** self.nbits - 1
        g = 1.0 / math.sqrt(x.numel() * qp)
        scale = grad_scale(self.get_lsq_scale(), g)
        view_shape = [1] * x.ndim
        view_shape[channel_axis] = scale.numel()
        scale = scale.reshape(view_shape)
        return round_pass((x / scale).clamp(qn, qp)) * scale

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        legacy_checkpoint = prefix + "_lsq_positive_scale_version" not in state_dict
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if not self.enabled:
            self.set_lsq_scale(LSQ_SCALE_EPS)
            if self.zero_point is not None:
                self.zero_point.data.zero_()
            self.init_state.zero_()
            self.signed.zero_()
        elif legacy_checkpoint:
            # Legacy ActLSQ was a pass-through, so its saved parameters contain no
            # meaningful calibration. Force a fresh training-mode initialization.
            self.set_lsq_scale(1.0)
            if self.zero_point is not None:
                self.zero_point.data.zero_()
            self.init_state.zero_()
            self.signed.zero_()
