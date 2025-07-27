import math
import torch
import causal_conv1d_cuda
import selective_scan_cuda
import torch.nn.functional as F

from torch import nn
from einops import rearrange, repeat
from torch.cuda.amp import custom_bwd, custom_fwd


class SSM(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, delta, A, B, C, D, z, delta_bias, delta_softplus):
        """
            inputs: (batch, in_channels_dec, seqlen)
            z: gated unit
        """
        if not z.is_contiguous():
            z, inputs = z.contiguous(), inputs.contiguous()

        L = inputs.shape[-1]

        if not A.is_complex():
            B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        else:
            B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
            C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        if D is not None:
            D = D.contiguous()

        out, scan_intermediates, out_z = selective_scan_cuda.fwd(
            inputs, delta, A, B, C, D, z, delta_bias, delta_softplus
        )

        ctx.delta_softplus = delta_softplus
        ctx.save_for_backward(inputs, A, B, C, D, delta, z, delta_bias, scan_intermediates, out)

        return out_z

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        """
            dout: (batch, in_channels_dec, seqlen)
        """

        inputs, A, B, C, D, delta, z, delta_bias, scan_intermediates, out = ctx.saved_tensors

        dz = torch.empty_like(z)
        dinputs, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
            inputs, delta, A, B, C, D, z, delta_bias, dout.contiguous(), scan_intermediates, out, dz, ctx.delta_softplus,
            True  # option to recompute out_z
        )

        dD = dD if D is not None else None
        if not A.is_complex():
            dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
            dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
        else:
            dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()

        return dinputs, ddelta, dA, dB, dC, dD, dz, ddelta_bias, None


def ssm(x, delta, A, B=None, C=None, D=None, z=None,
        delta_bias=None, delta_softplus=True):

    return SSM.apply(x, delta, A, B, C, D, z, delta_bias, delta_softplus)


class CausalConv(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, conv1d_weight, conv1d_bias):
        """
            inputs: (batch, in_channels_dec, seqlen)
        """

        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
        conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
            inputs, conv1d_weight, conv1d_bias, None, True
        )

        ctx.save_for_backward(inputs, conv1d_weight, conv1d_bias, conv1d_out)

        return conv1d_out

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        """
            dout: (batch, in_channels_dec, seqlen)
        """
        inputs, conv1d_weight, conv1d_bias, conv1d_out = ctx.saved_tensors

        dinputs = torch.empty_like(inputs)
        dinputs, dconv1d_weight, dconv1d_bias = causal_conv1d_cuda.causal_conv1d_bwd(
            inputs, conv1d_weight, conv1d_bias, dout, None, dinputs, True
        )

        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")

        return dinputs, dconv1d_weight, dconv1d_bias


def causalconv(x, conv1d_weight=None, conv1d_bias=None):

    return CausalConv.apply(x, conv1d_weight, conv1d_bias)


class MultiheadSSM(nn.Module):
    def __init__(
            self,
            in_channels,
            num_heads,
            d_state=16,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            bias=False,
            layer_idx=None,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MultiheadSSM, self).__init__()

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.in_channels)
        self.h_channels = self.d_inner // num_heads
        self.dt_rank = math.ceil(self.in_channels / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.in_channels, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, self.in_channels, bias=bias, **factory_kwargs)

        self.x_proj = nn.Linear(
            self.h_channels, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.h_channels, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.h_channels, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.h_channels,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.h_channels, device=device, dtype=torch.float32))  # Keep in fp32
        self.D._no_weight_decay = True

    def forward(self, inputs):
        assert not self.d_inner % self.num_heads

        # (h_channels, d_state)
        A = -torch.exp(self.A_log.float())
        x, z = self.in_proj(inputs).chunk(2, dim=-1)
        x = rearrange(x, 'b l (h t) -> (b h) t l', h=self.num_heads)
        z = rearrange(z, 'b l (h t) -> (b h) t l', h=self.num_heads)

        x_dbl = F.linear(rearrange(x, 'b t l -> (b l) t'), self.x_proj.weight)
        delta = rearrange(self.dt_proj.weight @ x_dbl[:, :self.dt_rank].t(), "t (b l) -> b t l", l=x.shape[-1])
        B, C = x_dbl[:, self.dt_rank:self.dt_rank + self.d_state], x_dbl[:, -self.d_state:]
        if self.D is not None:
            D = self.D.contiguous()

        state = ssm(
            x, delta, A, B, C, D, z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True
        )

        return self.out_proj(rearrange(state, '(b h) t l -> b l (h t)', h=self.num_heads))
