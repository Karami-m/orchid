"""
Util functions for Orchid block.

Some of the blocks are (modified) from
the code of Hyena: https://github.com/HazyResearch/safari/tree/main
including PositionalEmbedding(), ExponentialModulation(), and fixed long convolution filter ConvKernel_Static()
"""

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from orchid_standalone.utils.nn import Activation, OptimModule
from orchid_standalone.utils.dct_ops import dct1, idct1, dct, idct


class Sin(nn.Module):
    def __init__(self, dim, w=10, w_mod=1, train_freq=True):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim)) if train_freq else w * torch.ones(1, dim)
        self.w_mod = w_mod

    def forward(self, x):
        return torch.sin(self.w_mod * self.freq * x)


class PositionalEmbedding(OptimModule):
    def __init__(self, emb_dim: int, seq_len: int, lr_pos_emb: float = 1e-5, **kwargs):
        """Complex exponential positional embeddings for Hyena filters."""
        super().__init__()

        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]  # 1, L, 1

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
            # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len  # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.register("z", z, lr=lr_pos_emb)
        self.register("t", t, lr=0.0)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(OptimModule):
    def __init__(
            self,
            d_model,
            fast_decay_pct=0.3,
            slow_decay_pct=1.5,
            target=1e-2,
            modulation_lr=0.0,
            modulate: bool = True,
            shift: float = 0.0,
            **kwargs
    ):
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register("deltas", deltas, lr=modulation_lr)

    def forward(self, t, x):
        if self.modulate:
            decay = torch.exp(-t * self.deltas.abs())
            x = x * (decay + self.shift)
        return x


def transform(x, mode='fft', inverse=False, fft_size=2, dim=-1):
    mode = (mode, None) if isinstance(mode, str) else mode

    if 'fft' in mode[0]:
        if not '-o' in mode[0]:
            if not inverse:
                return torch.fft.rfft(x, n=fft_size, dim=dim) / fft_size
            else:
                return torch.fft.irfft(x, n=fft_size, dim=dim, norm='forward')
        elif '-o' in mode[0]:
            if not inverse:
                return torch.fft.rfft(x, n=fft_size, dim=dim, norm='ortho')
            else:
                return torch.fft.irfft(x, n=fft_size, dim=dim, norm='ortho')

    elif mode[0] == 'dct1':
        if not inverse:
            return dct1(x, dim=dim)
        else:
            return idct1(x, dim=dim)

    elif mode[0] in ['dct', 'dct2']:
        if not inverse:
            return dct(x, dim=dim)
        else:
            return idct(x, dim=dim)

    elif 'dctx' in mode[0]:
        dctx_fn = mode[1]
        return dctx_fn(x, inverse=inverse, dim=dim)

class ConvKernel_Static(OptimModule):
    def __init__(
            self,
            d_model,
            emb_dim=3,  # dim of input to MLP, augments with positional encoding
            order=16,  # width of the implicit MLP
            seq_len=1024,
            lr=1e-3,
            lr_pos_emb=1e-5,
            dropout=0.0,
            w=1,  # frequency of periodic activations
            w_mod=1,  # non-learnable modification of w
            wd=0,  # weight decay of kernel parameters
            bias=True,
            num_inner_mlps=2,
            linear_mixer=False,
            modulate: bool = True,
            normalized=False,
            bidirectional=False,
            dxt='fft',
            chang_initialize=None,
            **kwargs,
    ):
        """
        Creates Implicit long filter with modulation for static global convolution.

        Args:
            d_model: number of channels in the input
            emb_dim: dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP

        """
        super().__init__()

        self.d_model = d_model
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.modulate = modulate
        self.use_bias = bias
        self.bidirectional = bidirectional
        self.dxt = dxt

        if ('dct' in dxt) and bidirectional:
            raise ValueError('for DCT, biderectional should be False')

        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(dropout)

        act = Sin(dim=order, w=w, w_mod=w_mod)
        assert (
                emb_dim % 2 != 0 and emb_dim >= 3
        ), "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr_pos_emb)

        # uses a variable number of inner linear layers
        if linear_mixer is False:
            self.implicit_filter = nn.Sequential(
                nn.Linear(emb_dim, order),
                act,
            )
            for i in range(num_inner_mlps):
                self.implicit_filter.append(nn.Linear(order, order))
                self.implicit_filter.append(act)
            self.implicit_filter.append(nn.Linear(order, d_model, bias=False))
        else:
            self.implicit_filter = nn.Sequential(
                nn.Linear(emb_dim, d_model, bias=False),
            )

        if self.bidirectional:
            self.implicit_filter_rev = nn.Sequential(
                nn.Linear(emb_dim, order),
                act,
            )
            for i in range(num_inner_mlps):
                self.implicit_filter_rev.append(nn.Linear(order, order))
                self.implicit_filter_rev.append(act)
            self.implicit_filter_rev.append(nn.Linear(order, d_model, bias=False))

        self.modulation = ExponentialModulation(d_model, **kwargs)

        self.normalized = normalized
        for c in self.implicit_filter.children():
            for name, v in c.state_dict().items():
                optim = {"weight_decay": wd, "lr": lr}
                setattr(getattr(c, name), "_optim", optim)

        self.separable = True
        self.chang_initialize = chang_initialize
        self.register_buffer("initialized", torch.zeros(1).bool(), persistent=False)

    def filter(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        self.chang_initialization(z)
        h = self.implicit_filter(z)
        if self.modulate:
            h = self.modulation(t, h)
        if self.normalized:
            h = h / torch.norm(h, dim=-1, p=1, keepdim=True)
        return h

    def filter_rev(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter_rev(z)
        if self.modulate:
            h = self.modulation(t, h)
        if self.normalized:
            h = h / torch.norm(h, dim=-1, p=1, keepdim=True)
        return h

    def chang_initialization(self, x):
        if not self.initialized[0] and self.chang_initialize:
            # Initialization - Initialize the last layer of self.Kernel as in Chang et al. (2020)
            with torch.no_grad():
                kernel_size = x.shape[1]
                if kernel_size != self.seq_len:
                    warnings.warn("Kernel size does not match self.seq_len. Consider adjusting it.")

                if self.separable:
                    normalization_factor = kernel_size
                else:
                    normalization_factor = self.in_channels * kernel_size
                assert isinstance(self.implicit_filter[-1], nn.Linear), "out layer of conv kernel is not a Linear layer"
                nn.init.normal_(self.implicit_filter[-1].weight, std=math.sqrt(1.0 / normalization_factor))

                if 'all' in self.chang_initialize.lower():
                    for layer in self.implicit_filter[:-1]:
                        if isinstance(layer, nn.Linear):
                            nn.init.kaiming_uniform_(layer.weight)

                # self.implicit_filter[-1].weight.data *= math.sqrt(1.0 / normalization_factor)
            # Set the initialization flag to true
            self.initialized[0] = True

    def forward(self, x, L, k_fwd=None, k_rev=None, bias=None):
        if k_fwd is None:
            k_fwd = self.filter(L)
            if self.bidirectional and k_rev is None:
                k_rev = self.filter_rev(L)

        # Ensure compatibility with filters that return a tuple
        k_fwd = k_fwd[0] if type(k_fwd) is tuple else k_fwd
        if bias is None:
            bias = self.bias
        bias = bias if self.use_bias else 0 * bias

        if self.bidirectional:
            k_rev = k_rev[0] if type(k_rev) is tuple else k_rev
            k = F.pad(k_fwd, (0, L)) \
                + F.pad(k_rev.flip(-1), (L, 0))
        else:
            k = k_fwd

        y = self._fft_conv(
            x,
            k,
            bias,
        )

        return y.to(dtype=x.dtype)

    def _fft_conv(self, x, k, D, k_rev=None):

        seqlen = x.shape[-1]
        fft_size = (2 if self.bidirectional else 1) * seqlen
        k_f = transform(k, mode=self.dxt, fft_size=fft_size)

        if k_rev is not None:
            k_rev_f = transform(k_rev, mode=self.dxt, fft_size=fft_size)
            k_f = k_f + k_rev_f.conj()
        u_f = transform(x.to(dtype=k.dtype), mode=self.dxt, fft_size=fft_size)

        if (len(x.shape) > 3) and x.dim() != k_f.dim(): k_f = k_f.unsqueeze(1)

        y = transform(u_f * k_f,
                      inverse=True,
                      mode=self.dxt,
                      fft_size=fft_size)[..., :seqlen]

        out = y + x * D
        return out.to(dtype=x.dtype)


class GlobalConvolution(OptimModule):
    def __init__(
            self,
            d_model,
            seq_len=1024,
            bias=True,
            alpha_adapt=1.,
            dxt='fft',
    ):
        """

        """
        super().__init__()
        self.d_model = d_model
        self.use_bias = bias
        self.dxt = (dxt, None) if isinstance(dxt, str) else dxt

        self.seq_len = seq_len
        self.alpha_adapt = alpha_adapt

        self.bias = None

    def forward(self, x, L, k=None, h_adapt=None, bias=None, *args, **kwargs):
        """
        Apply Global convolution to the input x.

        Args:
            x: input tensor of shape (batch_size, seq_len, d_model)
            L: length of the input sequence,
            k: static convolution kernel
            h_adapt: dictionary containing the adaptive kernel parameters
            bias: bias tensor of shape (batch_size, seq_len, d_model)

        """
        # Ensure compatibility with filters that return a tuple
        k = k[0] if type(k) is tuple else k
        if bias is None: bias = self.bias
        bias = bias if self.use_bias else 0 * bias

        y = self._fft_conv(
            x,
            k_fix=k, h_adapt=h_adapt, D=bias,
            dropout_mask=None, gelu=False,
        )

        return y

    def _fft_conv(self, x, k_fix, h_adapt, D, dropout_mask, gelu=True, k_rev=None):

        seqlen = x.shape[-1]
        k = k_fix
        if h_adapt['td'] == 'time':
            if not isinstance(h_adapt['h'], tuple):
                k_adapt = h_adapt['h']
            else:
                raise ValueError("")
            k = k + self.alpha_adapt * k_adapt

        k_f = transform(k,
                        mode=self.dxt,
                        fft_size= seqlen)

        if h_adapt['td'] == 'freq':
            if not isinstance(h_adapt['h'], tuple):
                k_f_adapt = h_adapt['h']
            else:
                k_f_adapt = h_adapt['h'][0].conj() * h_adapt['h'][1]
            k_f = k_f + self.alpha_adapt * k_f_adapt

        if k_rev is not None:
            k_rev_f = transform(k_rev,
                                mode=self.dxt,
                                fft_size=seqlen)
            k_f = k_f + k_rev_f.conj()
        u_f = transform(x.to(dtype=k_fix.dtype),
                        mode=self.dxt,
                        fft_size=seqlen)

        if (len(x.shape) > 3) and x.dim() != k_f.dim(): k_f = k_f.unsqueeze(1)

        y = transform(u_f * k_f,
                      inverse=True,
                      mode=self.dxt,
                      fft_size=seqlen)[..., :seqlen]

        out = y + x * D
        if gelu:
            out = F.gelu(out)
        if dropout_mask is not None:
            return (out * rearrange(dropout_mask, 'b H -> b H 1')).to(dtype=x.dtype)
        else:
            return out.to(dtype=x.dtype)


class CustomNorm(nn.Module):
    def __init__(self, norm_type, num_features):
        super().__init__()
        self.norm_type = norm_type
        if self.norm_type == "layernorm":
            self.norm_conv_kernel = nn.LayerNorm(normalized_shape=num_features)
        elif self.norm_type == "batchnorm":
            self.norm_conv_kernel = nn.BatchNorm1d(num_features=num_features)
        else:
            self.norm_conv_kernel = nn.Identity()

    def forward(self, u_f):
        if self.norm_type == "none":
            return u_f

        if self.norm_type == "batchnorm":
            u_f = rearrange(u_f, 'b l d -> b d l')
        u_f = self.norm_conv_kernel(u_f)
        if self.norm_type == "batchnorm":
            u_f = rearrange(u_f, 'b d l -> b l d')

        return u_f

class CustomConv(OptimModule):
    def __init__(
            self,
            d_in,
            d_hid,
            conv_filter_size,
            n_layer=1,
            bias=True,
            norm_type='layernorm',
            nn_type='convt',
            in_factor=1,
            out_factor=1,
            num_conv_head=1,
            is_causal=False,
            cfg=None,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hid = d_hid
        self.conv_filter_size = conv_filter_size
        self.n_layer = n_layer
        self.bias = bias
        self.nn_type = nn_type
        self.mlp = 'mlp' in nn_type
        self.norm_type = norm_type
        self.cfg = cfg
        self.in_factor = in_factor
        self.out_factor = out_factor
        self.num_conv_head = num_conv_head
        if is_causal:
            self._padding = self.conv_filter_size - 1
        else:
            self._padding = (self.conv_filter_size - 1) // 2

        self.padding_mode = cfg.get('padding_mode', 'zeros')

        if self.mlp:
            self._create_mlp_conv_model()
        else:
            self._create_conv_model()

    def _create_conv_model(self):
        _conv1d_ls = []
        _norm_ls = []
        total_width = self.d_hid * self.out_factor * self.num_conv_head
        for l in range(self.n_layer):
            if l == 0:
                _conv1d_ls.append(nn.Conv1d(
                    in_channels=self.d_in * self.in_factor,
                    out_channels=total_width,
                    kernel_size=self.conv_filter_size,
                    groups=self.d_hid * self.num_conv_head,
                    padding=self._padding,
                    padding_mode=self.padding_mode,
                    bias=self.bias
                ))

            else:
                _conv1d_ls.append(nn.Conv1d(
                    in_channels=total_width,
                    out_channels=total_width,
                    kernel_size=self.conv_filter_size,
                    groups=self.d_hid * self.num_conv_head,
                    padding=self._padding,
                    padding_mode=self.padding_mode,
                    bias=self.bias
                ))

            _norm_ls.append(CustomNorm(norm_type=self.norm_type,
                                       num_features=total_width))

        self.conv1d_ls = nn.ModuleList(_conv1d_ls)
        self.norm_ls = nn.ModuleList(_norm_ls)

        self.activation = Activation(self.cfg.activation)

        for c in self.conv1d_ls:
            for name, v in c.state_dict().items():
                optim = {"weight_decay": self.cfg.wd, "lr": self.cfg.lr}
                setattr(getattr(c, name), "_optim", optim)

    def _create_mlp_conv_model(self):
        _filter_depthwise_ls = []
        _conv1d_ls = []
        _norm_ls = []
        total_width = self.d_hid * self.out_factor * self.num_conv_head
        for l in range(self.n_layer):
            if l == 0:
                _filter_frq_depthwise = nn.Linear(self.d_in * self.in_factor,
                                                  total_width,
                                                  bias=self.bias)
            else:
                _filter_frq_depthwise = nn.Linear(total_width, total_width, bias=self.bias)
            _filter_depthwise_ls.append(_filter_frq_depthwise)

            _conv1d_ls.append(nn.Conv1d(
                in_channels=total_width,
                out_channels=total_width,
                kernel_size=self.conv_filter_size,
                groups=total_width,
                padding=self._padding,
                padding_mode=self.padding_mode,
                bias=self.bias
            ))
            _norm_ls.append(CustomNorm(norm_type=self.norm_type,
                                       num_features=total_width))

        self.filter_depthwise_ls = nn.ModuleList(_filter_depthwise_ls)
        self.conv1d_ls = nn.ModuleList(_conv1d_ls)
        self.norm_ls = nn.ModuleList(_norm_ls)

        self.activation = Activation(self.cfg.activation)

        for c in self.filter_depthwise_ls + self.conv1d_ls:
            for name, v in c.state_dict().items():
                optim = {"weight_decay": self.cfg.wd, "lr": self.cfg.lr}
                setattr(getattr(c, name), "_optim", optim)

    def forward(self, x):
        if self.mlp:
            return self._forward_mlp_conv_model(x)
        else:
            return self._forward_conv_model(x)

    def _forward_conv_model(self, x):
        _dim = -2
        seq_len = x.size(_dim)
        is_complex = torch.is_complex(x)
        if is_complex:
            dim = -1
            (x_Re, x_Im) = x.real(), x.imag()
            raise NotImplementedError
            # y = interleave cat
        else:
            y = x

        for l in range(self.n_layer):
            y = rearrange(y, 'b l d -> b d l')
            y = self.conv1d_ls[l](y)[..., :seq_len]
            y = rearrange(y, 'b d l -> b l d')
            if l < self.n_layer - 1:
                y = self.activation(y)
            y = self.norm_ls[l](y)

        if is_complex:
            dim = -1
            # (y, y_Im) = torch.split(u_f, int(u_f.shape[dim]/2), dim=dim)
            (y_Re, y_Im) = torch.tensor_split(y, 2, dim=dim)
            y = y_Re + 1j * y_Im

        return y

    def _forward_mlp_conv_model(self, x):
        _dim = -2
        seq_len = x.size(_dim)
        is_complex = torch.is_complex(x)
        if is_complex:
            dim = -1
            (x_Re, x_Im) = x.real(), x.imag()
            raise NotImplementedError
            # y = interleave cat
        else:
            y = x
        for l in range(self.n_layer):
            y = self.filter_depthwise_ls[l](y)
            y = rearrange(y, 'b l d -> b d l')
            y = self.conv1d_ls[l](y)[..., :seq_len]
            y = rearrange(y, 'b d l -> b l d')
            if l < self.n_layer - 1:
                y = self.activation(y)
            y = self.norm_ls[l](y)

        if is_complex:
            dim = -1
            # (y, y_Im) = torch.split(u_f, int(u_f.shape[dim]/2), dim=dim)
            (y_Re, y_Im) = torch.tensor_split(y, 2, dim=dim)
            y = y_Re + 1j * y_Im

        return y


class ConvKernel_Time_Frequency(nn.Module):
    def __init__(
            self,
            cfg,
            d_model,
            num_conv_filter,
            l_max,
            inner_factor=1,
            num_heads=1,
            causal=True,
            dxt='fft',
            **kwargs
    ):
        """
        The conditioning network makes the convolution kernel.
        It operates in both time and frequency domain as explained in the paper
        """
        super().__init__()
        self.cfg = cfg
        self.inner_factor = inner_factor
        # self.n_layer = cfg.n_layer
        self.num_conv_filter = num_conv_filter
        self.d_model = d_model
        self.l_max = l_max
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_heads
        self.type_conv_kernel = self.cfg.type
        self.dxt = (dxt, None) if isinstance(dxt, str) else dxt
        self.nn_type = self.cfg.get('nn', 'convtf')  # ['mlp-conv1d', 'convt', 'conv2t']

        self.is_causal = causal
        self.twosided = self.is_causal

        if self.type_conv_kernel in ['freq', 'absfreq']:
            self._create_conv_kernel_freq_model()
        elif self.type_conv_kernel in ['corrTime', 'corrTime2', 'corrFreq']:
            self._create_kq_corr_model()

    def _create_conv_kernel_freq_model(self):
        if self.nn_type in ['convt', 'conv2t', 'convtf', 'conv2tf']:
            _in_factor = _out_factor = 1
        elif self.nn_type in ['convf', 'conv2f']:
            if 'dct' in self.dxt[0]:
                _in_factor = _out_factor = 1
            else:
                _in_factor = 1 if self.type_conv_kernel == 'absfreq' else 2
                _out_factor = 2

        self.nn_model = CustomConv(
            d_in=self.d_model,
            d_hid=self.d_model,
            conv_filter_size=self.cfg.conv_filter_size,
            n_layer=self.cfg.n_layer,
            bias=self.cfg.bias,
            norm_type=self.cfg.norm,
            nn_type=self.nn_type,
            in_factor=_in_factor,
            out_factor=_out_factor,
            cfg=self.cfg,
            num_conv_head=self.num_conv_filter,
        )

        if self.nn_type in ['convtf', 'conv2tf']:
            if 'dct' in self.dxt[0]:
                _in_factor = _out_factor = 1
            else:
                _in_factor = 1 if self.type_conv_kernel == 'absfreq' else 2
                _out_factor = 2
            self.nn_model2 = CustomConv(
                d_in=self.d_model,
                d_hid=self.d_model,
                conv_filter_size=self.cfg.conv_filter_size,
                n_layer=self.cfg.n_layer,
                bias=self.cfg.bias,
                norm_type=self.cfg.norm,
                nn_type=self.nn_type,
                in_factor=_in_factor,
                out_factor=_out_factor,
                cfg=self.cfg,
                num_conv_head=self.num_conv_filter,
            )

    def _create_kq_corr_model(self):
        _in_factor = _out_factor = 1

        self.nn_model_k = CustomConv(
            d_in=self.d_model,
            d_hid=self.d_model,
            conv_filter_size=self.cfg.conv_filter_size,
            n_layer=self.cfg.n_layer,
            bias=self.cfg.bias,
            norm_type=self.cfg.norm,
            nn_type=self.nn_type,
            in_factor=_in_factor,
            out_factor=_out_factor,
            cfg=self.cfg,
            num_conv_head=self.num_conv_filter,
        )

        self.nn_model_q = CustomConv(
            d_in=self.d_model,
            d_hid=self.d_model,
            conv_filter_size=self.cfg.conv_filter_size,
            n_layer=self.cfg.n_layer,
            bias=self.cfg.bias,
            norm_type=self.cfg.norm,
            nn_type=self.nn_type,
            in_factor=_in_factor,
            out_factor=_out_factor,
            cfg=self.cfg,
            num_conv_head=self.num_conv_filter,
        )

        self.activation_q = Activation(self.cfg.actq.split('_')[0])
        self.activation_q_on_abs = self.cfg.actq.split('_')[-1] == 'abs'

        if self.nn_type in ['convtf', 'conv2tf']:
            if 'dct' in self.dxt[0]:
                _in_factor = _out_factor = 1
            else:
                _in_factor = _out_factor = 2

            self.nn_model_frq = CustomConv(
                d_in=self.d_model,
                d_hid=self.d_model,
                conv_filter_size=self.cfg.conv_filter_size,
                n_layer=self.cfg.n_layer,
                bias=self.cfg.bias,
                norm_type=self.cfg.norm,
                nn_type=self.nn_type,
                in_factor=_in_factor,
                out_factor=_out_factor,
                cfg=self.cfg,
                num_conv_head=self.num_conv_filter,
            )

    def forward(self, u):
        if self.type_conv_kernel in ['freq', 'absfreq']:
            return self._get_conv_kernel_freq(u)
        elif self.type_conv_kernel in ['corrTime']:
            return self._get_kq_corr(u)
        else:
            raise ValueError('invalid type_conv_kernel')

    def _get_conv_kernel_freq(self, u):
        _dim = -2
        l = seqlen = u.size(_dim)

        if self.nn_type in ['convt', 'conv2t', 'convtf', 'conv2tf']:
            u = self.nn_model(u)

        u = transform(u,
                      mode=self.dxt,
                      fft_size=(2 if self.twosided else 1) * seqlen,
                      dim=_dim)

        if self.type_conv_kernel == 'absfreq':
            u = u.real ** 2. + u.imag ** 2. if 'fft' in self.dxt[0] else \
                u.abs()
        elif self.type_conv_kernel == 'freq':
            u = torch.cat([u.real, u.imag], dim=-1) if 'fft' in self.dxt[0] else \
                u

        if self.nn_type in ['convf', 'conv2f']:
            u = self.nn_model(u)
        elif self.nn_type in ['convtf', 'conv2tf']:
            u = self.nn_model2(u)

        if self.nn_type in ['convf', 'conv2f', 'convtf', 'conv2tf'] and ('fft' in self.dxt[0]):
            u = torch.complex(u[..., 0::2], u[..., 1::2])

        h_f = rearrange(u,
                        # 'b (ho v) (z l) -> b ho v z l', z=self.num_blocks,
                        'b (l) (ho v) -> b ho v l',
                        ho=self.num_heads,
                        v=self.head_dim * self.num_conv_filter
                        )
        if not self.is_causal:
            h_f = rearrange(h_f, 'b ho (v o) l -> o b ho v l', ho=self.num_heads, o=self.num_conv_filter)
            if self.num_heads == 1:
                h_f = h_f.squeeze(2)
            h_adapt_ls = []
            for o in range(self.num_conv_filter):
                h_adapt_ = {'h': h_f[o], 'td': 'freq'}
                h_adapt_ls.append(h_adapt_)

        else:
            h = transform(h_f,
                          mode=self.dxt,
                          inverse=True,
                          fft_size=2 * seqlen)
            h = h[..., : seqlen]
            # h_ls = h.split(self.d_model, dim=2)
            h = rearrange(h, 'b ho (v o) l -> o b ho v l', ho=self.num_heads, o=self.num_conv_filter)

            h_adapt_ls = []
            for o in range(self.num_conv_filter):
                h_adapt_ = {'h': h[o], 'td': 'time'}
                h_adapt_ls.append(h_adapt_)

        return h_adapt_ls

    def _get_kq_corr(self, u):
        _dim = -2
        l = seqlen = u.size(_dim)

        k = self.nn_model_k(u)
        q = self.nn_model_q(u)

        k = rearrange(k, 'b (l) (ho v o)-> o b ho v l',
                      ho=self.num_heads,
                      v=self.head_dim,
                      o=self.num_conv_filter)
        if self.num_heads == 1: k = k.squeeze(2)
        k = transform(k,
                      mode=self.dxt,
                      fft_size=(2 if self.twosided else 1) * seqlen)

        if self.cfg.actq in ['softmax', 'smax']:
            q = self.activation_q(q)
        q = rearrange(q, 'b (l) (ho v o)-> o b ho v l',
                      ho=self.num_heads,
                      v=self.head_dim,
                      o=self.num_conv_filter)
        if self.num_heads == 1: q = q.squeeze(2)
        q = transform(q,
                      mode=self.dxt,
                      fft_size=(2 if self.twosided else 1) * seqlen)
        if not self.cfg.actq in ['softmax', 'smax']:
            if 'fft' in self.dxt[0]:
                if self.activation_q_on_abs:
                    q = get_activation_on_abs(q, self.activation_q)
                else:
                    q = self.activation_q(q.real) + 1j * self.activation_q(q.imag)
            else:
                if self.activation_q_on_abs:
                    q = get_activation_on_abs(q, self.activation_q)
                else:
                    q = self.activation_q(q)

        if self.nn_type in ['convtf', 'conv2tf']:
            h_f = k.conj() * q
            if self.num_heads > 1: raise NotImplementedError("here num_heads should be 1")
            h_f = rearrange(h_f, 'o b v l -> b (l) (v o)',
                            v=self.head_dim,
                            o=self.num_conv_filter)

            if 'fft' in self.dxt[0]:
                h_f = torch.cat([h_f.real, h_f.imag], dim=-1)

            h_f = self.nn_model_frq(h_f)

            if 'fft' in self.dxt[0]:
                dim_ = h_f.shape[-1] // 2
                h_f = torch.complex(h_f[..., 0:dim_], h_f[..., -dim_:])

            h_f = rearrange(h_f, 'b (l) (v o)-> o b v l',
                            v=self.head_dim,
                            o=self.num_conv_filter)

            h_adapt_ls = []
            for o in range(self.num_conv_filter):
                h_adapt_ = {'h': h_f[o], 'td': 'freq'}
                h_adapt_ls.append(h_adapt_)
            return h_adapt_ls

        h_adapt_ls = []
        for o in range(self.num_conv_filter):
            h_adapt_ = {'h': (k[o], q[o]), 'td': 'freq'}
            h_adapt_ls.append(h_adapt_)
        return h_adapt_ls


def get_activation_on_abs(x, activation_fn):
    x_abs = x.abs()
    y = activation_fn(x_abs) * x / (torch.clamp(x_abs, min=1e-6))
    return y