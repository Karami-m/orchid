
"""
A simple standalone implementation of the Orchid block presented in https://arxiv.org/abs/2402.18508
Some of the blocks are (modified) from
the code of Hyena: https://github.com/HazyResearch/safari/tree/main
"""

import torch
import torch.nn as nn
from einops import rearrange
from easydict import EasyDict as edict


from orchid_standalone.utils.nn import Activation
from orchid_standalone.orchid_utils import ConvKernel_Time_Frequency, ConvKernel_Static, GlobalConvolution

conv_kernel_cfg = edict({
        'type': 'absfreq',        # [freq, absfreq, corrTime]
        'nn': 'convtf',           # ['convtf', 'convf', 'conv2f', 'convt', 'conv2t']
        'conv_filter_size': 3,
        'padding_mode': 'zeros',  # [zeros, reflect]
        'norm': 'layernorm',      # [none, layernorm, batchnorm]
        'n_layer': 1,
        'activation': 'gelu',     # intermediate activation if n_layer >1
        'actq': 'id',             # used in cross- crrelation only (corrTime) [id, gelu, sigmoid, tanh, softmax]
        'wd': None,               # None: use the same weight decay that was used in the optimization
        'lr': None,               # None: use the same lr as the optimization.lr
        'bias': True
    })

class OrchidOperator(nn.Module):
    def __init__(
            self,
            d_model,
            l_max=128,
            filter_order=64,
            filter_dropout=0.0,
            num_heads=1,
            residual_long_conv = 'sconv',
            short_filter_order=3,
            order=2,
            actchain="id",
            chain="*-c-*",
            conv_kernel=conv_kernel_cfg,
            **filter_args,
    ):
        r"""

        Args:
            d_model (int): Dimension of the input and output embeddings (width of the layer)
            l_max: (int): Maximum input sequence length.
            order: (int): Depth of the Hyena recurrence. Defaults to 2
            filter_order: (int): Width of the FFN parametrizing the implicit filter. Defaults to 64
            num_heads: (int): Number of heads.
            dropout: (float): Dropout probability. Defaults to 0.0
            filter_dropout: (float): Dropout probability for the filter. Defaults to 0.0
            short_filter_order: (int): Length of the explicit input convolutional filter. Defaults to 3
            activation: (str): type of act between kernel output and FF (default identity)
            residual_long_conv: (str): type of long residual conv on the residual path (default 'sconv')
            actchain: (str): type of act between kernel output and FF (default 'id')
            conv_kernel: (dict): type of kernel (default 'ImRe': 'cat', 'norm': 'layernorm'')
            **filter_args: (dict): other filter args

        """
        super().__init__()

        self.d_model = d_model
        self.l_max = l_max
        self.channels = 1
        self.residual_long_conv = residual_long_conv
        self.short_filter_order = short_filter_order

        # self.dxt = dxt
        self.chain = chain
        self.order = order
        self.activation_chain = Activation(actchain)
        self.conv_kernel = conv_kernel

        # setup projections
        self.in_linear = nn.Linear(d_model, (self.order + 1) * d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        # setup short conv1d
        total_width = self.d_model * (self.order + 1)
        self.conv1d_short = nn.Conv1d(
            in_channels=total_width,
            out_channels=total_width,
            kernel_size=self.short_filter_order,
            groups=total_width,
            padding=self.short_filter_order - 1,
        )

        if not self.chain in ["*-c-*", "*c*"]:
            raise NotImplementedError

        # setup fixed conv
        self.num_conv_filter = self.order - 1 if self.chain in ["*-c-*", "*c*"] else self.order
        assert self.num_conv_filter >= 1, f'num_conv_filter must be at least 1, (got {self.num_conv_filter})'
        self.fixed_conv = ConvKernel_Static(
            self.d_model * self.num_conv_filter,
            order=filter_order,
            seq_len=self.l_max,
            dropout=filter_dropout,
            **filter_args
        )


        if self.residual_long_conv in ['sconv']:
            _filter_args = filter_args

            self.fixed_conv_res = ConvKernel_Static(
                self.d_model * self.num_conv_filter,
                order=filter_order,
                seq_len=self.l_max,
                dropout=filter_dropout,
                **_filter_args
            )

        # setup conditioning network (adaptive conv kernel)
        self.conv_kernel_nn = ConvKernel_Time_Frequency(
            cfg=self.conv_kernel,
            d_model=self.d_model,
            num_conv_filter=self.num_conv_filter,
            l_max=self.l_max,
            num_heads=num_heads,
            **filter_args
        )

        # setup adaptive conv
        self.adaptive_conv = GlobalConvolution(
            self.d_model * self.num_conv_filter,
            seq_len=self.l_max,
            dxt = filter_args['dxt']
        )

        self.register_buffer("kernel_fixed", torch.zeros(1), persistent=False)
        self.register_buffer("kernel_adapt", torch.zeros(1), persistent=False)

    def forward(self, u, **kwargs):
        """
        Args:
            u: (Tensor): input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: output tensor of shape (batch_size, seq_len, d_model)
        """

        # u is B x L
        L = u.size(-2)

        # in projection
        u_orig = u
        u = self.in_linear(u)

        # adaptive conv kernel: generated by conditioning network
        h_adapt_ls = self.conv_kernel_nn(u[:, :, -self.d_model:])

        u = rearrange(u, "b l d -> b d l")
        # short filter
        uc = self.conv1d_short(u)[..., :L]

        *x, y = uc.split(self.d_model, dim=1)

        k = self.fixed_conv.filter(L, device=u.device)

        # Buffers to compute regularizer_global_conv
        self.kernel_fixed = k
        self.kernel_adapt = h_adapt_ls[0]['h']

        # `c` is always 1 by default
        k = rearrange(k, 'c l (v o) -> c o v l', v=self.d_model, o=self.num_conv_filter)[0]
        bias = rearrange(self.fixed_conv.bias, '(v o) -> o v', v=self.d_model, o=self.num_conv_filter)


        for o, x_i in enumerate(reversed(x[1:])):
            # pointwise multiplication (Hadamard product)
            y = y * x_i

            # Global convolution
            y = self.adaptive_conv(
                y, L,
                k=k[o], h_adapt=h_adapt_ls[o],
                bias=bias[o, :, None])

        # final pointwise multiplication (Hadamard product) of the chain
        y = y * x[0]

        if self.residual_long_conv in ['sconv']:
            k2 = self.fixed_conv_res.filter(L, device=u.device)
            k2 = rearrange(k2, "c l d -> c d l")[0]

            y_r = self.fixed_conv_res(
                u_orig.transpose(-1, -2), L, k_fwd=k2,
                bias=self.fixed_conv_res.bias[None, :, None])
            y = y + y_r

        y = y.transpose(-1, -2)
        y = self.out_linear(y)
        return y

    @property
    def d_output(self):
        return self.d_model