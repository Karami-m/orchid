
"""
nn components, in particular activation functions.
Some of the blocks are (modified) from
the code of Hyena: https://github.com/HazyResearch/safari/tree/main
"""

from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from opt_einsum import contract


class OptimModule(nn.Module):
    """ Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters """

    def register(self, name, tensor, lr=None, wd=0.0):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {}
            if lr is not None: optim["lr"] = lr
            if wd is not None: optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)

class TransposedLN(nn.Module):
    """ LayerNorm module over second dimension
    Assumes shape (B, D, L), where L can be 1 or more axis

    This is slow and a dedicated CUDA/Triton implementation shuld provide substantial end-to-end speedup
    """
    def __init__(self, d, scalar=True):
        super().__init__()
        self.scalar = scalar
        if self.scalar:
            self.m = nn.Parameter(torch.zeros(1))
            self.s = nn.Parameter(torch.ones(1))
            setattr(self.m, "_optim", {"weight_decay": 0.0})
            setattr(self.s, "_optim", {"weight_decay": 0.0})
        else:
            self.ln = nn.LayerNorm(d)

    def forward(self, x):
        if self.scalar:
            # calc. stats over D dim / channels
            s, m = torch.std_mean(x, dim=1, unbiased=False, keepdim=True)
            y = (self.s/s) * (x-m+self.m)
        else:
            # move channel to last axis, apply layer_norm, then move channel back to second axis
            _x = self.ln(rearrange(x, 'b d ... -> b ... d'))
            y = rearrange(_x, 'b ... d -> b d ...')
        return y
def Activation(activation=None, size=None, dim=-1):
    if activation in [ None, 'id', 'identity', 'linear' ]:
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU()
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'softplus':
        return nn.Softplus()
    elif activation in ['softmax', 'smax']:
        return nn.Softmax(dim=1)
    elif activation == 'ln':
        return TransposedLN(dim)
    elif 'sigmoid+' in activation:
        shift = float(activation.split('+')[-1])
        return lambda x: nn.Sigmoid()(x - shift)
    elif activation == 'rational':
        return lambda x: x/(x + 1)
    elif 'softshrink+' in activation:
        _lambda = float(activation.split('+')[-1])
        return nn.Softshrink(_lambda)
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))
