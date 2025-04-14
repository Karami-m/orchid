"""
Implementations of several types of Discrete Cosine Transforms with various reductions to FFT.

"""

import torch
import torch.nn as nn
import numpy as np
import scipy.fft

class DCTX(nn.Module):
    def __init__(self, type, N, norm='backward'):
        super().__init__()

        self.N = N
        self.norm = norm

        self.mode = {
            'dctx' :1,
            'dctx1':1,
            'dctx2':2,
            'dctx4':4,
            'dctxd':0,
        }[type]

        # self.dct = DCT(N, norm=norm)
        # self.idct = IDCT(N, norm=norm)
        self.add_module('dct', DCT(N, norm=norm, mode=self.mode))
        self.add_module('idct', IDCT(N, norm=norm, mode=self.mode))

        self._name = type

    def forward(self, x, inverse=False, dim=-1):

        if dim != -1:
            x = x.transpose(-1, dim)

        if not inverse:
            y = self.dct(x, mode=self.mode)
        else:
            y = self.idct(x, mode=self.mode)

        if dim != -1:
            y = y.transpose(-1, dim)
        return y

    # def __eq__(self, other):
    #     return self._name == other


class DCT(nn.Module):
    """ Reductions adapted from https://dsp.stackexchange.com/questions/2807/fast-cosine-transform-via-fft """

    def __init__(self, N, norm='backward', mode=0):
        super().__init__()

        self.N = N
        self.mode = mode
        self.norm = norm

        if self.mode == 0:
            # Materialize DCT matrix
            P = scipy.fft.dct(np.eye(N), norm=norm, type=2).T
            Pr = torch.tensor(np.real(P), dtype=torch.float)
            Pi = torch.tensor(np.imag(P), dtype=torch.float)
            self.register_buffer('Pr', Pr) # half shift
            self.register_buffer('Pi', Pi) # half shift

        if self.mode in [1, 2]:
            # TODO take care of normalization
            Q = np.exp(-1j * np.pi / (2 * self.N) * np.arange(self.N))
            Qr = torch.tensor(np.real(Q), dtype=torch.float)
            Qi = torch.tensor(np.imag(Q), dtype=torch.float)
            self.register_buffer('Qr', Qr) # half shift
            self.register_buffer('Qi', Qi) # half shift
            if self.norm == "ortho":
                scale = np.ones(self.N)
                scale[0] /= np.sqrt(N) * 2
                scale[1:] /= np.sqrt(N / 2) * 2
                scale = torch.tensor(scale, dtype=torch.float)
                self.register_buffer('scale', scale)  # half shift

    def forward(self, x, mode=2):
        if mode == 0:
            return self.forward_dense(x)
        elif mode == 1:
            return self.forward_n(x)
        elif mode == 2:
            return self.forward_2n(x)
        elif mode == 4:
            return self.forward_4n(x)

    def forward_dense(self, x):
        """ Baseline DCT type II - matmul by DCT matrix """
        y = (torch.complex(self.Pr, self.Pi).to(x) @ x.unsqueeze(-1)).squeeze(-1)
        return y

    def forward_4n(self, x):
        """ DCT type II - reduction to FFT size 4N """
        assert self.N == x.shape[-1]
        x = torch.cat([x, x.flip(-1)], dim=-1)
        z = torch.zeros_like(x)
        x = torch.stack([z, x], dim=-1)
        x = x.view(x.shape[:-2] + (-1,))
        y = torch.fft.fft(x)
        y = y[..., :self.N]
        if torch.is_complex(x):
            return y
        else:
            return torch.real(y)

    def forward_2n(self, x):
        """ DCT type II - reduction to FFT size 2N mirrored

        The reduction from the DSP forum is not quite correct in the complex input case.
        halfshift(FFT[a, b, c, d, d, c, b, a]) -> [A, B, C, D, 0, -D, -C, -B]
        In the case of real input, the intermediate step after FFT has form [A, B, C, D, 0, D*, C*, B*]
        """
        assert self.N == x.shape[-1]
        x = torch.cat([x, x.flip(-1)], dim=-1)
        y = torch.fft.fft(x)[..., :self.N]
        y = y * torch.complex(self.Qr, self.Qi)
        if self.norm == "ortho":
            y = y*self.scale
        if torch.is_complex(x):
            return y
        else:
            return torch.real(y)

    def forward_n(self, x):
        """ DCT type II - reduction to size N """
        assert self.N == x.shape[-1]
        x = torch.cat([x[..., 0::2], x[..., 1::2].flip(-1)], dim=-1)
        y = torch.fft.fft(x)
        y = y * 2 * torch.complex(self.Qr, self.Qi)
        if torch.is_complex(x):
            y = torch.cat([y[..., :1], (y[..., 1:] + 1j * y[..., 1:].flip(-1)) / 2], dim=-1) # TODO in-place sum
        else:
            y = torch.real(y)
        return y

class IDCT(nn.Module):
    def __init__(self, N, norm='backward', mode=0):
        super().__init__()

        self.N = N
        self.mode = mode
        self.norm = norm

        if self.mode == 0:
            # Materialize DCT matrix
            P = np.linalg.inv(scipy.fft.dct(np.eye(N), norm=norm, type=2).T)
            Pr = torch.tensor(np.real(P), dtype=torch.float)
            Pi = torch.tensor(np.imag(P), dtype=torch.float)
            self.register_buffer('Pr', Pr) # half shift
            self.register_buffer('Pi', Pi) # half shift

        if self.mode in [1, 2]:
            # TODO take care of normalization
            Q = np.exp(-1j * np.pi / (2 * self.N) * np.arange(2*self.N))
            Qr = torch.tensor(np.real(Q), dtype=torch.float)
            Qi = torch.tensor(np.imag(Q), dtype=torch.float)
            self.register_buffer('Qr', Qr) # half shift
            self.register_buffer('Qi', Qi) # half shift
            if self.norm == "ortho":
                scale = np.ones(self.N)
                scale[0] *= np.sqrt(N) * 2
                scale[1:] *= np.sqrt(N / 2) * 2
                scale = torch.tensor(scale, dtype=torch.float)
                self.register_buffer('scale', scale)  # half shift

    def forward(self, x, mode=2):
        if mode == 0:
            return self.forward_dense(x)
        elif mode == 1:
            return self.forward_n(x)
        elif mode == 2:
            return self.forward_2n(x)
        elif mode == 4:
            return self.forward_4n(x)

    def forward_dense(self, x):
        """ Baseline DCT type II - matmul by DCT matrix """
        y = (torch.complex(self.Pr, self.Pi).to(x) @ x.unsqueeze(-1)).squeeze(-1)
        return y

    def forward_4n(self, x):
        """ DCT type II - reduction to FFT size 4N """
        assert self.N == x.shape[-1]
        z = x.new_zeros(x.shape[:-1] + (1,))
        x = torch.cat([x, z, -x.flip(-1), -x[..., 1:], z, x[..., 1:].flip(-1)], dim=-1)
        y = torch.fft.ifft(x)
        y = y[..., 1:2*self.N:2]
        if torch.is_complex(x):
            return y
        else:
            return torch.real(y)

    def forward_2n(self, x):
        """ DCT type II - reduction to FFT size 2N mirrored """
        assert self.N == x.shape[-1]
        if self.norm == "ortho":
            x = x*self.scale

        z = x.new_zeros(x.shape[:-1] + (1,))
        x = torch.cat([x, z, -x[..., 1:].flip(-1)], dim=-1)
        x = x / torch.complex(self.Qr, self.Qi)
        y = torch.fft.ifft(x)[..., :self.N]
        if torch.is_complex(x):
            return y
        else:
            return torch.real(y)

    def forward_n(self, x):
        """ DCT type II - reduction to size N """
        assert self.N == x.shape[-1]
        raise NotImplementedError # Straightforward by inverting operations of DCT-II reduction

def test_dct_ii():
    N = 8
    dct = DCT(N)

    baseline = dct.forward_dense
    methods = [dct.forward_4n, dct.forward_2n, dct.forward_n]

    # Real case
    print("DCT-II Real input")
    x = torch.randn(1, N)
    y = baseline(x)
    print(y)
    for fn in methods:
        y_ = fn(x)
        print("err", torch.norm(y-y_))

    # Complex case
    print("DCT-II Complex input")
    x = torch.randn(N) + 1j * torch.randn(N)
    y = baseline(x)
    print(y)
    for fn in methods:
        y_ = fn(x)
        print("err", torch.norm(y-y_))

def test_dct_iii():
    N = 8
    dct = IDCT(N)

    baseline = dct.forward_dense
    methods = [dct.forward_4n, dct.forward_2n]

    # Real case
    print("DCT-III Real input")
    x = torch.randn(1, N)
    y = baseline(x)
    print(y)
    for fn in methods:
        y_ = fn(x)
        print("err", torch.norm(y-y_))

    # Complex case
    print("DCT-III Complex input")
    # x = torch.randn(N) + 1j * torch.randn(N)
    x = 1j * torch.ones(N)
    y = baseline(x)
    print(y)
    for fn in methods:
        y_ = fn(x)
        print("err", torch.norm(y-y_))
