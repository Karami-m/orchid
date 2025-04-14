
import torch
from easydict import EasyDict as edict

from orchid_standalone.orchid_standalone import OrchidOperator
from orchid_standalone.utils.dxt import DCTX


seq_len = 128
batch_size = 2
d_model = 8
lr = 1e-3
weight_decay = 0.05
# lr = None
# weight_decay = None

dxt= 'dctxd-o'

if 'dctx' in dxt:
    # global dctx
    dctx = DCTX(type=dxt.split('-')[0],
                N=seq_len,
                norm='ortho' if '-o' in dxt else 'backward')
    # self.add_module('dctx', dctx)  # self.dctx = dctx
    dxt = (dxt, dctx)

layer = OrchidOperator(
    d_model = d_model,
    l_max = seq_len,
    filter_order= 128,      # width of the implicit MLP
    residual_long_conv='sconv', # {None, sconv}
    order=2,
    short_filter_order= 5,
    emb_dim= 33,           # for cifar it was 3, for larger imagenet it was 33
    num_inner_mlps= 1,     # for cifar it was 2, for larger imagenet it was 33
    w= 10,                 # frequency of periodic activations
    wd= 0,                 # weight decay of kernel parameters
    lr = lr,               # learning rate of kernel parameters
    lr_pos_emb=1e-5,
    dxt= dxt,
    conv_kernel=edict({
        'type': 'absfreq',        # [freq, absfreq, corrTime]
        'nn': 'convtf',           # ['convtf', 'convf', 'conv2f', 'convt', 'conv2t']
        'conv_filter_size': 3,
        'padding_mode': 'zeros',  # [zeros, reflect]
        'norm': 'layernorm',      # [none, layernorm, batchnorm]
        'n_layer': 1,
        'activation': 'gelu',     # intermediate activation if n_layer >1
        'actq': 'id',             # used in cross- crrelation only (corrTime) [id, gelu, sigmoid, tanh, softmax]
        'wd': weight_decay,
        'lr': lr,
        'bias': True
    })
)

x = torch.randn(batch_size, seq_len, d_model)
y = layer(x)
print(y.shape)
print(y)