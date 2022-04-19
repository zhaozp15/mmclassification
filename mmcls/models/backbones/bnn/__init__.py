from .mfnet import MFNet
from .multifea import MultiFea
from .xnornet import XnorNet

bnn_networks = [
    'MFNet',
    'MultiFea',
    'XnorNet',
]

__all__ = bnn_networks + ['bnn_networks']