from .mfnet import MFNet
from .multifea import MultiFea
from .resnet18 import ResNet18
from .xnornet import XnorNet

bnn_networks = [
    'MFNet',
    'MultiFea',
    'ResNet18',
    'XnorNet',
]

__all__ = bnn_networks + ['bnn_networks']