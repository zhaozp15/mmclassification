import torch
import torch.nn as nn
import torch.nn.functional as F
from ..bricks.binary_convs import BLConv2d

class InputScale(nn.Module):
    def __init__(self,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super(InputScale, self).__init__()
        self.ks = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(
            torch.full(
                (1, 1, self.ks, self.ks),
                1 / (self.ks * self.ks)),
            requires_grad=False)

    def forward(self, x):
        A = x.mean(dim=1, keepdim=True)
        out = F.conv2d(
            A,
            self.weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=1,
            groups=1)
        return out

class XnorBlock(nn.Module):
    '''XNORNet的block

    Args:
        scale_x_mode (str): 输入缩放因子的使用方法
            'none': 不使用输入缩放因子
            'k': 使用论文中计算的输入缩放因子矩阵
            'k_detach': 使用输入缩放矩阵，且该矩阵不参与反传
        scale_x_mode (str): 权重缩放因子的使用方法
            'none': 不使用权重缩放因子
            'mean': 使用绝对值均值计算权重缩放因子
            'mean_detach': 绝对值均值计算，且缩放因子不参与反传
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 downsample=None,
                 scale_x_mode='none',
                 scale_w_mode='mean',
                 **kwargs):
        super(XnorBlock, self).__init__()
        self.scale_x_mode = scale_x_mode
        self.scale_x_detach = 'detach' in scale_x_mode
        if scale_x_mode != 'none':
            self.scale1 = InputScale(
                kernel_size=3,
                stride=stride,
                padding=1)
            self.scale2 = InputScale(
                kernel_size=3,
                stride=1,
                padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = BLConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            scale_w_mode=scale_w_mode,
            **kwargs)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = BLConv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            scale_w_mode=scale_w_mode,
            **kwargs)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        if self.scale_x_mode == 'none':
            out = self.conv1(out)
        else:
            K1 = self.scale1(out)
            out = self.conv1(out)
            out = out * self._detach(K1)
        out = self.relu1(out)
        out = self.bn2(out)
        if self.scale_x_mode == 'none':
            out = self.conv1(out)
        else:
            K2 = self.scale2(out)
            out = self.conv2(out)
            out = out * self._detach(K2)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu2(out)

        return out

    def _detach(self, x):
        if self.scale_x_detach:
            breakpoint()
            return x.detach()
        else:
            breakpoint()
            return x
