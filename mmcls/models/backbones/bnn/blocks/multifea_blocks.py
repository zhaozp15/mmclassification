import torch
import torch.nn as nn
from ..bricks.acts import build_act
from ..bricks.binary_convs import BLConv2d, TAConv2d
from ..bricks.feature_expand import FeaExpand


class MultiFeaBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 nonlinear=('identity', 'hardtanh'),
                 fexpand_num=1, fexpand_mode='1', fexpand_thres=None,
                 **kwargs):
        super(MultiFeaBlock, self).__init__()
        self.out_channels = out_channels
        self.stride = stride
        self.downsample = downsample
        self.fexpand_num = fexpand_num
        self.fexpand1 = FeaExpand(expansion=fexpand_num, mode=fexpand_mode, in_channels=in_channels, thres=fexpand_thres)
        self.conv1 = BLConv2d(in_channels * fexpand_num, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.nonlinear11 = build_act(nonlinear[0], out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear12 = build_act(nonlinear[1], out_channels)
        self.fexpand2 = FeaExpand(expansion=fexpand_num, mode=fexpand_mode, in_channels=out_channels, thres=fexpand_thres)
        self.conv2 = BLConv2d(out_channels * fexpand_num, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.nonlinear21 = build_act(nonlinear[0], out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear22 = build_act(nonlinear[1], out_channels)

    def forward(self, x):
        identity = x

        out = self.fexpand1(x)
        out = self.conv1(out)
        out = self.nonlinear11(out)
        out = self.bn1(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.nonlinear12(out)

        identity = out
        out = self.fexpand2(out)
        out = self.conv2(out)
        out = self.nonlinear21(out)
        out = self.bn2(out)
        out += identity
        out = self.nonlinear22(out)

        return out


class MultiFeaSimBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 nonlinear=('identity', 'hardtanh'),
                 fexpand_num=2, fexpand_mode='indepedent', fexpand_thres=(-0.6, 0.6),
                 **kwargs):
        super(MultiFeaSimBlock, self).__init__()
        self.out_channels = out_channels
        self.stride = stride
        self.downsample = downsample
        self.fexpand_num = fexpand_num
        self.fexpand1 = FeaExpand(expansion=fexpand_num, mode=fexpand_mode, in_channels=in_channels, thres=fexpand_thres)
        self.conv1 = BLConv2d(in_channels * fexpand_num, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.nonlinear11 = build_act(nonlinear[0], out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear12 = build_act(nonlinear[1], out_channels)
        self.fexpand2 = FeaExpand(expansion=fexpand_num, mode=fexpand_mode, in_channels=out_channels, thres=fexpand_thres)
        self.conv2 = BLConv2d(out_channels * fexpand_num, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.nonlinear21 = build_act(nonlinear[0], out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear22 = build_act(nonlinear[1], out_channels)

    def forward(self, x):
        identity = x

        cos_sim1, out = self.fexpand1(x)
        out = self.conv1(out)
        out = self.nonlinear11(out)
        out = self.bn1(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.nonlinear12(out)

        identity = out
        cos_sim2, out = self.fexpand2(out)
        out = self.conv2(out)
        out = self.nonlinear21(out)
        out = self.bn2(out)
        out += identity
        out = self.nonlinear22(out)

        cos_sim = cos_sim1 + cos_sim2

        return cos_sim, out


class MF1Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1,
                 fexpand_num=2, fexpand_mode='5', fexpand_thres=(-0.55, 0.55),
                 nonlinear=('prelu', 'identity'), shortcut='identity', ahead_fexpand='identity',
                 **kwargs):
        super(MF1Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.fexpand_num = fexpand_num

        if self.stride == 2:
            self.pooling = nn.AvgPool2d(2, 2)
        self.shortcut1 = build_act(shortcut, in_channels)
        self.ahead_fexpand1 = build_act(ahead_fexpand, in_channels)
        self.fexpand1 = FeaExpand(expansion=fexpand_num, mode=fexpand_mode, in_channels=in_channels, thres=fexpand_thres)
        self.conv_3x3 = BLConv2d(in_channels * fexpand_num, in_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.nonlinear11 = build_act(nonlinear[0], in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.nonlinear12 = build_act(nonlinear[1], in_channels)
        self.shortcut2 = build_act(shortcut, out_channels)
        self.ahead_fexpand2 = build_act(ahead_fexpand, in_channels)
        self.fexpand2 = FeaExpand(expansion=fexpand_num, mode=fexpand_mode, in_channels=in_channels, thres=fexpand_thres)
        self.conv_1x1 = BLConv2d(in_channels * fexpand_num, out_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
        self.nonlinear21 = build_act(nonlinear[0], out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear22 = build_act(nonlinear[1], out_channels)

    def forward(self, x):
        # conv 3x3 (降采样)
        if self.stride == 2:
            identity = self.pooling(x)
        else:
            identity = x
        identity = self.shortcut1(identity)
        out1 = self.ahead_fexpand1(x)
        out1 = self.fexpand1(out1)
        out1 = self.conv_3x3(out1)
        out1 = self.nonlinear11(out1)
        out1 = self.bn1(out1)
        out1 += identity
        out1 = self.nonlinear12(out1)

        # conv 1x1 (升维)
        if self.in_channels == self.out_channels:
            identity = out1
        else:
            assert self.in_channels * 2 == self.out_channels
            identity = torch.cat([out1] * 2, dim=1)
        identity = self.shortcut2(identity)
        out2 = self.ahead_fexpand2(out1)
        out2 = self.fexpand2(out2)
        out2 = self.conv_1x1(out2)
        out2 = self.nonlinear21(out2)
        out2 = self.bn2(out2)
        out2 += identity
        out2 = self.nonlinear22(out2)

        return out2


class MF1SimBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1,
                 fexpand_num=2, fexpand_mode='9-indepedent', fexpand_thres=(-0.6, 0.6),
                 nonlinear=('prelu', 'identity'), shortcut='identity', ahead_fexpand='identity',
                 **kwargs):
        super(MF1SimBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.fexpand_num = fexpand_num

        if self.stride == 2:
            self.pooling = nn.AvgPool2d(2, 2)
        self.shortcut1 = build_act(shortcut, in_channels)
        self.ahead_fexpand1 = build_act(ahead_fexpand, in_channels)
        self.fexpand1 = FeaExpand(expansion=fexpand_num, mode=fexpand_mode, in_channels=in_channels, thres=fexpand_thres)
        self.conv_3x3 = BLConv2d(in_channels * fexpand_num, in_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.nonlinear11 = build_act(nonlinear[0], in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.nonlinear12 = build_act(nonlinear[1], in_channels)
        self.shortcut2 = build_act(shortcut, out_channels)
        self.ahead_fexpand2 = build_act(ahead_fexpand, in_channels)
        self.fexpand2 = FeaExpand(expansion=fexpand_num, mode=fexpand_mode, in_channels=in_channels, thres=fexpand_thres)
        self.conv_1x1 = BLConv2d(in_channels * fexpand_num, out_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
        self.nonlinear21 = build_act(nonlinear[0], out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear22 = build_act(nonlinear[1], out_channels)

    def forward(self, x):
        # conv 3x3 (降采样)
        if self.stride == 2:
            identity = self.pooling(x)
        else:
            identity = x
        identity = self.shortcut1(identity)
        out1 = self.ahead_fexpand1(x)
        cos_sim1, out1 = self.fexpand1(out1)
        out1 = self.conv_3x3(out1)
        out1 = self.nonlinear11(out1)
        out1 = self.bn1(out1)
        out1 += identity
        out1 = self.nonlinear12(out1)

        # conv 1x1 (升维)
        if self.in_channels == self.out_channels:
            identity = out1
        else:
            assert self.in_channels * 2 == self.out_channels
            identity = torch.cat([out1] * 2, dim=1)
        identity = self.shortcut2(identity)
        out2 = self.ahead_fexpand2(out1)
        cos_sim2, out2 = self.fexpand2(out2)
        out2 = self.conv_1x1(out2)
        out2 = self.nonlinear21(out2)
        out2 = self.bn2(out2)
        out2 += identity
        out2 = self.nonlinear22(out2)

        cos_sim = cos_sim1 + cos_sim2

        return out2, cos_sim


class MF11Block(nn.Module):
    '''不使用multifea与dprelu'''
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super(MF11Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.fexpand_num = 1

        if self.stride == 2:
            self.pooling = nn.AvgPool2d(2, 2)
        self.fexpand1 = FeaExpand(expansion=self.fexpand_num, mode='1', thres=(-0.55, 0.55))
        self.conv_3x3 = BLConv2d(in_channels * self.fexpand_num, in_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.nonlinear1 = nn.PReLU(in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.fexpand2 = FeaExpand(expansion=self.fexpand_num, mode='1', thres=(-0.55, 0.55))
        self.conv_1x1 = BLConv2d(in_channels * self.fexpand_num, out_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
        self.nonlinear2 = nn.PReLU(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # conv 3x3 (降采样)
        if self.stride == 2:
            identity = self.pooling(x)
        else:
            identity = x
        out1 = self.fexpand1(x)
        out1 = self.conv_3x3(out1)
        out1 = self.nonlinear1(out1)
        out1 = self.bn1(out1)
        out1 += identity

        # conv 1x1 (升维)
        if self.in_channels == self.out_channels:
            identity = out1
        else:
            assert self.in_channels * 2 == self.out_channels
            identity = torch.cat([out1] * 2, dim=1)
        out2 = self.fexpand2(out1)
        out2 = self.conv_1x1(out2)
        out2 = self.nonlinear2(out2)
        out2 = self.bn2(out2)
        out2 += identity

        return out2


class MF12Block(nn.Module):
    '''不使用multifea'''
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super(MF12Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.fexpand_num = 1

        if self.stride == 2:
            self.pooling = nn.AvgPool2d(2, 2)
        self.fexpand1 = FeaExpand(expansion=self.fexpand_num, mode='1', thres=(-0.55, 0.55))
        self.conv_3x3 = BLConv2d(in_channels * self.fexpand_num, in_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.nonlinear11 = nn.PReLU(in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.nonlinear12 = DPReLU(in_channels)
        
        self.fexpand2 = FeaExpand(expansion=self.fexpand_num, mode='1', thres=(-0.55, 0.55))
        self.conv_1x1 = BLConv2d(in_channels * self.fexpand_num, out_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
        self.nonlinear21 = nn.PReLU(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear22 = DPReLU(out_channels)

    def forward(self, x):
        # conv 3x3 (降采样)
        if self.stride == 2:
            identity = self.pooling(x)
        else:
            identity = x
        out1 = self.fexpand1(x)
        out1 = self.conv_3x3(out1)
        out1 = self.nonlinear11(out1)
        out1 = self.bn1(out1)
        out1 += identity
        out1 = self.nonlinear12(out1)

        # conv 1x1 (升维)
        if self.in_channels == self.out_channels:
            identity = out1
        else:
            assert self.in_channels * 2 == self.out_channels
            identity = torch.cat([out1] * 2, dim=1)
        out2 = self.fexpand2(out1)
        out2 = self.conv_1x1(out2)
        out2 = self.nonlinear21(out2)
        out2 = self.bn2(out2)
        out2 += identity
        out2 = self.nonlinear22(out2)

        return out2


class MF3Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, nonlinear=('prelu', 'identity'), groups=2, **kwargs):
        super(MF3Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.fexpand_num = 2

        if self.stride == 2:
            self.pooling = nn.AvgPool2d(2, 2)
        self.fexpand1 = FeaExpand(expansion=self.fexpand_num, mode='5', thres=(-0.55, 0.55))
        self.conv_3x3 = BLConv2d(in_channels * self.fexpand_num, in_channels * self.fexpand_num,
                                 kernel_size=3, stride=stride, padding=1, bias=False, groups=groups, **kwargs)
        self.nonlinear11 = self._build_act(nonlinear[0], in_channels * self.fexpand_num)
        self.bn1 = nn.BatchNorm2d(in_channels * self.fexpand_num)
        self.nonlinear12 = self._build_act(nonlinear[1], in_channels * self.fexpand_num)
        # self.fexpand2 = FeaExpand(expansion=self.fexpand_num, mode='5', thres=(-0.55, 0.55))
        self.conv_1x1 = BLConv2d(in_channels * self.fexpand_num, out_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
        self.nonlinear21 = self._build_act(nonlinear[0], out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear22 = self._build_act(nonlinear[1], out_channels)

    def _build_act(self, act_name, channels):
        if act_name == 'identity':
            return nn.Sequential()
        elif act_name == 'abs':
            return torch.abs
        elif act_name == 'prelu':
            return nn.PReLU(channels)
        elif act_name == 'prelu_pi=1':
            return nn.PReLU(channels, init=1.0)
        elif act_name == 'prelu_one_pi=1':
            return nn.PReLU(1, init=1.0)
        elif act_name == 'rprelu':
            return RPRelu(channels)
        elif act_name == 'rprelu_pi=1':
            return RPRelu(channels, prelu_init=1.0)
        elif act_name == 'bp':
            return CfgLayer(channels, config='bp', prelu_init=1.0)
        elif act_name == 'pb':
            return CfgLayer(channels, config='pb', prelu_init=1.0)
        elif act_name == 'bias':
            return CfgLayer(channels, config='b')
        elif act_name == 'bpbpb':
            return CfgLayer(channels, config='bpbpb', prelu_init=1.0)
        elif act_name == 'dprelu':
            return DPReLU(channels)
        elif act_name == 'nprelu':
            return NPReLU(channels)
        elif act_name == 'scale':
            return LearnableScale(channels)
        elif act_name == 'scale_one':
            return LearnableScale(1)
        elif act_name == 'prelu_shortcut':
            return PReLUsc(channels, init=1.0)
        else:
            return act_name_map[act_name]()

    def forward(self, x):
        # conv 3x3 (降采样)
        out1 = self.fexpand1(x)
        if self.stride == 2:
            identity = self.pooling(out1)
        else:
            identity = out1
        out1 = self.conv_3x3(out1)
        out1 = self.nonlinear11(out1)
        out1 = self.bn1(out1)
        out1 += identity
        out1 = self.nonlinear12(out1)

        # conv 1x1 (升维)
        if self.in_channels == self.out_channels:
            identity = x
        else:
            assert self.in_channels * 2 == self.out_channels
            identity = out1
        # out2 = self.fexpand2(out1)
        out2 = self.conv_1x1(out1)
        out2 = self.nonlinear21(out2)
        out2 = self.bn2(out2)
        out2 += identity
        out2 = self.nonlinear22(out2)

        return out2


class MF1s1Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, nonlinear=('prelu', 'identity'), **kwargs):
        super(MF1s1Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.fexpand_num = 2

        if self.stride == 2:
            self.pooling = nn.AvgPool2d(2, 2)
        self.scale1 = LearnableScale(in_channels)
        self.fexpand1 = FeaExpand(expansion=self.fexpand_num, mode='5', thres=(-0.55, 0.55))
        self.conv_3x3 = BLConv2d(in_channels * self.fexpand_num, in_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.nonlinear11 = self._build_act(nonlinear[0], in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.nonlinear12 = self._build_act(nonlinear[1], in_channels)
        self.scale2 = LearnableScale(in_channels)
        self.fexpand2 = FeaExpand(expansion=self.fexpand_num, mode='5', thres=(-0.55, 0.55))
        self.conv_1x1 = BLConv2d(in_channels * self.fexpand_num, out_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
        self.nonlinear21 = self._build_act(nonlinear[0], out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear22 = self._build_act(nonlinear[1], out_channels)

    def _build_act(self, act_name, channels):
        if act_name == 'identity':
            return nn.Sequential()
        elif act_name == 'abs':
            return torch.abs
        elif act_name == 'prelu':
            return nn.PReLU(channels)
        elif act_name == 'prelu_pi=1':
            return nn.PReLU(channels, init=1.0)
        elif act_name == 'prelu_one_pi=1':
            return nn.PReLU(1, init=1.0)
        elif act_name == 'rprelu':
            return RPRelu(channels)
        elif act_name == 'rprelu_pi=1':
            return RPRelu(channels, prelu_init=1.0)
        elif act_name == 'bp':
            return CfgLayer(channels, config='bp', prelu_init=1.0)
        elif act_name == 'pb':
            return CfgLayer(channels, config='pb', prelu_init=1.0)
        elif act_name == 'bias':
            return CfgLayer(channels, config='b')
        elif act_name == 'bpbpb':
            return CfgLayer(channels, config='bpbpb', prelu_init=1.0)
        elif act_name == 'dprelu':
            return DPReLU(channels)
        elif act_name == 'nprelu':
            return NPReLU(channels)
        elif act_name == 'scale':
            return LearnableScale(channels)
        elif act_name == 'scale_one':
            return LearnableScale(1)
        elif act_name == 'prelu_shortcut':
            return PReLUsc(channels, init=1.0)
        else:
            return act_name_map[act_name]()

    def forward(self, x):
        # conv 3x3 (降采样)
        if self.stride == 2:
            identity = self.pooling(x)
        else:
            identity = x
        out1 = self.scale1(x)
        out1 = self.fexpand1(out1)
        out1 = self.conv_3x3(out1)
        out1 = self.nonlinear11(out1)
        out1 = self.bn1(out1)
        out1 += identity
        out1 = self.nonlinear12(out1)

        # conv 1x1 (升维)
        if self.in_channels == self.out_channels:
            identity = out1
        else:
            assert self.in_channels * 2 == self.out_channels
            identity = torch.cat([out1] * 2, dim=1)
        out2 = self.scale2(out1)
        out2 = self.fexpand2(out2)
        out2 = self.conv_1x1(out2)
        out2 = self.nonlinear21(out2)
        out2 = self.bn2(out2)
        out2 += identity
        out2 = self.nonlinear22(out2)

        return out2


class MF1s2Block(nn.Module):
    ''' 在shortcut上加上scale '''
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, nonlinear=('prelu', 'identity'), **kwargs):
        super(MF1s2Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.fexpand_num = 2

        if self.stride == 2:
            self.pooling = nn.AvgPool2d(2, 2)
        self.scale1 = LearnableScale(in_channels)
        self.fexpand1 = FeaExpand(expansion=self.fexpand_num, mode='5', thres=(-0.55, 0.55))
        self.conv_3x3 = BLConv2d(in_channels * self.fexpand_num, in_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.nonlinear11 = self._build_act(nonlinear[0], in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.nonlinear12 = self._build_act(nonlinear[1], in_channels)
        self.scale2 = LearnableScale(out_channels)
        self.fexpand2 = FeaExpand(expansion=self.fexpand_num, mode='5', thres=(-0.55, 0.55))
        self.conv_1x1 = BLConv2d(in_channels * self.fexpand_num, out_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
        self.nonlinear21 = self._build_act(nonlinear[0], out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear22 = self._build_act(nonlinear[1], out_channels)

    def _build_act(self, act_name, channels):
        if act_name == 'identity':
            return nn.Sequential()
        elif act_name == 'abs':
            return torch.abs
        elif act_name == 'prelu':
            return nn.PReLU(channels)
        elif act_name == 'prelu_pi=1':
            return nn.PReLU(channels, init=1.0)
        elif act_name == 'prelu_one_pi=1':
            return nn.PReLU(1, init=1.0)
        elif act_name == 'rprelu':
            return RPRelu(channels)
        elif act_name == 'rprelu_pi=1':
            return RPRelu(channels, prelu_init=1.0)
        elif act_name == 'bp':
            return CfgLayer(channels, config='bp', prelu_init=1.0)
        elif act_name == 'pb':
            return CfgLayer(channels, config='pb', prelu_init=1.0)
        elif act_name == 'bias':
            return CfgLayer(channels, config='b')
        elif act_name == 'bpbpb':
            return CfgLayer(channels, config='bpbpb', prelu_init=1.0)
        elif act_name == 'dprelu':
            return DPReLU(channels)
        elif act_name == 'nprelu':
            return NPReLU(channels)
        elif act_name == 'scale':
            return LearnableScale(channels)
        elif act_name == 'scale_one':
            return LearnableScale(1)
        elif act_name == 'prelu_shortcut':
            return PReLUsc(channels, init=1.0)
        else:
            return act_name_map[act_name]()

    def forward(self, x):
        # conv 3x3 (降采样)
        if self.stride == 2:
            identity = self.pooling(x)
        else:
            identity = x
        identity = self.scale1(identity)
        out1 = self.fexpand1(x)
        out1 = self.conv_3x3(out1)
        out1 = self.nonlinear11(out1)
        out1 = self.bn1(out1)
        out1 += identity
        out1 = self.nonlinear12(out1)

        # conv 1x1 (升维)
        if self.in_channels == self.out_channels:
            identity = out1
        else:
            assert self.in_channels * 2 == self.out_channels
            identity = torch.cat([out1] * 2, dim=1)
        identity = self.scale2(identity)
        out2 = self.fexpand2(out1)
        out2 = self.conv_1x1(out2)
        out2 = self.nonlinear21(out2)
        out2 = self.bn2(out2)
        out2 += identity
        out2 = self.nonlinear22(out2)

        return out2


class MF6Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, nonlinear=('prelu', 'identity'), shortcut='identity', **kwargs):
        super(MF6Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.fexpand_num = 2

        if self.stride == 2:
            self.pooling = nn.AvgPool2d(2, 2)
        self.shortcut1 = self._build_act(shortcut, in_channels)
        self.fexpand1 = FeaExpand(expansion=self.fexpand_num, mode='5', thres=(-0.55, 0.55))
        self.conv_3x3 = BLConv2d(in_channels * self.fexpand_num, in_channels, kernel_size=3, stride=stride, padding=1, bias=False, groups=2, **kwargs)
        self.nonlinear11 = self._build_act(nonlinear[0], in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.nonlinear12 = self._build_act(nonlinear[1], in_channels)
        self.shortcut2 = self._build_act(shortcut, out_channels)
        # self.fexpand2 = FeaExpand(expansion=self.fexpand_num, mode='5', thres=(-0.55, 0.55))
        self.conv_1x1 = BLConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
        self.nonlinear21 = self._build_act(nonlinear[0], out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear22 = self._build_act(nonlinear[1], out_channels)

    def _build_act(self, act_name, channels):
        if act_name == 'identity':
            return nn.Sequential()
        elif act_name == 'abs':
            return torch.abs
        elif act_name == 'prelu':
            return nn.PReLU(channels)
        elif act_name == 'prelu_pi=1':
            return nn.PReLU(channels, init=1.0)
        elif act_name == 'prelu_one_pi=1':
            return nn.PReLU(1, init=1.0)
        elif act_name == 'rprelu':
            return RPRelu(channels)
        elif act_name == 'rprelu_pi=1':
            return RPRelu(channels, prelu_init=1.0)
        elif act_name == 'bp':
            return CfgLayer(channels, config='bp', prelu_init=1.0)
        elif act_name == 'pb':
            return CfgLayer(channels, config='pb', prelu_init=1.0)
        elif act_name == 'bias':
            return CfgLayer(channels, config='b')
        elif act_name == 'bpbpb':
            return CfgLayer(channels, config='bpbpb', prelu_init=1.0)
        elif act_name == 'dprelu':
            return DPReLU(channels)
        elif act_name == 'nprelu':
            return NPReLU(channels)
        elif act_name == 'scale':
            return LearnableScale(channels)
        elif act_name == 'scale3':
            return LearnableScale3(channels)
        elif act_name == 'scale_one':
            return LearnableScale(1)
        elif act_name == 'prelu_shortcut':
            return PReLUsc(channels, init=1.0)
        elif act_name == 'scale_sum':
            return ScaleSum(channels)
        elif act_name == 'ls+ss':
            return LSaddSS(channels)
        else:
            return act_name_map[act_name]()

    def forward(self, x):
        # conv 3x3 (降采样)
        if self.stride == 2:
            identity = self.pooling(x)
        else:
            identity = x
        identity = self.shortcut1(identity)
        out1 = self.fexpand1(x)
        out1 = self.conv_3x3(out1)
        out1 = self.nonlinear11(out1)
        out1 = self.bn1(out1)
        out1 += identity
        out1 = self.nonlinear12(out1)

        # conv 1x1 (升维)
        if self.in_channels == self.out_channels:
            identity = out1
        else:
            assert self.in_channels * 2 == self.out_channels
            identity = torch.cat([out1] * 2, dim=1)
        identity = self.shortcut2(identity)
        # out2 = self.fexpand2(out1)
        out2 = self.conv_1x1(out1)
        out2 = self.nonlinear21(out2)
        out2 = self.bn2(out2)
        out2 += identity
        out2 = self.nonlinear22(out2)

        return out2


class MF1terBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, mode='5',
        nonlinear=('prelu', 'identity'), shortcut='identity', ahead_fexpand='identity',
        **kwargs):
        super(MF1terBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.fexpand_num = 3
        self.mode = mode

        if self.stride == 2:
            self.pooling = nn.AvgPool2d(2, 2)
        self.shortcut1 = self._build_act(shortcut, in_channels)
        self.ahead_fexpand1 = self._build_act(ahead_fexpand, in_channels)
        # self.fexpand1 = FeaExpand(expansion=self.fexpand_num, mode=mode, thres=(-0.55, 0.55))
        self.conv_3x3 = TAConv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, bias=False, 
                                  thres=(-0.55, 0.55), **kwargs)
        self.nonlinear11 = self._build_act(nonlinear[0], in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.nonlinear12 = self._build_act(nonlinear[1], in_channels)
        self.shortcut2 = self._build_act(shortcut, out_channels)
        self.ahead_fexpand2 = self._build_act(ahead_fexpand, in_channels)
        # self.fexpand2 = FeaExpand(expansion=self.fexpand_num, mode=mode, thres=(-0.55, 0.55))
        self.conv_1x1 = TAConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False,
                                 thres=(-0.55, 0.55), **kwargs)
        self.nonlinear21 = self._build_act(nonlinear[0], out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear22 = self._build_act(nonlinear[1], out_channels)

    def _build_act(self, act_name, channels):
        if act_name == 'identity':
            return nn.Sequential()
        elif act_name == 'abs':
            return torch.abs
        elif act_name == 'prelu':
            return nn.PReLU(channels)
        elif act_name == 'prelu_pi=1':
            return nn.PReLU(channels, init=1.0)
        elif act_name == 'prelu_one_pi=1':
            return nn.PReLU(1, init=1.0)
        elif act_name == 'rprelu':
            return RPRelu(channels)
        elif act_name == 'rprelu_pi=1':
            return RPRelu(channels, prelu_init=1.0)
        elif act_name == 'bp':
            return CfgLayer(channels, config='bp', prelu_init=1.0)
        elif act_name == 'pb':
            return CfgLayer(channels, config='pb', prelu_init=1.0)
        elif act_name == 'bias':
            return CfgLayer(channels, config='b')
        elif act_name == 'bpbpb':
            return CfgLayer(channels, config='bpbpb', prelu_init=1.0)
        elif act_name == 'dprelu':
            return DPReLU(channels)
        elif act_name == 'nprelu':
            return NPReLU(channels)
        elif act_name == 'scale':
            return LearnableScale(channels)
        elif act_name == 'scale3':
            return LearnableScale3(channels)
        elif act_name == 'scale_one':
            return LearnableScale(1)
        elif act_name == 'prelu_shortcut':
            return PReLUsc(channels, init=1.0)
        elif act_name == 'scale_sum':
            return ScaleSum(channels)
        elif act_name == 'ls+ss':
            return LSaddSS(channels)
        else:
            return act_name_map[act_name]()

    def forward(self, x):
        # conv 3x3 (降采样)
        if self.stride == 2:
            identity = self.pooling(x)
        else:
            identity = x
        identity = self.shortcut1(identity)
        out1 = self.ahead_fexpand1(x)
        # out1 = self.fexpand1(out1)
        out1 = self.conv_3x3(out1)
        out1 = self.nonlinear11(out1)
        out1 = self.bn1(out1)
        out1 += identity
        out1 = self.nonlinear12(out1)

        # conv 1x1 (升维)
        if self.in_channels == self.out_channels:
            identity = out1
        else:
            assert self.in_channels * 2 == self.out_channels
            identity = torch.cat([out1] * 2, dim=1)
        identity = self.shortcut2(identity)
        out2 = self.ahead_fexpand2(out1)
        # out2 = self.fexpand2(out2)
        out2 = self.conv_1x1(out2)
        out2 = self.nonlinear21(out2)
        out2 = self.bn2(out2)
        out2 += identity
        out2 = self.nonlinear22(out2)

        return out2