import torch
import torch.nn as nn
from .binary_convs import BLConv2d, RAConv2d
from .binary_functions import act_name_map, FeaExpand, LearnableScale, LearnableScale3, RPRelu, CfgLayer, DPReLU, NPReLU, PReLUsc


class ReActBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super(ReActBlock, self).__init__()

        self.move1 = LearnableBias(in_channels)
        self.conv_3x3 = RAConv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.rprelu1 = RPRelu(in_channels)

        self.move2 = LearnableBias(in_channels)
        if in_channels == out_channels:
            self.conv_1x1 = RAConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.conv_1x1_down1 = RAConv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.conv_1x1_down2 = RAConv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.bn2_1 = nn.BatchNorm2d(in_channels)
            self.bn2_2 = nn.BatchNorm2d(in_channels)
        self.rprelu2 = RPRelu(out_channels)

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.in_channels != self.out_channels:
            self.pooling = nn.AvgPool2d(2, 2)
        
    def forward(self, x):
        # 3x3 conv
        out1 = self.move1(x)
        out1 = self.conv_3x3(out1)
        out1 = self.bn1(out1)

        if self.stride == 2:
            x = self.pooling(x)

        out1 += x
        out1 = self.rprelu1(out1)

        # 1x1 conv
        out2 = self.move2(out1)
        if self.in_channels == self.out_channels:
            out2 = self.conv_1x1(out2)
            out2 = self.bn2(out2)
            out2 += out1
        else:
            assert self.in_channels * 2 == self.out_channels
            out2_1 = self.conv_1x1_down1(out2)
            out2_2 = self.conv_1x1_down2(out2)
            out2_1 = self.bn2_1(out2_1)
            out2_2 = self.bn2_2(out2_2)
            out2_1 += out1
            out2_2 += out1
            out2 = torch.cat([out2_1, out2_2], dim=1)
        out2 = self.rprelu2(out2)

        return out2


class ReAct1Block(nn.Module):
    '''' reactnet block without rsign'''

    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super(ReAct1Block, self).__init__()

        self.conv_3x3 = RAConv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.rprelu1 = RPRelu(in_channels)

        if in_channels == out_channels:
            self.conv_1x1 = RAConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.conv_1x1_down1 = RAConv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.conv_1x1_down2 = RAConv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.bn2_1 = nn.BatchNorm2d(in_channels)
            self.bn2_2 = nn.BatchNorm2d(in_channels)
        self.rprelu2 = RPRelu(out_channels)

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.in_channels != self.out_channels:
            self.pooling = nn.AvgPool2d(2, 2)

    def forward(self, x):
        # 3x3 conv
        out1 = self.conv_3x3(x)
        out1 = self.bn1(out1)

        if self.stride == 2:
            x = self.pooling(x)

        out1 += x
        out1 = self.rprelu1(out1)

        # 1x1 conv
        if self.in_channels == self.out_channels:
            out2 = self.conv_1x1(out1)
            out2 = self.bn2(out2)
            out2 += out1
        else:
            assert self.in_channels * 2 == self.out_channels
            out2_1 = self.conv_1x1_down1(out1)
            out2_2 = self.conv_1x1_down2(out1)
            out2_1 = self.bn2_1(out2_1)
            out2_2 = self.bn2_2(out2_2)
            out2_1 += out1
            out2_2 += out1
            out2 = torch.cat([out2_1, out2_2], dim=1)
        out2 = self.rprelu2(out2)

        return out2


class ReActBaseBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super(ReActBaseBlock, self).__init__()

        self.conv_3x3 = RAConv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(in_channels)
        # self.rprelu1 = RPRelu(in_channels)

        if in_channels == out_channels:
            self.conv_1x1 = RAConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.conv_1x1_down1 = RAConv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.conv_1x1_down2 = RAConv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.bn2_1 = nn.BatchNorm2d(in_channels)
            self.bn2_2 = nn.BatchNorm2d(in_channels)
        # self.rprelu2 = RPRelu(out_channels)

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.in_channels != self.out_channels:
            self.pooling = nn.AvgPool2d(2, 2)

    def forward(self, x):
        # 3x3 conv
        out1 = self.conv_3x3(x)
        out1 = self.bn1(out1)

        if self.stride == 2:
            x = self.pooling(x)

        out1 += x
        # out1 = self.rprelu1(out1)

        # 1x1 conv
        if self.in_channels == self.out_channels:
            out2 = self.conv_1x1(out1)
            out2 = self.bn2(out2)
            out2 += out1
        else:
            assert self.in_channels * 2 == self.out_channels
            out2_1 = self.conv_1x1_down1(out1)
            out2_2 = self.conv_1x1_down2(out1)
            out2_1 = self.bn2_1(out2_1)
            out2_2 = self.bn2_2(out2_2)
            out2_1 += out1
            out2_2 += out1
            out2 = torch.cat([out2_1, out2_2], dim=1)
        # out2 = self.rprelu2(out2)

        return out2


class ReActBaseGBaBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, n=1, **kwargs):
        super(ReActBaseGBaBlock, self).__init__()

        self.groups = n
        self.alpha = []
        self.move1 = nn.ModuleList()
        self.conv_3x3 = nn.ModuleList()
        self.bn1 = nn.ModuleList()
        
        for i in range(self.groups):
            self.alpha.append(-1 + (i + 1) * 2 / (self.groups + 1))
            self.conv_3x3.append(RAConv2d(in_channels, in_channels, kernel_size=3,
                stride=stride, padding=1, bias=False, groups=self.groups, **kwargs))
            self.bn1.append(nn.BatchNorm2d(in_channels))

        if in_channels == out_channels:
            self.conv_1x1 = RAConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.conv_1x1_down1 = RAConv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.conv_1x1_down2 = RAConv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.bn2_1 = nn.BatchNorm2d(in_channels)
            self.bn2_2 = nn.BatchNorm2d(in_channels)

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.in_channels != self.out_channels:
            self.pooling = nn.AvgPool2d(2, 2)

    def forward(self, x):
        # 3x3 conv
        x_max = torch.max(abs(x))
        x_list = []
        for i in range(self.groups):
            a = self.alpha[i] * x_max
            x_list.append(x + a)
        
        out1 = 0
        for i in range(self.groups):
            x_list[i] = self.conv_3x3[i](x_list[i])
            x_list[i] = self.bn1[i](x_list[i])
            out1 += x_list[i]

        if self.stride == 2:
            x = self.pooling(x)

        out1 += x

        # 1x1 conv
        if self.in_channels == self.out_channels:
            out2 = self.conv_1x1(out1)
            out2 = self.bn2(out2)
            out2 += out1
        else:
            assert self.in_channels * 2 == self.out_channels
            out2_1 = self.conv_1x1_down1(out1)
            out2_2 = self.conv_1x1_down2(out1)
            out2_1 = self.bn2_1(out2_1)
            out2_2 = self.bn2_2(out2_2)
            out2_1 += out1
            out2_2 += out1
            out2 = torch.cat([out2_1, out2_2], dim=1)

        return out2


class ReActBaseGBa4Block(ReActBaseGBaBlock):

    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super(ReActBaseGBa4Block, self).__init__(in_channels, out_channels, stride, n=4, **kwargs)


class ReActGBaBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, n=1, **kwargs):
        super(ReActGBaBlock, self).__init__()

        self.groups = n
        self.alpha = []
        self.move1 = nn.ModuleList()
        self.conv_3x3 = nn.ModuleList()
        self.bn1 = nn.ModuleList()
        
        for i in range(self.groups):
            self.alpha.append(-1 + (i + 1) * 2 / (self.groups + 1))
            self.move1.append(LearnableBias(in_channels))
            self.conv_3x3.append(RAConv2d(in_channels, in_channels, kernel_size=3,
                stride=stride, padding=1, bias=False, groups=self.groups, **kwargs))
            self.bn1.append(nn.BatchNorm2d(in_channels))
        self.rprelu1 = RPRelu(in_channels)

        self.move2 = LearnableBias(in_channels)
        if in_channels == out_channels:
            self.conv_1x1 = RAConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.conv_1x1_down1 = RAConv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.conv_1x1_down2 = RAConv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.bn2_1 = nn.BatchNorm2d(in_channels)
            self.bn2_2 = nn.BatchNorm2d(in_channels)
        self.rprelu2 = RPRelu(out_channels)

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.in_channels != self.out_channels:
            self.pooling = nn.AvgPool2d(2, 2)

    def forward(self, x):
        # 3x3 conv
        x_max = torch.max(abs(x))
        x_list = []
        for i in range(self.groups):
            a = self.alpha[i] * x_max
            x_list.append(x + a)
        
        out1 = 0
        for i in range(self.groups):
            x_list[i] = self.move1[i](x_list[i])
            x_list[i] = self.conv_3x3[i](x_list[i])
            x_list[i] = self.bn1[i](x_list[i])
            out1 += x_list[i]

        if self.stride == 2:
            x = self.pooling(x)

        out1 += x
        out1 = self.rprelu1(out1)

        # 1x1 conv
        out2 = self.move2(out1)
        if self.in_channels == self.out_channels:
            out2 = self.conv_1x1(out2)
            out2 = self.bn2(out2)
            out2 += out1
        else:
            assert self.in_channels * 2 == self.out_channels
            out2_1 = self.conv_1x1_down1(out2)
            out2_2 = self.conv_1x1_down2(out2)
            out2_1 = self.bn2_1(out2_1)
            out2_2 = self.bn2_2(out2_2)
            out2_1 += out1
            out2_2 += out1
            out2 = torch.cat([out2_1, out2_2], dim=1)
        out2 = self.rprelu2(out2)

        return out2


class ReActGBa4Block(ReActGBaBlock):

    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super(ReActGBa4Block, self).__init__(in_channels, out_channels, stride, n=4, **kwargs)


class ReAct1GBaBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, n=1, **kwargs):
        super(ReAct1GBaBlock, self).__init__()

        self.groups = n
        self.alpha = []
        self.move1 = nn.ModuleList()
        self.conv_3x3 = nn.ModuleList()
        self.bn1 = nn.ModuleList()
        
        for i in range(self.groups):
            self.alpha.append(-1 + (i + 1) * 2 / (self.groups + 1))
            self.conv_3x3.append(RAConv2d(in_channels, in_channels, kernel_size=3,
                stride=stride, padding=1, bias=False, groups=self.groups, **kwargs))
            self.bn1.append(nn.BatchNorm2d(in_channels))
        self.rprelu1 = RPRelu(in_channels)

        if in_channels == out_channels:
            self.conv_1x1 = RAConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.conv_1x1_down1 = RAConv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.conv_1x1_down2 = RAConv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.bn2_1 = nn.BatchNorm2d(in_channels)
            self.bn2_2 = nn.BatchNorm2d(in_channels)
        self.rprelu2 = RPRelu(out_channels)

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.in_channels != self.out_channels:
            self.pooling = nn.AvgPool2d(2, 2)

    def forward(self, x):
        # 3x3 conv
        x_max = torch.max(abs(x))
        x_list = []
        for i in range(self.groups):
            a = self.alpha[i] * x_max
            x_list.append(x + a)
        
        out1 = 0
        for i in range(self.groups):
            x_list[i] = self.conv_3x3[i](x_list[i])
            x_list[i] = self.bn1[i](x_list[i])
            out1 += x_list[i]

        if self.stride == 2:
            x = self.pooling(x)

        out1 += x
        out1 = self.rprelu1(out1)

        # 1x1 conv
        if self.in_channels == self.out_channels:
            out2 = self.conv_1x1(out1)
            out2 = self.bn2(out2)
            out2 += out1
        else:
            assert self.in_channels * 2 == self.out_channels
            out2_1 = self.conv_1x1_down1(out1)
            out2_2 = self.conv_1x1_down2(out1)
            out2_1 = self.bn2_1(out2_1)
            out2_2 = self.bn2_2(out2_2)
            out2_1 += out1
            out2_2 += out1
            out2 = torch.cat([out2_1, out2_2], dim=1)
        out2 = self.rprelu2(out2)

        return out2


class ReAct1GBa4Block(ReAct1GBaBlock):

    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super(ReAct1GBa4Block, self).__init__(in_channels, out_channels, stride, n=4, **kwargs)


class ReActGSBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, n=1, **kwargs):
        super(ReActGSBlock, self).__init__()

        self.groups = n
        self.alpha = []
        for i in range(self.groups):
            self.alpha.append(-1 + (i + 1) * 2 / (self.groups + 1))
        self.move1 = LearnableBias(in_channels)
        self.conv_3x3 = RAConv2d(in_channels, in_channels, kernel_size=3,
            stride=stride, padding=1, bias=False, groups=self.groups, **kwargs)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.rprelu1 = RPRelu(in_channels)

        self.move2 = LearnableBias(in_channels)
        if in_channels == out_channels:
            self.conv_1x1 = RAConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.conv_1x1_down1 = RAConv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.conv_1x1_down2 = RAConv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.bn2_1 = nn.BatchNorm2d(in_channels)
            self.bn2_2 = nn.BatchNorm2d(in_channels)
        self.rprelu2 = RPRelu(out_channels)

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.in_channels != self.out_channels:
            self.pooling = nn.AvgPool2d(2, 2)

    def forward(self, x):
        # 3x3 conv
        x_max = torch.max(abs(x))
        x_group = x.split(self.in_channels // self.groups, dim=1)
        assert len(x_group) == self.groups
        x_group_list = []
        for i in range(self.groups):
            a = self.alpha[i] * x_max
            x_group_list.append(x_group[i] + a)
        x = torch.cat(x_group_list, dim=1)

        out1 = self.move1(x)
        out1 = self.conv_3x3(out1)
        out1 = self.bn1(out1)

        if self.stride == 2:
            x = self.pooling(x)

        out1 += x

        out1 = self.rprelu1(out1)

        # 1x1 conv
        out2 = self.move2(out1)

        if self.in_channels == self.out_channels:
            out2 = self.conv_1x1(out2)
            out2 = self.bn2(out2)
            out2 += out1
        else:
            assert self.in_channels * 2 == self.out_channels

            out2_1 = self.conv_1x1_down1(out2)
            out2_2 = self.conv_1x1_down2(out2)
            out2_1 = self.bn2_1(out2_1)
            out2_2 = self.bn2_2(out2_2)
            out2_1 += out1
            out2_2 += out1
            out2 = torch.cat([out2_1, out2_2], dim=1)

        out2 = self.rprelu2(out2)

        return out2


class ReActGS4Block(ReActGSBlock):

    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super(ReActGS4Block, self).__init__(in_channels, out_channels, stride, n=4, **kwargs)


class ReAct1GSBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, n=1, **kwargs):
        super(ReAct1GSBlock, self).__init__()

        self.groups = n
        self.alpha = []
        for i in range(self.groups):
            self.alpha.append(-1 + (i + 1) * 2 / (self.groups + 1))
        self.conv_3x3 = RAConv2d(in_channels, in_channels, kernel_size=3,
            stride=stride, padding=1, bias=False, groups=self.groups, **kwargs)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.rprelu1 = RPRelu(in_channels)

        if in_channels == out_channels:
            self.conv_1x1 = RAConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.conv_1x1_down1 = RAConv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.conv_1x1_down2 = RAConv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.bn2_1 = nn.BatchNorm2d(in_channels)
            self.bn2_2 = nn.BatchNorm2d(in_channels)
        self.rprelu2 = RPRelu(out_channels)

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.in_channels != self.out_channels:
            self.pooling = nn.AvgPool2d(2, 2)

    def forward(self, x):
        # 3x3 conv
        x_max = torch.max(abs(x))
        x_group = x.split(self.in_channels // self.groups, dim=1)
        assert len(x_group) == self.groups
        x_group_list = []
        for i in range(self.groups):
            a = self.alpha[i] * x_max
            x_group_list.append(x_group[i] + a)
        x = torch.cat(x_group_list, dim=1)

        out1 = self.conv_3x3(x)
        out1 = self.bn1(out1)

        if self.stride == 2:
            x = self.pooling(x)

        out1 += x

        out1 = self.rprelu1(out1)

        # 1x1 conv
        if self.in_channels == self.out_channels:
            out2 = self.conv_1x1(out1)
            out2 = self.bn2(out2)
            out2 += out1
        else:
            assert self.in_channels * 2 == self.out_channels

            out2_1 = self.conv_1x1_down1(out1)
            out2_2 = self.conv_1x1_down2(out1)
            out2_1 = self.bn2_1(out2_1)
            out2_2 = self.bn2_2(out2_2)
            out2_1 += out1
            out2_2 += out1
            out2 = torch.cat([out2_1, out2_2], dim=1)

        out2 = self.rprelu2(out2)

        return out2


class ReAct1GS4Block(ReAct1GSBlock):

    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super(ReAct1GS4Block, self).__init__(in_channels, out_channels, stride, n=4, **kwargs)


class ReActFATBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super(ReActFATBlock, self).__init__()

        self.move1 = LearnableBias(in_channels)
        self.conv_3x3 = RAFATConv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.rprelu1 = RPRelu(in_channels)

        self.move2 = LearnableBias(in_channels)

        if in_channels == out_channels:
            self.conv_1x1 = RAFATConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.conv_1x1_down1 = RAFATConv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.conv_1x1_down2 = RAFATConv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.bn2_1 = nn.BatchNorm2d(in_channels)
            self.bn2_2 = nn.BatchNorm2d(in_channels)

        self.rprelu2 = RPRelu(out_channels)

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.in_channels != self.out_channels:
            self.pooling = nn.AvgPool2d(2, 2)

    def forward(self, x):
        # 3x3 conv
        out1 = self.move1(x)

        out1 = self.conv_3x3(out1)
        out1 = self.bn1(out1)

        if self.stride == 2:
            x = self.pooling(x)

        out1 += x

        out1 = self.rprelu1(out1)

        # 1x1 conv
        out2 = self.move2(out1)

        if self.in_channels == self.out_channels:
            out2 = self.conv_1x1(out2)
            out2 = self.bn2(out2)
            out2 += out1
        else:
            assert self.in_channels * 2 == self.out_channels

            out2_1 = self.conv_1x1_down1(out2)
            out2_2 = self.conv_1x1_down2(out2)
            out2_1 = self.bn2_1(out2_1)
            out2_2 = self.bn2_2(out2_2)
            out2_1 += out1
            out2_2 += out1
            out2 = torch.cat([out2_1, out2_2], dim=1)

        out2 = self.rprelu2(out2)

        return out2


class ReActMFG4Block(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, n=4, **kwargs):
        super(ReActMFG4Block, self).__init__()

        self.groups = n
        self.alpha = [-0.7, -0.3, 0.3, 0.7]
        # self.move1 = nn.ModuleList()
        self.conv_3x3 = nn.ModuleList()
        self.bn1 = nn.ModuleList()
        
        for i in range(self.groups):
            # self.move1.append(LearnableBias(in_channels))
            self.conv_3x3.append(RAConv2d(in_channels, in_channels, kernel_size=3,
                stride=stride, padding=1, bias=False, groups=self.groups, **kwargs))
            self.bn1.append(nn.BatchNorm2d(in_channels))
        self.rprelu1 = RPRelu(in_channels)

        self.move2 = LearnableBias(in_channels)
        if in_channels == out_channels:
            self.conv_1x1 = RAConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.conv_1x1_down1 = RAConv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.conv_1x1_down2 = RAConv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
            self.bn2_1 = nn.BatchNorm2d(in_channels)
            self.bn2_2 = nn.BatchNorm2d(in_channels)
        self.rprelu2 = RPRelu(out_channels)

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.in_channels != self.out_channels:
            self.pooling = nn.AvgPool2d(2, 2)

    def forward(self, x):
        # 3x3 conv
        x_list = []
        for i in range(self.groups):
            x_list.append(x + self.alpha[i])
        
        out1 = 0
        for i in range(self.groups):
            # x_list[i] = self.move1[i](x_list[i])
            x_list[i] = self.conv_3x3[i](x_list[i])
            x_list[i] = self.bn1[i](x_list[i])
            out1 += x_list[i]

        if self.stride == 2:
            x = self.pooling(x)

        out1 += x
        out1 = self.rprelu1(out1)

        # 1x1 conv
        out2 = self.move2(out1)
        if self.in_channels == self.out_channels:
            out2 = self.conv_1x1(out2)
            out2 = self.bn2(out2)
            out2 += out1
        else:
            assert self.in_channels * 2 == self.out_channels
            out2_1 = self.conv_1x1_down1(out2)
            out2_2 = self.conv_1x1_down2(out2)
            out2_1 = self.bn2_1(out2_1)
            out2_2 = self.bn2_2(out2_2)
            out2_1 += out1
            out2_2 += out1
            out2 = torch.cat([out2_1, out2_2], dim=1)
        out2 = self.rprelu2(out2)

        return out2


class MF1Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, nonlinear=('prelu', 'identity'), shortcut='identity', **kwargs):
        super(MF1Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        if self.stride == 2:
            self.pooling = nn.AvgPool2d(2, 2)
        self.shortcut1 = self._build_act(shortcut, in_channels)
        self.conv_3x3 = BLConv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.nonlinear11 = self._build_act(nonlinear[0], in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.nonlinear12 = self._build_act(nonlinear[1], in_channels)
        self.shortcut2 = self._build_act(shortcut, out_channels)
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
        else:
            return act_name_map[act_name]()

    def forward(self, x):
        # conv 3x3 (降采样)
        if self.stride == 2:
            identity = self.pooling(x)
        else:
            identity = x
        identity = self.shortcut1(identity)
        out1 = self.conv_3x3(x)
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
        out2 = self.conv_1x1(out1)
        out2 = self.nonlinear21(out2)
        out2 = self.bn2(out2)
        out2 += identity
        out2 = self.nonlinear22(out2)

        return out2