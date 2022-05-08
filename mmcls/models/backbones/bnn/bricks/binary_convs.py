import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .acts import (IRNetSign, RANetActSign, RANetWSign, TernarySign,
    STESign, PolySign,
    STESignFake, PolySignFake)


class BaseBinaryConv2d(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 binary_type=(True, True),
                 **kwargs):
        super(BaseBinaryConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias)
        self.binary_type = binary_type

    def binary_weight(self, x):
        pass

    def binary_input(self, w):
        pass

    def forward(self, input):
        x = self.binary_input(input) if self.binary_type[0] else input
        w = self.binary_weight(self.weight) if self.binary_type[1] else self.weight
        output =  F.conv2d(
            x,
            w,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups)
        return output


class BLConv2d(BaseBinaryConv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 binary_type=(True, True),
                 sign_x='poly',
                 sign_w='ste',
                 scale_w_mode='mean_detach',
                 **kwargs):
        super(BLConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride, padding,
            dilation,
            groups,
            bias,
            binary_type,
            **kwargs)
        sign_methods = {
            'poly': PolySign,
            'ste': STESign,
            'poly_fake': PolySignFake,
            'ste_fake': STESignFake,
        }
        self.sign_x = sign_methods[sign_x]()
        self.sign_w = sign_methods[sign_w]()

        self.scale_w_mode = scale_w_mode
        self.scale_w_methods = {
            'mean_detach': self.scale_mean_detach,
            'mean': self.scale_mean,
            'none': self.scale_none,
        }

    def binary_input(self, x):
        return self.sign_x(x)

    def binary_weight(self, w):
        bw = self.sign_w(w)
        sw = self.scale_w_methods[self.scale_w_mode](w)
        return bw * sw

    def scale_mean_detach(self, w):
        return w.abs().mean(dim=(1, 2, 3), keepdim=True).detach()

    def scale_mean(self, w):
        return w.abs().mean(dim=(1, 2, 3), keepdim=True)
    
    def scale_none(self, w):
        return 1.0


class IRConv2d(BaseBinaryConv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 binary_type=(True, True), **kwargs):
        super(IRConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, binary_type, **kwargs)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()

    def binary_input(self, x):
        return IRNetSign().apply(x, self.k, self.t)

    def binary_weight(self, w):
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        sw = torch.pow(torch.tensor([2] * bw.size(0)).cuda().float(),
                       (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(
            bw.size(0), 1, 1, 1).detach()
        bw = IRNetSign().apply(bw, self.k, self.t)
        return bw * sw

    def ede(self, k, t):
        self.k = k
        self.t = t


class IRG3swConv2d(BaseBinaryConv2d):
    ''' group sign with the same w '''
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 binary_type=(True, True), **kwargs):
        super(IRG3swConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, binary_type, **kwargs)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()

    def binary_input(self, x):
        return IRNetSign().apply(x, self.k, self.t)

    def binary_weight(self, w):
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        sw = torch.pow(torch.tensor([2] * bw.size(0)).cuda().float(),
                       (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(
            bw.size(0), 1, 1, 1).detach()
        bw = IRNetSign().apply(bw, self.k, self.t)
        return bw * sw

    def ede(self, k, t):
        self.k = k
        self.t = t
    
    def forward(self, x):
        x_max = torch.max(abs(x))
        x1 = x - x_max / 2
        x2 = x
        x3 = x + x_max / 2
        x1 = self.binary_input(x1) if self.binary_type[0] else x1
        x2 = self.binary_input(x2) if self.binary_type[0] else x2
        x3 = self.binary_input(x3) if self.binary_type[0] else x3
        w = self.binary_weight(self.weight) if self.binary_type[1] else self.weight
        out1 =  F.conv2d(x1, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        out2 =  F.conv2d(x2, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        out3 =  F.conv2d(x3, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        out = out1 + out2 + out3
        return out


class RAConv2d(BaseBinaryConv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 binary_type=(True, True), **kwargs):
        super(RAConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, binary_type, **kwargs)
        
        self.sign_a = RANetActSign()
        self.sign_w = RANetWSign()

    def binary_input(self, x):
        return self.sign_a(x)

    def binary_weight(self, w):
        bw = self.sign_w(w)
        sw = torch.mean(torch.mean(torch.mean(abs(w),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True).detach()
        return bw * sw


class ANDConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ANDConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()

    def forward(self, input):
        w = self.weight
        a = input
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        sw = torch.pow(torch.tensor([2]*bw.size(0)).cuda().float(), (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(bw.size(0), 1, 1, 1).detach()
        bw = (IRNetSign().apply(bw, self.k, self.t) + 1.0) / 2.0
        ba = (IRNetSign().apply(a, self.k, self.t) + 1.0) / 2.0
        bw = bw * sw
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output


class BLSTEConv2d(BaseBinaryConv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 binary_type=(True, True), clip=1, **kwargs):
        super(BLSTEConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, binary_type, **kwargs)
        
        self.sign_a = STESign(clip=clip)
        self.sign_w = STESign(clip=clip)

    def binary_input(self, x):
        return self.sign_a(x)

    def binary_weight(self, w):
        bw = self.sign_w(w)
        sw = torch.mean(torch.mean(torch.mean(abs(w),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True).detach()
        return bw * sw


class STEConv2d(BaseBinaryConv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 binary_type=(True, True), clip=1, **kwargs):
        super(STEConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, binary_type, **kwargs)
        
        self.sign_a = STESign(clip=clip)
        self.sign_w = STESign(clip=clip)

    def binary_input(self, x):
        return self.sign_a(x)

    def binary_weight(self, w):
        return self.sign_w(w)


class BConvWS2d(BaseBinaryConv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 binary_type=(True, True), **kwargs):
        super(BConvWS2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, binary_type, **kwargs)
        
        self.sign_a = RANetActSign()
        self.sign_w = RANetWSign()

    def binary_input(self, x):
        return self.sign_a(x)

    def binary_weight(self, w):
        mean=  w.mean(dim=(1, 2, 3), keepdim=True)
        std = w.std(dim=(1, 2, 3), keepdim=True)
        w = (w - mean) / (std + 1e-5)
        bw = self.sign_w(w)
        sw = w.abs().mean(dim=(1, 2, 3), keepdim=True).detach()
        # sw = torch.pow(torch.tensor([2] * w.size(0)).cuda().float(),
        #                (torch.log(w.abs().view(w.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(
        #     w.size(0), 1, 1, 1).detach()

        return bw * sw


class TAConv2d(BaseBinaryConv2d):
    '''ternary weight conv'''
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 binary_type=(True, True), thres=(-0.5, 0.5), **kwargs):
        super(TAConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, binary_type, **kwargs)
        
        self.sign_a = TernarySign(thres)
        self.sign_w = RANetWSign()

    def binary_input(self, x):
        return self.sign_a(x)

    def binary_weight(self, w):
        bw = self.sign_w(w)
        sw = w.abs().mean(dim=(1, 2, 3), keepdim=True).detach()
        # sw = torch.mean(torch.mean(torch.mean(abs(w),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True).detach()
        return bw * sw