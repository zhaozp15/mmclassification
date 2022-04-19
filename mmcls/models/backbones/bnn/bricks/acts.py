import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class IRNetSign(Function):
    """Sign function from IR-Net, which can add EDE progress"""
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None


class RANetActSign(nn.Module):
    """ReActNet's activation sign function"""
    def __init__(self):
        super(RANetActSign, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out


class RANetWSign(nn.Module):
    """ReActNet's weight sign function"""
    def __init__(self, clip=1):
        super(RANetWSign, self).__init__()
        self.clip = clip

    def forward(self, x):
        binary_weights_no_grad = torch.sign(x)
        cliped_weights = torch.clamp(x, -self.clip, self.clip)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights

        return binary_weights


class TernarySign(nn.Module):
    """ternary sign function"""
    def __init__(self, thres=(-0.5, 0.5)):
        super(TernarySign, self).__init__()
        self.thres = thres

    def forward(self, x):
        out_forward = 0.5 * torch.sign(x + self.thres[0]) + 0.5 + torch.sign(x - self.thres[1])
        mask1 = (x < -1).float()
        mask2 = (x < self.thres[0]).float()
        mask3 = (x < 0).float()
        mask4 = (x < self.thres[1]).float()
        mask5 = (x < 1).float()
        out1 = (-1) * mask1 + (2*x*x + 4*x + 1) * (1 - mask1)
        out2 = out1 * mask2 + -2*x*x * (1 - mask2)
        out3 = out2 * mask3 + 2*x*x * (1 - mask3)
        out4 = out3 * mask4 + (-2*x*x + 4*x -1) * (1 - mask4)
        out5 = out4 * mask5 + 1 * (1 - mask4)

        out = out_forward.detach() - out5.detach() + out5

        return out


class STESign(nn.Module):
    """a sign function using STE"""
    def __init__(self, clip=1):
        super(STESign, self).__init__()
        assert clip > 0
        self.clip = clip

    def forward(self, x):
        out_no_grad = torch.sign(x)
        cliped_out = torch.clamp(x, -self.clip, self.clip)
        out = out_no_grad.detach() - cliped_out.detach() + cliped_out

        return out


class LearnableBias(nn.Module):
    def __init__(self, channels, init=0, groups=1):
        super(LearnableBias, self).__init__()
        self.groups = groups
        if groups == 1:
            self.learnable_bias = nn.Parameter(torch.ones(1, channels, 1, 1) * init, requires_grad=True)
        else:
            learnable_bias_list = [nn.Parameter(torch.ones(1, channels // groups, 1, 1) * init, requires_grad=True) for i in range(groups)]
            self.learnable_bias = torch.cat(learnable_bias_list, dim=1)

    def forward(self, x):
        out = x + self.learnable_bias.expand_as(x)
        return out


class LearnableScale(nn.Module):
    def __init__(self, channels, init=1.0):
        super(LearnableScale, self).__init__()
        self.channels = channels
        self.learnable_scale = nn.Parameter(torch.ones(1, channels, 1, 1) * init, requires_grad=True)

    def forward(self, x):
        out = x * self.learnable_scale.expand_as(x)

        return out


class ScaleSum(nn.Module):
    def __init__(self, channels, init=1.0):
        super(ScaleSum, self).__init__()
        self.channels = channels
        self.learnable_scale = nn.Parameter(torch.ones(1, channels, 1, 1) * init, requires_grad=True)

    def forward(self, x):
        scaled = x * self.learnable_scale
        sum = scaled.sum(dim=1, keepdim=True) / self.channels
        out = sum.expand_as(x)

        return out


class LSaddSS(nn.Module):
    def __init__(self, channels, init=1.0):
        super(LSaddSS, self).__init__()
        self.channels = channels
        self.scale = LearnableScale(channels)
        self.scale_sum = ScaleSum(channels)

    def forward(self, x):
        out = self.scale(x) / 2 + self.scale_sum(x) / 2

        return out


class RPRelu(nn.Module):
    """RPRelu form ReActNet"""
    def __init__(self, in_channels, bias_init=0.0, prelu_init=0.25, **kwargs):
        super(RPRelu, self).__init__()
        self.bias1 = LearnableBias(in_channels, init=bias_init)
        self.prelu = nn.PReLU(in_channels, init=prelu_init)
        self.bias2 = LearnableBias(in_channels, init=bias_init)

    def forward(self, x):
        x = self.bias1(x)
        x = self.prelu(x)
        x = self.bias2(x)
        return x


class DPReLU(nn.Module):
    """ relu with double parameters """
    def __init__(self, channels, init_pos=1.0, init_neg=1.0):
        super(DPReLU, self).__init__()
        self.channels = channels
        self.w_pos = nn.Parameter(torch.ones(1, channels, 1, 1) * init_pos, requires_grad=True)
        self.w_neg = nn.Parameter(torch.ones(1, channels, 1, 1) * init_neg, requires_grad=True)

    def forward(self, x):
        # w_pos * max(0, x) + w_neg * min(0, x)
        out = self.w_pos * F.relu(x) - self.w_neg * F.relu(-x)

        return out


class NPReLU(nn.Module):
    def __init__(self, channels, init=1.0):
        super(NPReLU, self).__init__()
        self.channels = channels
        self.w_neg = nn.Parameter(torch.ones(1, channels, 1, 1) * init, requires_grad=True)

    def forward(self, x):
        # max(0, x) + w_neg * min(0, x)
        out = F.relu(x) - self.w_neg * F.relu(-x)

        return out


class PReLUsc(nn.Module):
    def __init__(self, channels, init=1.0):
        super(PReLUsc, self).__init__()
        self.channels = channels
        self.prelu = nn.PReLU(channels, init=init)

    def forward(self, x):
        out = self.prelu(x)
        out = 0.5 * x + 0.5 * out

        return out


class CfgLayer(nn.Module):
    """Configurable layer"""

    def __init__(self, in_channels, config, bias_init=0.0, prelu_init=0.25, **kwargs):
        super(CfgLayer, self).__init__()
        self.act = nn.ModuleList()
        for c in config:
            if c == 'b':
                self.act.append(LearnableBias(in_channels, init=bias_init))
            elif c == 'p':
                self.act.append(nn.PReLU(in_channels, init=prelu_init))

    def forward(self, x):
        for act in self.act:
            x = act(x)
        return x


class LearnableScale3(nn.Module):
    hw_settings = {
        64: 56,
        128: 28,
        256: 14,
        512: 7,
    }
    def __init__(self, channels):
        super(LearnableScale3, self).__init__()
        self.channels = channels
        self.height = self.hw_settings[channels]
        self.width = self.hw_settings[channels]
        self.alpha = nn.Parameter(torch.ones(1, self.channels, 1, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(1, 1, self.height, 1), requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1, 1, 1, self.width), requires_grad=True)

    def forward(self, x):
        out = x * self.alpha.expand_as(x)
        out = out * self.beta.expand_as(x)
        out = out * self.gamma.expand_as(x)

        return out

# act functions that need no arguments
act_map_0 = {
        'hardtanh': nn.Hardtanh,
        'relu': nn.ReLU,
        'prelu_one': nn.PReLU,
        'mish': Mish,
    }

# act functins that need a 'channels' argument
act_map_1 = {
        'prelu': nn.PReLU,
        'rprelu': RPRelu,
        'scale': LearnableScale,
        'scale3': LearnableScale3,
        'dprelu': DPReLU,
        'nprelu': NPReLU,
    }

def build_act(act, channels):
    if act == 'identity':
        return nn.Sequential()
    elif act == 'abs':
        return torch.abs
    elif act in act_map_0:
        return act_map_0[act]()
    elif act in act_map_1:
        return act_map_1[act](channels)
    elif act == 'prelu_pi=1':
        return nn.PReLU(channels, init=1.0)
    elif act == 'prelu_one_pi=1':
        return nn.PReLU(1, init=1.0)
    elif act == 'rprelu_pi=1':
        return RPRelu(channels, prelu_init=1.0)
    elif act == 'dprelu_pi=0.25_1':
        return DPReLU(channels, init_pos=1.0, init_neg=0.25)
    else:
        raise ValueError(f'Unsupported activation function: {act}')