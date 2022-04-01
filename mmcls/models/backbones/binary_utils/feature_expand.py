from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F
from .functions import LearnableBias, LearnableScale, RANetActSign
from .binary_convs import BLConv2d

from scipy.stats import norm

class CopyBias2_1(Function):
    """
    一个(n,c,h,w)特征用两个bias复制成两个
    并修改反传梯度计算函数
    """
    @staticmethod
    def forward(ctx, x, thres):
        # ctx.save_for_backward(input, k, t)
        y = torch.cat(x + thres[0], x + thres[1])
        return y

    @staticmethod
    def backward(ctx, dy):
        dy1, dy2 = dy.chunk(chunks=2, dim=1)
        # input, k, t = ctx.saved_tensors
        dx = dy1 + dy2
        mask = dy1 > 0 and dy2 < 0
        dx[mask] = 0
        return dx, None

class CopyBias2_2(Function):
    """
    一个(n,c,h,w)特征用两个bias复制成两个
    并修改反传梯度计算函数
    """
    @staticmethod
    def forward(ctx, x, thres):
        # ctx.save_for_backward(input, k, t)
        y = torch.cat(x + thres[0], x + thres[1])
        return y

    @staticmethod
    def backward(ctx, dy):
        dy1, dy2 = dy.chunk(chunks=2, dim=1)
        # input, k, t = ctx.saved_tensors
        dx = dy1 + dy2
        mask = dy1 > 0 and dy2 < 0 and dx < 0
        dx[mask] = 0
        return dx, None

class FeaExpand(nn.Module):
    """expand feature map

    mode:
        1: 根据特征图绝对值最大值均匀选择阈值
        1c: 1的基础上分通道
        1c-m: 根据特征图的最大值和最小值,分通道计算阈值
        1nc-m: 根据特征图的最大值和最小值，分输入分通道计算阈值
        2: 仅限于2张特征图，第1张不变，第2张绝对值小的映射为+1，绝对值大的映射为-1
        3: 根据均值方差选择阈值（一个batch中的所有图片计算一个均值和方差）
        3n: 3的基础上分输入，每个输入图片计算自己的均值方差
        3c: 3的基础上分通道计算均值方差（类似bn）
        3nc: 3的基础上既分输入也分通道计算均值方差
        4: 使用1的值初始化的可学习的阈值
        4c: 4的基础上分通道
        4g*: 4的基础上分组（是4和4c的折中）
        4s: 可学习的是阈值的系数，而不是阈值本身
        4s-a: 可学习系数使用sigmoid函数计算，多个阈值共用一个系数
        4sc-a: 4s-a的分通道版本
        4s-a-n: 4s-a的每个阈值拥有自己的可学习系数的版本
        4sc-b: 输入特征图先经过分通道的可学习scale，再使用固定阈值
        5: 手动设置阈值
        5re: 专门针对expand为2的情况，将负阈值得到的结果乘-1
        5-3: 在5的基础上增加一份特征图，其中绝对值小的数映射为+1，绝对值大的映射为-1
        5-mean: 以特征图的均值为轴对称应用手动设置的阈值
        5-bp: 将反向传播中不可能实现的情况的梯度置0
        6: 按照数值的个数均匀选择阈值，由直方图计算得到
        7: 根据输入计算的自适应阈值
        8: 使用conv进行通道数扩增
        8b: 在8的基础上增加bn层
        8ab: 在8的基础上增加bn层和激活层，顺序为先激活后bn
        8ba: 在8的基础上增加bn层和激活层，顺序为先bn后激活
        82: 仅限于2张特征图，第1张不变，第2张使用conv计算得到
        8bin: 使用二值conv进行通道数扩增
        9-symmetric: 2个阈值通过余弦相似度得到单独的loss进行学习，2个阈值是对称的
        9-independent: 2个阈值是独立的
        10-symmetric: 可学习阈值，初始值通过thres传入，2个阈值是对称的
        10c-symmetric: 在10-symmetric的基础上分通道
        11a: loss_sim只与thres相关，loss_cls只与weight相关
             loss_sim = f(thres)  loss_cls = f(weight)
        11b: loss_sim只与thres相关，loss_cls与thres和weight都相关
             loss_sim = f(thres)  loss_cls = f(weight, thres)
        11c: loss_sim和loss_cls与weight和thres都相关
             loss_sim = f(weight, thres)  loss_cls = f(weight, thres) 
    """

    def __init__(self, expansion=3, mode='1', in_channels=None, thres=None):
        super(FeaExpand, self).__init__()
        self.expansion = expansion
        self.mode = mode
        self.in_channels = in_channels
        self.thres = thres

        self.mode_methods = {
            '1': (self.init_1, self.forward_1),
            '5': (self.init_5, self.forward_5),
            '9-symmetric': (self.init_9_symmetric, self.forward_9_symmetric),
            '10-symmetric': (self.init_10_symmetric, self.forward_10_symmetric),
            '10c-symmetric': (self.init_10c_symmetric, self.forward_10_symmetric),
            '10sim-symmetric': (self.init_9_symmetric, self.forward_10sim_symmetric),
        }

        try:
            # 根据mode设定需要的成员变量
            self.mode_methods[mode][0]()
        except KeyError:
            print(f"init method of mode {self.mode} isn't implenment")

    def forward(self, x):
        if self.expansion == 1 and self.mode != '5':
            return x

        try:
            # 根据mode执行前传过程，要求返回值是一个张量列表
            # 或者返回元组的最后一个元素是张量列表
            out = self.mode_methods[self.mode][1](x)
            
        except KeyError:
            print(f"forward method of mode {self.mode} isn't implenment")
              
        if isinstance(out, list):
            return torch.cat(out, dim=1)
        else:
            return tuple(out[:-1]) + (torch.cat(out[-1], dim=1), )

    # '1'
    def init_1(self):
        self.alpha = [-1 + (i + 1) * 2 / (self.expansion + 1) for i in range(self.expansion)]
    
    def forward_1(self, x):
        x_max = x.abs().max()
        out = [x + alpha * x_max for alpha in self.alpha]
        return out
    
    # '5'
    def init_5(self):
        assert len(self.thres) == self.expansion
    
    def forward_5(self, x):
        return [x + t for t in self.thres]

    # '9-symmetric'
    def init_9_symmetric(self):
        init = self.thres[0]
        # learnable_thres
        self.lt = nn.Parameter(torch.ones(1, 1, 1, 1) * init, requires_grad=True)
        self.sign1 = RANetActSign()
        self.sign2 = RANetActSign()
    
    def forward_9_symmetric(self, x):
        # 单独复制出特征图用于计算相似度loss
        N = x.shape[0]
        C = x.shape[1]
        x_detach = x.detach()
        fea_bin1 = self.sign1(x_detach + self.lt).reshape(N, C, -1)
        fea_bin2 = self.sign2(x_detach - self.lt).reshape(N, C, -1)
        cos_sim = F.cosine_similarity(fea_bin1, fea_bin2, dim=2)
        cos_sim = cos_sim.abs().sum() / N

        fea1 = x + self.lt.detach()
        fea2 = x - self.lt.detach()
        out = [fea1, fea2]
        return cos_sim, out

    # '10-symmetric'
    def init_10_symmetric(self):
        init = self.thres[0]
        # learnable_thres
        self.lt = nn.Parameter(torch.ones(1, 1, 1, 1) * init, requires_grad=True)
    
    def forward_10_symmetric(self, x):
        out = [x + self.lt, x - self.lt]
        return out
    
    # '10c-symmetric'
    def init_10c_symmetric(self):
        init = self.thres[0]
        # learnable_thres
        self.lt = nn.Parameter(torch.ones(1, self.in_channels, 1, 1) * init, requires_grad=True)
    
    # '10sim-symmetric'
    def forward_10sim_symmetric(self, x):
        N = x.shape[0]
        C = x.shape[1]
        fea_bin1 = self.sign1(x + self.lt).reshape(N, C, -1)
        fea_bin2 = self.sign2(x - self.lt).reshape(N, C, -1)
        cos_sim = F.cosine_similarity(fea_bin1, fea_bin2, dim=2)
        cos_sim = cos_sim.abs().sum() / N

        out = [x + self.lt, x - self.lt]
        return cos_sim, out
