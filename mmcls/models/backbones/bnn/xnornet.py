# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init)
from mmcv.utils.parrots_wrapper import _BatchNorm

from ...builder import BACKBONES
from ..base_backbone import BaseBackbone

from .bricks.acts import build_act
from .blocks.xnor_block import XnorBlock, XnorWBlock


@BACKBONES.register_module()
class XnorNet(BaseBackbone):
    """XNORNet

    Args:
        arch (string): Network architecture.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Output channels of the stem layer. Default: 64.
        base_channels (int): Middle channels of the first stage. Default: 64.
        num_stages (int): Stages of the network. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages.
            Default: ``(3, )``.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.

    Example:
        >>> from mmcls.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        'xnornet_1': (XnorBlock, (2, 2, 2, 2)),
        'xnornet_float': (XnorWBlock, (2, 2, 2, 2)),
    }

    def __init__(self,
                 arch,
                 binary_type=(True, False),
                 stem_act='relu',
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(3, ),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ],
                 **kwargs):
        super(XnorNet, self).__init__(init_cfg)

        self.arch = arch
        if self.arch not in self.arch_settings:
            raise KeyError(f'invalid arch {arch} for XnorNet')
        self.binary_type = binary_type
        self.stem_act_name = stem_act
 
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        # 设置block类型和每个stage的block数
        self.block, stage_blocks = self.arch_settings[arch]
        self.stage_blocks = stage_blocks[:num_stages]

        self._make_stem_layer(in_channels, stem_channels)

        self.layers = nn.ModuleList()
        _in_channels = stem_channels
        _out_channels = base_channels
        # 对每个stage循环
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            # 对当前stage中的每个block循环
            for j in range(num_blocks):
                # 只有每个stage第一个block的stride可以不是1
                if j >= 1: stride = 1
                # 如果当前是升维降采样block，获得shortcut模块
                if stride != 1 or _in_channels != _out_channels:
                    downsample = self._build_downsample(_in_channels, _out_channels, stride, avg_down)
                else:
                    downsample = None
                self.layers.append(
                    self.block(
                        in_channels=_in_channels,
                        out_channels=_out_channels,
                        stride=stride,
                        dilation=dilation,
                        downsample=downsample,
                        binary_type=binary_type,
                        **kwargs
                    ))
                _in_channels = _out_channels
            # 下一个stage的输出通道数翻倍
            _out_channels *= 2

        self._freeze_stages()

        # self.feat_dim = res_layer[-1].out_channels

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)
    
    def _build_downsample(self, in_channels, out_channels, stride, avg_down=False):
        downsample = []
        conv_stride = stride
        if avg_down and stride != 1:
            conv_stride = 1
            downsample.append(
                nn.AvgPool2d(
                    kernel_size=stride,
                    stride=stride,
                    ceil_mode=True,
                    count_include_pad=False))
        downsample.extend([
            build_conv_layer(
                self.conv_cfg,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=conv_stride,
                bias=False),
            build_norm_layer(self.norm_cfg, out_channels)[1]
        ])
        downsample = nn.Sequential(*downsample)

        return downsample

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                ConvModule(
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.stem_act = build_act(self.stem_act_name, stem_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        super(XnorNet, self).init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress zero_init_residual if use pretrained model.
            return

        # if self.zero_init_residual:
        #     for m in self.modules():
        #         # if isinstance(m, Bottleneck):
        #         #     constant_init(m.norm3, 0)
        #         # elif isinstance(m, BasicBlock):
        #         #     constant_init(m.norm2, 0)
        #         constant_init(m.norm2, 0)

    def forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.stem_act(x)
        x = self.maxpool(x)

        for layer in self.layers:
            x = layer(x)

        return (x, )

    def train(self, mode=True):
        super(XnorNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
