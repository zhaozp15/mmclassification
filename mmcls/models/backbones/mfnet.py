import logging

import torch.nn as nn

from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init)
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.runner import load_checkpoint

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from .binary_utils.multifea_blocks import MF1Block, MF11Block, MF12Block, MF3Block, MF1s1Block, MF1s2Block, MF6Block, MF1terBlock, MF1LearnableBlock


@BACKBONES.register_module()
class MFNet(BaseBackbone):
    """MobileNet architecture"""

    # Parameters to build layers. 4 parameters are needed to construct a
    # layer, from left to right: channel, num_blocks, stride.
    arch_settings = {
        'mf_1': (MF1Block, [[64, 2, 2], [128, 3, 2], [256, 7, 2], [512, 1, 1], [1024, 1, 2]]),
        'mf_1s1': (MF1s1Block, [[64, 2, 2], [128, 3, 2], [256, 7, 2], [512, 1, 1], [1024, 1, 2]]),
        'mf_1s2': (MF1s2Block, [[64, 2, 2], [128, 3, 2], [256, 7, 2], [512, 1, 1], [1024, 1, 2]]),
        'mf_2': (MF1Block, [[64, 2, 2], [128, 2, 2], [256, 3, 2], [512, 1, 1], [1024, 1, 2]]),
        'mf_2_c2': (MF11Block, [[90, 2, 2], [180, 2, 2], [360, 3, 2], [720, 1, 1], [1440, 1, 2]]),
        'mf_3': (MF3Block, [[64, 2, 2], [128, 3, 2], [256, 7, 2], [512, 1, 1], [1024, 1, 2]]),
        'mf_4': (MF1Block, [[64, 2, 2], [128, 4, 2], [256, 8, 2], [512, 1, 1], [1024, 2, 2]]),
        'mf_5': (MF1Block, [[64, 2, 2], [128, 5, 2], [256, 11, 2], [512, 1, 1], [1024, 1, 2]]),
        'mf_5a': (MF1Block, [[64, 2, 2], [128, 8, 2], [256, 8, 2], [512, 1, 1], [1024, 1, 2]]),
        'mf_5b': (MF1Block, [[64, 3, 2], [128, 6, 2], [256, 9, 2], [512, 1, 1], [1024, 1, 2]]),
        'mf_5c': (MF1Block, [[64, 6, 2], [128, 6, 2], [256, 6, 2], [512, 1, 1], [1024, 1, 2]]),
        'mf_5d': (MF1Block, [[64, 2, 2], [128, 6, 2], [256, 10, 2], [512, 1, 1], [1024, 1, 2]]),
        'mf_5_c2': (MF12Block, [[90, 2, 2], [180, 5, 2], [360, 11, 2], [720, 1, 1], [1440, 1, 2]]),
        'mf_5_c4': (MF12Block, [[128, 2, 2], [256, 5, 2], [512, 11, 2], [512, 1, 1], [1024, 1, 2]]),
        'mf_5_c2_wodprelu': (MF11Block, [[90, 2, 2], [180, 5, 2], [360, 11, 2], [720, 1, 1], [1440, 1, 2]]),
        'mf_5_c4_wodprelu': (MF11Block, [[128, 2, 2], [256, 5, 2], [512, 11, 2], [512, 1, 1], [1024, 1, 2]]),
        'mf_6': (MF12Block, [[64, 2, 2], [128, 3, 2], [256, 9, 2], [512, 1, 1], [1024, 1, 2]]),
        'mf_6_mfg': (MF6Block, [[64, 2, 2], [128, 3, 2], [256, 9, 2], [512, 1, 1], [1024, 1, 2]]),
        'mf_5_ter': (MF1terBlock, [[64, 2, 2], [128, 5, 2], [256, 11, 2], [512, 1, 1], [1024, 1, 2]]),
        'mf_4_sim': (MF1LearnableBlock, [[64, 2, 2], [128, 4, 2], [256, 8, 2], [512, 1, 1], [1024, 2, 2]]),
    }

    def __init__(self,
                 arch,
                 binary_type=(True, False),
                 fea_num=2,
                 fexpand_mode='5',
                 thres=(-0.55, 0.55),
                 block_act=('prelu', 'identity'),
                 shortcut_act='identity',
                 af_act='identity',
                 binary_type_cfg=None,
                 out_indices=(4,),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 stem_conv_ks=7,
                 num_stages=5,
                 in_channels=3,
                 stem_channels=32,
                 norm_eval=False,
                 with_cp=False,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ],
                 **kwargs):
        super(MFNet, self).__init__(init_cfg)
        if frozen_stages not in range(-1, 6):
            raise ValueError('frozen_stages must be in range(-1, 6). '
                             f'But received {frozen_stages}')
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 5
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.in_channels = stem_channels

        self.stem_conv = build_conv_layer(
            self.conv_cfg,
            in_channels,
            stem_channels,
            kernel_size=stem_conv_ks,
            stride=2,
            padding=stem_conv_ks // 2,
            bias=False)
        self.stem_act = nn.PReLU(stem_channels)
        self.stem_bn = nn.BatchNorm2d(stem_channels)

        self.layers = nn.ModuleList()
        self.block, self.layers_cfg = self.arch_settings[arch]
        in_channels = self.in_channels
        for i, layer_cfg in enumerate(self.layers_cfg):
            out_channels, num_blocks, stride = layer_cfg
            for j in range(num_blocks):
                if j >= 1: stride = 1
                # breakpoint()
                self.layers.append(
                    self.block(
                        in_channels,
                        out_channels,
                        stride,
                        binary_type=binary_type,
                        fea_num=fea_num,
                        fexpand_mode=fexpand_mode,
                        thres=thres,
                        nonlinear=block_act,
                        shortcut=shortcut_act,
                        ahead_fexpand=af_act,
                        **kwargs))
                # breakpoint
                in_channels = out_channels
        
        self._freeze_stages()

    def make_layer(self, out_channels, num_blocks, stride, binary_type=(True, False), **kwargs):
        layers = []
        for i in range(num_blocks):
            if isinstance(binary_type[0], bool):
                block_binary_type = binary_type
            elif isinstance(binary_type[0], tuple):
                block_binary_type = binary_type[i]
            if i >= 1:
                stride = 1
            layers.append(
                self.block(
                    self.in_channels,
                    out_channels,
                    stride,
                    binary_type=block_binary_type,
                    **kwargs))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def init_weights(self, pretrained=None):
        super(MFNet, self).init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            return

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.stem_act(x)
        x = self.stem_bn(x)

        # only for compute cos_sim loss
        if self.block == MF1LearnableBlock:
            cos_sim = 0
            for layer in self.layers:
                x, layer_cos_sim = layer(x)
                cos_sim += layer_cos_sim

            return cos_sim, (x, )

        for layer in self.layers:
            x = layer(x)

        return (x, )

    def _freeze_stages(self):
        if self.frozen_stages != -1:
            for param in self.stem_conv.parameters():
                param.requires_grad = False
            for param in self.stem_bn.parameters():
                param.requires_grad = False
            for param in self.stem_act.parameters():
                param.requires_grad = False
            for i in range(1, self.frozen_stages + 1):
                layer = getattr(self, f'layer{i}')
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(MFNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
