# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict

import torch
import torch.distributed as dist

from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from ..utils.augment import Augments
from .image import ImageClassifier

warnings.simplefilter('once')


@CLASSIFIERS.register_module()
class ImageClassifierSimWO(ImageClassifier):
    '''
    ImageClassifierSim中相似度命名为loss_sim，会被_parse_losses默认算作总loss的一部分
    ImageClassifierSimWO中相似度命名为为sim，不会被默认算入总loss中
    '''

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 loss_sim=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super(ImageClassifierSimWO, self).__init__(backbone, neck, head, 
                                                 pretrained, train_cfg, init_cfg)
        self.loss_sim_weight = loss_sim.loss_weight

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck."""
        cos_sim, x = self.backbone(img)
        if self.return_tuple:
            if not isinstance(x, tuple):
                x = (x, )
                warnings.warn(
                    'We will force all backbones to return a tuple in the '
                    'future. Please check your backbone and wrap the output '
                    'as a tuple.', DeprecationWarning)
        else:
            if isinstance(x, tuple):
                x = x[-1]
        if self.with_neck:
            x = self.neck(x)
        return cos_sim, x

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                shoulf be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)

        cos_sim, x = self.extract_feat(img)

        losses = dict()
        try:
            loss = self.head.forward_train(x, gt_label)
        except TypeError as e:
            if 'not tuple' in str(e) and self.return_tuple:
                return TypeError(
                    'Seems the head cannot handle tuple input. We have '
                    'changed all backbones\' output to a tuple. Please '
                    'update your custom head\'s forward function. '
                    'Temporarily, you can set "return_tuple=False" in '
                    'your backbone config to disable this feature.')
            raise e

        losses.update(loss)
        losses['loss_cls'] = losses.pop('loss')
        losses['sim'] = cos_sim * self.loss_sim_weight

        return losses

    def simple_test(self, img, img_metas):
        """Test without augmentation."""
        cos_sim, x = self.extract_feat(img)

        try:
            res = self.head.simple_test(x)
        except TypeError as e:
            if 'not tuple' in str(e) and self.return_tuple:
                return TypeError(
                    'Seems the head cannot handle tuple input. We have '
                    'changed all backbones\' output to a tuple. Please '
                    'update your custom head\'s forward function. '
                    'Temporarily, you can set "return_tuple=False" in '
                    'your backbone config to disable this feature.')
            raise e

        return res

    # def _parse_losses_sim(self, losses):
    #     log_vars = OrderedDict()
    #     for loss_name, loss_value in losses.items():
    #         if isinstance(loss_value, torch.Tensor):
    #             log_vars[loss_name] = loss_value.mean()
    #         elif isinstance(loss_value, list):
    #             log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
    #         elif isinstance(loss_value, dict):
    #             for name, value in loss_value.items():
    #                 log_vars[name] = value
    #         else:
    #             raise TypeError(
    #                 f'{loss_name} is not a tensor or list of tensors')

    #     # 计算包含特征相似度的loss
    #     loss = log_vars['loss_cls'] + log_vars['sim']

    #     log_vars['loss'] = loss
    #     for loss_name, loss_value in log_vars.items():
    #         # reduce loss when distributed training
    #         if dist.is_available() and dist.is_initialized():
    #             loss_value = loss_value.data.clone()
    #             dist.all_reduce(loss_value.div_(dist.get_world_size()))
    #         log_vars[loss_name] = loss_value.item()

    #     return loss, log_vars