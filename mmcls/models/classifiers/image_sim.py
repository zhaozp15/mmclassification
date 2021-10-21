# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings

from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from ..utils.augment import Augments
from .image import ImageClassifier

warnings.simplefilter('once')


@CLASSIFIERS.register_module()
class ImageClassifierSim(ImageClassifier):

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 loss_sim=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super(ImageClassifierSim, self).__init__(backbone, neck, head, 
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

        # losses.update(loss)
        losses['loss_cls'] = loss['loss']

        losses['loss_sim'] = cos_sim * self.loss_sim_weight

        return losses

    def simple_test(self, img, img_metas):
        """Test without augmentation."""
        x = self.extract_feat(img)

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
