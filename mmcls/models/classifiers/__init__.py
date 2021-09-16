# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseClassifier
from .image import ImageClassifier
from .distiller import DistillingImageClassifier

__all__ = ['BaseClassifier', 'ImageClassifier']

__all__ += ['DistillingImageClassifier']
