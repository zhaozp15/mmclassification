# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseClassifier
from .image import ImageClassifier
from .distiller import DistillingImageClassifier
from .image_sim import ImageClassifierSim
from .image_sim_wo import ImageClassifierSimWO

__all__ = ['BaseClassifier', 'ImageClassifier']

__all__ += ['DistillingImageClassifier', 'ImageClassifierSim', 'ImageClassifierSimWO']
