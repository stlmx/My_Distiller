# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseClassifier
from .hugging_face import HuggingFaceClassifier
from .image import ImageClassifier
from .timm import TimmClassifier
from .distillation import Distiller

__all__ = [
    'BaseClassifier', 'ImageClassifier', 'TimmClassifier',
    'HuggingFaceClassifier', 'Distiller'
]
