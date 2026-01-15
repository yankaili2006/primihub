"""
Federated Learning Feature Encoding Module
联邦学习特征编码模块

支持多种特征编码方式
"""

from .base import (
    FLFeatureEncoderBase,
    FLOneHotEncoder,
    FLLabelEncoder,
    FLTargetEncoder,
    FLHashEncoder,
    FLEmbeddingEncoder,
)
from .host import FeatureEncodingHost
from .guest import FeatureEncodingGuest

__all__ = [
    "FLFeatureEncoderBase",
    "FLOneHotEncoder",
    "FLLabelEncoder",
    "FLTargetEncoder",
    "FLHashEncoder",
    "FLEmbeddingEncoder",
    "FeatureEncodingHost",
    "FeatureEncodingGuest",
]
