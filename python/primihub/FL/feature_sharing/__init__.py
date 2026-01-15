"""
Federated Learning Feature Sharing Module
联邦学习特征分享模块

支持安全的特征分享和交换
"""

from .base import (
    FeatureSharingBase,
    SecureFeatureSharing,
    PartialFeatureSharing,
    FeatureAggregation,
)
from .host import FeatureSharingHost
from .guest import FeatureSharingGuest

__all__ = [
    "FeatureSharingBase",
    "SecureFeatureSharing",
    "PartialFeatureSharing",
    "FeatureAggregation",
    "FeatureSharingHost",
    "FeatureSharingGuest",
]
