"""
Federated Learning Feature Alignment Module
联邦学习特征对齐模块

支持跨方特征对齐和映射
"""

from .base import (
    FeatureAlignmentBase,
    StatisticalAlignment,
    DistributionAlignment,
    SchemaAlignment,
)
from .host import FeatureAlignmentHost
from .guest import FeatureAlignmentGuest

__all__ = [
    "FeatureAlignmentBase",
    "StatisticalAlignment",
    "DistributionAlignment",
    "SchemaAlignment",
    "FeatureAlignmentHost",
    "FeatureAlignmentGuest",
]
