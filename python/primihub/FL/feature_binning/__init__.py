"""
Federated Learning Feature Binning Module
联邦学习特征装仓模块

支持联邦场景下的特征分箱
"""

from .base import (
    FeatureBinningBase,
    EqualWidthBinning,
    EqualFrequencyBinning,
    OptimalBinning,
    WOEBinning,
)
from .host import FeatureBinningHost
from .guest import FeatureBinningGuest

__all__ = [
    "FeatureBinningBase",
    "EqualWidthBinning",
    "EqualFrequencyBinning",
    "OptimalBinning",
    "WOEBinning",
    "FeatureBinningHost",
    "FeatureBinningGuest",
]
