"""
Federated Learning Sample Expansion Module
联邦学习样本列扩展模块

支持特征扩展和样本增强
"""

from .base import (
    SampleExpansionBase,
    PolynomialExpansion,
    InteractionExpansion,
    CrossFeatureExpansion,
)
from .host import SampleExpansionHost
from .guest import SampleExpansionGuest

__all__ = [
    "SampleExpansionBase",
    "PolynomialExpansion",
    "InteractionExpansion",
    "CrossFeatureExpansion",
    "SampleExpansionHost",
    "SampleExpansionGuest",
]
