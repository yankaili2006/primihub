"""
Federated Learning Feature Similarity Module
联邦学习特征相似度分析模块

支持跨方特征相似度计算和分析
"""

from .base import (
    FeatureSimilarityBase,
    CosineSimilarity,
    PearsonCorrelation,
    MutualInformation,
    JaccardSimilarity,
)
from .host import FeatureSimilarityHost
from .guest import FeatureSimilarityGuest

__all__ = [
    "FeatureSimilarityBase",
    "CosineSimilarity",
    "PearsonCorrelation",
    "MutualInformation",
    "JaccardSimilarity",
    "FeatureSimilarityHost",
    "FeatureSimilarityGuest",
]
