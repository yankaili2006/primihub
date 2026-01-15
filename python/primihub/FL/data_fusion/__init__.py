"""
Federated Learning Data Fusion Module
联邦学习数据融合模块

支持多种数据融合策略:
- 横向联邦数据融合 (HFL)
- 纵向联邦数据融合 (VFL)
- 安全多方数据融合
"""

from .base import (
    DataFusionBase,
    HorizontalDataFusion,
    VerticalDataFusion,
    SecureDataFusion,
)
from .host import DataFusionHost
from .guest import DataFusionGuest
from .coordinator import DataFusionCoordinator

__all__ = [
    "DataFusionBase",
    "HorizontalDataFusion",
    "VerticalDataFusion",
    "SecureDataFusion",
    "DataFusionHost",
    "DataFusionGuest",
    "DataFusionCoordinator",
]
