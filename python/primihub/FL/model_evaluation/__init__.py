"""
Federated Model Evaluation Module
联邦学习模型评估模块

提供联邦学习场景下的模型评估功能，支持分类和回归模型的多种评估指标。

支持的评估类型：
- 分类评估: accuracy, f1, precision, recall, auc, roc, ks
- 回归评估: mse, rmse, mae, r2, mape 等
"""

from .base import (
    FLEvaluatorBase,
    ClassificationEvaluator,
    RegressionEvaluator,
)
from .host import ModelEvaluationHost
from .guest import ModelEvaluationGuest

__all__ = [
    "FLEvaluatorBase",
    "ClassificationEvaluator",
    "RegressionEvaluator",
    "ModelEvaluationHost",
    "ModelEvaluationGuest",
]
