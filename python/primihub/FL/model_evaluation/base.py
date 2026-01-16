"""
Federated Model Evaluation Base Classes
联邦模型评估基础类

实现联邦学习场景下的模型评估功能。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import (
    # Classification metrics
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    log_loss,
    # Regression metrics
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

logger = logging.getLogger(__name__)


class FLEvaluatorBase(ABC):
    """
    联邦模型评估基类

    定义评估的基本接口。
    """

    def __init__(self, task_type: str = "classification"):
        """
        初始化评估器

        Args:
            task_type: 任务类型 (classification, regression)
        """
        self.task_type = task_type
        self._metrics: Dict[str, Any] = {}

    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                 **kwargs) -> Dict[str, Any]:
        """
        评估模型

        Args:
            y_true: 真实标签
            y_pred: 预测值/预测概率

        Returns:
            评估指标字典
        """
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """获取评估指标"""
        return self._metrics

    def _validate_inputs(self, y_true: np.ndarray, y_pred: np.ndarray):
        """验证输入数据"""
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred)

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Length mismatch: y_true has {len(y_true)} samples, "
                f"y_pred has {len(y_pred)} samples"
            )

        return y_true, y_pred


class ClassificationEvaluator(FLEvaluatorBase):
    """
    分类模型评估器

    支持二分类和多分类评估。
    """

    # 支持的指标
    SUPPORTED_METRICS = [
        "accuracy", "acc",
        "f1", "f1_score",
        "precision",
        "recall",
        "auc", "roc_auc",
        "roc",
        "ks",
        "log_loss", "logloss",
        "confusion_matrix",
        "classification_report",
    ]

    def __init__(self, multiclass: bool = False, threshold: float = 0.5):
        """
        初始化分类评估器

        Args:
            multiclass: 是否是多分类问题
            threshold: 二分类阈值
        """
        super().__init__(task_type="classification")
        self.multiclass = multiclass
        self.threshold = threshold

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: Optional[np.ndarray] = None,
        metrics: Optional[List[str]] = None,
        prefix: str = "",
    ) -> Dict[str, Any]:
        """
        评估分类模型

        Args:
            y_true: 真实标签
            y_pred: 预测类别（如果y_score不为None，则忽略）
            y_score: 预测概率/得分
            metrics: 要计算的指标列表
            prefix: 指标名称前缀

        Returns:
            评估指标字典
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)

        if metrics is None:
            metrics = ["accuracy", "f1", "precision", "recall", "auc"]

        # 如果提供了概率得分，根据阈值计算预测类别
        if y_score is not None:
            y_score = np.asarray(y_score)
            if self.multiclass:
                y_pred = np.argmax(y_score, axis=1)
            else:
                y_pred = (y_score > self.threshold).astype(int)
        else:
            y_score = y_pred  # 用于需要概率的指标

        self._metrics = {}

        for metric in metrics:
            metric_lower = metric.lower()
            key = f"{prefix}{metric}" if prefix else metric

            try:
                if metric_lower in ["accuracy", "acc"]:
                    self._metrics[key] = float(accuracy_score(y_true, y_pred))

                elif metric_lower in ["f1", "f1_score"]:
                    avg = "macro" if self.multiclass else "binary"
                    self._metrics[key] = float(f1_score(y_true, y_pred, average=avg))

                elif metric_lower == "precision":
                    avg = "macro" if self.multiclass else "binary"
                    self._metrics[key] = float(precision_score(y_true, y_pred, average=avg))

                elif metric_lower == "recall":
                    avg = "macro" if self.multiclass else "binary"
                    self._metrics[key] = float(recall_score(y_true, y_pred, average=avg))

                elif metric_lower in ["auc", "roc_auc"]:
                    if y_score is not None:
                        if self.multiclass:
                            self._metrics[key] = float(
                                roc_auc_score(y_true, y_score, multi_class="ovr")
                            )
                        else:
                            self._metrics[key] = float(roc_auc_score(y_true, y_score))

                elif metric_lower == "roc" and not self.multiclass:
                    if y_score is not None:
                        fpr, tpr, thresholds = roc_curve(y_true, y_score)
                        thresholds[0] = float(thresholds[1] + 1.0)
                        self._metrics[f"{prefix}fpr"] = fpr.tolist()
                        self._metrics[f"{prefix}tpr"] = tpr.tolist()
                        self._metrics[f"{prefix}thresholds"] = thresholds.tolist()

                elif metric_lower == "ks" and not self.multiclass:
                    if y_score is not None:
                        pos_idx = y_true == 1
                        if np.any(pos_idx) and np.any(~pos_idx):
                            ks_stat = ks_2samp(
                                y_score[pos_idx], y_score[~pos_idx]
                            ).statistic
                            self._metrics[key] = float(ks_stat)

                elif metric_lower in ["log_loss", "logloss"]:
                    if y_score is not None:
                        self._metrics[key] = float(log_loss(y_true, y_score))

                elif metric_lower == "confusion_matrix":
                    cm = confusion_matrix(y_true, y_pred)
                    self._metrics[key] = cm.tolist()

                elif metric_lower == "classification_report":
                    report = classification_report(y_true, y_pred, output_dict=True)
                    self._metrics[key] = report

            except Exception as e:
                logger.warning(f"Failed to compute metric '{metric}': {e}")

        # 打印指标
        for key, value in self._metrics.items():
            if not isinstance(value, (list, dict)):
                logger.info(f"{key}: {value:.6f}")

        return self._metrics


class RegressionEvaluator(FLEvaluatorBase):
    """
    回归模型评估器
    """

    # 支持的指标
    SUPPORTED_METRICS = [
        "mse", "mean_squared_error",
        "rmse", "root_mean_squared_error",
        "mae", "mean_absolute_error",
        "r2", "r2_score",
        "mape", "mean_absolute_percentage_error",
        "ev", "explained_variance",
        "max_error", "maxe",
        "medae", "median_absolute_error",
        "msle", "mean_squared_log_error",
    ]

    def __init__(self):
        """初始化回归评估器"""
        super().__init__(task_type="regression")

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: Optional[List[str]] = None,
        prefix: str = "",
    ) -> Dict[str, Any]:
        """
        评估回归模型

        Args:
            y_true: 真实值
            y_pred: 预测值
            metrics: 要计算的指标列表
            prefix: 指标名称前缀

        Returns:
            评估指标字典
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)

        if metrics is None:
            metrics = ["mse", "rmse", "mae", "r2", "mape"]

        self._metrics = {}

        for metric in metrics:
            metric_lower = metric.lower()
            key = f"{prefix}{metric}" if prefix else metric

            try:
                if metric_lower in ["mse", "mean_squared_error"]:
                    self._metrics[key] = float(mean_squared_error(y_true, y_pred))

                elif metric_lower in ["rmse", "root_mean_squared_error"]:
                    self._metrics[key] = float(
                        mean_squared_error(y_true, y_pred, squared=False)
                    )

                elif metric_lower in ["mae", "mean_absolute_error"]:
                    self._metrics[key] = float(mean_absolute_error(y_true, y_pred))

                elif metric_lower in ["r2", "r2_score"]:
                    self._metrics[key] = float(r2_score(y_true, y_pred))

                elif metric_lower in ["mape", "mean_absolute_percentage_error"]:
                    self._metrics[key] = float(
                        mean_absolute_percentage_error(y_true, y_pred)
                    )

                elif metric_lower in ["ev", "explained_variance"]:
                    self._metrics[key] = float(explained_variance_score(y_true, y_pred))

                elif metric_lower in ["max_error", "maxe"]:
                    self._metrics[key] = float(max_error(y_true, y_pred))

                elif metric_lower in ["medae", "median_absolute_error"]:
                    self._metrics[key] = float(median_absolute_error(y_true, y_pred))

                elif metric_lower in ["msle", "mean_squared_log_error"]:
                    # 需要确保值非负
                    if np.all(y_true >= 0) and np.all(y_pred >= 0):
                        self._metrics[key] = float(mean_squared_log_error(y_true, y_pred))
                    else:
                        logger.warning(
                            f"Skipping MSLE: requires non-negative values"
                        )

            except Exception as e:
                logger.warning(f"Failed to compute metric '{metric}': {e}")

        # 打印指标
        for key, value in self._metrics.items():
            logger.info(f"{key}: {value:.6f}")

        return self._metrics


class SecureEvaluator:
    """
    安全评估器

    支持在不泄露原始数据的情况下进行联邦模型评估。
    使用秘密共享或同态加密技术。
    """

    def __init__(self, method: str = "aggregation"):
        """
        初始化安全评估器

        Args:
            method: 安全计算方法 (aggregation, secret_sharing)
        """
        self.method = method

    def secure_aggregate_predictions(
        self,
        local_predictions: List[np.ndarray],
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        安全聚合多方预测结果

        Args:
            local_predictions: 各方的本地预测
            weights: 聚合权重

        Returns:
            聚合后的预测
        """
        if weights is None:
            weights = [1.0 / len(local_predictions)] * len(local_predictions)

        # 简单加权聚合（实际应用中应使用安全聚合协议）
        aggregated = np.zeros_like(local_predictions[0])
        for pred, weight in zip(local_predictions, weights):
            aggregated += weight * np.asarray(pred)

        return aggregated

    def compute_secure_metrics(
        self,
        aggregated_predictions: np.ndarray,
        y_true: np.ndarray,
        task_type: str = "classification"
    ) -> Dict[str, Any]:
        """
        基于聚合预测计算评估指标

        Args:
            aggregated_predictions: 聚合后的预测
            y_true: 真实标签（仅由持有标签的一方提供）
            task_type: 任务类型

        Returns:
            评估指标
        """
        if task_type == "classification":
            evaluator = ClassificationEvaluator()
            return evaluator.evaluate(y_true, aggregated_predictions)
        else:
            evaluator = RegressionEvaluator()
            return evaluator.evaluate(y_true, aggregated_predictions)
