"""
Federated Learning Metrics Modeling Base Classes
联邦学习指标建模分析基础类
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class MetricsModelingBase(ABC):
    """指标建模基类"""

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
    ):
        self.FL_type = FL_type
        self.role = role
        self.channel = channel

    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算指标"""
        pass


class FederatedMetrics(MetricsModelingBase):
    """
    联邦指标计算

    支持在联邦场景下计算各种评估指标
    """

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        task_type: str = "classification",
    ):
        """
        Args:
            task_type: 任务类型 ('classification', 'regression')
        """
        super().__init__(FL_type, role, channel)
        self.task_type = task_type

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算本地指标"""
        if self.task_type == "classification":
            return self._classification_metrics(y_true, y_pred)
        else:
            return self._regression_metrics(y_true, y_pred)

    def compute_federated(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """计算联邦指标"""
        local_metrics = self.compute(y_true, y_pred)
        local_n = len(y_true)

        if not self.channel:
            return local_metrics

        if self.role == "server":
            return self._aggregate_metrics(local_metrics, local_n)
        elif self.role == "client":
            return self._send_local_metrics(local_metrics, local_n)

        return local_metrics

    def _classification_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """分类指标"""
        y_pred_binary = (y_pred > 0.5).astype(int) if y_pred.max() <= 1 else y_pred

        # 准确率
        accuracy = np.mean(y_true == y_pred_binary)

        # 混淆矩阵元素
        tp = np.sum((y_true == 1) & (y_pred_binary == 1))
        tn = np.sum((y_true == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true == 1) & (y_pred_binary == 0))

        # 精确率、召回率、F1
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        # AUC (简化计算)
        auc = self._compute_auc(y_true, y_pred)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }

    def _regression_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """回归指标"""
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)

        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
        }

    def _compute_auc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算AUC"""
        try:
            sorted_indices = np.argsort(y_pred)[::-1]
            y_true_sorted = y_true[sorted_indices]

            tpr_list = []
            fpr_list = []

            n_pos = np.sum(y_true == 1)
            n_neg = np.sum(y_true == 0)

            tp = 0
            fp = 0

            for label in y_true_sorted:
                if label == 1:
                    tp += 1
                else:
                    fp += 1
                tpr_list.append(tp / (n_pos + 1e-10))
                fpr_list.append(fp / (n_neg + 1e-10))

            auc = np.trapz(tpr_list, fpr_list)
            return abs(auc)
        except Exception:
            return 0.5

    def _aggregate_metrics(
        self, local_metrics: Dict, local_n: int
    ) -> Dict[str, float]:
        """聚合联邦指标"""
        self.channel.send_all("request_metrics", True)

        all_metrics = self.channel.recv_all("local_metrics")
        all_n = self.channel.recv_all("local_n")

        total_n = local_n + sum(all_n.values())
        aggregated = {}

        for key in local_metrics:
            if key in ["tp", "tn", "fp", "fn"]:
                # 累加计数
                aggregated[key] = local_metrics[key]
                for party in all_metrics:
                    if key in all_metrics[party]:
                        aggregated[key] += all_metrics[party][key]
            else:
                # 加权平均
                weighted_sum = local_metrics[key] * local_n
                for party in all_metrics:
                    if key in all_metrics[party]:
                        weighted_sum += all_metrics[party][key] * all_n[party]
                aggregated[key] = weighted_sum / total_n

        # 重新计算基于聚合计数的指标
        if "tp" in aggregated:
            tp, tn, fp, fn = aggregated["tp"], aggregated["tn"], aggregated["fp"], aggregated["fn"]
            aggregated["precision"] = tp / (tp + fp + 1e-10)
            aggregated["recall"] = tp / (tp + fn + 1e-10)
            aggregated["f1"] = 2 * aggregated["precision"] * aggregated["recall"] / (
                aggregated["precision"] + aggregated["recall"] + 1e-10
            )
            aggregated["accuracy"] = (tp + tn) / (tp + tn + fp + fn + 1e-10)

        return aggregated

    def _send_local_metrics(
        self, local_metrics: Dict, local_n: int
    ) -> Dict[str, float]:
        """发送本地指标"""
        self.channel.recv("request_metrics")
        self.channel.send("local_metrics", local_metrics)
        self.channel.send("local_n", local_n)
        return local_metrics


class ModelPerformanceAnalyzer:
    """
    模型性能分析器

    分析模型在各种维度上的性能
    """

    def __init__(self, task_type: str = "classification"):
        self.task_type = task_type
        self.metrics_calculator = FederatedMetrics(task_type=task_type)

    def analyze(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        全面分析模型性能

        Args:
            y_true: 真实标签
            y_pred: 预测值
            groups: 分组标签（可选）

        Returns:
            分析报告
        """
        report = {
            "overall": self.metrics_calculator.compute(y_true, y_pred),
        }

        # 按组分析
        if groups is not None:
            report["by_group"] = self._analyze_by_group(y_true, y_pred, groups)

        # 按预测置信度分析
        if self.task_type == "classification":
            report["by_confidence"] = self._analyze_by_confidence(y_true, y_pred)

        # 误差分析
        report["error_analysis"] = self._error_analysis(y_true, y_pred)

        return report

    def _analyze_by_group(
        self, y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray
    ) -> Dict[str, Dict]:
        """按组分析"""
        unique_groups = np.unique(groups)
        group_metrics = {}

        for g in unique_groups:
            mask = groups == g
            group_metrics[str(g)] = self.metrics_calculator.compute(
                y_true[mask], y_pred[mask]
            )

        return group_metrics

    def _analyze_by_confidence(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Dict]:
        """按置信度分析"""
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        confidence_metrics = {}

        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i + 1]
            mask = (y_pred >= low) & (y_pred < high)

            if np.sum(mask) > 0:
                key = f"{low:.1f}-{high:.1f}"
                confidence_metrics[key] = {
                    "count": int(np.sum(mask)),
                    "accuracy": float(np.mean(y_true[mask] == (y_pred[mask] > 0.5).astype(int))),
                }

        return confidence_metrics

    def _error_analysis(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """误差分析"""
        if self.task_type == "classification":
            y_pred_binary = (y_pred > 0.5).astype(int)
            errors = y_true != y_pred_binary

            return {
                "error_rate": float(np.mean(errors)),
                "error_count": int(np.sum(errors)),
                "false_positive_rate": float(np.mean((y_true == 0) & (y_pred_binary == 1))),
                "false_negative_rate": float(np.mean((y_true == 1) & (y_pred_binary == 0))),
            }
        else:
            errors = y_true - y_pred
            return {
                "mean_error": float(np.mean(errors)),
                "std_error": float(np.std(errors)),
                "max_error": float(np.max(np.abs(errors))),
                "error_percentiles": {
                    "25": float(np.percentile(np.abs(errors), 25)),
                    "50": float(np.percentile(np.abs(errors), 50)),
                    "75": float(np.percentile(np.abs(errors), 75)),
                    "95": float(np.percentile(np.abs(errors), 95)),
                },
            }


class FeatureImportanceAnalyzer:
    """
    特征重要性分析器

    分析特征对模型的重要性
    """

    def __init__(self, method: str = "permutation"):
        """
        Args:
            method: 分析方法 ('permutation', 'correlation', 'variance')
        """
        self.method = method

    def analyze(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        model: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        分析特征重要性

        Args:
            X: 特征矩阵
            y: 标签
            feature_names: 特征名称
            model: 训练好的模型（可选）

        Returns:
            特征重要性分析结果
        """
        n_features = X.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        if self.method == "permutation" and model is not None:
            importance = self._permutation_importance(X, y, model)
        elif self.method == "correlation":
            importance = self._correlation_importance(X, y)
        elif self.method == "variance":
            importance = self._variance_importance(X)
        else:
            importance = self._correlation_importance(X, y)

        # 排序
        sorted_idx = np.argsort(importance)[::-1]

        return {
            "importance_scores": {
                feature_names[i]: float(importance[i]) for i in range(n_features)
            },
            "ranking": [feature_names[i] for i in sorted_idx],
            "top_features": [feature_names[i] for i in sorted_idx[:10]],
        }

    def _permutation_importance(
        self, X: np.ndarray, y: np.ndarray, model: Any
    ) -> np.ndarray:
        """排列重要性"""
        baseline_score = self._score(model, X, y)
        n_features = X.shape[1]
        importance = np.zeros(n_features)

        for i in range(n_features):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            permuted_score = self._score(model, X_permuted, y)
            importance[i] = baseline_score - permuted_score

        return importance

    def _correlation_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """相关性重要性"""
        n_features = X.shape[1]
        importance = np.zeros(n_features)

        for i in range(n_features):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            importance[i] = abs(corr) if not np.isnan(corr) else 0

        return importance

    def _variance_importance(self, X: np.ndarray) -> np.ndarray:
        """方差重要性"""
        return np.var(X, axis=0)

    def _score(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """计算模型分数"""
        try:
            if hasattr(model, "score"):
                return model.score(X, y)
            elif hasattr(model, "predict"):
                y_pred = model.predict(X)
                return -np.mean((y - y_pred) ** 2)
        except Exception:
            pass
        return 0.0
