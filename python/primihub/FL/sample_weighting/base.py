"""
Federated Learning Sample Weighting Base Classes
联邦学习样本加权基础类
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)


class SampleWeightingBase(ABC):
    """样本加权基类"""

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
    ):
        self.FL_type = FL_type
        self.role = role
        self.channel = channel
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """拟合权重计算器"""
        pass

    @abstractmethod
    def compute_weights(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """计算样本权重"""
        pass


class ClassWeighting(SampleWeightingBase):
    """
    类别加权

    根据类别分布计算样本权重
    """

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        strategy: str = "balanced",
        class_weights: Optional[Dict] = None,
    ):
        """
        Args:
            strategy: 加权策略 ('balanced', 'custom')
            class_weights: 自定义类别权重
        """
        super().__init__(FL_type, role, channel)
        self.strategy = strategy
        self.class_weights = class_weights
        self.class_weights_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """计算类别权重"""
        classes, counts = np.unique(y, return_counts=True)
        n_samples = len(y)
        n_classes = len(classes)

        if self.strategy == "balanced":
            # 平衡类别权重
            weights = n_samples / (n_classes * counts)
            self.class_weights_ = dict(zip(classes, weights))

        elif self.strategy == "custom" and self.class_weights:
            self.class_weights_ = self.class_weights

        else:
            self.class_weights_ = {c: 1.0 for c in classes}

        # 联邦场景下同步
        if self.FL_type == "H" and self.channel:
            self._sync_class_weights(classes, counts)

        self._is_fitted = True
        return self

    def compute_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """计算样本权重"""
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        weights = np.ones(len(y))
        for i, label in enumerate(y):
            weights[i] = self.class_weights_.get(label, 1.0)

        return weights

    def _sync_class_weights(self, classes: np.ndarray, counts: np.ndarray):
        """同步类别权重"""
        if self.role == "client":
            self.channel.send("local_classes", classes.tolist())
            self.channel.send("local_counts", counts.tolist())
            global_weights = self.channel.recv("global_class_weights")
            self.class_weights_ = global_weights

        elif self.role == "server":
            all_classes = self.channel.recv_all("local_classes")
            all_counts = self.channel.recv_all("local_counts")

            # 合并类别计数
            merged_counts = {}
            for c, cnt in zip(classes, counts):
                merged_counts[c] = merged_counts.get(c, 0) + cnt

            for party in all_classes:
                for c, cnt in zip(all_classes[party], all_counts[party]):
                    merged_counts[c] = merged_counts.get(c, 0) + cnt

            # 计算全局权重
            total_samples = sum(merged_counts.values())
            n_classes = len(merged_counts)
            global_weights = {
                c: total_samples / (n_classes * cnt)
                for c, cnt in merged_counts.items()
            }

            self.class_weights_ = global_weights
            self.channel.send_all("global_class_weights", global_weights)


class ImportanceWeighting(SampleWeightingBase):
    """
    重要性加权

    根据样本重要性计算权重
    """

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        method: str = "density",
        temperature: float = 1.0,
    ):
        """
        Args:
            method: 重要性计算方法 ('density', 'distance', 'gradient')
            temperature: 温度参数
        """
        super().__init__(FL_type, role, channel)
        self.method = method
        self.temperature = temperature
        self.reference_data_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """拟合重要性计算器"""
        self.reference_data_ = X
        self._is_fitted = True
        return self

    def compute_weights(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """计算样本重要性权重"""
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        if self.method == "density":
            return self._density_weights(X)
        elif self.method == "distance":
            return self._distance_weights(X)
        elif self.method == "gradient":
            return self._gradient_weights(X, y)
        else:
            return np.ones(len(X))

    def _density_weights(self, X: np.ndarray) -> np.ndarray:
        """基于密度的权重"""
        n_samples = X.shape[0]
        weights = np.zeros(n_samples)

        # 计算每个样本的局部密度
        for i in range(n_samples):
            distances = np.linalg.norm(X - X[i], axis=1)
            k = min(10, n_samples - 1)
            k_nearest = np.sort(distances)[1:k+1]
            density = 1.0 / (np.mean(k_nearest) + 1e-10)
            weights[i] = density

        # 归一化并应用温度
        weights = np.exp(-weights / self.temperature)
        weights = weights / weights.sum() * len(weights)

        return weights

    def _distance_weights(self, X: np.ndarray) -> np.ndarray:
        """基于距离的权重"""
        center = np.mean(X, axis=0)
        distances = np.linalg.norm(X - center, axis=1)

        # 距离越远权重越大（关注边界样本）
        weights = np.exp(distances / (np.std(distances) + 1e-10) / self.temperature)
        weights = weights / weights.sum() * len(weights)

        return weights

    def _gradient_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """基于梯度的权重（简化实现）"""
        if y is None:
            return np.ones(len(X))

        # 简化：使用残差作为重要性
        mean_y = np.mean(y)
        residuals = np.abs(y - mean_y)

        weights = residuals / (np.sum(residuals) + 1e-10) * len(y)
        return weights


class DistributionWeighting(SampleWeightingBase):
    """
    分布加权

    用于处理数据分布偏移
    """

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        target_distribution: str = "uniform",
    ):
        """
        Args:
            target_distribution: 目标分布 ('uniform', 'global', 'custom')
        """
        super().__init__(FL_type, role, channel)
        self.target_distribution = target_distribution
        self.source_density_ = None
        self.target_density_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """估计分布"""
        # 估计源分布密度
        self.source_density_ = self._estimate_density(X)

        if self.target_distribution == "uniform":
            n_samples = len(X)
            self.target_density_ = np.ones(n_samples) / n_samples
        else:
            self.target_density_ = self.source_density_

        # 联邦场景下同步
        if self.FL_type == "H" and self.channel:
            self._sync_distributions(X)

        self._is_fitted = True
        return self

    def compute_weights(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """计算分布权重"""
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        # 重要性比例
        weights = self.target_density_ / (self.source_density_ + 1e-10)

        # 裁剪极端值
        weights = np.clip(weights, 0.1, 10.0)

        # 归一化
        weights = weights / weights.mean()

        return weights

    def _estimate_density(self, X: np.ndarray) -> np.ndarray:
        """估计密度"""
        n_samples = X.shape[0]
        densities = np.zeros(n_samples)

        # 使用KDE估计
        bandwidth = np.std(X) * (n_samples ** (-1/5))

        for i in range(n_samples):
            distances = np.linalg.norm(X - X[i], axis=1)
            kernel_values = np.exp(-distances**2 / (2 * bandwidth**2 + 1e-10))
            densities[i] = np.mean(kernel_values)

        return densities

    def _sync_distributions(self, X: np.ndarray):
        """同步分布信息"""
        if self.target_distribution == "global" and self.channel:
            if self.role == "client":
                # 发送本地统计量
                self.channel.send("local_mean", np.mean(X, axis=0).tolist())
                self.channel.send("local_std", np.std(X, axis=0).tolist())
                self.channel.send("local_n", len(X))

                # 接收全局统计量
                global_mean = np.array(self.channel.recv("global_mean"))
                global_std = np.array(self.channel.recv("global_std"))

                # 根据全局分布调整目标密度
                local_mean = np.mean(X, axis=0)
                local_std = np.std(X, axis=0)

                # 计算与全局分布的差异
                diff = np.linalg.norm(local_mean - global_mean) / (np.linalg.norm(global_std) + 1e-10)
                adjustment = np.exp(-diff)
                self.target_density_ = self.source_density_ * adjustment

            elif self.role == "server":
                all_means = self.channel.recv_all("local_mean")
                all_stds = self.channel.recv_all("local_std")
                all_n = self.channel.recv_all("local_n")

                # 计算全局统计量
                total_n = sum(all_n.values())
                global_mean = np.zeros_like(np.mean(X, axis=0))
                for party, mean in all_means.items():
                    global_mean += np.array(mean) * all_n[party]
                global_mean /= total_n

                global_std = np.std(X, axis=0)  # 简化

                self.channel.send_all("global_mean", global_mean.tolist())
                self.channel.send_all("global_std", global_std.tolist())
