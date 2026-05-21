"""
Federated Learning Feature Alignment Base Classes
联邦学习特征对齐基础类
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureAlignmentBase(ABC):
    """特征对齐基类"""

    def __init__(
        self,
        FL_type: str = "V",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
    ):
        self.FL_type = FL_type
        self.role = role
        self.channel = channel
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, reference: Optional[np.ndarray] = None):
        """拟合对齐器"""
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """执行对齐"""
        pass

    def fit_transform(
        self, X: np.ndarray, reference: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """拟合并转换"""
        self.fit(X, reference)
        return self.transform(X)


class StatisticalAlignment(FeatureAlignmentBase):
    """
    统计对齐

    基于统计量对齐不同方的特征
    """

    def __init__(
        self,
        FL_type: str = "V",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        method: str = "zscore",
    ):
        """
        Args:
            method: 对齐方法 ('zscore', 'minmax', 'rank')
        """
        super().__init__(FL_type, role, channel)
        self.method = method

        self.local_mean_ = None
        self.local_std_ = None
        self.local_min_ = None
        self.local_max_ = None
        self.global_mean_ = None
        self.global_std_ = None
        self.global_min_ = None
        self.global_max_ = None

    def fit(self, X: np.ndarray, reference: Optional[np.ndarray] = None):
        """
        计算对齐参数

        Args:
            X: 本地数据
            reference: 参考数据（可选）
        """
        # 计算本地统计量
        self.local_mean_ = np.nanmean(X, axis=0)
        self.local_std_ = np.nanstd(X, axis=0)
        self.local_min_ = np.nanmin(X, axis=0)
        self.local_max_ = np.nanmax(X, axis=0)

        # 联邦场景下同步
        if self.channel:
            self._sync_statistics(X.shape[0])
        else:
            self.global_mean_ = self.local_mean_
            self.global_std_ = self.local_std_
            self.global_min_ = self.local_min_
            self.global_max_ = self.local_max_

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        执行对齐

        Args:
            X: 输入数据

        Returns:
            对齐后的数据
        """
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        if self.method == "zscore":
            # Z-score 标准化对齐
            local_normalized = (X - self.local_mean_) / (self.local_std_ + 1e-10)
            aligned = local_normalized * self.global_std_ + self.global_mean_
            return aligned

        elif self.method == "minmax":
            # Min-Max 对齐
            local_range = self.local_max_ - self.local_min_ + 1e-10
            local_normalized = (X - self.local_min_) / local_range
            global_range = self.global_max_ - self.global_min_
            aligned = local_normalized * global_range + self.global_min_
            return aligned

        elif self.method == "rank":
            # 排名对齐
            result = np.zeros_like(X)
            for col in range(X.shape[1]):
                ranks = np.argsort(np.argsort(X[:, col]))
                result[:, col] = ranks / (len(ranks) - 1)
            return result

        return X

    def _sync_statistics(self, n_samples: int):
        """同步统计量"""
        if self.role == "host":
            # 发送本地统计量
            self.channel.send_all("local_stats", {
                "mean": self.local_mean_.tolist(),
                "std": self.local_std_.tolist(),
                "min": self.local_min_.tolist(),
                "max": self.local_max_.tolist(),
                "n": n_samples,
            })

            # 接收guest统计量并计算全局
            guest_stats = self.channel.recv_all("guest_stats")

            all_means = [self.local_mean_ * n_samples]
            all_n = [n_samples]

            for party, stats in guest_stats.items():
                if stats:
                    all_means.append(np.array(stats["mean"]) * stats["n"])
                    all_n.append(stats["n"])

            total_n = sum(all_n)
            self.global_mean_ = np.sum(all_means, axis=0) / total_n
            self.global_std_ = self.local_std_  # 简化
            self.global_min_ = np.minimum.reduce(
                [self.local_min_] + [np.array(s["min"]) for s in guest_stats.values() if s]
            )
            self.global_max_ = np.maximum.reduce(
                [self.local_max_] + [np.array(s["max"]) for s in guest_stats.values() if s]
            )

            # 分发全局统计量
            self.channel.send_all("global_stats", {
                "mean": self.global_mean_.tolist(),
                "std": self.global_std_.tolist(),
                "min": self.global_min_.tolist(),
                "max": self.global_max_.tolist(),
            })

        elif self.role == "guest":
            self.channel.recv("local_stats")
            self.channel.send("guest_stats", {
                "mean": self.local_mean_.tolist(),
                "std": self.local_std_.tolist(),
                "min": self.local_min_.tolist(),
                "max": self.local_max_.tolist(),
                "n": n_samples,
            })

            global_stats = self.channel.recv("global_stats")
            self.global_mean_ = np.array(global_stats["mean"])
            self.global_std_ = np.array(global_stats["std"])
            self.global_min_ = np.array(global_stats["min"])
            self.global_max_ = np.array(global_stats["max"])


class DistributionAlignment(FeatureAlignmentBase):
    """
    分布对齐

    对齐特征的分布
    """

    def __init__(
        self,
        FL_type: str = "V",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        n_quantiles: int = 100,
    ):
        super().__init__(FL_type, role, channel)
        self.n_quantiles = n_quantiles
        self.local_quantiles_ = None
        self.target_quantiles_ = None

    def fit(self, X: np.ndarray, reference: Optional[np.ndarray] = None):
        """
        计算分位数

        Args:
            X: 本地数据
            reference: 目标分布参考
        """
        quantile_points = np.linspace(0, 100, self.n_quantiles)

        self.local_quantiles_ = np.percentile(X, quantile_points, axis=0)

        if reference is not None:
            self.target_quantiles_ = np.percentile(reference, quantile_points, axis=0)
        else:
            self.target_quantiles_ = self.local_quantiles_

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        分布对齐转换

        Args:
            X: 输入数据

        Returns:
            对齐后的数据
        """
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        result = np.zeros_like(X)

        for col in range(X.shape[1]):
            # 找到每个值在本地分位数中的位置
            local_q = self.local_quantiles_[:, col]
            target_q = self.target_quantiles_[:, col]

            for i, val in enumerate(X[:, col]):
                # 找到最近的分位数索引
                idx = np.searchsorted(local_q, val)
                idx = min(idx, len(target_q) - 1)
                result[i, col] = target_q[idx]

        return result


class SchemaAlignment(FeatureAlignmentBase):
    """
    模式对齐

    对齐特征模式（名称、类型）
    """

    def __init__(
        self,
        FL_type: str = "V",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
    ):
        super().__init__(FL_type, role, channel)
        self.local_schema_ = None
        self.global_schema_ = None
        self.mapping_ = None

    def fit(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        feature_types: Optional[List[str]] = None,
    ):
        """
        拟合模式对齐

        Args:
            X: 输入数据
            feature_names: 特征名称
            feature_types: 特征类型
        """
        n_features = X.shape[1]

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        if feature_types is None:
            feature_types = ["numeric"] * n_features

        self.local_schema_ = {
            "names": feature_names,
            "types": feature_types,
            "n_features": n_features,
        }

        # 联邦场景下同步模式
        if self.channel:
            self._sync_schema()
        else:
            self.global_schema_ = self.local_schema_
            self.mapping_ = list(range(n_features))

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        应用模式对齐

        Args:
            X: 输入数据

        Returns:
            对齐后的数据
        """
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        if self.mapping_ is None or len(self.mapping_) == X.shape[1]:
            return X

        # 按映射重排列
        return X[:, self.mapping_]

    def _sync_schema(self):
        """同步模式"""
        if self.role == "host":
            self.channel.send_all("local_schema", self.local_schema_)
            guest_schemas = self.channel.recv_all("guest_schema")

            # 合并模式
            all_names = set(self.local_schema_["names"])
            for schema in guest_schemas.values():
                if schema:
                    all_names.update(schema["names"])

            self.global_schema_ = {
                "names": sorted(list(all_names)),
                "n_features": len(all_names),
            }

            # 建立映射
            self.mapping_ = [
                self.global_schema_["names"].index(name)
                if name in self.global_schema_["names"] else -1
                for name in self.local_schema_["names"]
            ]

            self.channel.send_all("global_schema", self.global_schema_)

        elif self.role == "guest":
            self.channel.recv("local_schema")
            self.channel.send("guest_schema", self.local_schema_)
            self.global_schema_ = self.channel.recv("global_schema")

            self.mapping_ = [
                self.global_schema_["names"].index(name)
                if name in self.global_schema_["names"] else -1
                for name in self.local_schema_["names"]
            ]


class FeatureMapper:
    """
    特征映射器

    建立不同方特征之间的映射关系
    """

    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.mapping_ = None

    def build_mapping(
        self,
        local_features: List[str],
        remote_features: List[str],
        similarity_matrix: Optional[np.ndarray] = None,
    ) -> Dict[str, str]:
        """
        建立特征映射

        Args:
            local_features: 本地特征名
            remote_features: 远程特征名
            similarity_matrix: 相似度矩阵

        Returns:
            映射字典
        """
        self.mapping_ = {}

        if similarity_matrix is not None:
            for i, local_name in enumerate(local_features):
                best_match_idx = np.argmax(similarity_matrix[i])
                best_similarity = similarity_matrix[i, best_match_idx]

                if best_similarity >= self.similarity_threshold:
                    self.mapping_[local_name] = remote_features[best_match_idx]
        else:
            # 基于名称相似度
            for local_name in local_features:
                for remote_name in remote_features:
                    if self._name_similarity(local_name, remote_name) >= self.similarity_threshold:
                        self.mapping_[local_name] = remote_name
                        break

        return self.mapping_

    def _name_similarity(self, name1: str, name2: str) -> float:
        """计算名称相似度"""
        name1_lower = name1.lower().replace("_", "")
        name2_lower = name2.lower().replace("_", "")

        if name1_lower == name2_lower:
            return 1.0

        # 简单的编辑距离相似度
        max_len = max(len(name1_lower), len(name2_lower))
        if max_len == 0:
            return 1.0

        common = sum(c1 == c2 for c1, c2 in zip(name1_lower, name2_lower))
        return common / max_len
