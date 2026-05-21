"""
Federated Learning Feature Similarity Base Classes
联邦学习特征相似度基础类
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureSimilarityBase(ABC):
    """特征相似度基类"""

    def __init__(
        self,
        FL_type: str = "V",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
    ):
        """
        Args:
            FL_type: 联邦学习类型 ('V' 纵向)
            role: 角色
            channel: 通信通道
        """
        self.FL_type = FL_type
        self.role = role
        self.channel = channel

    @abstractmethod
    def compute(
        self, X: np.ndarray, Y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """计算相似度"""
        pass

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """标准化向量"""
        norm = np.linalg.norm(X, axis=0, keepdims=True)
        return X / (norm + 1e-10)


class CosineSimilarity(FeatureSimilarityBase):
    """
    余弦相似度

    计算特征向量之间的余弦相似度
    """

    def compute(
        self, X: np.ndarray, Y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        计算余弦相似度

        Args:
            X: 特征矩阵 (n_samples, n_features_x)
            Y: 另一方特征矩阵 (n_samples, n_features_y)

        Returns:
            相似度矩阵 (n_features_x, n_features_y)
        """
        X_norm = self._normalize(X)

        if Y is None:
            Y_norm = X_norm
        else:
            Y_norm = self._normalize(Y)

        return np.dot(X_norm.T, Y_norm)

    def compute_secure(
        self, X: np.ndarray, channel: Any
    ) -> np.ndarray:
        """
        安全计算余弦相似度（跨方）

        Args:
            X: 本地特征
            channel: 通信通道

        Returns:
            相似度矩阵
        """
        X_norm = self._normalize(X)

        if self.role == "host":
            # 发送标准化后的特征
            channel.send("X_norm", X_norm.tolist())
            # 接收对方标准化后的特征
            Y_norm = np.array(channel.recv("Y_norm"))
            # 计算相似度
            similarity = np.dot(X_norm.T, Y_norm)
            return similarity

        elif self.role == "guest":
            Y_norm = np.array(channel.recv("X_norm"))
            channel.send("Y_norm", X_norm.tolist())
            similarity = np.dot(X_norm.T, Y_norm)
            return similarity

        return np.array([])


class PearsonCorrelation(FeatureSimilarityBase):
    """
    皮尔逊相关系数

    计算特征之间的线性相关性
    """

    def compute(
        self, X: np.ndarray, Y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        计算皮尔逊相关系数

        Args:
            X: 特征矩阵
            Y: 另一方特征矩阵

        Returns:
            相关系数矩阵
        """
        # 中心化
        X_centered = X - np.mean(X, axis=0)

        if Y is None:
            Y_centered = X_centered
        else:
            Y_centered = Y - np.mean(Y, axis=0)

        # 计算协方差
        cov = np.dot(X_centered.T, Y_centered) / (X.shape[0] - 1)

        # 计算标准差
        std_x = np.std(X, axis=0, ddof=1)
        if Y is None:
            std_y = std_x
        else:
            std_y = np.std(Y, axis=0, ddof=1)

        # 计算相关系数
        correlation = cov / (np.outer(std_x, std_y) + 1e-10)

        return correlation

    def compute_secure(
        self, X: np.ndarray, channel: Any
    ) -> np.ndarray:
        """安全计算皮尔逊相关系数"""
        n = X.shape[0]
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0, ddof=1)
        X_centered = X - X_mean

        if self.role == "host":
            # 交换统计量
            channel.send("n_samples", n)
            channel.send("X_centered", X_centered.tolist())
            channel.send("X_std", X_std.tolist())

            Y_centered = np.array(channel.recv("Y_centered"))
            Y_std = np.array(channel.recv("Y_std"))

            # 计算协方差
            cov = np.dot(X_centered.T, Y_centered) / (n - 1)
            correlation = cov / (np.outer(X_std, Y_std) + 1e-10)

            return correlation

        elif self.role == "guest":
            channel.recv("n_samples")
            Y_centered = np.array(channel.recv("X_centered"))
            Y_std = np.array(channel.recv("X_std"))

            channel.send("Y_centered", X_centered.tolist())
            channel.send("Y_std", X_std.tolist())

            cov = np.dot(X_centered.T, Y_centered) / (n - 1)
            correlation = cov / (np.outer(X_std, Y_std) + 1e-10)

            return correlation

        return np.array([])


class MutualInformation(FeatureSimilarityBase):
    """
    互信息

    计算特征之间的互信息
    """

    def __init__(
        self,
        FL_type: str = "V",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        n_bins: int = 10,
    ):
        super().__init__(FL_type, role, channel)
        self.n_bins = n_bins

    def compute(
        self, X: np.ndarray, Y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        计算互信息

        Args:
            X: 特征矩阵
            Y: 另一方特征矩阵

        Returns:
            互信息矩阵
        """
        if Y is None:
            Y = X

        n_features_x = X.shape[1]
        n_features_y = Y.shape[1]

        mi_matrix = np.zeros((n_features_x, n_features_y))

        for i in range(n_features_x):
            for j in range(n_features_y):
                mi_matrix[i, j] = self._mutual_info(X[:, i], Y[:, j])

        return mi_matrix

    def _mutual_info(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算两个变量的互信息"""
        # 离散化
        x_bins = np.digitize(x, np.linspace(x.min(), x.max(), self.n_bins))
        y_bins = np.digitize(y, np.linspace(y.min(), y.max(), self.n_bins))

        # 计算联合分布
        joint_hist, _, _ = np.histogram2d(x_bins, y_bins, bins=self.n_bins)
        joint_prob = joint_hist / joint_hist.sum()

        # 计算边缘分布
        p_x = joint_prob.sum(axis=1)
        p_y = joint_prob.sum(axis=0)

        # 计算互信息
        mi = 0.0
        for i in range(len(p_x)):
            for j in range(len(p_y)):
                if joint_prob[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += joint_prob[i, j] * np.log(
                        joint_prob[i, j] / (p_x[i] * p_y[j])
                    )

        return mi


class JaccardSimilarity(FeatureSimilarityBase):
    """
    Jaccard 相似度

    用于类别特征的相似度计算
    """

    def compute(
        self, X: np.ndarray, Y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        计算 Jaccard 相似度

        Args:
            X: 特征矩阵（二值化后）
            Y: 另一方特征矩阵

        Returns:
            相似度矩阵
        """
        if Y is None:
            Y = X

        # 转换为布尔类型
        X_bool = X.astype(bool)
        Y_bool = Y.astype(bool)

        n_features_x = X.shape[1]
        n_features_y = Y.shape[1]

        similarity = np.zeros((n_features_x, n_features_y))

        for i in range(n_features_x):
            for j in range(n_features_y):
                intersection = np.sum(X_bool[:, i] & Y_bool[:, j])
                union = np.sum(X_bool[:, i] | Y_bool[:, j])
                if union > 0:
                    similarity[i, j] = intersection / union

        return similarity


class FeatureSimilarityAnalyzer:
    """
    特征相似度分析器

    综合多种相似度度量进行分析
    """

    def __init__(
        self,
        methods: List[str] = ["cosine", "pearson"],
        FL_type: str = "V",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
    ):
        """
        Args:
            methods: 使用的相似度方法列表
            FL_type: 联邦学习类型
            role: 角色
            channel: 通信通道
        """
        self.methods = methods
        self.FL_type = FL_type
        self.role = role
        self.channel = channel

        self.calculators = {
            "cosine": CosineSimilarity(FL_type, role, channel),
            "pearson": PearsonCorrelation(FL_type, role, channel),
            "mi": MutualInformation(FL_type, role, channel),
            "jaccard": JaccardSimilarity(FL_type, role, channel),
        }

    def analyze(
        self, X: np.ndarray, Y: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        执行相似度分析

        Args:
            X: 本地特征
            Y: 对方特征

        Returns:
            各方法的相似度矩阵
        """
        results = {}

        for method in self.methods:
            if method in self.calculators:
                calculator = self.calculators[method]
                results[method] = calculator.compute(X, Y)
            else:
                logger.warning(f"未知的相似度方法: {method}")

        return results

    def find_similar_features(
        self,
        similarity_matrix: np.ndarray,
        threshold: float = 0.8,
        feature_names_x: Optional[List[str]] = None,
        feature_names_y: Optional[List[str]] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        找出相似的特征对

        Args:
            similarity_matrix: 相似度矩阵
            threshold: 相似度阈值
            feature_names_x: X的特征名
            feature_names_y: Y的特征名

        Returns:
            相似特征对列表 [(feature_x, feature_y, similarity), ...]
        """
        similar_pairs = []

        n_x, n_y = similarity_matrix.shape

        if feature_names_x is None:
            feature_names_x = [f"x_{i}" for i in range(n_x)]
        if feature_names_y is None:
            feature_names_y = [f"y_{j}" for j in range(n_y)]

        for i in range(n_x):
            for j in range(n_y):
                if abs(similarity_matrix[i, j]) >= threshold:
                    similar_pairs.append(
                        (feature_names_x[i], feature_names_y[j], float(similarity_matrix[i, j]))
                    )

        # 按相似度排序
        similar_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        return similar_pairs
