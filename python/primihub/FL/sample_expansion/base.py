"""
Federated Learning Sample Expansion Base Classes
联邦学习样本列扩展基础类
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
from itertools import combinations, combinations_with_replacement
import logging

logger = logging.getLogger(__name__)


class SampleExpansionBase(ABC):
    """样本列扩展基类"""

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
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """拟合扩展器"""
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """扩展特征"""
        pass

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)


class PolynomialExpansion(SampleExpansionBase):
    """
    多项式扩展

    生成多项式特征
    """

    def __init__(
        self,
        FL_type: str = "V",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = True,
    ):
        """
        Args:
            degree: 多项式阶数
            interaction_only: 只生成交互项
            include_bias: 是否包含偏置项
        """
        super().__init__(FL_type, role, channel)
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.n_input_features_ = None
        self.n_output_features_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """计算输出特征数量"""
        self.n_input_features_ = X.shape[1]
        self.n_output_features_ = self._count_output_features()
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """生成多项式特征"""
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        n_samples, n_features = X.shape
        result = []

        # 偏置项
        if self.include_bias:
            result.append(np.ones((n_samples, 1)))

        # 一次项
        result.append(X)

        # 高次项
        for d in range(2, self.degree + 1):
            if self.interaction_only:
                # 只有交互项
                for combo in combinations(range(n_features), d):
                    feature = np.prod(X[:, combo], axis=1, keepdims=True)
                    result.append(feature)
            else:
                # 所有组合
                for combo in combinations_with_replacement(range(n_features), d):
                    feature = np.prod(X[:, list(combo)], axis=1, keepdims=True)
                    result.append(feature)

        return np.hstack(result)

    def _count_output_features(self) -> int:
        """计算输出特征数量"""
        n = self.n_input_features_
        count = 0

        if self.include_bias:
            count += 1

        # 一次项
        count += n

        # 高次项
        for d in range(2, self.degree + 1):
            if self.interaction_only:
                from math import comb
                count += comb(n, d)
            else:
                from math import comb
                count += comb(n + d - 1, d)

        return count


class InteractionExpansion(SampleExpansionBase):
    """
    交互特征扩展

    生成特征之间的交互项
    """

    def __init__(
        self,
        FL_type: str = "V",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        interaction_pairs: Optional[List[Tuple[int, int]]] = None,
        include_original: bool = True,
    ):
        """
        Args:
            interaction_pairs: 指定的交互对，None表示所有对
            include_original: 是否包含原始特征
        """
        super().__init__(FL_type, role, channel)
        self.interaction_pairs = interaction_pairs
        self.include_original = include_original

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """确定交互对"""
        n_features = X.shape[1]

        if self.interaction_pairs is None:
            self.interaction_pairs = list(combinations(range(n_features), 2))

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """生成交互特征"""
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        result = []

        if self.include_original:
            result.append(X)

        for i, j in self.interaction_pairs:
            interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
            result.append(interaction)

        return np.hstack(result)


class CrossFeatureExpansion(SampleExpansionBase):
    """
    跨方特征扩展

    生成跨方的特征交互（用于VFL）
    """

    def __init__(
        self,
        FL_type: str = "V",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        expansion_type: str = "product",
        secure: bool = True,
    ):
        """
        Args:
            expansion_type: 扩展类型 ('product', 'concat', 'sum')
            secure: 是否使用安全计算
        """
        super().__init__(FL_type, role, channel)
        self.expansion_type = expansion_type
        self.secure = secure

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """准备跨方扩展"""
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """执行跨方特征扩展"""
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        if not self.channel:
            return X

        if self.role == "host":
            return self._host_transform(X)
        elif self.role == "guest":
            return self._guest_transform(X)

        return X

    def _host_transform(self, X: np.ndarray) -> np.ndarray:
        """主机端扩展"""
        # 发送本地特征统计
        if self.secure:
            # 安全计算模式：只交换加密的中间结果
            self.channel.send_all("host_feature_shape", X.shape)
            guest_shapes = self.channel.recv_all("guest_feature_shape")

            # 扩展维度
            total_features = X.shape[1]
            for shape in guest_shapes.values():
                if shape:
                    total_features += shape[1]

            # 返回原始特征（跨方交互在训练时计算）
            return X
        else:
            # 非安全模式：直接交换特征
            self.channel.send_all("host_features", X.tolist())
            guest_features = self.channel.recv_all("guest_features")

            result = [X]
            for features in guest_features.values():
                if features is not None:
                    result.append(np.array(features))

            return np.hstack(result)

    def _guest_transform(self, X: np.ndarray) -> np.ndarray:
        """访客端扩展"""
        if self.secure:
            self.channel.recv("host_feature_shape")
            self.channel.send("guest_feature_shape", X.shape)
            return X
        else:
            self.channel.recv("host_features")
            self.channel.send("guest_features", X.tolist())
            return X


class FeatureAugmentation(SampleExpansionBase):
    """
    特征增强

    通过各种变换增强特征
    """

    def __init__(
        self,
        FL_type: str = "V",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        transformations: List[str] = None,
    ):
        """
        Args:
            transformations: 变换列表 ['log', 'sqrt', 'square', 'reciprocal']
        """
        super().__init__(FL_type, role, channel)
        self.transformations = transformations or ["log", "sqrt"]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """准备变换"""
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """应用变换增强特征"""
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        result = [X]

        for transform in self.transformations:
            if transform == "log":
                # 对数变换（处理非正值）
                augmented = np.log1p(np.abs(X))
                result.append(augmented)

            elif transform == "sqrt":
                # 平方根变换
                augmented = np.sqrt(np.abs(X))
                result.append(augmented)

            elif transform == "square":
                # 平方变换
                augmented = X ** 2
                result.append(augmented)

            elif transform == "reciprocal":
                # 倒数变换
                augmented = 1.0 / (X + 1e-10)
                result.append(augmented)

            elif transform == "exp":
                # 指数变换（限制范围）
                augmented = np.exp(np.clip(X, -10, 10))
                result.append(augmented)

        return np.hstack(result)
