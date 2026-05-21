"""
Federated Learning Feature Sharing Base Classes
联邦学习特征分享基础类
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureSharingBase(ABC):
    """特征分享基类"""

    def __init__(
        self,
        FL_type: str = "V",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
    ):
        self.FL_type = FL_type
        self.role = role
        self.channel = channel

    @abstractmethod
    def share(self, X: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """分享特征"""
        pass

    @abstractmethod
    def receive(self) -> Dict[str, np.ndarray]:
        """接收特征"""
        pass


class SecureFeatureSharing(FeatureSharingBase):
    """
    安全特征分享

    使用加密技术保护分享的特征
    """

    def __init__(
        self,
        FL_type: str = "V",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        encryption_type: str = "additive_mask",
        noise_scale: float = 0.1,
    ):
        """
        Args:
            encryption_type: 加密类型 ('additive_mask', 'multiplicative_mask', 'dp')
            noise_scale: 噪声规模
        """
        super().__init__(FL_type, role, channel)
        self.encryption_type = encryption_type
        self.noise_scale = noise_scale
        self._masks = {}

    def share(
        self,
        X: np.ndarray,
        feature_indices: Optional[List[int]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        安全分享特征

        Args:
            X: 特征数据
            feature_indices: 要分享的特征索引

        Returns:
            分享结果
        """
        if feature_indices is not None:
            data_to_share = X[:, feature_indices]
        else:
            data_to_share = X

        # 添加保护
        protected_data = self._add_protection(data_to_share)

        if self.channel:
            self.channel.send("shared_features", protected_data.tolist())
            self.channel.send("feature_shape", data_to_share.shape)

        return {"local_data": data_to_share, "protected_data": protected_data}

    def receive(self) -> Dict[str, np.ndarray]:
        """
        接收分享的特征

        Returns:
            接收到的特征
        """
        if self.channel:
            protected_data = np.array(self.channel.recv("shared_features"))
            shape = self.channel.recv("feature_shape")

            # 移除保护（如果可能）
            if self.encryption_type == "dp":
                # DP 保护不可逆
                return {"received_data": protected_data, "is_protected": True}
            else:
                return {"received_data": protected_data, "is_protected": True}

        return {}

    def _add_protection(self, data: np.ndarray) -> np.ndarray:
        """添加隐私保护"""
        if self.encryption_type == "additive_mask":
            mask = np.random.randn(*data.shape) * self.noise_scale
            self._masks["additive"] = mask
            return data + mask

        elif self.encryption_type == "multiplicative_mask":
            mask = np.random.randn(*data.shape) * self.noise_scale + 1
            self._masks["multiplicative"] = mask
            return data * mask

        elif self.encryption_type == "dp":
            # 差分隐私噪声
            noise = np.random.laplace(0, self.noise_scale, data.shape)
            return data + noise

        return data

    def remove_protection(self, protected_data: np.ndarray) -> np.ndarray:
        """移除保护（仅限本方）"""
        if self.encryption_type == "additive_mask" and "additive" in self._masks:
            return protected_data - self._masks["additive"]

        elif self.encryption_type == "multiplicative_mask" and "multiplicative" in self._masks:
            return protected_data / self._masks["multiplicative"]

        return protected_data


class PartialFeatureSharing(FeatureSharingBase):
    """
    部分特征分享

    只分享部分特征或特征的统计信息
    """

    def __init__(
        self,
        FL_type: str = "V",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        sharing_ratio: float = 0.5,
        sharing_method: str = "random",
    ):
        """
        Args:
            sharing_ratio: 分享比例
            sharing_method: 分享方法 ('random', 'importance', 'statistics')
        """
        super().__init__(FL_type, role, channel)
        self.sharing_ratio = sharing_ratio
        self.sharing_method = sharing_method
        self.shared_indices_ = None

    def share(
        self,
        X: np.ndarray,
        importance_scores: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        部分分享特征

        Args:
            X: 特征数据
            importance_scores: 特征重要性分数

        Returns:
            分享结果
        """
        n_features = X.shape[1]
        n_share = int(n_features * self.sharing_ratio)

        if self.sharing_method == "random":
            self.shared_indices_ = np.random.choice(n_features, n_share, replace=False)

        elif self.sharing_method == "importance" and importance_scores is not None:
            self.shared_indices_ = np.argsort(importance_scores)[-n_share:]

        elif self.sharing_method == "statistics":
            # 分享统计信息而非原始数据
            stats = {
                "mean": np.mean(X, axis=0).tolist(),
                "std": np.std(X, axis=0).tolist(),
                "min": np.min(X, axis=0).tolist(),
                "max": np.max(X, axis=0).tolist(),
            }
            if self.channel:
                self.channel.send("feature_statistics", stats)
            return {"statistics": stats}

        else:
            self.shared_indices_ = np.arange(min(n_share, n_features))

        shared_data = X[:, self.shared_indices_]

        if self.channel:
            self.channel.send("shared_features", shared_data.tolist())
            self.channel.send("shared_indices", self.shared_indices_.tolist())

        return {"shared_data": shared_data, "shared_indices": self.shared_indices_}

    def receive(self) -> Dict[str, np.ndarray]:
        """接收部分特征"""
        if self.channel:
            shared_data = np.array(self.channel.recv("shared_features"))
            shared_indices = self.channel.recv("shared_indices")
            return {"received_data": shared_data, "indices": shared_indices}
        return {}


class FeatureAggregation(FeatureSharingBase):
    """
    特征聚合

    聚合多方的特征
    """

    def __init__(
        self,
        FL_type: str = "V",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        aggregation_method: str = "concat",
    ):
        """
        Args:
            aggregation_method: 聚合方法 ('concat', 'average', 'weighted')
        """
        super().__init__(FL_type, role, channel)
        self.aggregation_method = aggregation_method

    def share(self, X: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """分享特征用于聚合"""
        if self.channel:
            self.channel.send("features_for_aggregation", X.tolist())
        return {"local_features": X}

    def receive(self) -> Dict[str, np.ndarray]:
        """接收特征"""
        if self.channel:
            if hasattr(self.channel, "recv_all"):
                all_features = self.channel.recv_all("features_for_aggregation")
                return {party: np.array(features) for party, features in all_features.items()}
            else:
                features = np.array(self.channel.recv("features_for_aggregation"))
                return {"received": features}
        return {}

    def aggregate(
        self,
        feature_dict: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        聚合特征

        Args:
            feature_dict: 各方特征字典
            weights: 权重字典

        Returns:
            聚合后的特征
        """
        features_list = list(feature_dict.values())

        if not features_list:
            return np.array([])

        if self.aggregation_method == "concat":
            return np.concatenate(features_list, axis=1)

        elif self.aggregation_method == "average":
            return np.mean(np.stack(features_list), axis=0)

        elif self.aggregation_method == "weighted" and weights:
            parties = list(feature_dict.keys())
            weight_array = np.array([weights.get(p, 1.0) for p in parties])
            weight_array = weight_array / weight_array.sum()

            weighted_sum = np.zeros_like(features_list[0])
            for f, w in zip(features_list, weight_array):
                weighted_sum += f * w

            return weighted_sum

        return features_list[0]


class SecretSharingFeatures:
    """
    秘密分享特征

    使用秘密分享协议分享特征
    """

    def __init__(self, n_shares: int = 2, prime: int = 2**31 - 1):
        """
        Args:
            n_shares: 分享数量
            prime: 素数模
        """
        self.n_shares = n_shares
        self.prime = prime

    def create_shares(self, X: np.ndarray) -> List[np.ndarray]:
        """
        创建秘密分享

        Args:
            X: 原始数据

        Returns:
            分享列表
        """
        # 简化的加法秘密分享
        shares = []
        remaining = X.copy().astype(np.int64)

        for i in range(self.n_shares - 1):
            share = np.random.randint(0, self.prime, X.shape, dtype=np.int64)
            shares.append(share)
            remaining = (remaining - share) % self.prime

        shares.append(remaining)

        return shares

    def reconstruct(self, shares: List[np.ndarray]) -> np.ndarray:
        """
        重构数据

        Args:
            shares: 分享列表

        Returns:
            原始数据
        """
        result = np.zeros_like(shares[0])
        for share in shares:
            result = (result + share) % self.prime

        return result
