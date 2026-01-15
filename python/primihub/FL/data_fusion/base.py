"""
Federated Learning Data Fusion Base Classes
联邦学习数据融合基础类

提供数据融合的核心算法实现
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)


class DataFusionBase(ABC):
    """数据融合基类"""

    def __init__(
        self,
        fusion_method: str = "average",
        weights: Optional[List[float]] = None,
        secure: bool = False,
    ):
        """
        初始化数据融合器

        Args:
            fusion_method: 融合方法 ('average', 'weighted', 'concat', 'stack')
            weights: 权重列表，用于加权融合
            secure: 是否启用安全计算
        """
        self.fusion_method = fusion_method
        self.weights = weights
        self.secure = secure
        self._is_fitted = False

    @abstractmethod
    def fuse(self, data_list: List[np.ndarray]) -> np.ndarray:
        """执行数据融合"""
        pass

    def validate_inputs(self, data_list: List[np.ndarray]) -> bool:
        """验证输入数据"""
        if not data_list:
            raise ValueError("数据列表不能为空")

        if self.fusion_method in ["average", "weighted", "stack"]:
            shapes = [d.shape for d in data_list]
            if len(set(shapes)) > 1:
                raise ValueError(f"数据形状不一致: {shapes}")

        if self.fusion_method == "weighted" and self.weights:
            if len(self.weights) != len(data_list):
                raise ValueError("权重数量与数据源数量不匹配")

        return True


class HorizontalDataFusion(DataFusionBase):
    """
    横向联邦数据融合

    用于具有相同特征、不同样本的数据融合
    """

    def __init__(
        self,
        fusion_method: str = "concat",
        weights: Optional[List[float]] = None,
        secure: bool = False,
        sample_alignment: bool = False,
    ):
        """
        Args:
            fusion_method: 融合方法 ('concat', 'stack', 'average')
            weights: 权重
            secure: 是否安全计算
            sample_alignment: 是否进行样本对齐
        """
        super().__init__(fusion_method, weights, secure)
        self.sample_alignment = sample_alignment

    def fuse(self, data_list: List[np.ndarray]) -> np.ndarray:
        """
        执行横向数据融合

        Args:
            data_list: 来自各参与方的数据列表

        Returns:
            融合后的数据
        """
        self.validate_inputs(data_list)

        if self.fusion_method == "concat":
            return self._concat_fusion(data_list)
        elif self.fusion_method == "stack":
            return self._stack_fusion(data_list)
        elif self.fusion_method == "average":
            return self._average_fusion(data_list)
        else:
            raise ValueError(f"不支持的融合方法: {self.fusion_method}")

    def _concat_fusion(self, data_list: List[np.ndarray]) -> np.ndarray:
        """纵向拼接（沿样本维度）"""
        return np.concatenate(data_list, axis=0)

    def _stack_fusion(self, data_list: List[np.ndarray]) -> np.ndarray:
        """堆叠融合"""
        return np.stack(data_list, axis=0)

    def _average_fusion(self, data_list: List[np.ndarray]) -> np.ndarray:
        """平均融合（用于统计量）"""
        if self.weights:
            weights = np.array(self.weights)
            weights = weights / weights.sum()
            return np.average(np.stack(data_list), axis=0, weights=weights)
        return np.mean(np.stack(data_list), axis=0)


class VerticalDataFusion(DataFusionBase):
    """
    纵向联邦数据融合

    用于具有相同样本、不同特征的数据融合
    """

    def __init__(
        self,
        fusion_method: str = "concat",
        weights: Optional[List[float]] = None,
        secure: bool = False,
        feature_alignment: bool = True,
        id_column: Optional[str] = None,
    ):
        """
        Args:
            fusion_method: 融合方法 ('concat', 'weighted_concat')
            weights: 特征权重
            secure: 是否安全计算
            feature_alignment: 是否进行特征对齐
            id_column: ID列名，用于样本对齐
        """
        super().__init__(fusion_method, weights, secure)
        self.feature_alignment = feature_alignment
        self.id_column = id_column

    def fuse(self, data_list: List[np.ndarray]) -> np.ndarray:
        """
        执行纵向数据融合

        Args:
            data_list: 来自各参与方的数据列表

        Returns:
            融合后的数据
        """
        self.validate_inputs(data_list)

        if self.fusion_method == "concat":
            return self._concat_fusion(data_list)
        elif self.fusion_method == "weighted_concat":
            return self._weighted_concat_fusion(data_list)
        else:
            raise ValueError(f"不支持的融合方法: {self.fusion_method}")

    def _concat_fusion(self, data_list: List[np.ndarray]) -> np.ndarray:
        """横向拼接（沿特征维度）"""
        return np.concatenate(data_list, axis=1)

    def _weighted_concat_fusion(self, data_list: List[np.ndarray]) -> np.ndarray:
        """加权拼接"""
        if self.weights is None:
            return self._concat_fusion(data_list)

        weighted_data = []
        for data, weight in zip(data_list, self.weights):
            weighted_data.append(data * weight)

        return np.concatenate(weighted_data, axis=1)


class SecureDataFusion(DataFusionBase):
    """
    安全数据融合

    支持加密状态下的数据融合
    """

    def __init__(
        self,
        fusion_method: str = "secure_average",
        weights: Optional[List[float]] = None,
        encryption_type: str = "paillier",
    ):
        """
        Args:
            fusion_method: 融合方法
            weights: 权重
            encryption_type: 加密类型 ('paillier', 'ckks')
        """
        super().__init__(fusion_method, weights, secure=True)
        self.encryption_type = encryption_type
        self.public_key = None
        self.private_key = None

    def set_keys(self, public_key: Any, private_key: Optional[Any] = None):
        """设置加密密钥"""
        self.public_key = public_key
        self.private_key = private_key

    def fuse(self, data_list: List[np.ndarray]) -> np.ndarray:
        """
        执行安全数据融合

        Args:
            data_list: 加密数据列表

        Returns:
            融合后的加密/解密数据
        """
        if self.fusion_method == "secure_average":
            return self._secure_average_fusion(data_list)
        elif self.fusion_method == "secure_sum":
            return self._secure_sum_fusion(data_list)
        else:
            raise ValueError(f"不支持的安全融合方法: {self.fusion_method}")

    def _secure_average_fusion(self, data_list: List) -> np.ndarray:
        """安全平均融合"""
        n = len(data_list)
        if n == 0:
            raise ValueError("数据列表为空")

        # 在加密域中进行求和
        result = data_list[0]
        for data in data_list[1:]:
            result = result + data

        # 除以参与方数量
        if hasattr(result, '__truediv__'):
            return result / n
        else:
            # 如果不支持除法，返回求和结果
            return result

    def _secure_sum_fusion(self, data_list: List) -> Any:
        """安全求和融合"""
        if not data_list:
            raise ValueError("数据列表为空")

        result = data_list[0]
        for data in data_list[1:]:
            result = result + data

        return result


class FeatureDataFusion:
    """
    特征级数据融合

    支持特征层面的融合操作
    """

    def __init__(
        self,
        feature_selection: Optional[List[int]] = None,
        aggregation_method: str = "concat",
    ):
        """
        Args:
            feature_selection: 要融合的特征索引
            aggregation_method: 聚合方法
        """
        self.feature_selection = feature_selection
        self.aggregation_method = aggregation_method

    def fuse_features(
        self,
        data_list: List[np.ndarray],
        feature_indices_list: Optional[List[List[int]]] = None,
    ) -> np.ndarray:
        """
        融合指定特征

        Args:
            data_list: 数据列表
            feature_indices_list: 每个数据源要选择的特征索引

        Returns:
            融合后的特征数据
        """
        selected_features = []

        for i, data in enumerate(data_list):
            if feature_indices_list and i < len(feature_indices_list):
                indices = feature_indices_list[i]
                selected = data[:, indices] if indices else data
            else:
                selected = data

            selected_features.append(selected)

        if self.aggregation_method == "concat":
            return np.concatenate(selected_features, axis=1)
        elif self.aggregation_method == "stack":
            return np.stack(selected_features, axis=0)
        else:
            raise ValueError(f"不支持的聚合方法: {self.aggregation_method}")


class StatisticalDataFusion:
    """
    统计数据融合

    用于融合来自多方的统计信息
    """

    def __init__(self, stat_type: str = "mean"):
        """
        Args:
            stat_type: 统计类型 ('mean', 'sum', 'var', 'std', 'count')
        """
        self.stat_type = stat_type

    def fuse_statistics(
        self,
        stats_list: List[Dict[str, np.ndarray]],
        sample_counts: Optional[List[int]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        融合统计信息

        Args:
            stats_list: 统计信息列表
            sample_counts: 各方样本数量

        Returns:
            融合后的统计信息
        """
        if not stats_list:
            return {}

        result = {}

        # 获取所有统计键
        all_keys = set()
        for stats in stats_list:
            all_keys.update(stats.keys())

        for key in all_keys:
            values = [s[key] for s in stats_list if key in s]

            if self.stat_type == "mean":
                if sample_counts:
                    # 加权平均
                    total_samples = sum(sample_counts)
                    weighted_sum = sum(
                        v * c for v, c in zip(values, sample_counts)
                    )
                    result[key] = weighted_sum / total_samples
                else:
                    result[key] = np.mean(np.stack(values), axis=0)

            elif self.stat_type == "sum":
                result[key] = np.sum(np.stack(values), axis=0)

            elif self.stat_type == "var":
                # 合并方差（需要均值和样本数）
                result[key] = np.var(np.stack(values), axis=0)

            elif self.stat_type == "count":
                result[key] = np.sum(values)

        return result
