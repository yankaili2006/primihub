"""
Feature Encoding Base Classes
特征编码基础类

提供各种特征编码功能。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEncoderBase(ABC):
    """特征编码基类"""

    def __init__(self, columns: Optional[List[str]] = None):
        """
        初始化特征编码器

        Args:
            columns: 要编码的列，None表示自动检测分类列
        """
        self.columns = columns
        self._is_fitted = False
        self._encoding_maps = {}

    @abstractmethod
    def fit(self, data: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureEncoderBase':
        """拟合编码器"""
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """转换数据"""
        pass

    def fit_transform(self, data: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """拟合并转换数据"""
        self.fit(data, y)
        return self.transform(data)

    def get_encoding_maps(self) -> Dict[str, Any]:
        """获取编码映射"""
        return self._encoding_maps

    def _get_categorical_columns(self, data: pd.DataFrame) -> List[str]:
        """获取要编码的分类列"""
        if self.columns:
            return [c for c in self.columns if c in data.columns]
        # 自动检测分类列
        return data.select_dtypes(include=['object', 'category']).columns.tolist()


class OneHotEncoder(FeatureEncoderBase):
    """
    独热编码器

    将分类变量转换为二进制向量。
    """

    def __init__(self, columns: Optional[List[str]] = None,
                 drop_first: bool = False,
                 handle_unknown: str = "ignore",
                 max_categories: Optional[int] = None):
        """
        初始化独热编码器

        Args:
            columns: 要编码的列
            drop_first: 是否删除第一个类别（避免多重共线性）
            handle_unknown: 未知类别处理方式（ignore, error）
            max_categories: 最大类别数，超过则只保留频率最高的
        """
        super().__init__(columns)
        self.drop_first = drop_first
        self.handle_unknown = handle_unknown
        self.max_categories = max_categories
        self._categories = {}

    def fit(self, data: pd.DataFrame, y: Optional[pd.Series] = None) -> 'OneHotEncoder':
        """拟合独热编码器"""
        logger.info("Fitting OneHotEncoder")

        cols = self._get_categorical_columns(data)

        for col in cols:
            categories = data[col].value_counts()

            if self.max_categories and len(categories) > self.max_categories:
                # 只保留频率最高的类别
                categories = categories.head(self.max_categories)

            self._categories[col] = categories.index.tolist()

        self._encoding_maps = {"categories": self._categories}
        self._is_fitted = True
        logger.info(f"Fitted OneHotEncoder for {len(cols)} columns")
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """独热编码转换"""
        if not self._is_fitted:
            raise RuntimeError("OneHotEncoder is not fitted")

        result = data.copy()
        cols_to_drop = []

        for col, categories in self._categories.items():
            if col not in result.columns:
                continue

            for i, cat in enumerate(categories):
                if self.drop_first and i == 0:
                    continue

                new_col = f"{col}_{cat}"
                result[new_col] = (result[col] == cat).astype(int)

                # 处理未知类别
                if self.handle_unknown == "ignore":
                    unknown_mask = ~result[col].isin(categories)
                    result.loc[unknown_mask, new_col] = 0

            cols_to_drop.append(col)

        result = result.drop(columns=cols_to_drop)
        return result


class LabelEncoder(FeatureEncoderBase):
    """
    标签编码器

    将分类变量转换为整数。
    """

    def __init__(self, columns: Optional[List[str]] = None):
        super().__init__(columns)
        self._label_maps = {}

    def fit(self, data: pd.DataFrame, y: Optional[pd.Series] = None) -> 'LabelEncoder':
        """拟合标签编码器"""
        logger.info("Fitting LabelEncoder")

        cols = self._get_categorical_columns(data)

        for col in cols:
            unique_values = data[col].dropna().unique()
            self._label_maps[col] = {val: i for i, val in enumerate(unique_values)}

        self._encoding_maps = {"label_maps": self._label_maps}
        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """标签编码转换"""
        if not self._is_fitted:
            raise RuntimeError("LabelEncoder is not fitted")

        result = data.copy()

        for col, mapping in self._label_maps.items():
            if col not in result.columns:
                continue

            result[col] = result[col].map(mapping)
            # 未知值设为-1
            result[col] = result[col].fillna(-1).astype(int)

        return result

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """逆标签编码"""
        if not self._is_fitted:
            raise RuntimeError("LabelEncoder is not fitted")

        result = data.copy()

        for col, mapping in self._label_maps.items():
            if col not in result.columns:
                continue

            inverse_map = {v: k for k, v in mapping.items()}
            result[col] = result[col].map(inverse_map)

        return result


class OrdinalEncoder(FeatureEncoderBase):
    """
    顺序编码器

    将有序分类变量转换为整数，保持顺序关系。
    """

    def __init__(self, columns: Optional[List[str]] = None,
                 order_mapping: Optional[Dict[str, List]] = None):
        """
        初始化顺序编码器

        Args:
            columns: 要编码的列
            order_mapping: 每列的顺序映射，格式为 {列名: [值1, 值2, ...]}
        """
        super().__init__(columns)
        self.order_mapping = order_mapping or {}
        self._ordinal_maps = {}

    def fit(self, data: pd.DataFrame, y: Optional[pd.Series] = None) -> 'OrdinalEncoder':
        """拟合顺序编码器"""
        logger.info("Fitting OrdinalEncoder")

        cols = self._get_categorical_columns(data)

        for col in cols:
            if col in self.order_mapping:
                # 使用指定的顺序
                order = self.order_mapping[col]
            else:
                # 按出现频率排序
                order = data[col].value_counts().index.tolist()

            self._ordinal_maps[col] = {val: i for i, val in enumerate(order)}

        self._encoding_maps = {"ordinal_maps": self._ordinal_maps}
        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """顺序编码转换"""
        if not self._is_fitted:
            raise RuntimeError("OrdinalEncoder is not fitted")

        result = data.copy()

        for col, mapping in self._ordinal_maps.items():
            if col not in result.columns:
                continue

            result[col] = result[col].map(mapping)
            # 未知值设为-1
            result[col] = result[col].fillna(-1).astype(int)

        return result


class TargetEncoder(FeatureEncoderBase):
    """
    目标编码器

    使用目标变量的均值来编码分类变量。
    """

    def __init__(self, columns: Optional[List[str]] = None,
                 smoothing: float = 1.0,
                 min_samples: int = 1):
        """
        初始化目标编码器

        Args:
            columns: 要编码的列
            smoothing: 平滑参数
            min_samples: 最小样本数
        """
        super().__init__(columns)
        self.smoothing = smoothing
        self.min_samples = min_samples
        self._target_maps = {}
        self._global_mean = 0.0

    def fit(self, data: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TargetEncoder':
        """拟合目标编码器"""
        if y is None:
            raise ValueError("TargetEncoder requires target variable y")

        logger.info("Fitting TargetEncoder")

        self._global_mean = y.mean()
        cols = self._get_categorical_columns(data)

        for col in cols:
            # 计算每个类别的目标均值
            category_stats = data.groupby(col)[y.name].agg(['mean', 'count'])
            category_stats.columns = ['mean', 'count']

            # 应用平滑
            smoothed_mean = (
                category_stats['count'] * category_stats['mean'] +
                self.smoothing * self._global_mean
            ) / (category_stats['count'] + self.smoothing)

            self._target_maps[col] = smoothed_mean.to_dict()

        self._encoding_maps = {
            "target_maps": self._target_maps,
            "global_mean": self._global_mean,
        }
        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """目标编码转换"""
        if not self._is_fitted:
            raise RuntimeError("TargetEncoder is not fitted")

        result = data.copy()

        for col, mapping in self._target_maps.items():
            if col not in result.columns:
                continue

            result[col] = result[col].map(mapping)
            # 未知值使用全局均值
            result[col] = result[col].fillna(self._global_mean)

        return result


class BinaryEncoder(FeatureEncoderBase):
    """
    二进制编码器

    将分类变量转换为二进制表示。
    """

    def __init__(self, columns: Optional[List[str]] = None):
        super().__init__(columns)
        self._binary_maps = {}
        self._n_bits = {}

    def fit(self, data: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BinaryEncoder':
        """拟合二进制编码器"""
        logger.info("Fitting BinaryEncoder")

        cols = self._get_categorical_columns(data)

        for col in cols:
            unique_values = data[col].dropna().unique()
            n_unique = len(unique_values)
            n_bits = max(1, int(np.ceil(np.log2(n_unique + 1))))

            self._binary_maps[col] = {val: i for i, val in enumerate(unique_values)}
            self._n_bits[col] = n_bits

        self._encoding_maps = {"binary_maps": self._binary_maps, "n_bits": self._n_bits}
        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """二进制编码转换"""
        if not self._is_fitted:
            raise RuntimeError("BinaryEncoder is not fitted")

        result = data.copy()
        cols_to_drop = []

        for col, mapping in self._binary_maps.items():
            if col not in result.columns:
                continue

            n_bits = self._n_bits[col]
            encoded = result[col].map(mapping).fillna(0).astype(int)

            # 转换为二进制
            for i in range(n_bits):
                new_col = f"{col}_bit{i}"
                result[new_col] = (encoded >> i) & 1

            cols_to_drop.append(col)

        result = result.drop(columns=cols_to_drop)
        return result


class FrequencyEncoder(FeatureEncoderBase):
    """
    频率编码器

    使用类别出现的频率来编码分类变量。
    """

    def __init__(self, columns: Optional[List[str]] = None,
                 normalize: bool = True):
        """
        初始化频率编码器

        Args:
            columns: 要编码的列
            normalize: 是否归一化频率到[0,1]
        """
        super().__init__(columns)
        self.normalize = normalize
        self._freq_maps = {}

    def fit(self, data: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FrequencyEncoder':
        """拟合频率编码器"""
        logger.info("Fitting FrequencyEncoder")

        cols = self._get_categorical_columns(data)

        for col in cols:
            freq = data[col].value_counts()
            if self.normalize:
                freq = freq / len(data)
            self._freq_maps[col] = freq.to_dict()

        self._encoding_maps = {"freq_maps": self._freq_maps}
        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """频率编码转换"""
        if not self._is_fitted:
            raise RuntimeError("FrequencyEncoder is not fitted")

        result = data.copy()

        for col, mapping in self._freq_maps.items():
            if col not in result.columns:
                continue

            result[col] = result[col].map(mapping)
            # 未知值设为0
            result[col] = result[col].fillna(0)

        return result
