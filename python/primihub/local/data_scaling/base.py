"""
Data Scaling Base Classes
数据缩放基础类

提供各种数据缩放/标准化功能。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataScalerBase(ABC):
    """数据缩放基类"""

    def __init__(self, columns: Optional[List[str]] = None):
        """
        初始化数据缩放器

        Args:
            columns: 要缩放的列，None表示所有数值列
        """
        self.columns = columns
        self._is_fitted = False
        self._params = {}

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'DataScalerBase':
        """拟合缩放器"""
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """转换数据"""
        pass

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """拟合并转换数据"""
        self.fit(data)
        return self.transform(data)

    @abstractmethod
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """逆转换"""
        pass

    def get_params(self) -> Dict[str, Any]:
        """获取拟合参数"""
        return self._params

    def _get_numeric_columns(self, data: pd.DataFrame) -> List[str]:
        """获取要处理的数值列"""
        if self.columns:
            return [c for c in self.columns if c in data.columns]
        return data.select_dtypes(include=[np.number]).columns.tolist()


class StandardScaler(DataScalerBase):
    """
    标准化缩放器

    将数据转换为均值为0，标准差为1的分布。
    公式: z = (x - mean) / std
    """

    def __init__(self, columns: Optional[List[str]] = None,
                 with_mean: bool = True,
                 with_std: bool = True):
        """
        初始化标准化缩放器

        Args:
            columns: 要缩放的列
            with_mean: 是否中心化（减均值）
            with_std: 是否缩放（除标准差）
        """
        super().__init__(columns)
        self.with_mean = with_mean
        self.with_std = with_std
        self._means = {}
        self._stds = {}

    def fit(self, data: pd.DataFrame) -> 'StandardScaler':
        """拟合标准化参数"""
        logger.info("Fitting StandardScaler")

        cols = self._get_numeric_columns(data)

        for col in cols:
            col_data = data[col].dropna()
            self._means[col] = col_data.mean()
            self._stds[col] = col_data.std()
            if self._stds[col] == 0:
                self._stds[col] = 1.0

        self._params = {"means": self._means, "stds": self._stds}
        self._is_fitted = True
        logger.info(f"Fitted StandardScaler for {len(cols)} columns")
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """标准化转换"""
        if not self._is_fitted:
            raise RuntimeError("StandardScaler is not fitted")

        result = data.copy()
        cols = self._get_numeric_columns(data)

        for col in cols:
            if col in self._means and col in self._stds:
                if self.with_mean:
                    result[col] = result[col] - self._means[col]
                if self.with_std:
                    result[col] = result[col] / self._stds[col]

        return result

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """逆标准化"""
        if not self._is_fitted:
            raise RuntimeError("StandardScaler is not fitted")

        result = data.copy()
        cols = self._get_numeric_columns(data)

        for col in cols:
            if col in self._means and col in self._stds:
                if self.with_std:
                    result[col] = result[col] * self._stds[col]
                if self.with_mean:
                    result[col] = result[col] + self._means[col]

        return result


class MinMaxScaler(DataScalerBase):
    """
    最小-最大缩放器

    将数据缩放到指定范围。
    公式: x_scaled = (x - min) / (max - min) * (feature_range[1] - feature_range[0]) + feature_range[0]
    """

    def __init__(self, columns: Optional[List[str]] = None,
                 feature_range: tuple = (0, 1)):
        """
        初始化最小-最大缩放器

        Args:
            columns: 要缩放的列
            feature_range: 目标范围
        """
        super().__init__(columns)
        self.feature_range = feature_range
        self._mins = {}
        self._maxs = {}

    def fit(self, data: pd.DataFrame) -> 'MinMaxScaler':
        """拟合最小-最大参数"""
        logger.info("Fitting MinMaxScaler")

        cols = self._get_numeric_columns(data)

        for col in cols:
            col_data = data[col].dropna()
            self._mins[col] = col_data.min()
            self._maxs[col] = col_data.max()
            if self._mins[col] == self._maxs[col]:
                self._maxs[col] = self._mins[col] + 1.0

        self._params = {"mins": self._mins, "maxs": self._maxs}
        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """最小-最大转换"""
        if not self._is_fitted:
            raise RuntimeError("MinMaxScaler is not fitted")

        result = data.copy()
        cols = self._get_numeric_columns(data)
        range_min, range_max = self.feature_range

        for col in cols:
            if col in self._mins and col in self._maxs:
                data_range = self._maxs[col] - self._mins[col]
                result[col] = (result[col] - self._mins[col]) / data_range
                result[col] = result[col] * (range_max - range_min) + range_min

        return result

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """逆最小-最大转换"""
        if not self._is_fitted:
            raise RuntimeError("MinMaxScaler is not fitted")

        result = data.copy()
        cols = self._get_numeric_columns(data)
        range_min, range_max = self.feature_range

        for col in cols:
            if col in self._mins and col in self._maxs:
                data_range = self._maxs[col] - self._mins[col]
                result[col] = (result[col] - range_min) / (range_max - range_min)
                result[col] = result[col] * data_range + self._mins[col]

        return result


class MaxAbsScaler(DataScalerBase):
    """
    最大绝对值缩放器

    将数据按最大绝对值缩放到[-1, 1]范围。
    公式: x_scaled = x / max(|x|)
    """

    def __init__(self, columns: Optional[List[str]] = None):
        super().__init__(columns)
        self._max_abs = {}

    def fit(self, data: pd.DataFrame) -> 'MaxAbsScaler':
        """拟合最大绝对值参数"""
        logger.info("Fitting MaxAbsScaler")

        cols = self._get_numeric_columns(data)

        for col in cols:
            col_data = data[col].dropna()
            self._max_abs[col] = col_data.abs().max()
            if self._max_abs[col] == 0:
                self._max_abs[col] = 1.0

        self._params = {"max_abs": self._max_abs}
        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """最大绝对值转换"""
        if not self._is_fitted:
            raise RuntimeError("MaxAbsScaler is not fitted")

        result = data.copy()
        cols = self._get_numeric_columns(data)

        for col in cols:
            if col in self._max_abs:
                result[col] = result[col] / self._max_abs[col]

        return result

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """逆最大绝对值转换"""
        if not self._is_fitted:
            raise RuntimeError("MaxAbsScaler is not fitted")

        result = data.copy()
        cols = self._get_numeric_columns(data)

        for col in cols:
            if col in self._max_abs:
                result[col] = result[col] * self._max_abs[col]

        return result


class RobustScaler(DataScalerBase):
    """
    鲁棒缩放器

    使用中位数和四分位距进行缩放，对异常值更鲁棒。
    公式: x_scaled = (x - median) / IQR
    """

    def __init__(self, columns: Optional[List[str]] = None,
                 with_centering: bool = True,
                 with_scaling: bool = True,
                 quantile_range: tuple = (25.0, 75.0)):
        """
        初始化鲁棒缩放器

        Args:
            columns: 要缩放的列
            with_centering: 是否中心化（减中位数）
            with_scaling: 是否缩放（除IQR）
            quantile_range: 分位数范围
        """
        super().__init__(columns)
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self._medians = {}
        self._iqrs = {}

    def fit(self, data: pd.DataFrame) -> 'RobustScaler':
        """拟合鲁棒缩放参数"""
        logger.info("Fitting RobustScaler")

        cols = self._get_numeric_columns(data)
        q_low, q_high = self.quantile_range

        for col in cols:
            col_data = data[col].dropna()
            self._medians[col] = col_data.median()
            q1 = col_data.quantile(q_low / 100.0)
            q3 = col_data.quantile(q_high / 100.0)
            self._iqrs[col] = q3 - q1
            if self._iqrs[col] == 0:
                self._iqrs[col] = 1.0

        self._params = {"medians": self._medians, "iqrs": self._iqrs}
        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """鲁棒缩放转换"""
        if not self._is_fitted:
            raise RuntimeError("RobustScaler is not fitted")

        result = data.copy()
        cols = self._get_numeric_columns(data)

        for col in cols:
            if col in self._medians and col in self._iqrs:
                if self.with_centering:
                    result[col] = result[col] - self._medians[col]
                if self.with_scaling:
                    result[col] = result[col] / self._iqrs[col]

        return result

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """逆鲁棒缩放"""
        if not self._is_fitted:
            raise RuntimeError("RobustScaler is not fitted")

        result = data.copy()
        cols = self._get_numeric_columns(data)

        for col in cols:
            if col in self._medians and col in self._iqrs:
                if self.with_scaling:
                    result[col] = result[col] * self._iqrs[col]
                if self.with_centering:
                    result[col] = result[col] + self._medians[col]

        return result


class Normalizer(DataScalerBase):
    """
    归一化器

    将每个样本（行）归一化为单位范数。
    """

    def __init__(self, columns: Optional[List[str]] = None,
                 norm: str = "l2"):
        """
        初始化归一化器

        Args:
            columns: 要归一化的列
            norm: 范数类型（l1, l2, max）
        """
        super().__init__(columns)
        self.norm = norm

    def fit(self, data: pd.DataFrame) -> 'Normalizer':
        """归一化器不需要拟合"""
        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """归一化转换"""
        result = data.copy()
        cols = self._get_numeric_columns(data)

        if not cols:
            return result

        numeric_data = result[cols].values

        if self.norm == "l1":
            norms = np.abs(numeric_data).sum(axis=1, keepdims=True)
        elif self.norm == "l2":
            norms = np.sqrt((numeric_data ** 2).sum(axis=1, keepdims=True))
        elif self.norm == "max":
            norms = np.abs(numeric_data).max(axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown norm: {self.norm}")

        norms[norms == 0] = 1.0
        normalized = numeric_data / norms

        for i, col in enumerate(cols):
            result[col] = normalized[:, i]

        return result

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """归一化不可逆"""
        raise NotImplementedError("Normalizer does not support inverse_transform")
