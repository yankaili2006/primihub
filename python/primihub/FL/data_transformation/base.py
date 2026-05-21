"""
Federated Learning Data Transformation Base Classes
联邦学习数据转换基础类
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class DataTransformationBase(ABC):
    """数据转换基类"""

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
        """拟合转换器"""
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """转换数据"""
        pass

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    @abstractmethod
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """逆转换"""
        pass


class LogTransformer(DataTransformationBase):
    """
    对数转换

    支持多种对数变换
    """

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        method: str = "log1p",
        base: float = np.e,
    ):
        """
        Args:
            method: 转换方法 ('log', 'log1p', 'log10')
            base: 对数底数
        """
        super().__init__(FL_type, role, channel)
        self.method = method
        self.base = base
        self.shift_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """计算转换参数"""
        if self.method == "log":
            # 需要处理非正值
            min_val = np.min(X)
            if min_val <= 0:
                self.shift_ = abs(min_val) + 1
            else:
                self.shift_ = 0

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """对数转换"""
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        if self.method == "log1p":
            return np.log1p(np.abs(X))
        elif self.method == "log":
            return np.log(X + self.shift_) / np.log(self.base)
        elif self.method == "log10":
            return np.log10(np.abs(X) + 1)
        else:
            return np.log1p(np.abs(X))

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """逆对数转换"""
        if self.method == "log1p":
            return np.expm1(X)
        elif self.method == "log":
            return np.power(self.base, X) - self.shift_
        elif self.method == "log10":
            return np.power(10, X) - 1
        else:
            return np.expm1(X)


class BoxCoxTransformer(DataTransformationBase):
    """
    Box-Cox转换

    使数据更接近正态分布
    """

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        standardize: bool = True,
    ):
        super().__init__(FL_type, role, channel)
        self.standardize = standardize
        self.lambdas_ = None
        self.shift_ = None
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """估计Box-Cox参数"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_features = X.shape[1]
        self.lambdas_ = np.zeros(n_features)
        self.shift_ = np.zeros(n_features)

        for col in range(n_features):
            col_data = X[:, col]

            # 确保数据为正
            min_val = np.min(col_data)
            if min_val <= 0:
                self.shift_[col] = abs(min_val) + 1
                col_data = col_data + self.shift_[col]

            # 估计lambda参数
            try:
                _, self.lambdas_[col] = stats.boxcox(col_data[~np.isnan(col_data)])
            except Exception:
                self.lambdas_[col] = 1.0

        # 联邦场景下同步
        if self.FL_type == "H" and self.channel:
            self._sync_lambdas()

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Box-Cox转换"""
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        result = np.zeros_like(X, dtype=np.float64)

        for col in range(X.shape[1]):
            if col < len(self.lambdas_):
                col_data = X[:, col] + self.shift_[col]
                lmbda = self.lambdas_[col]

                if abs(lmbda) < 1e-10:
                    result[:, col] = np.log(col_data)
                else:
                    result[:, col] = (np.power(col_data, lmbda) - 1) / lmbda
            else:
                result[:, col] = X[:, col]

        if self.standardize:
            self.mean_ = np.mean(result, axis=0)
            self.std_ = np.std(result, axis=0)
            result = (result - self.mean_) / (self.std_ + 1e-10)

        return result

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """逆Box-Cox转换"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        result = X.copy()

        if self.standardize:
            result = result * (self.std_ + 1e-10) + self.mean_

        for col in range(result.shape[1]):
            if col < len(self.lambdas_):
                lmbda = self.lambdas_[col]

                if abs(lmbda) < 1e-10:
                    result[:, col] = np.exp(result[:, col])
                else:
                    result[:, col] = np.power(result[:, col] * lmbda + 1, 1 / lmbda)

                result[:, col] = result[:, col] - self.shift_[col]

        return result

    def _sync_lambdas(self):
        """同步lambda参数"""
        if self.role == "client":
            self.channel.send("local_lambdas", self.lambdas_.tolist())
            global_lambdas = self.channel.recv("global_lambdas")
            self.lambdas_ = np.array(global_lambdas)

        elif self.role == "server":
            all_lambdas = self.channel.recv_all("local_lambdas")
            # 取平均
            global_lambdas = self.lambdas_.copy()
            count = 1
            for party_lambdas in all_lambdas.values():
                global_lambdas += np.array(party_lambdas)
                count += 1
            global_lambdas /= count

            self.lambdas_ = global_lambdas
            self.channel.send_all("global_lambdas", global_lambdas.tolist())


class YeoJohnsonTransformer(DataTransformationBase):
    """
    Yeo-Johnson转换

    类似Box-Cox但支持负值
    """

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        standardize: bool = True,
    ):
        super().__init__(FL_type, role, channel)
        self.standardize = standardize
        self.lambdas_ = None
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """估计Yeo-Johnson参数"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_features = X.shape[1]
        self.lambdas_ = np.zeros(n_features)

        for col in range(n_features):
            col_data = X[:, col]
            valid_data = col_data[~np.isnan(col_data)]

            try:
                _, self.lambdas_[col] = stats.yeojohnson(valid_data)
            except Exception:
                self.lambdas_[col] = 1.0

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Yeo-Johnson转换"""
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        result = np.zeros_like(X, dtype=np.float64)

        for col in range(X.shape[1]):
            if col < len(self.lambdas_):
                result[:, col] = self._yeo_johnson_transform(X[:, col], self.lambdas_[col])
            else:
                result[:, col] = X[:, col]

        if self.standardize:
            self.mean_ = np.mean(result, axis=0)
            self.std_ = np.std(result, axis=0)
            result = (result - self.mean_) / (self.std_ + 1e-10)

        return result

    def _yeo_johnson_transform(self, x: np.ndarray, lmbda: float) -> np.ndarray:
        """单列Yeo-Johnson转换"""
        result = np.zeros_like(x)

        pos_mask = x >= 0
        neg_mask = ~pos_mask

        if abs(lmbda) < 1e-10:
            result[pos_mask] = np.log1p(x[pos_mask])
        else:
            result[pos_mask] = (np.power(x[pos_mask] + 1, lmbda) - 1) / lmbda

        if abs(lmbda - 2) < 1e-10:
            result[neg_mask] = -np.log1p(-x[neg_mask])
        else:
            result[neg_mask] = -(np.power(-x[neg_mask] + 1, 2 - lmbda) - 1) / (2 - lmbda)

        return result

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """逆Yeo-Johnson转换"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        result = X.copy()

        if self.standardize:
            result = result * (self.std_ + 1e-10) + self.mean_

        for col in range(result.shape[1]):
            if col < len(self.lambdas_):
                result[:, col] = self._yeo_johnson_inverse(result[:, col], self.lambdas_[col])

        return result

    def _yeo_johnson_inverse(self, x: np.ndarray, lmbda: float) -> np.ndarray:
        """逆Yeo-Johnson转换"""
        result = np.zeros_like(x)

        for i, val in enumerate(x):
            if val >= 0:
                if abs(lmbda) < 1e-10:
                    result[i] = np.expm1(val)
                else:
                    result[i] = np.power(val * lmbda + 1, 1 / lmbda) - 1
            else:
                if abs(lmbda - 2) < 1e-10:
                    result[i] = -np.expm1(-val)
                else:
                    result[i] = 1 - np.power(-val * (2 - lmbda) + 1, 1 / (2 - lmbda))

        return result


class RankTransformer(DataTransformationBase):
    """
    排名转换

    将数据转换为排名或分位数
    """

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        output_distribution: str = "uniform",
        n_quantiles: int = 1000,
    ):
        """
        Args:
            output_distribution: 输出分布 ('uniform', 'normal')
            n_quantiles: 分位数数量
        """
        super().__init__(FL_type, role, channel)
        self.output_distribution = output_distribution
        self.n_quantiles = n_quantiles
        self.quantiles_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """计算分位数"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.quantiles_ = []
        quantile_points = np.linspace(0, 100, self.n_quantiles)

        for col in range(X.shape[1]):
            col_data = X[:, col]
            quantiles = np.percentile(col_data[~np.isnan(col_data)], quantile_points)
            self.quantiles_.append(quantiles)

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """排名转换"""
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        result = np.zeros_like(X, dtype=np.float64)

        for col in range(X.shape[1]):
            if col < len(self.quantiles_):
                # 找到每个值的分位数位置
                ranks = np.searchsorted(self.quantiles_[col], X[:, col]) / self.n_quantiles
                ranks = np.clip(ranks, 0, 1)

                if self.output_distribution == "normal":
                    # 转换为正态分布
                    result[:, col] = stats.norm.ppf(np.clip(ranks, 0.001, 0.999))
                else:
                    result[:, col] = ranks
            else:
                result[:, col] = X[:, col]

        return result

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """逆排名转换"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        result = np.zeros_like(X, dtype=np.float64)

        for col in range(X.shape[1]):
            if col < len(self.quantiles_):
                if self.output_distribution == "normal":
                    ranks = stats.norm.cdf(X[:, col])
                else:
                    ranks = X[:, col]

                # 从分位数查找原始值
                indices = (ranks * (self.n_quantiles - 1)).astype(int)
                indices = np.clip(indices, 0, self.n_quantiles - 1)
                result[:, col] = self.quantiles_[col][indices]
            else:
                result[:, col] = X[:, col]

        return result
