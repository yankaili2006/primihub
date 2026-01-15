"""
Federated Learning Feature Imputation Base Classes
联邦学习特征填充基础类
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class FLImputerBase(ABC):
    """联邦学习缺失值填充基类"""

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        missing_values: float = np.nan,
    ):
        self.FL_type = FL_type
        self.role = role
        self.channel = channel
        self.missing_values = missing_values
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """拟合填充器"""
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """填充缺失值"""
        pass

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def _get_mask(self, X: np.ndarray) -> np.ndarray:
        """获取缺失值掩码"""
        if np.isnan(self.missing_values):
            return np.isnan(X)
        else:
            return X == self.missing_values


class FLMeanImputer(FLImputerBase):
    """
    联邦学习均值填充器

    使用全局均值填充缺失值
    """

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        missing_values: float = np.nan,
    ):
        super().__init__(FL_type, role, channel, missing_values)
        self.statistics_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """计算均值"""
        mask = self._get_mask(X)
        n_samples = X.shape[0] - np.sum(mask, axis=0)
        local_sum = np.nansum(X, axis=0)

        if self.FL_type == "H" and self.channel:
            self.statistics_ = self._federated_mean(local_sum, n_samples)
        else:
            self.statistics_ = local_sum / np.maximum(n_samples, 1)

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """填充缺失值"""
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        result = X.copy()
        mask = self._get_mask(X)

        for col in range(X.shape[1]):
            result[mask[:, col], col] = self.statistics_[col]

        return result

    def _federated_mean(self, local_sum: np.ndarray, n_samples: np.ndarray) -> np.ndarray:
        """联邦均值计算"""
        if self.role == "client":
            self.channel.send("local_sum", local_sum.tolist())
            self.channel.send("n_samples", n_samples.tolist())
            return np.array(self.channel.recv("global_mean"))

        elif self.role == "server":
            all_sums = self.channel.recv_all("local_sum")
            all_n = self.channel.recv_all("n_samples")

            global_sum = local_sum.copy()
            global_n = n_samples.copy()

            for party in all_sums:
                global_sum += np.array(all_sums[party])
                global_n += np.array(all_n[party])

            global_mean = global_sum / np.maximum(global_n, 1)
            self.channel.send_all("global_mean", global_mean.tolist())

            return global_mean

        return local_sum / np.maximum(n_samples, 1)


class FLMedianImputer(FLImputerBase):
    """
    联邦学习中位数填充器

    使用中位数填充缺失值（近似计算）
    """

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        missing_values: float = np.nan,
        n_quantiles: int = 100,
    ):
        super().__init__(FL_type, role, channel, missing_values)
        self.n_quantiles = n_quantiles
        self.statistics_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """计算中位数"""
        self.statistics_ = np.nanmedian(X, axis=0)

        if self.FL_type == "H" and self.channel:
            self.statistics_ = self._federated_median(X)

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """填充缺失值"""
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        result = X.copy()
        mask = self._get_mask(X)

        for col in range(X.shape[1]):
            result[mask[:, col], col] = self.statistics_[col]

        return result

    def _federated_median(self, X: np.ndarray) -> np.ndarray:
        """联邦中位数计算（使用分位数近似）"""
        # 使用分位数草图进行近似
        quantile_points = np.linspace(0, 100, self.n_quantiles)
        local_quantiles = np.percentile(X, quantile_points, axis=0)

        if self.role == "client":
            self.channel.send("local_quantiles", local_quantiles.tolist())
            return np.array(self.channel.recv("global_median"))

        elif self.role == "server":
            all_quantiles = self.channel.recv_all("local_quantiles")

            # 合并分位数（简化：取平均）
            merged = local_quantiles.copy()
            for q in all_quantiles.values():
                merged += np.array(q)
            merged /= (len(all_quantiles) + 1)

            # 取中位数（第50分位数）
            median_idx = self.n_quantiles // 2
            global_median = merged[median_idx]

            self.channel.send_all("global_median", global_median.tolist())
            return global_median

        return np.nanmedian(X, axis=0)


class FLKNNImputer(FLImputerBase):
    """
    联邦学习KNN填充器

    使用K近邻填充缺失值
    """

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        missing_values: float = np.nan,
        n_neighbors: int = 5,
    ):
        super().__init__(FL_type, role, channel, missing_values)
        self.n_neighbors = n_neighbors
        self._fitted_data = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """存储非缺失数据"""
        self._fitted_data = X.copy()
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """使用KNN填充"""
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        result = X.copy()
        mask = self._get_mask(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if mask[i, j]:
                    result[i, j] = self._knn_impute(i, j, X)

        return result

    def _knn_impute(self, row_idx: int, col_idx: int, X: np.ndarray) -> float:
        """KNN填充单个值"""
        # 找到col_idx列非缺失的行
        valid_mask = ~self._get_mask(self._fitted_data[:, col_idx])
        valid_data = self._fitted_data[valid_mask]

        if len(valid_data) == 0:
            return 0.0

        # 计算距离（只用非缺失的特征）
        row = X[row_idx].copy()
        row_mask = ~self._get_mask(row)
        row[~row_mask] = 0

        distances = []
        for valid_row in valid_data:
            valid_row_copy = valid_row.copy()
            valid_row_copy[~row_mask] = 0
            dist = np.sqrt(np.sum((row[row_mask] - valid_row_copy[row_mask]) ** 2))
            distances.append(dist)

        distances = np.array(distances)
        k = min(self.n_neighbors, len(distances))
        nearest_idx = np.argsort(distances)[:k]

        return np.mean(valid_data[nearest_idx, col_idx])


class FLIterativeImputer(FLImputerBase):
    """
    联邦学习迭代填充器

    使用迭代方法填充缺失值
    """

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        missing_values: float = np.nan,
        max_iter: int = 10,
        tol: float = 1e-3,
    ):
        super().__init__(FL_type, role, channel, missing_values)
        self.max_iter = max_iter
        self.tol = tol
        self.initial_imputer_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """拟合迭代填充器"""
        # 使用均值做初始填充
        self.initial_imputer_ = FLMeanImputer(
            FL_type=self.FL_type, role=self.role, channel=self.channel
        )
        self.initial_imputer_.fit(X)

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """迭代填充"""
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        mask = self._get_mask(X)
        result = self.initial_imputer_.transform(X)

        for iteration in range(self.max_iter):
            prev_result = result.copy()

            # 逐列更新
            for col in range(X.shape[1]):
                col_mask = mask[:, col]
                if not np.any(col_mask):
                    continue

                # 用其他列预测当前列
                other_cols = [i for i in range(X.shape[1]) if i != col]
                X_train = result[~col_mask][:, other_cols]
                y_train = result[~col_mask, col]

                X_pred = result[col_mask][:, other_cols]

                if len(X_train) > 0 and len(X_pred) > 0:
                    # 简单线性回归
                    try:
                        coeffs = np.linalg.lstsq(
                            np.c_[np.ones(len(X_train)), X_train],
                            y_train,
                            rcond=None
                        )[0]
                        predictions = np.c_[np.ones(len(X_pred)), X_pred] @ coeffs
                        result[col_mask, col] = predictions
                    except Exception:
                        pass

            # 检查收敛
            diff = np.max(np.abs(result - prev_result))
            if diff < self.tol:
                break

        return result
