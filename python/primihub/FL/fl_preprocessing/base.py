"""
Federated Learning Preprocessing Base Classes
联邦学习预处理基础类

提供联邦学习场景下的数据清洗、异常检测、数据验证等功能
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class FLPreprocessBase(ABC):
    """联邦学习预处理基类"""

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
    ):
        """
        初始化预处理器

        Args:
            FL_type: 联邦学习类型 ('H' 水平, 'V' 纵向)
            role: 角色 ('client', 'server', 'host', 'guest')
            channel: 通信通道
        """
        self.FL_type = FL_type
        self.role = role
        self.channel = channel
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """拟合预处理器"""
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """转换数据"""
        pass

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """拟合并转换数据"""
        self.fit(X, y)
        return self.transform(X)


class FLDataCleaner(FLPreprocessBase):
    """
    联邦学习数据清洗器

    支持多种数据清洗操作
    """

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        remove_duplicates: bool = True,
        handle_missing: str = "drop",
        missing_threshold: float = 0.5,
        remove_constant: bool = True,
    ):
        """
        Args:
            remove_duplicates: 是否移除重复行
            handle_missing: 缺失值处理方式 ('drop', 'fill', 'keep')
            missing_threshold: 缺失值比例阈值
            remove_constant: 是否移除常量列
        """
        super().__init__(FL_type, role, channel)
        self.remove_duplicates = remove_duplicates
        self.handle_missing = handle_missing
        self.missing_threshold = missing_threshold
        self.remove_constant = remove_constant

        self.valid_columns_ = None
        self.fill_values_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        分析数据并确定清洗策略

        Args:
            X: 输入数据
            y: 标签 (可选)
        """
        n_samples, n_features = X.shape

        # 计算每列的缺失值比例
        missing_ratio = np.isnan(X).sum(axis=0) / n_samples

        # 确定有效列
        valid_mask = missing_ratio < self.missing_threshold

        # 移除常量列
        if self.remove_constant:
            with np.errstate(all='ignore'):
                std = np.nanstd(X, axis=0)
                valid_mask &= std > 1e-10

        self.valid_columns_ = np.where(valid_mask)[0]

        # 计算填充值
        if self.handle_missing == "fill":
            self.fill_values_ = np.nanmean(X, axis=0)

        # 联邦场景下的同步
        if self.FL_type == "H" and self.channel:
            self._sync_valid_columns()

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        执行数据清洗

        Args:
            X: 输入数据

        Returns:
            清洗后的数据
        """
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        result = X.copy()

        # 只保留有效列
        if self.valid_columns_ is not None:
            result = result[:, self.valid_columns_]

        # 处理缺失值
        if self.handle_missing == "fill" and self.fill_values_ is not None:
            fill_vals = self.fill_values_[self.valid_columns_]
            for i in range(result.shape[1]):
                mask = np.isnan(result[:, i])
                result[mask, i] = fill_vals[i]
        elif self.handle_missing == "drop":
            mask = ~np.any(np.isnan(result), axis=1)
            result = result[mask]

        # 移除重复行
        if self.remove_duplicates:
            result = np.unique(result, axis=0)

        return result

    def _sync_valid_columns(self):
        """在联邦场景中同步有效列"""
        if self.role == "client":
            self.channel.send("valid_columns", self.valid_columns_.tolist())
            global_valid = self.channel.recv("global_valid_columns")
            self.valid_columns_ = np.array(global_valid)
        elif self.role == "server":
            all_valid = self.channel.recv_all("valid_columns")
            # 取交集
            global_valid = set(self.valid_columns_.tolist())
            for client_valid in all_valid.values():
                global_valid &= set(client_valid)
            global_valid = sorted(list(global_valid))
            self.valid_columns_ = np.array(global_valid)
            self.channel.send_all("global_valid_columns", global_valid)


class FLOutlierDetector(FLPreprocessBase):
    """
    联邦学习异常值检测器

    支持多种异常检测方法
    """

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        method: str = "zscore",
        threshold: float = 3.0,
        contamination: float = 0.1,
    ):
        """
        Args:
            method: 检测方法 ('zscore', 'iqr', 'isolation_forest')
            threshold: 阈值
            contamination: 异常比例
        """
        super().__init__(FL_type, role, channel)
        self.method = method
        self.threshold = threshold
        self.contamination = contamination

        self.mean_ = None
        self.std_ = None
        self.q1_ = None
        self.q3_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        计算异常检测参数

        Args:
            X: 输入数据
        """
        if self.method == "zscore":
            self.mean_ = np.nanmean(X, axis=0)
            self.std_ = np.nanstd(X, axis=0)

            # 联邦场景下聚合
            if self.FL_type == "H" and self.channel:
                self._sync_zscore_params(X.shape[0])

        elif self.method == "iqr":
            self.q1_ = np.nanpercentile(X, 25, axis=0)
            self.q3_ = np.nanpercentile(X, 75, axis=0)

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        标记并处理异常值

        Args:
            X: 输入数据

        Returns:
            处理后的数据（异常值被移除）
        """
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        outlier_mask = self.detect_outliers(X)
        return X[~outlier_mask]

    def detect_outliers(self, X: np.ndarray) -> np.ndarray:
        """
        检测异常值

        Args:
            X: 输入数据

        Returns:
            布尔数组，True 表示异常
        """
        if self.method == "zscore":
            with np.errstate(all='ignore'):
                z_scores = np.abs((X - self.mean_) / (self.std_ + 1e-10))
                outliers = np.any(z_scores > self.threshold, axis=1)

        elif self.method == "iqr":
            iqr = self.q3_ - self.q1_
            lower = self.q1_ - 1.5 * iqr
            upper = self.q3_ + 1.5 * iqr
            outliers = np.any((X < lower) | (X > upper), axis=1)

        else:
            outliers = np.zeros(X.shape[0], dtype=bool)

        return outliers

    def _sync_zscore_params(self, n_samples: int):
        """同步 Z-score 参数"""
        if self.role == "client":
            self.channel.send("local_sum", (self.mean_ * n_samples).tolist())
            self.channel.send("local_sq_sum", ((self.std_**2 + self.mean_**2) * n_samples).tolist())
            self.channel.send("n_samples", n_samples)

            self.mean_ = np.array(self.channel.recv("global_mean"))
            self.std_ = np.array(self.channel.recv("global_std"))

        elif self.role == "server":
            all_sums = self.channel.recv_all("local_sum")
            all_sq_sums = self.channel.recv_all("local_sq_sum")
            all_n = self.channel.recv_all("n_samples")

            total_n = sum(all_n.values())
            global_sum = np.sum([np.array(s) for s in all_sums.values()], axis=0)
            global_sq_sum = np.sum([np.array(s) for s in all_sq_sums.values()], axis=0)

            self.mean_ = global_sum / total_n
            self.std_ = np.sqrt(global_sq_sum / total_n - self.mean_**2)

            self.channel.send_all("global_mean", self.mean_.tolist())
            self.channel.send_all("global_std", self.std_.tolist())


class FLDataValidator(FLPreprocessBase):
    """
    联邦学习数据验证器

    验证数据质量和一致性
    """

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        check_types: bool = True,
        check_range: bool = True,
        check_consistency: bool = True,
    ):
        """
        Args:
            check_types: 检查数据类型
            check_range: 检查数值范围
            check_consistency: 检查数据一致性
        """
        super().__init__(FL_type, role, channel)
        self.check_types = check_types
        self.check_range = check_range
        self.check_consistency = check_consistency

        self.dtypes_ = None
        self.min_values_ = None
        self.max_values_ = None
        self.validation_report_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        学习数据特征

        Args:
            X: 输入数据
        """
        self.dtypes_ = X.dtype
        self.min_values_ = np.nanmin(X, axis=0)
        self.max_values_ = np.nanmax(X, axis=0)

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        验证并返回数据

        Args:
            X: 输入数据

        Returns:
            验证通过的数据
        """
        self.validation_report_ = self.validate(X)

        if self.validation_report_["valid"]:
            return X
        else:
            logger.warning(f"数据验证警告: {self.validation_report_['warnings']}")
            return X

    def validate(self, X: np.ndarray) -> Dict[str, Any]:
        """
        验证数据

        Args:
            X: 输入数据

        Returns:
            验证报告
        """
        report = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "stats": {},
        }

        # 检查数据形状
        if X.ndim != 2:
            report["errors"].append(f"数据维度错误: 期望2维, 实际{X.ndim}维")
            report["valid"] = False

        # 检查数据类型
        if self.check_types and self.dtypes_ is not None:
            if X.dtype != self.dtypes_:
                report["warnings"].append(
                    f"数据类型不一致: 期望{self.dtypes_}, 实际{X.dtype}"
                )

        # 检查数值范围
        if self.check_range and self.min_values_ is not None:
            current_min = np.nanmin(X, axis=0)
            current_max = np.nanmax(X, axis=0)

            if np.any(current_min < self.min_values_ * 0.9):
                report["warnings"].append("部分特征值低于历史最小值")

            if np.any(current_max > self.max_values_ * 1.1):
                report["warnings"].append("部分特征值高于历史最大值")

        # 检查缺失值
        missing_count = np.isnan(X).sum()
        missing_ratio = missing_count / X.size
        report["stats"]["missing_ratio"] = float(missing_ratio)

        if missing_ratio > 0.3:
            report["warnings"].append(f"缺失值比例过高: {missing_ratio:.2%}")

        # 检查重复行
        unique_rows = len(np.unique(X, axis=0))
        duplicate_ratio = 1 - unique_rows / len(X)
        report["stats"]["duplicate_ratio"] = float(duplicate_ratio)

        if duplicate_ratio > 0.5:
            report["warnings"].append(f"重复行比例过高: {duplicate_ratio:.2%}")

        return report


class FLFeatureFilter(FLPreprocessBase):
    """
    联邦学习特征过滤器

    基于统计指标过滤特征
    """

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        variance_threshold: float = 0.0,
        correlation_threshold: float = 0.95,
        missing_threshold: float = 0.5,
    ):
        """
        Args:
            variance_threshold: 方差阈值
            correlation_threshold: 相关性阈值
            missing_threshold: 缺失值阈值
        """
        super().__init__(FL_type, role, channel)
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.missing_threshold = missing_threshold

        self.selected_features_ = None
        self.feature_variances_ = None
        self.feature_correlations_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        分析特征并确定过滤策略

        Args:
            X: 输入数据
        """
        n_features = X.shape[1]
        selected = np.ones(n_features, dtype=bool)

        # 方差过滤
        self.feature_variances_ = np.nanvar(X, axis=0)
        selected &= self.feature_variances_ > self.variance_threshold

        # 缺失值过滤
        missing_ratio = np.isnan(X).sum(axis=0) / X.shape[0]
        selected &= missing_ratio < self.missing_threshold

        # 相关性过滤 (移除高度相关的特征)
        valid_X = X[:, selected]
        if valid_X.shape[1] > 1:
            corr_matrix = np.corrcoef(valid_X.T)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

            to_remove = set()
            for i in range(corr_matrix.shape[0]):
                for j in range(i + 1, corr_matrix.shape[1]):
                    if abs(corr_matrix[i, j]) > self.correlation_threshold:
                        to_remove.add(j)

            valid_indices = np.where(selected)[0]
            for idx in to_remove:
                selected[valid_indices[idx]] = False

        self.selected_features_ = np.where(selected)[0]
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        过滤特征

        Args:
            X: 输入数据

        Returns:
            过滤后的数据
        """
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        return X[:, self.selected_features_]
