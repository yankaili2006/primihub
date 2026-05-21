"""
Data Statistics Base Classes
数据统计基础类

提供各种数据统计分析功能。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DataStatisticsBase(ABC):
    """数据统计基类"""

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        计算统计信息

        Args:
            data: 输入数据

        Returns:
            统计结果字典
        """
        pass


class DescriptiveStatistics(DataStatisticsBase):
    """
    描述性统计

    计算基本的描述性统计量：均值、中位数、标准差、最值、分位数等。
    """

    def __init__(self, columns: Optional[List[str]] = None,
                 percentiles: List[float] = None):
        """
        初始化描述性统计

        Args:
            columns: 要统计的列名列表，None表示所有数值列
            percentiles: 分位数列表，默认为[0.25, 0.5, 0.75]
        """
        self.columns = columns
        self.percentiles = percentiles or [0.25, 0.5, 0.75]

    def compute(self, data: pd.DataFrame) -> Dict[str, Any]:
        """计算描述性统计"""
        logger.info("Computing descriptive statistics")

        # 选择数值列
        if self.columns:
            numeric_data = data[self.columns].select_dtypes(include=[np.number])
        else:
            numeric_data = data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            logger.warning("No numeric columns found")
            return {}

        result = {
            "summary": {},
            "shape": {"rows": len(data), "columns": len(data.columns)},
            "dtypes": data.dtypes.astype(str).to_dict(),
        }

        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            if len(col_data) == 0:
                continue

            col_stats = {
                "count": int(len(col_data)),
                "missing": int(data[col].isna().sum()),
                "missing_rate": float(data[col].isna().mean()),
                "mean": float(col_data.mean()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "range": float(col_data.max() - col_data.min()),
                "median": float(col_data.median()),
                "mode": float(col_data.mode().iloc[0]) if len(col_data.mode()) > 0 else None,
                "variance": float(col_data.var()),
                "skewness": float(col_data.skew()),
                "kurtosis": float(col_data.kurtosis()),
                "percentiles": {
                    f"p{int(p*100)}": float(col_data.quantile(p))
                    for p in self.percentiles
                },
                "unique_count": int(col_data.nunique()),
            }
            result["summary"][col] = col_stats

        logger.info(f"Computed statistics for {len(result['summary'])} columns")
        return result


class DistributionAnalysis(DataStatisticsBase):
    """
    分布分析

    分析数据的分布特征，包括正态性检验、直方图统计等。
    """

    def __init__(self, columns: Optional[List[str]] = None,
                 n_bins: int = 20,
                 test_normality: bool = True):
        """
        初始化分布分析

        Args:
            columns: 要分析的列名列表
            n_bins: 直方图分箱数
            test_normality: 是否进行正态性检验
        """
        self.columns = columns
        self.n_bins = n_bins
        self.test_normality = test_normality

    def compute(self, data: pd.DataFrame) -> Dict[str, Any]:
        """计算分布统计"""
        logger.info("Computing distribution analysis")

        if self.columns:
            numeric_data = data[self.columns].select_dtypes(include=[np.number])
        else:
            numeric_data = data.select_dtypes(include=[np.number])

        result = {"distributions": {}}

        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            if len(col_data) < 3:
                continue

            col_dist = {
                "histogram": self._compute_histogram(col_data),
            }

            # 正态性检验
            if self.test_normality and len(col_data) >= 8:
                try:
                    # Shapiro-Wilk检验（样本量限制为5000）
                    sample = col_data.sample(min(5000, len(col_data)), random_state=42)
                    shapiro_stat, shapiro_p = stats.shapiro(sample)
                    col_dist["normality_test"] = {
                        "shapiro_wilk": {
                            "statistic": float(shapiro_stat),
                            "p_value": float(shapiro_p),
                            "is_normal": shapiro_p > 0.05,
                        }
                    }

                    # Kolmogorov-Smirnov检验
                    ks_stat, ks_p = stats.kstest(col_data, 'norm',
                                                  args=(col_data.mean(), col_data.std()))
                    col_dist["normality_test"]["ks_test"] = {
                        "statistic": float(ks_stat),
                        "p_value": float(ks_p),
                        "is_normal": ks_p > 0.05,
                    }
                except Exception as e:
                    logger.warning(f"Normality test failed for {col}: {e}")

            result["distributions"][col] = col_dist

        return result

    def _compute_histogram(self, data: pd.Series) -> Dict[str, Any]:
        """计算直方图统计"""
        counts, bin_edges = np.histogram(data, bins=self.n_bins)
        return {
            "counts": counts.tolist(),
            "bin_edges": bin_edges.tolist(),
            "bin_centers": ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist(),
        }


class CorrelationAnalysis(DataStatisticsBase):
    """
    相关性分析

    计算特征之间的相关性。
    """

    def __init__(self, columns: Optional[List[str]] = None,
                 method: str = "pearson",
                 threshold: float = 0.0):
        """
        初始化相关性分析

        Args:
            columns: 要分析的列名列表
            method: 相关性计算方法（pearson, spearman, kendall）
            threshold: 相关性阈值，只返回绝对值大于此阈值的相关性
        """
        self.columns = columns
        self.method = method
        self.threshold = threshold

    def compute(self, data: pd.DataFrame) -> Dict[str, Any]:
        """计算相关性"""
        logger.info(f"Computing {self.method} correlation")

        if self.columns:
            numeric_data = data[self.columns].select_dtypes(include=[np.number])
        else:
            numeric_data = data.select_dtypes(include=[np.number])

        if numeric_data.shape[1] < 2:
            logger.warning("Need at least 2 numeric columns for correlation")
            return {}

        # 计算相关性矩阵
        corr_matrix = numeric_data.corr(method=self.method)

        result = {
            "method": self.method,
            "correlation_matrix": corr_matrix.to_dict(),
            "high_correlations": [],
        }

        # 找出高相关性对
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > self.threshold:
                    result["high_correlations"].append({
                        "feature1": col1,
                        "feature2": col2,
                        "correlation": float(corr_value),
                    })

        # 按相关性绝对值排序
        result["high_correlations"].sort(
            key=lambda x: abs(x["correlation"]), reverse=True
        )

        return result


class OutlierStatistics(DataStatisticsBase):
    """
    异常值统计

    检测和统计数据中的异常值。
    """

    def __init__(self, columns: Optional[List[str]] = None,
                 method: str = "iqr",
                 threshold: float = 1.5):
        """
        初始化异常值统计

        Args:
            columns: 要分析的列名列表
            method: 检测方法（iqr, zscore）
            threshold: 阈值（IQR倍数或Z分数）
        """
        self.columns = columns
        self.method = method
        self.threshold = threshold

    def compute(self, data: pd.DataFrame) -> Dict[str, Any]:
        """计算异常值统计"""
        logger.info(f"Computing outlier statistics using {self.method} method")

        if self.columns:
            numeric_data = data[self.columns].select_dtypes(include=[np.number])
        else:
            numeric_data = data.select_dtypes(include=[np.number])

        result = {"method": self.method, "threshold": self.threshold, "columns": {}}

        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            if len(col_data) == 0:
                continue

            if self.method == "iqr":
                outlier_mask, lower_bound, upper_bound = self._iqr_outliers(col_data)
            else:  # zscore
                outlier_mask, lower_bound, upper_bound = self._zscore_outliers(col_data)

            outlier_count = int(outlier_mask.sum())
            result["columns"][col] = {
                "outlier_count": outlier_count,
                "outlier_rate": float(outlier_count / len(col_data)),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "outlier_indices": np.where(outlier_mask)[0].tolist()[:100],  # 最多返回100个索引
            }

        return result

    def _iqr_outliers(self, data: pd.Series):
        """使用IQR方法检测异常值"""
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - self.threshold * iqr
        upper_bound = q3 + self.threshold * iqr
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        return outlier_mask, lower_bound, upper_bound

    def _zscore_outliers(self, data: pd.Series):
        """使用Z分数方法检测异常值"""
        z_scores = np.abs(stats.zscore(data))
        outlier_mask = z_scores > self.threshold
        mean, std = data.mean(), data.std()
        lower_bound = mean - self.threshold * std
        upper_bound = mean + self.threshold * std
        return outlier_mask, lower_bound, upper_bound


class MissingValueStatistics(DataStatisticsBase):
    """
    缺失值统计

    分析数据中的缺失值情况。
    """

    def __init__(self, columns: Optional[List[str]] = None):
        """
        初始化缺失值统计

        Args:
            columns: 要分析的列名列表
        """
        self.columns = columns

    def compute(self, data: pd.DataFrame) -> Dict[str, Any]:
        """计算缺失值统计"""
        logger.info("Computing missing value statistics")

        if self.columns:
            analysis_data = data[self.columns]
        else:
            analysis_data = data

        result = {
            "total_cells": int(analysis_data.size),
            "total_missing": int(analysis_data.isna().sum().sum()),
            "overall_missing_rate": float(analysis_data.isna().sum().sum() / analysis_data.size),
            "columns": {},
            "rows_with_missing": int(analysis_data.isna().any(axis=1).sum()),
            "complete_rows": int((~analysis_data.isna().any(axis=1)).sum()),
        }

        for col in analysis_data.columns:
            missing_count = int(analysis_data[col].isna().sum())
            result["columns"][col] = {
                "missing_count": missing_count,
                "missing_rate": float(missing_count / len(analysis_data)),
                "non_missing_count": int(len(analysis_data) - missing_count),
            }

        # 按缺失率排序
        result["columns_by_missing_rate"] = sorted(
            result["columns"].items(),
            key=lambda x: x[1]["missing_rate"],
            reverse=True
        )

        return result
