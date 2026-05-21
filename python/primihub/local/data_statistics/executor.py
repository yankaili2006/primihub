"""
Data Statistics Executor
数据统计执行器

单方数据统计任务的执行入口。
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from ..base import LocalBaseModel
from .base import (
    DescriptiveStatistics,
    DistributionAnalysis,
    CorrelationAnalysis,
    OutlierStatistics,
    MissingValueStatistics,
)

logger = logging.getLogger(__name__)


class DataStatisticsExecutor(LocalBaseModel):
    """
    数据统计执行器

    执行单方数据统计任务，支持多种统计分析类型。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _parse_params(self):
        """解析参数"""
        # 统计类型列表
        self.stat_types = self.common_params.get("stat_types", ["descriptive"])
        # 要分析的列
        self.columns = self.common_params.get("columns", None)
        # 输出路径
        self.output_path = self.common_params.get("output_path", "")
        # 相关性分析方法
        self.correlation_method = self.common_params.get("correlation_method", "pearson")
        # 相关性阈值
        self.correlation_threshold = self.common_params.get("correlation_threshold", 0.5)
        # 异常值检测方法
        self.outlier_method = self.common_params.get("outlier_method", "iqr")
        # 异常值阈值
        self.outlier_threshold = self.common_params.get("outlier_threshold", 1.5)
        # 分位数列表
        self.percentiles = self.common_params.get("percentiles", [0.25, 0.5, 0.75])
        # 直方图分箱数
        self.n_bins = self.common_params.get("n_bins", 20)

    def run(self) -> Dict[str, Any]:
        """执行数据统计"""
        logger.info("DataStatisticsExecutor: Starting data statistics")

        # 加载数据
        data, labels = self._load_data()
        if data.empty:
            logger.error("No data loaded")
            return {"error": "No data loaded"}

        logger.info(f"Data loaded: shape={data.shape}")

        # 执行各种统计
        result = {
            "data_shape": {"rows": len(data), "columns": len(data.columns)},
            "column_names": list(data.columns),
            "statistics": {},
        }

        stat_analyzers = {
            "descriptive": DescriptiveStatistics(
                columns=self.columns,
                percentiles=self.percentiles
            ),
            "distribution": DistributionAnalysis(
                columns=self.columns,
                n_bins=self.n_bins
            ),
            "correlation": CorrelationAnalysis(
                columns=self.columns,
                method=self.correlation_method,
                threshold=self.correlation_threshold
            ),
            "outlier": OutlierStatistics(
                columns=self.columns,
                method=self.outlier_method,
                threshold=self.outlier_threshold
            ),
            "missing": MissingValueStatistics(
                columns=self.columns
            ),
        }

        for stat_type in self.stat_types:
            if stat_type in stat_analyzers:
                logger.info(f"Computing {stat_type} statistics")
                try:
                    result["statistics"][stat_type] = stat_analyzers[stat_type].compute(data)
                except Exception as e:
                    logger.error(f"Error computing {stat_type}: {e}")
                    result["statistics"][stat_type] = {"error": str(e)}
            else:
                logger.warning(f"Unknown stat type: {stat_type}")

        # 保存结果
        if self.output_path:
            self._save_result(result, self.output_path)

        logger.info("DataStatisticsExecutor: Completed")
        return result
