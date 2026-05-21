"""
Data Cleaning Executor
数据清洗执行器

单方数据清洗任务的执行入口。
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from ..base import LocalBaseModel
from .base import (
    MissingValueHandler,
    DuplicateHandler,
    OutlierHandler,
    DataTypeConverter,
    ValueReplacer,
)

logger = logging.getLogger(__name__)


class DataCleaningExecutor(LocalBaseModel):
    """
    数据清洗执行器

    执行单方数据清洗任务，支持多种清洗操作的组合。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _parse_params(self):
        """解析参数"""
        # 清洗操作列表
        self.operations = self.common_params.get("operations", ["missing", "duplicate"])
        # 输出路径
        self.output_path = self.common_params.get("output_path", "")

        # 缺失值处理参数
        self.missing_strategy = self.common_params.get("missing_strategy", "drop")
        self.missing_fill_value = self.common_params.get("missing_fill_value", None)
        self.missing_columns = self.common_params.get("missing_columns", None)

        # 重复值处理参数
        self.duplicate_subset = self.common_params.get("duplicate_subset", None)
        self.duplicate_keep = self.common_params.get("duplicate_keep", "first")

        # 异常值处理参数
        self.outlier_method = self.common_params.get("outlier_method", "iqr")
        self.outlier_threshold = self.common_params.get("outlier_threshold", 1.5)
        self.outlier_strategy = self.common_params.get("outlier_strategy", "clip")
        self.outlier_columns = self.common_params.get("outlier_columns", None)

        # 类型转换参数
        self.type_mapping = self.common_params.get("type_mapping", {})
        self.auto_convert = self.common_params.get("auto_convert", False)

        # 值替换参数
        self.replacements = self.common_params.get("replacements", {})
        self.global_replacements = self.common_params.get("global_replacements", {})

    def run(self) -> Dict[str, Any]:
        """执行数据清洗"""
        logger.info("DataCleaningExecutor: Starting data cleaning")

        # 加载数据
        data, labels = self._load_data()
        if data.empty:
            logger.error("No data loaded")
            return {"error": "No data loaded"}

        logger.info(f"Data loaded: shape={data.shape}")

        result_data = data
        reports = {}

        # 定义清洗器
        cleaners = {
            "missing": MissingValueHandler(
                strategy=self.missing_strategy,
                columns=self.missing_columns,
                fill_value=self.missing_fill_value,
            ),
            "duplicate": DuplicateHandler(
                subset=self.duplicate_subset,
                keep=self.duplicate_keep,
            ),
            "outlier": OutlierHandler(
                method=self.outlier_method,
                threshold=self.outlier_threshold,
                strategy=self.outlier_strategy,
                columns=self.outlier_columns,
            ),
            "type_convert": DataTypeConverter(
                type_mapping=self.type_mapping,
                auto_convert=self.auto_convert,
            ),
            "replace": ValueReplacer(
                replacements=self.replacements,
                global_replacements=self.global_replacements,
            ),
        }

        # 按顺序执行清洗操作
        for op in self.operations:
            if op in cleaners:
                logger.info(f"Executing {op} cleaning")
                try:
                    result_data = cleaners[op].clean(result_data)
                    reports[op] = cleaners[op].get_report()
                except Exception as e:
                    logger.error(f"Error in {op} cleaning: {e}")
                    reports[op] = {"error": str(e)}
            else:
                logger.warning(f"Unknown operation: {op}")

        result = {
            "original_shape": {"rows": len(data), "columns": len(data.columns)},
            "cleaned_shape": {"rows": len(result_data), "columns": len(result_data.columns)},
            "reports": reports,
        }

        # 保存结果
        if self.output_path:
            if self.output_path.endswith('.csv'):
                result_data.to_csv(self.output_path, index=False)
                logger.info(f"Cleaned data saved to: {self.output_path}")
            else:
                self._save_result({"data": result_data, "report": result}, self.output_path)

        logger.info("DataCleaningExecutor: Completed")
        return result
