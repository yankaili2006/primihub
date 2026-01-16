"""
Data Scaling Executor
数据缩放执行器

单方数据缩放任务的执行入口。
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from ..base import LocalBaseModel
from .base import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    Normalizer,
)

logger = logging.getLogger(__name__)


class DataScalingExecutor(LocalBaseModel):
    """
    数据缩放执行器

    执行单方数据缩放任务，支持多种缩放方法。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _parse_params(self):
        """解析参数"""
        # 缩放方法
        self.method = self.common_params.get("method", "standard")
        # 要缩放的列
        self.columns = self.common_params.get("columns", None)
        # 输出路径
        self.output_path = self.common_params.get("output_path", "")
        # 模型输出路径（保存缩放器参数）
        self.model_output_path = self.common_params.get("model_output_path", "")

        # StandardScaler参数
        self.with_mean = self.common_params.get("with_mean", True)
        self.with_std = self.common_params.get("with_std", True)

        # MinMaxScaler参数
        self.feature_range = tuple(self.common_params.get("feature_range", [0, 1]))

        # RobustScaler参数
        self.with_centering = self.common_params.get("with_centering", True)
        self.with_scaling = self.common_params.get("with_scaling", True)
        self.quantile_range = tuple(self.common_params.get("quantile_range", [25.0, 75.0]))

        # Normalizer参数
        self.norm = self.common_params.get("norm", "l2")

    def run(self) -> Dict[str, Any]:
        """执行数据缩放"""
        logger.info(f"DataScalingExecutor: Starting data scaling with method={self.method}")

        # 加载数据
        data, labels = self._load_data()
        if data.empty:
            logger.error("No data loaded")
            return {"error": "No data loaded"}

        logger.info(f"Data loaded: shape={data.shape}")

        # 创建缩放器
        scalers = {
            "standard": StandardScaler(
                columns=self.columns,
                with_mean=self.with_mean,
                with_std=self.with_std,
            ),
            "minmax": MinMaxScaler(
                columns=self.columns,
                feature_range=self.feature_range,
            ),
            "maxabs": MaxAbsScaler(columns=self.columns),
            "robust": RobustScaler(
                columns=self.columns,
                with_centering=self.with_centering,
                with_scaling=self.with_scaling,
                quantile_range=self.quantile_range,
            ),
            "normalize": Normalizer(
                columns=self.columns,
                norm=self.norm,
            ),
        }

        if self.method not in scalers:
            logger.error(f"Unknown scaling method: {self.method}")
            return {"error": f"Unknown method: {self.method}"}

        scaler = scalers[self.method]

        # 执行缩放
        try:
            scaled_data = scaler.fit_transform(data)
            scaler_params = scaler.get_params()
        except Exception as e:
            logger.error(f"Error scaling data: {e}")
            return {"error": str(e)}

        result = {
            "method": self.method,
            "original_shape": {"rows": len(data), "columns": len(data.columns)},
            "scaled_shape": {"rows": len(scaled_data), "columns": len(scaled_data.columns)},
            "scaler_params": scaler_params,
        }

        # 保存缩放后的数据
        if self.output_path:
            if self.output_path.endswith('.csv'):
                scaled_data.to_csv(self.output_path, index=False)
                logger.info(f"Scaled data saved to: {self.output_path}")
            else:
                self._save_result({"data": scaled_data, "params": scaler_params}, self.output_path)

        # 保存缩放器参数
        if self.model_output_path:
            self._save_result(scaler_params, self.model_output_path)
            logger.info(f"Scaler params saved to: {self.model_output_path}")

        logger.info("DataScalingExecutor: Completed")
        return result
