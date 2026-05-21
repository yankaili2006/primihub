"""
Feature Binning Executor
特征分箱执行器

单方特征分箱任务的执行入口。
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from ..base import LocalBaseModel
from .base import (
    EqualWidthBinner,
    EqualFrequencyBinner,
    KMeansBinner,
    DecisionTreeBinner,
    CustomBinner,
)

logger = logging.getLogger(__name__)


class FeatureBinningExecutor(LocalBaseModel):
    """
    特征分箱执行器

    执行单方特征分箱任务，支持多种分箱方法。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _parse_params(self):
        """解析参数"""
        # 分箱方法
        self.method = self.common_params.get("method", "equal_width")
        # 要分箱的列
        self.columns = self.common_params.get("columns", None)
        # 分箱数量
        self.n_bins = self.common_params.get("n_bins", 5)
        # 输出路径
        self.output_path = self.common_params.get("output_path", "")
        # 模型输出路径
        self.model_output_path = self.common_params.get("model_output_path", "")

        # 分箱标签
        self.labels = self.common_params.get("labels", None)

        # 自定义分箱边界
        self.custom_bin_edges = self.common_params.get("custom_bin_edges", {})

        # DecisionTreeBinner参数
        self.min_samples_leaf = self.common_params.get("min_samples_leaf", 50)

        # 随机种子
        self.random_state = self.common_params.get("random_state", 42)

    def run(self) -> Dict[str, Any]:
        """执行特征分箱"""
        logger.info(f"FeatureBinningExecutor: Starting feature binning with method={self.method}")

        # 加载数据
        data, labels = self._load_data()
        if data.empty:
            logger.error("No data loaded")
            return {"error": "No data loaded"}

        logger.info(f"Data loaded: shape={data.shape}")

        # 创建分箱器
        binners = {
            "equal_width": EqualWidthBinner(
                columns=self.columns,
                n_bins=self.n_bins,
                labels=self.labels,
            ),
            "equal_frequency": EqualFrequencyBinner(
                columns=self.columns,
                n_bins=self.n_bins,
                labels=self.labels,
            ),
            "kmeans": KMeansBinner(
                columns=self.columns,
                n_bins=self.n_bins,
                random_state=self.random_state,
            ),
            "decision_tree": DecisionTreeBinner(
                columns=self.columns,
                n_bins=self.n_bins,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            ),
            "custom": CustomBinner(
                bin_edges=self.custom_bin_edges,
            ),
        }

        if self.method not in binners:
            logger.error(f"Unknown binning method: {self.method}")
            return {"error": f"Unknown method: {self.method}"}

        binner = binners[self.method]

        # 执行分箱
        try:
            if self.method == "decision_tree":
                if labels is None:
                    logger.error("Decision tree binning requires target variable")
                    return {"error": "Target variable required for decision tree binning"}
                binned_data = binner.fit_transform(data, labels)
            else:
                binned_data = binner.fit_transform(data)

            bin_edges = binner.get_bin_edges()
        except Exception as e:
            logger.error(f"Error binning data: {e}")
            return {"error": str(e)}

        result = {
            "method": self.method,
            "n_bins": self.n_bins,
            "original_shape": {"rows": len(data), "columns": len(data.columns)},
            "binned_shape": {"rows": len(binned_data), "columns": len(binned_data.columns)},
            "bin_edges": bin_edges,
        }

        # 保存分箱后的数据
        if self.output_path:
            if self.output_path.endswith('.csv'):
                binned_data.to_csv(self.output_path, index=False)
                logger.info(f"Binned data saved to: {self.output_path}")
            else:
                self._save_result({"data": binned_data, "bin_edges": bin_edges}, self.output_path)

        # 保存分箱边界
        if self.model_output_path:
            self._save_result(bin_edges, self.model_output_path)
            logger.info(f"Bin edges saved to: {self.model_output_path}")

        logger.info("FeatureBinningExecutor: Completed")
        return result
