"""
Feature Selection Executor
特征筛选执行器

单方特征筛选任务的执行入口。
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from ..base import LocalBaseModel
from .base import (
    VarianceSelector,
    CorrelationSelector,
    MutualInfoSelector,
    ChiSquareSelector,
    RFESelector,
    LassoSelector,
)

logger = logging.getLogger(__name__)


class FeatureSelectionExecutor(LocalBaseModel):
    """
    特征筛选执行器

    执行单方特征筛选任务，支持多种筛选方法。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _parse_params(self):
        """解析参数"""
        # 筛选方法
        self.method = self.common_params.get("method", "variance")
        # 要选择的特征数量
        self.n_features = self.common_params.get("n_features", None)
        # 筛选阈值
        self.threshold = self.common_params.get("threshold", None)
        # 输出路径
        self.output_path = self.common_params.get("output_path", "")
        # 模型输出路径
        self.model_output_path = self.common_params.get("model_output_path", "")

        # CorrelationSelector参数
        self.correlation_method = self.common_params.get("correlation_method", "pearson")
        self.selection_mode = self.common_params.get("selection_mode", "target")

        # MutualInfoSelector参数
        self.task = self.common_params.get("task", "auto")

        # RFESelector参数
        self.estimator = self.common_params.get("estimator", "logistic")
        self.step = self.common_params.get("step", 1)

        # LassoSelector参数
        self.alpha = self.common_params.get("alpha", 1.0)

        # 随机种子
        self.random_state = self.common_params.get("random_state", 42)

    def run(self) -> Dict[str, Any]:
        """执行特征筛选"""
        logger.info(f"FeatureSelectionExecutor: Starting feature selection with method={self.method}")

        # 加载数据
        data, labels = self._load_data()
        if data.empty:
            logger.error("No data loaded")
            return {"error": "No data loaded"}

        logger.info(f"Data loaded: shape={data.shape}")

        # 创建筛选器
        selectors = {
            "variance": VarianceSelector(
                threshold=self.threshold or 0.0,
            ),
            "correlation": CorrelationSelector(
                threshold=self.threshold or 0.1,
                method=self.selection_mode,
                correlation_method=self.correlation_method,
            ),
            "mutual_info": MutualInfoSelector(
                n_features=self.n_features,
                threshold=self.threshold,
                task=self.task,
                random_state=self.random_state,
            ),
            "chi_square": ChiSquareSelector(
                n_features=self.n_features,
                threshold=self.threshold,
            ),
            "rfe": RFESelector(
                n_features=self.n_features or 10,
                estimator=self.estimator,
                step=self.step,
            ),
            "lasso": LassoSelector(
                alpha=self.alpha,
                threshold=self.threshold or 0.0,
                n_features=self.n_features,
            ),
        }

        if self.method not in selectors:
            logger.error(f"Unknown selection method: {self.method}")
            return {"error": f"Unknown method: {self.method}"}

        selector = selectors[self.method]

        # 执行特征筛选
        try:
            if self.method in ["variance"]:
                selected_data = selector.fit_transform(data)
            else:
                if labels is None:
                    logger.error(f"{self.method} selection requires target variable")
                    return {"error": "Target variable required"}
                selected_data = selector.fit_transform(data, labels)

            selected_features = selector.get_selected_features()
            feature_scores = selector.get_feature_scores()
        except Exception as e:
            logger.error(f"Error selecting features: {e}")
            return {"error": str(e)}

        result = {
            "method": self.method,
            "original_features": len(data.columns),
            "selected_features": len(selected_features),
            "selected_feature_names": selected_features,
            "feature_scores": feature_scores,
        }

        # 保存筛选后的数据
        if self.output_path:
            if self.output_path.endswith('.csv'):
                selected_data.to_csv(self.output_path, index=False)
                logger.info(f"Selected data saved to: {self.output_path}")
            else:
                self._save_result({"data": selected_data, "features": selected_features}, self.output_path)

        # 保存筛选结果
        if self.model_output_path:
            self._save_result({
                "selected_features": selected_features,
                "feature_scores": feature_scores,
            }, self.model_output_path)

        logger.info("FeatureSelectionExecutor: Completed")
        return result
