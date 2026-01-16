"""
Feature Derivation Executor
特征衍生执行器

单方特征衍生任务的执行入口。
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from ..base import LocalBaseModel
from .base import (
    PolynomialDeriver,
    InteractionDeriver,
    AggregationDeriver,
    DateTimeDeriver,
    MathDeriver,
    WindowDeriver,
)

logger = logging.getLogger(__name__)


class FeatureDerivationExecutor(LocalBaseModel):
    """
    特征衍生执行器

    执行单方特征衍生任务，支持多种衍生方法。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _parse_params(self):
        """解析参数"""
        # 衍生方法列表
        self.methods = self.common_params.get("methods", ["polynomial"])
        # 要处理的列
        self.columns = self.common_params.get("columns", None)
        # 输出路径
        self.output_path = self.common_params.get("output_path", "")

        # PolynomialDeriver参数
        self.degree = self.common_params.get("degree", 2)
        self.include_bias = self.common_params.get("include_bias", False)

        # InteractionDeriver参数
        self.interaction_type = self.common_params.get("interaction_type", "multiply")
        self.max_interactions = self.common_params.get("max_interactions", 10)

        # AggregationDeriver参数
        self.group_columns = self.common_params.get("group_columns", [])
        self.agg_columns = self.common_params.get("agg_columns", [])
        self.agg_functions = self.common_params.get("agg_functions", ['mean', 'sum'])

        # DateTimeDeriver参数
        self.datetime_columns = self.common_params.get("datetime_columns", None)
        self.extract_features = self.common_params.get("extract_features", None)

        # MathDeriver参数
        self.math_functions = self.common_params.get("math_functions", ['log1p', 'sqrt'])

        # WindowDeriver参数
        self.window_sizes = self.common_params.get("window_sizes", [3, 5, 7])
        self.window_functions = self.common_params.get("window_functions", ['mean', 'std'])

    def run(self) -> Dict[str, Any]:
        """执行特征衍生"""
        logger.info(f"FeatureDerivationExecutor: Starting feature derivation with methods={self.methods}")

        # 加载数据
        data, labels = self._load_data()
        if data.empty:
            logger.error("No data loaded")
            return {"error": "No data loaded"}

        logger.info(f"Data loaded: shape={data.shape}")

        result_data = data
        all_derived_columns = []
        derivation_report = {}

        # 创建衍生器
        derivers = {
            "polynomial": PolynomialDeriver(
                columns=self.columns,
                degree=self.degree,
                include_bias=self.include_bias,
            ),
            "interaction": InteractionDeriver(
                columns=self.columns,
                interaction_type=self.interaction_type,
                max_interactions=self.max_interactions,
            ),
            "aggregation": AggregationDeriver(
                group_columns=self.group_columns,
                agg_columns=self.agg_columns or self.columns or [],
                agg_functions=self.agg_functions,
            ) if self.group_columns else None,
            "datetime": DateTimeDeriver(
                columns=self.datetime_columns,
                extract_features=self.extract_features,
            ),
            "math": MathDeriver(
                columns=self.columns,
                functions=self.math_functions,
            ),
            "window": WindowDeriver(
                columns=self.columns,
                window_sizes=self.window_sizes,
                functions=self.window_functions,
            ),
        }

        # 按顺序执行衍生操作
        for method in self.methods:
            if method not in derivers or derivers[method] is None:
                if method == "aggregation" and not self.group_columns:
                    logger.warning("Aggregation requires group_columns parameter")
                else:
                    logger.warning(f"Unknown derivation method: {method}")
                continue

            logger.info(f"Executing {method} derivation")

            try:
                deriver = derivers[method]
                result_data = deriver.derive(result_data)
                derived_cols = deriver.get_derived_columns()

                all_derived_columns.extend(derived_cols)
                derivation_report[method] = {
                    "derived_count": len(derived_cols),
                    "derived_columns": derived_cols,
                }
            except Exception as e:
                logger.error(f"Error in {method} derivation: {e}")
                derivation_report[method] = {"error": str(e)}

        result = {
            "original_shape": {"rows": len(data), "columns": len(data.columns)},
            "derived_shape": {"rows": len(result_data), "columns": len(result_data.columns)},
            "total_derived_features": len(all_derived_columns),
            "derivation_report": derivation_report,
        }

        # 保存衍生后的数据
        if self.output_path:
            if self.output_path.endswith('.csv'):
                result_data.to_csv(self.output_path, index=False)
                logger.info(f"Derived data saved to: {self.output_path}")
            else:
                self._save_result({"data": result_data, "derived_columns": all_derived_columns}, self.output_path)

        logger.info("FeatureDerivationExecutor: Completed")
        return result
