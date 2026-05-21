"""
Feature Encoding Executor
特征编码执行器

单方特征编码任务的执行入口。
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from ..base import LocalBaseModel
from .base import (
    OneHotEncoder,
    LabelEncoder,
    OrdinalEncoder,
    TargetEncoder,
    BinaryEncoder,
    FrequencyEncoder,
)

logger = logging.getLogger(__name__)


class FeatureEncodingExecutor(LocalBaseModel):
    """
    特征编码执行器

    执行单方特征编码任务，支持多种编码方法。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _parse_params(self):
        """解析参数"""
        # 编码方法
        self.method = self.common_params.get("method", "onehot")
        # 要编码的列
        self.columns = self.common_params.get("columns", None)
        # 输出路径
        self.output_path = self.common_params.get("output_path", "")
        # 模型输出路径
        self.model_output_path = self.common_params.get("model_output_path", "")

        # OneHotEncoder参数
        self.drop_first = self.common_params.get("drop_first", False)
        self.handle_unknown = self.common_params.get("handle_unknown", "ignore")
        self.max_categories = self.common_params.get("max_categories", None)

        # OrdinalEncoder参数
        self.order_mapping = self.common_params.get("order_mapping", {})

        # TargetEncoder参数
        self.smoothing = self.common_params.get("smoothing", 1.0)
        self.min_samples = self.common_params.get("min_samples", 1)

        # FrequencyEncoder参数
        self.normalize = self.common_params.get("normalize", True)

    def run(self) -> Dict[str, Any]:
        """执行特征编码"""
        logger.info(f"FeatureEncodingExecutor: Starting feature encoding with method={self.method}")

        # 加载数据
        data, labels = self._load_data()
        if data.empty:
            logger.error("No data loaded")
            return {"error": "No data loaded"}

        logger.info(f"Data loaded: shape={data.shape}")

        # 如果有标签，添加到数据中以便目标编码使用
        if labels is not None:
            data_with_labels = data.copy()
            data_with_labels['__target__'] = labels
        else:
            data_with_labels = data

        # 创建编码器
        encoders = {
            "onehot": OneHotEncoder(
                columns=self.columns,
                drop_first=self.drop_first,
                handle_unknown=self.handle_unknown,
                max_categories=self.max_categories,
            ),
            "label": LabelEncoder(columns=self.columns),
            "ordinal": OrdinalEncoder(
                columns=self.columns,
                order_mapping=self.order_mapping,
            ),
            "target": TargetEncoder(
                columns=self.columns,
                smoothing=self.smoothing,
                min_samples=self.min_samples,
            ),
            "binary": BinaryEncoder(columns=self.columns),
            "frequency": FrequencyEncoder(
                columns=self.columns,
                normalize=self.normalize,
            ),
        }

        if self.method not in encoders:
            logger.error(f"Unknown encoding method: {self.method}")
            return {"error": f"Unknown method: {self.method}"}

        encoder = encoders[self.method]

        # 执行编码
        try:
            if self.method == "target" and labels is not None:
                encoded_data = encoder.fit_transform(data, labels)
            else:
                encoded_data = encoder.fit_transform(data)

            encoding_maps = encoder.get_encoding_maps()
        except Exception as e:
            logger.error(f"Error encoding data: {e}")
            return {"error": str(e)}

        result = {
            "method": self.method,
            "original_shape": {"rows": len(data), "columns": len(data.columns)},
            "encoded_shape": {"rows": len(encoded_data), "columns": len(encoded_data.columns)},
            "encoding_maps": encoding_maps,
        }

        # 保存编码后的数据
        if self.output_path:
            if self.output_path.endswith('.csv'):
                encoded_data.to_csv(self.output_path, index=False)
                logger.info(f"Encoded data saved to: {self.output_path}")
            else:
                self._save_result({"data": encoded_data, "maps": encoding_maps}, self.output_path)

        # 保存编码器参数
        if self.model_output_path:
            self._save_result(encoding_maps, self.model_output_path)
            logger.info(f"Encoding maps saved to: {self.model_output_path}")

        logger.info("FeatureEncodingExecutor: Completed")
        return result
