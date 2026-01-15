"""
Federated Learning Preprocessing Guest
联邦学习预处理访客端
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

from primihub.FL.utils.base import BaseModel
from primihub.FL.utils.net_work import GrpcClient
from primihub.FL.utils.file import save_pickle_file
from primihub.FL.utils.dataset import read_data

from .base import FLDataCleaner, FLOutlierDetector, FLDataValidator, FLFeatureFilter

logger = logging.getLogger(__name__)


class FLPreprocessGuest(BaseModel):
    """联邦学习预处理访客端"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parse_params()

    def _parse_params(self):
        """解析参数"""
        # 通用参数
        self.preprocess_steps = self.common_params.get("preprocess_steps", ["clean"])
        self.output_path = self.common_params.get("output_path", "")

        # 数据清洗参数
        self.remove_duplicates = self.common_params.get("remove_duplicates", True)
        self.handle_missing = self.common_params.get("handle_missing", "fill")
        self.missing_threshold = self.common_params.get("missing_threshold", 0.5)

        # 异常检测参数
        self.outlier_method = self.common_params.get("outlier_method", "zscore")
        self.outlier_threshold = self.common_params.get("outlier_threshold", 3.0)

        # 特征过滤参数
        self.variance_threshold = self.common_params.get("variance_threshold", 0.0)
        self.correlation_threshold = self.common_params.get("correlation_threshold", 0.95)

        # 角色参数
        self.data_info = self.role_params.get("data", {})
        self.selected_columns = self.role_params.get("selected_columns", [])

    def run(self):
        """执行预处理"""
        logger.info("FLPreprocessGuest: 开始预处理任务")

        # 建立与host的通信通道
        host_party = None
        for party, role in self.roles.items():
            if role == "host":
                host_party = party
                break

        host_channel = None
        if host_party:
            host_channel = GrpcClient(
                self.node_info["self_party"],
                host_party,
                self.node_info,
                self.task_info,
            )

        # 加载数据
        data = self._load_data()
        logger.info(f"FLPreprocessGuest: 加载数据 shape={data.shape}")

        # 执行预处理步骤
        result = data.copy()

        for step in self.preprocess_steps:
            if step == "clean":
                result = self._clean_data(result, host_channel)
            elif step == "outlier":
                result = self._detect_outliers(result, host_channel)
            elif step == "validate":
                result = self._validate_data(result, host_channel)
            elif step == "filter":
                result = self._filter_features(result, host_channel)

            logger.info(f"FLPreprocessGuest: 完成 {step} 步骤, shape={result.shape}")

        # 保存结果
        self._save_results(result)

        logger.info("FLPreprocessGuest: 预处理任务完成")
        return result

    def _load_data(self) -> np.ndarray:
        """加载数据"""
        if self.data_info:
            data = read_data(
                data_info=self.data_info,
                selected_column=self.selected_columns if self.selected_columns else None,
                droped_column=None,
            )
            return np.array(data, dtype=np.float64)
        return np.array([])

    def _clean_data(
        self, data: np.ndarray, host_channel: Optional[GrpcClient]
    ) -> np.ndarray:
        """数据清洗"""
        cleaner = FLDataCleaner(
            FL_type="V",
            role="guest",
            channel=host_channel,
            remove_duplicates=self.remove_duplicates,
            handle_missing=self.handle_missing,
            missing_threshold=self.missing_threshold,
        )

        return cleaner.fit_transform(data)

    def _detect_outliers(
        self, data: np.ndarray, host_channel: Optional[GrpcClient]
    ) -> np.ndarray:
        """异常检测"""
        detector = FLOutlierDetector(
            FL_type="V",
            role="guest",
            channel=host_channel,
            method=self.outlier_method,
            threshold=self.outlier_threshold,
        )

        detector.fit(data)
        outlier_mask = detector.detect_outliers(data)
        return data[~outlier_mask]

    def _validate_data(
        self, data: np.ndarray, host_channel: Optional[GrpcClient]
    ) -> np.ndarray:
        """数据验证"""
        validator = FLDataValidator(
            FL_type="V",
            role="guest",
            channel=host_channel,
        )

        return validator.fit_transform(data)

    def _filter_features(
        self, data: np.ndarray, host_channel: Optional[GrpcClient]
    ) -> np.ndarray:
        """特征过滤"""
        filter_ = FLFeatureFilter(
            FL_type="V",
            role="guest",
            channel=host_channel,
            variance_threshold=self.variance_threshold,
            correlation_threshold=self.correlation_threshold,
            missing_threshold=self.missing_threshold,
        )

        return filter_.fit_transform(data)

    def _save_results(self, data: np.ndarray):
        """保存结果"""
        if self.output_path:
            save_pickle_file(data, self.output_path)
            logger.info(f"FLPreprocessGuest: 结果已保存到 {self.output_path}")
