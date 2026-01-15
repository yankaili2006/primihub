"""
Federated Learning Preprocessing Host
联邦学习预处理主机端
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

from primihub.FL.utils.base import BaseModel
from primihub.FL.utils.net_work import GrpcClient, MultiGrpcClients
from primihub.FL.utils.file import save_json_file, save_pickle_file
from primihub.FL.utils.dataset import read_data

from .base import FLDataCleaner, FLOutlierDetector, FLDataValidator, FLFeatureFilter

logger = logging.getLogger(__name__)


class FLPreprocessHost(BaseModel):
    """联邦学习预处理主机端"""

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
        logger.info("FLPreprocessHost: 开始预处理任务")

        # 建立通信通道
        parties = self.roles.keys()
        guest_parties = [p for p in parties if self.roles[p] == "guest"]

        guest_channel = None
        if guest_parties:
            guest_channel = MultiGrpcClients(
                self.node_info["self_party"],
                guest_parties,
                self.node_info,
                self.task_info,
            )

        # 加载数据
        data = self._load_data()
        logger.info(f"FLPreprocessHost: 加载数据 shape={data.shape}")

        # 执行预处理步骤
        result = data.copy()
        preprocess_info = {}

        for step in self.preprocess_steps:
            if step == "clean":
                result, info = self._clean_data(result, guest_channel)
                preprocess_info["clean"] = info
            elif step == "outlier":
                result, info = self._detect_outliers(result, guest_channel)
                preprocess_info["outlier"] = info
            elif step == "validate":
                result, info = self._validate_data(result, guest_channel)
                preprocess_info["validate"] = info
            elif step == "filter":
                result, info = self._filter_features(result, guest_channel)
                preprocess_info["filter"] = info

            logger.info(f"FLPreprocessHost: 完成 {step} 步骤, shape={result.shape}")

        # 保存结果
        self._save_results(result, preprocess_info)

        logger.info("FLPreprocessHost: 预处理任务完成")
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
        self, data: np.ndarray, guest_channel: Optional[MultiGrpcClients]
    ) -> tuple:
        """数据清洗"""
        cleaner = FLDataCleaner(
            FL_type="V",
            role="host",
            channel=guest_channel,
            remove_duplicates=self.remove_duplicates,
            handle_missing=self.handle_missing,
            missing_threshold=self.missing_threshold,
        )

        result = cleaner.fit_transform(data)

        info = {
            "original_shape": data.shape,
            "result_shape": result.shape,
            "valid_columns": cleaner.valid_columns_.tolist() if cleaner.valid_columns_ is not None else [],
        }

        return result, info

    def _detect_outliers(
        self, data: np.ndarray, guest_channel: Optional[MultiGrpcClients]
    ) -> tuple:
        """异常检测"""
        detector = FLOutlierDetector(
            FL_type="V",
            role="host",
            channel=guest_channel,
            method=self.outlier_method,
            threshold=self.outlier_threshold,
        )

        detector.fit(data)
        outlier_mask = detector.detect_outliers(data)
        result = data[~outlier_mask]

        info = {
            "outliers_count": int(outlier_mask.sum()),
            "outliers_ratio": float(outlier_mask.mean()),
        }

        return result, info

    def _validate_data(
        self, data: np.ndarray, guest_channel: Optional[MultiGrpcClients]
    ) -> tuple:
        """数据验证"""
        validator = FLDataValidator(
            FL_type="V",
            role="host",
            channel=guest_channel,
        )

        result = validator.fit_transform(data)
        info = validator.validation_report_ or {}

        return result, info

    def _filter_features(
        self, data: np.ndarray, guest_channel: Optional[MultiGrpcClients]
    ) -> tuple:
        """特征过滤"""
        filter_ = FLFeatureFilter(
            FL_type="V",
            role="host",
            channel=guest_channel,
            variance_threshold=self.variance_threshold,
            correlation_threshold=self.correlation_threshold,
            missing_threshold=self.missing_threshold,
        )

        result = filter_.fit_transform(data)

        info = {
            "original_features": data.shape[1],
            "selected_features": len(filter_.selected_features_),
            "selected_indices": filter_.selected_features_.tolist(),
        }

        return result, info

    def _save_results(self, data: np.ndarray, info: Dict):
        """保存结果"""
        if self.output_path:
            save_pickle_file(data, self.output_path)
            save_json_file(info, self.output_path.replace(".pkl", "_info.json"))
            logger.info(f"FLPreprocessHost: 结果已保存到 {self.output_path}")
