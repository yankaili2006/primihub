"""
Federated Learning Feature Alignment Host
联邦学习特征对齐主机端
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

from primihub.FL.utils.base import BaseModel
from primihub.FL.utils.net_work import MultiGrpcClients
from primihub.FL.utils.file import save_json_file, save_pickle_file
from primihub.FL.utils.dataset import read_data

from .base import StatisticalAlignment, DistributionAlignment, SchemaAlignment

logger = logging.getLogger(__name__)


class FeatureAlignmentHost(BaseModel):
    """特征对齐主机端"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parse_params()

    def _parse_params(self):
        self.alignment_method = self.common_params.get("alignment_method", "zscore")
        self.output_path = self.common_params.get("output_path", "")

        self.data_info = self.role_params.get("data", {})
        self.feature_names = self.role_params.get("feature_names", [])

    def run(self):
        """执行特征对齐"""
        logger.info("FeatureAlignmentHost: 开始特征对齐")

        guest_parties = [p for p in self.roles if self.roles[p] == "guest"]
        guest_channel = MultiGrpcClients(
            self.node_info["self_party"],
            guest_parties,
            self.node_info,
            self.task_info,
        ) if guest_parties else None

        data = self._load_data()
        logger.info(f"FeatureAlignmentHost: 数据 shape={data.shape}")

        if self.alignment_method in ["zscore", "minmax", "rank"]:
            aligner = StatisticalAlignment(
                FL_type="V", role="host", channel=guest_channel, method=self.alignment_method
            )
        elif self.alignment_method == "distribution":
            aligner = DistributionAlignment(FL_type="V", role="host", channel=guest_channel)
        elif self.alignment_method == "schema":
            aligner = SchemaAlignment(FL_type="V", role="host", channel=guest_channel)
            aligner.fit(data, self.feature_names)
            aligned_data = aligner.transform(data)
            if self.output_path:
                save_pickle_file(aligned_data, self.output_path)
            return aligned_data
        else:
            raise ValueError(f"不支持的对齐方法: {self.alignment_method}")

        aligned_data = aligner.fit_transform(data)

        if self.output_path:
            save_pickle_file(aligned_data, self.output_path)

        logger.info("FeatureAlignmentHost: 特征对齐完成")
        return aligned_data

    def _load_data(self) -> np.ndarray:
        if self.data_info:
            data = read_data(data_info=self.data_info, selected_column=None, droped_column=None)
            return np.array(data, dtype=np.float64)
        return np.array([])
