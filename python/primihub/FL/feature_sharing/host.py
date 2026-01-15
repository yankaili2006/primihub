"""
Federated Learning Feature Sharing Host
联邦学习特征分享主机端
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

from primihub.FL.utils.base import BaseModel
from primihub.FL.utils.net_work import MultiGrpcClients
from primihub.FL.utils.file import save_pickle_file
from primihub.FL.utils.dataset import read_data

from .base import SecureFeatureSharing, PartialFeatureSharing, FeatureAggregation

logger = logging.getLogger(__name__)


class FeatureSharingHost(BaseModel):
    """特征分享主机端"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parse_params()

    def _parse_params(self):
        self.sharing_method = self.common_params.get("sharing_method", "secure")
        self.sharing_ratio = self.common_params.get("sharing_ratio", 0.5)
        self.noise_scale = self.common_params.get("noise_scale", 0.1)
        self.aggregation_method = self.common_params.get("aggregation_method", "concat")
        self.output_path = self.common_params.get("output_path", "")

        self.data_info = self.role_params.get("data", {})
        self.feature_indices = self.role_params.get("feature_indices", None)

    def run(self):
        """执行特征分享"""
        logger.info("FeatureSharingHost: 开始特征分享")

        guest_parties = [p for p in self.roles if self.roles[p] == "guest"]
        guest_channel = MultiGrpcClients(
            self.node_info["self_party"],
            guest_parties,
            self.node_info,
            self.task_info,
        ) if guest_parties else None

        data = self._load_data()
        logger.info(f"FeatureSharingHost: 数据 shape={data.shape}")

        if self.sharing_method == "secure":
            sharer = SecureFeatureSharing(
                FL_type="V", role="host", channel=guest_channel, noise_scale=self.noise_scale
            )
            result = sharer.share(data, self.feature_indices)

        elif self.sharing_method == "partial":
            sharer = PartialFeatureSharing(
                FL_type="V", role="host", channel=guest_channel, sharing_ratio=self.sharing_ratio
            )
            result = sharer.share(data)

        elif self.sharing_method == "aggregation":
            aggregator = FeatureAggregation(
                FL_type="V", role="host", channel=guest_channel,
                aggregation_method=self.aggregation_method
            )
            aggregator.share(data)
            received = aggregator.receive()
            received["host"] = data
            aggregated = aggregator.aggregate(received)
            result = {"aggregated_data": aggregated}

            if self.output_path:
                save_pickle_file(aggregated, self.output_path)

        else:
            raise ValueError(f"不支持的分享方法: {self.sharing_method}")

        logger.info("FeatureSharingHost: 特征分享完成")
        return result

    def _load_data(self) -> np.ndarray:
        if self.data_info:
            data = read_data(data_info=self.data_info, selected_column=None, droped_column=None)
            return np.array(data, dtype=np.float64)
        return np.array([])
