"""
Federated Learning Feature Sharing Guest
联邦学习特征分享访客端
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

from primihub.FL.utils.base import BaseModel
from primihub.FL.utils.net_work import GrpcClient
from primihub.FL.utils.file import save_pickle_file
from primihub.FL.utils.dataset import read_data

from .base import SecureFeatureSharing, PartialFeatureSharing, FeatureAggregation

logger = logging.getLogger(__name__)


class FeatureSharingGuest(BaseModel):
    """特征分享访客端"""

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
        logger.info("FeatureSharingGuest: 开始特征分享")

        host_party = None
        for party, role in self.roles.items():
            if role == "host":
                host_party = party
                break

        host_channel = GrpcClient(
            self.node_info["self_party"],
            host_party,
            self.node_info,
            self.task_info,
        ) if host_party else None

        data = self._load_data()
        logger.info(f"FeatureSharingGuest: 数据 shape={data.shape}")

        if self.sharing_method == "secure":
            sharer = SecureFeatureSharing(
                FL_type="V", role="guest", channel=host_channel, noise_scale=self.noise_scale
            )
            result = sharer.share(data, self.feature_indices)

        elif self.sharing_method == "partial":
            sharer = PartialFeatureSharing(
                FL_type="V", role="guest", channel=host_channel, sharing_ratio=self.sharing_ratio
            )
            result = sharer.share(data)

        elif self.sharing_method == "aggregation":
            aggregator = FeatureAggregation(
                FL_type="V", role="guest", channel=host_channel,
                aggregation_method=self.aggregation_method
            )
            aggregator.share(data)
            result = aggregator.receive()

        else:
            raise ValueError(f"不支持的分享方法: {self.sharing_method}")

        logger.info("FeatureSharingGuest: 特征分享完成")
        return result

    def _load_data(self) -> np.ndarray:
        if self.data_info:
            data = read_data(data_info=self.data_info, selected_column=None, droped_column=None)
            return np.array(data, dtype=np.float64)
        return np.array([])
