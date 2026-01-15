"""
Federated Learning Sample Expansion Host/Guest
联邦学习样本列扩展主机端和访客端
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

from primihub.FL.utils.base import BaseModel
from primihub.FL.utils.net_work import GrpcClient, MultiGrpcClients
from primihub.FL.utils.file import save_pickle_file
from primihub.FL.utils.dataset import read_data

from .base import PolynomialExpansion, InteractionExpansion, CrossFeatureExpansion, FeatureAugmentation

logger = logging.getLogger(__name__)


class SampleExpansionHost(BaseModel):
    """样本列扩展主机端"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parse_params()

    def _parse_params(self):
        self.expansion_method = self.common_params.get("expansion_method", "polynomial")
        self.degree = self.common_params.get("degree", 2)
        self.interaction_only = self.common_params.get("interaction_only", False)
        self.transformations = self.common_params.get("transformations", ["log", "sqrt"])
        self.output_path = self.common_params.get("output_path", "")

        self.data_info = self.role_params.get("data", {})

    def run(self):
        """执行样本列扩展"""
        logger.info("SampleExpansionHost: 开始样本列扩展")

        guest_parties = [p for p in self.roles if self.roles[p] == "guest"]
        guest_channel = MultiGrpcClients(
            self.node_info["self_party"],
            guest_parties,
            self.node_info,
            self.task_info,
        ) if guest_parties else None

        data = self._load_data()
        logger.info(f"SampleExpansionHost: 原始数据 shape={data.shape}")

        if self.expansion_method == "polynomial":
            expander = PolynomialExpansion(
                FL_type="V", role="host", channel=guest_channel,
                degree=self.degree, interaction_only=self.interaction_only
            )
        elif self.expansion_method == "interaction":
            expander = InteractionExpansion(FL_type="V", role="host", channel=guest_channel)
        elif self.expansion_method == "cross":
            expander = CrossFeatureExpansion(FL_type="V", role="host", channel=guest_channel)
        elif self.expansion_method == "augmentation":
            expander = FeatureAugmentation(
                FL_type="V", role="host", channel=guest_channel, transformations=self.transformations
            )
        else:
            raise ValueError(f"不支持的扩展方法: {self.expansion_method}")

        expanded_data = expander.fit_transform(data)
        logger.info(f"SampleExpansionHost: 扩展后数据 shape={expanded_data.shape}")

        if self.output_path:
            save_pickle_file(expanded_data, self.output_path)

        return expanded_data

    def _load_data(self) -> np.ndarray:
        if self.data_info:
            data = read_data(data_info=self.data_info, selected_column=None, droped_column=None)
            return np.array(data, dtype=np.float64)
        return np.array([])


class SampleExpansionGuest(BaseModel):
    """样本列扩展访客端"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parse_params()

    def _parse_params(self):
        self.expansion_method = self.common_params.get("expansion_method", "polynomial")
        self.degree = self.common_params.get("degree", 2)
        self.interaction_only = self.common_params.get("interaction_only", False)
        self.transformations = self.common_params.get("transformations", ["log", "sqrt"])
        self.output_path = self.common_params.get("output_path", "")

        self.data_info = self.role_params.get("data", {})

    def run(self):
        """执行样本列扩展"""
        logger.info("SampleExpansionGuest: 开始样本列扩展")

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
        logger.info(f"SampleExpansionGuest: 原始数据 shape={data.shape}")

        if self.expansion_method == "polynomial":
            expander = PolynomialExpansion(
                FL_type="V", role="guest", channel=host_channel,
                degree=self.degree, interaction_only=self.interaction_only
            )
        elif self.expansion_method == "interaction":
            expander = InteractionExpansion(FL_type="V", role="guest", channel=host_channel)
        elif self.expansion_method == "cross":
            expander = CrossFeatureExpansion(FL_type="V", role="guest", channel=host_channel)
        elif self.expansion_method == "augmentation":
            expander = FeatureAugmentation(
                FL_type="V", role="guest", channel=host_channel, transformations=self.transformations
            )
        else:
            raise ValueError(f"不支持的扩展方法: {self.expansion_method}")

        expanded_data = expander.fit_transform(data)

        if self.output_path:
            save_pickle_file(expanded_data, self.output_path)

        return expanded_data

    def _load_data(self) -> np.ndarray:
        if self.data_info:
            data = read_data(data_info=self.data_info, selected_column=None, droped_column=None)
            return np.array(data, dtype=np.float64)
        return np.array([])
