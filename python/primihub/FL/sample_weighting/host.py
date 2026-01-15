"""
Federated Learning Sample Weighting Host/Guest
联邦学习样本加权主机端和访客端
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

from primihub.FL.utils.base import BaseModel
from primihub.FL.utils.net_work import GrpcClient, MultiGrpcClients
from primihub.FL.utils.file import save_pickle_file, save_json_file
from primihub.FL.utils.dataset import read_data

from .base import ClassWeighting, ImportanceWeighting, DistributionWeighting

logger = logging.getLogger(__name__)


class SampleWeightingHost(BaseModel):
    """样本加权主机端"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parse_params()

    def _parse_params(self):
        self.weighting_method = self.common_params.get("weighting_method", "class")
        self.strategy = self.common_params.get("strategy", "balanced")
        self.temperature = self.common_params.get("temperature", 1.0)
        self.output_path = self.common_params.get("output_path", "")

        self.data_info = self.role_params.get("data", {})
        self.label_column = self.role_params.get("label_column", None)

    def run(self):
        """执行样本加权"""
        logger.info("SampleWeightingHost: 开始样本加权")

        guest_parties = [p for p in self.roles if self.roles[p] == "guest"]
        guest_channel = MultiGrpcClients(
            self.node_info["self_party"],
            guest_parties,
            self.node_info,
            self.task_info,
        ) if guest_parties else None

        data, labels = self._load_data()
        logger.info(f"SampleWeightingHost: 数据 shape={data.shape}")

        if self.weighting_method == "class":
            weighter = ClassWeighting(
                FL_type="H", role="server", channel=guest_channel, strategy=self.strategy
            )
        elif self.weighting_method == "importance":
            weighter = ImportanceWeighting(
                FL_type="H", role="server", channel=guest_channel, temperature=self.temperature
            )
        elif self.weighting_method == "distribution":
            weighter = DistributionWeighting(FL_type="H", role="server", channel=guest_channel)
        else:
            raise ValueError(f"不支持的加权方法: {self.weighting_method}")

        weighter.fit(data, labels)
        weights = weighter.compute_weights(data, labels)

        logger.info(f"SampleWeightingHost: 权重范围=[{weights.min():.4f}, {weights.max():.4f}]")

        if self.output_path:
            save_pickle_file(weights, self.output_path)

        return weights

    def _load_data(self):
        labels = None
        if self.data_info:
            data = read_data(data_info=self.data_info, selected_column=None, droped_column=None)
            data = np.array(data, dtype=np.float64)
            if self.label_column is not None and data.shape[1] > 1:
                labels = data[:, -1]
                data = data[:, :-1]
            return data, labels
        return np.array([]), None


class SampleWeightingGuest(BaseModel):
    """样本加权访客端"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parse_params()

    def _parse_params(self):
        self.weighting_method = self.common_params.get("weighting_method", "class")
        self.strategy = self.common_params.get("strategy", "balanced")
        self.temperature = self.common_params.get("temperature", 1.0)
        self.output_path = self.common_params.get("output_path", "")

        self.data_info = self.role_params.get("data", {})
        self.label_column = self.role_params.get("label_column", None)

    def run(self):
        """执行样本加权"""
        logger.info("SampleWeightingGuest: 开始样本加权")

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

        data, labels = self._load_data()

        if self.weighting_method == "class":
            weighter = ClassWeighting(
                FL_type="H", role="client", channel=host_channel, strategy=self.strategy
            )
        elif self.weighting_method == "importance":
            weighter = ImportanceWeighting(
                FL_type="H", role="client", channel=host_channel, temperature=self.temperature
            )
        elif self.weighting_method == "distribution":
            weighter = DistributionWeighting(FL_type="H", role="client", channel=host_channel)
        else:
            raise ValueError(f"不支持的加权方法: {self.weighting_method}")

        weighter.fit(data, labels)
        weights = weighter.compute_weights(data, labels)

        if self.output_path:
            save_pickle_file(weights, self.output_path)

        return weights

    def _load_data(self):
        labels = None
        if self.data_info:
            data = read_data(data_info=self.data_info, selected_column=None, droped_column=None)
            data = np.array(data, dtype=np.float64)
            if self.label_column is not None and data.shape[1] > 1:
                labels = data[:, -1]
                data = data[:, :-1]
            return data, labels
        return np.array([]), None
