"""
Federated Learning Data Transformation Host/Guest
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

from primihub.FL.utils.base import BaseModel
from primihub.FL.utils.net_work import GrpcClient, MultiGrpcClients
from primihub.FL.utils.file import save_pickle_file
from primihub.FL.utils.dataset import read_data

from .base import LogTransformer, BoxCoxTransformer, YeoJohnsonTransformer, RankTransformer

logger = logging.getLogger(__name__)


class DataTransformationHost(BaseModel):
    """数据转换主机端"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parse_params()

    def _parse_params(self):
        self.transform_method = self.common_params.get("transform_method", "log")
        self.standardize = self.common_params.get("standardize", True)
        self.output_distribution = self.common_params.get("output_distribution", "uniform")
        self.output_path = self.common_params.get("output_path", "")

        self.data_info = self.role_params.get("data", {})

    def run(self):
        """执行数据转换"""
        logger.info("DataTransformationHost: 开始数据转换")

        guest_parties = [p for p in self.roles if self.roles[p] == "guest"]
        guest_channel = MultiGrpcClients(
            self.node_info["self_party"],
            guest_parties,
            self.node_info,
            self.task_info,
        ) if guest_parties else None

        data = self._load_data()
        logger.info(f"DataTransformationHost: 数据 shape={data.shape}")

        if self.transform_method == "log":
            transformer = LogTransformer(FL_type="H", role="server", channel=guest_channel)
        elif self.transform_method == "boxcox":
            transformer = BoxCoxTransformer(
                FL_type="H", role="server", channel=guest_channel, standardize=self.standardize
            )
        elif self.transform_method == "yeojohnson":
            transformer = YeoJohnsonTransformer(
                FL_type="H", role="server", channel=guest_channel, standardize=self.standardize
            )
        elif self.transform_method == "rank":
            transformer = RankTransformer(
                FL_type="H", role="server", channel=guest_channel,
                output_distribution=self.output_distribution
            )
        else:
            raise ValueError(f"不支持的转换方法: {self.transform_method}")

        transformed_data = transformer.fit_transform(data)

        if self.output_path:
            save_pickle_file(transformed_data, self.output_path)
            save_pickle_file(transformer, self.output_path.replace(".pkl", "_transformer.pkl"))

        logger.info("DataTransformationHost: 数据转换完成")
        return transformed_data

    def _load_data(self) -> np.ndarray:
        if self.data_info:
            data = read_data(data_info=self.data_info, selected_column=None, droped_column=None)
            return np.array(data, dtype=np.float64)
        return np.array([])


class DataTransformationGuest(BaseModel):
    """数据转换访客端"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parse_params()

    def _parse_params(self):
        self.transform_method = self.common_params.get("transform_method", "log")
        self.standardize = self.common_params.get("standardize", True)
        self.output_distribution = self.common_params.get("output_distribution", "uniform")
        self.output_path = self.common_params.get("output_path", "")

        self.data_info = self.role_params.get("data", {})

    def run(self):
        """执行数据转换"""
        logger.info("DataTransformationGuest: 开始数据转换")

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

        if self.transform_method == "log":
            transformer = LogTransformer(FL_type="H", role="client", channel=host_channel)
        elif self.transform_method == "boxcox":
            transformer = BoxCoxTransformer(
                FL_type="H", role="client", channel=host_channel, standardize=self.standardize
            )
        elif self.transform_method == "yeojohnson":
            transformer = YeoJohnsonTransformer(
                FL_type="H", role="client", channel=host_channel, standardize=self.standardize
            )
        elif self.transform_method == "rank":
            transformer = RankTransformer(
                FL_type="H", role="client", channel=host_channel,
                output_distribution=self.output_distribution
            )
        else:
            raise ValueError(f"不支持的转换方法: {self.transform_method}")

        transformed_data = transformer.fit_transform(data)

        if self.output_path:
            save_pickle_file(transformed_data, self.output_path)

        return transformed_data

    def _load_data(self) -> np.ndarray:
        if self.data_info:
            data = read_data(data_info=self.data_info, selected_column=None, droped_column=None)
            return np.array(data, dtype=np.float64)
        return np.array([])
