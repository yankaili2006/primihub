"""
Federated Learning Feature Imputation Host/Guest
联邦学习特征填充主机端和访客端
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

from primihub.FL.utils.base import BaseModel
from primihub.FL.utils.net_work import GrpcClient, MultiGrpcClients
from primihub.FL.utils.file import save_pickle_file
from primihub.FL.utils.dataset import read_data

from .base import FLMeanImputer, FLMedianImputer, FLKNNImputer, FLIterativeImputer

logger = logging.getLogger(__name__)


class FeatureImputationHost(BaseModel):
    """特征填充主机端"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parse_params()

    def _parse_params(self):
        self.imputation_method = self.common_params.get("imputation_method", "mean")
        self.n_neighbors = self.common_params.get("n_neighbors", 5)
        self.max_iter = self.common_params.get("max_iter", 10)
        self.output_path = self.common_params.get("output_path", "")

        self.data_info = self.role_params.get("data", {})

    def run(self):
        """执行特征填充"""
        logger.info("FeatureImputationHost: 开始特征填充")

        guest_parties = [p for p in self.roles if self.roles[p] == "guest"]
        guest_channel = MultiGrpcClients(
            self.node_info["self_party"],
            guest_parties,
            self.node_info,
            self.task_info,
        ) if guest_parties else None

        data = self._load_data()
        logger.info(f"FeatureImputationHost: 数据 shape={data.shape}")
        logger.info(f"FeatureImputationHost: 缺失值数量={np.sum(np.isnan(data))}")

        if self.imputation_method == "mean":
            imputer = FLMeanImputer(FL_type="H", role="server", channel=guest_channel)
        elif self.imputation_method == "median":
            imputer = FLMedianImputer(FL_type="H", role="server", channel=guest_channel)
        elif self.imputation_method == "knn":
            imputer = FLKNNImputer(FL_type="H", role="server", channel=guest_channel, n_neighbors=self.n_neighbors)
        elif self.imputation_method == "iterative":
            imputer = FLIterativeImputer(FL_type="H", role="server", channel=guest_channel, max_iter=self.max_iter)
        else:
            raise ValueError(f"不支持的填充方法: {self.imputation_method}")

        imputed_data = imputer.fit_transform(data)

        if self.output_path:
            save_pickle_file(imputed_data, self.output_path)

        logger.info("FeatureImputationHost: 特征填充完成")
        return imputed_data

    def _load_data(self) -> np.ndarray:
        if self.data_info:
            data = read_data(data_info=self.data_info, selected_column=None, droped_column=None)
            return np.array(data, dtype=np.float64)
        return np.array([])


class FeatureImputationGuest(BaseModel):
    """特征填充访客端"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parse_params()

    def _parse_params(self):
        self.imputation_method = self.common_params.get("imputation_method", "mean")
        self.n_neighbors = self.common_params.get("n_neighbors", 5)
        self.max_iter = self.common_params.get("max_iter", 10)
        self.output_path = self.common_params.get("output_path", "")

        self.data_info = self.role_params.get("data", {})

    def run(self):
        """执行特征填充"""
        logger.info("FeatureImputationGuest: 开始特征填充")

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
        logger.info(f"FeatureImputationGuest: 数据 shape={data.shape}")

        if self.imputation_method == "mean":
            imputer = FLMeanImputer(FL_type="H", role="client", channel=host_channel)
        elif self.imputation_method == "median":
            imputer = FLMedianImputer(FL_type="H", role="client", channel=host_channel)
        elif self.imputation_method == "knn":
            imputer = FLKNNImputer(FL_type="H", role="client", channel=host_channel, n_neighbors=self.n_neighbors)
        elif self.imputation_method == "iterative":
            imputer = FLIterativeImputer(FL_type="H", role="client", channel=host_channel, max_iter=self.max_iter)
        else:
            raise ValueError(f"不支持的填充方法: {self.imputation_method}")

        imputed_data = imputer.fit_transform(data)

        if self.output_path:
            save_pickle_file(imputed_data, self.output_path)

        logger.info("FeatureImputationGuest: 特征填充完成")
        return imputed_data

    def _load_data(self) -> np.ndarray:
        if self.data_info:
            data = read_data(data_info=self.data_info, selected_column=None, droped_column=None)
            return np.array(data, dtype=np.float64)
        return np.array([])
