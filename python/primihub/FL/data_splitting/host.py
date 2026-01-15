"""
Federated Learning Data Splitting Host/Guest
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging

from primihub.FL.utils.base import BaseModel
from primihub.FL.utils.net_work import GrpcClient, MultiGrpcClients
from primihub.FL.utils.file import save_pickle_file
from primihub.FL.utils.dataset import read_data

from .base import TrainTestSplitter, KFoldSplitter, StratifiedSplitter, TimeSplitter

logger = logging.getLogger(__name__)


class DataSplittingHost(BaseModel):
    """数据分割主机端"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parse_params()

    def _parse_params(self):
        self.split_method = self.common_params.get("split_method", "train_test")
        self.test_size = self.common_params.get("test_size", 0.2)
        self.n_splits = self.common_params.get("n_splits", 5)
        self.shuffle = self.common_params.get("shuffle", True)
        self.random_state = self.common_params.get("random_state", 42)
        self.output_path = self.common_params.get("output_path", "")

        self.data_info = self.role_params.get("data", {})
        self.label_column = self.role_params.get("label_column", None)

    def run(self):
        """执行数据分割"""
        logger.info("DataSplittingHost: 开始数据分割")

        guest_parties = [p for p in self.roles if self.roles[p] == "guest"]
        guest_channel = MultiGrpcClients(
            self.node_info["self_party"],
            guest_parties,
            self.node_info,
            self.task_info,
        ) if guest_parties else None

        data, labels = self._load_data()
        logger.info(f"DataSplittingHost: 数据 shape={data.shape}")

        if self.split_method == "train_test":
            splitter = TrainTestSplitter(
                FL_type="H", role="server", channel=guest_channel,
                test_size=self.test_size, shuffle=self.shuffle, random_state=self.random_state
            )
            (X_train, y_train), (X_test, y_test) = splitter.split(data, labels)
            result = {
                "X_train": X_train, "y_train": y_train,
                "X_test": X_test, "y_test": y_test,
            }

        elif self.split_method == "kfold":
            splitter = KFoldSplitter(
                FL_type="H", role="server", channel=guest_channel,
                n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state
            )
            folds = list(splitter.split(data, labels))
            result = {"folds": folds, "n_splits": self.n_splits}

        elif self.split_method == "stratified":
            splitter = StratifiedSplitter(
                FL_type="H", role="server", channel=guest_channel,
                test_size=self.test_size, shuffle=self.shuffle, random_state=self.random_state
            )
            (X_train, y_train), (X_test, y_test) = splitter.split(data, labels)
            result = {
                "X_train": X_train, "y_train": y_train,
                "X_test": X_test, "y_test": y_test,
            }

        elif self.split_method == "time_series":
            splitter = TimeSplitter(
                FL_type="H", role="server", channel=guest_channel, n_splits=self.n_splits
            )
            folds = list(splitter.split(data, labels))
            result = {"folds": folds, "n_splits": self.n_splits}

        else:
            raise ValueError(f"不支持的分割方法: {self.split_method}")

        if self.output_path:
            save_pickle_file(result, self.output_path)

        logger.info("DataSplittingHost: 数据分割完成")
        return result

    def _load_data(self):
        labels = None
        if self.data_info:
            data = read_data(data_info=self.data_info, selected_column=None, droped_column=None)
            data = np.array(data, dtype=np.float64)
            if self.label_column is not None:
                labels = data[:, self.label_column]
                data = np.delete(data, self.label_column, axis=1)
            return data, labels
        return np.array([]), None


class DataSplittingGuest(BaseModel):
    """数据分割访客端"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parse_params()

    def _parse_params(self):
        self.split_method = self.common_params.get("split_method", "train_test")
        self.test_size = self.common_params.get("test_size", 0.2)
        self.n_splits = self.common_params.get("n_splits", 5)
        self.shuffle = self.common_params.get("shuffle", True)
        self.random_state = self.common_params.get("random_state", 42)
        self.output_path = self.common_params.get("output_path", "")

        self.data_info = self.role_params.get("data", {})

    def run(self):
        """执行数据分割"""
        logger.info("DataSplittingGuest: 开始数据分割")

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

        splitter = TrainTestSplitter(
            FL_type="H", role="client", channel=host_channel,
            test_size=self.test_size, shuffle=self.shuffle, random_state=self.random_state
        )
        (X_train, _), (X_test, _) = splitter.split(data)
        result = {"X_train": X_train, "X_test": X_test}

        if self.output_path:
            save_pickle_file(result, self.output_path)

        return result

    def _load_data(self) -> np.ndarray:
        if self.data_info:
            data = read_data(data_info=self.data_info, selected_column=None, droped_column=None)
            return np.array(data, dtype=np.float64)
        return np.array([])
