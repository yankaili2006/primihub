"""
Federated Learning Feature Binning Host/Guest
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

from primihub.FL.utils.base import BaseModel
from primihub.FL.utils.net_work import GrpcClient, MultiGrpcClients
from primihub.FL.utils.file import save_pickle_file, save_json_file
from primihub.FL.utils.dataset import read_data

from .base import EqualWidthBinning, EqualFrequencyBinning, OptimalBinning, WOEBinning

logger = logging.getLogger(__name__)


class FeatureBinningHost(BaseModel):
    """特征装仓主机端"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parse_params()

    def _parse_params(self):
        self.binning_method = self.common_params.get("binning_method", "equal_width")
        self.n_bins = self.common_params.get("n_bins", 10)
        self.output_path = self.common_params.get("output_path", "")

        self.data_info = self.role_params.get("data", {})
        self.label_column = self.role_params.get("label_column", None)

    def run(self):
        """执行特征装仓"""
        logger.info("FeatureBinningHost: 开始特征装仓")

        guest_parties = [p for p in self.roles if self.roles[p] == "guest"]
        guest_channel = MultiGrpcClients(
            self.node_info["self_party"],
            guest_parties,
            self.node_info,
            self.task_info,
        ) if guest_parties else None

        data, labels = self._load_data()
        logger.info(f"FeatureBinningHost: 数据 shape={data.shape}")

        if self.binning_method == "equal_width":
            binner = EqualWidthBinning(
                FL_type="H", role="server", channel=guest_channel, n_bins=self.n_bins
            )
            binned_data = binner.fit_transform(data)
        elif self.binning_method == "equal_frequency":
            binner = EqualFrequencyBinning(
                FL_type="H", role="server", channel=guest_channel, n_bins=self.n_bins
            )
            binned_data = binner.fit_transform(data)
        elif self.binning_method == "optimal":
            binner = OptimalBinning(
                FL_type="H", role="server", channel=guest_channel, n_bins=self.n_bins
            )
            binned_data = binner.fit_transform(data, labels)
        elif self.binning_method == "woe":
            binner = WOEBinning(
                FL_type="H", role="server", channel=guest_channel, n_bins=self.n_bins
            )
            binner.fit(data, labels)
            binned_data = binner.transform_woe(data)
        else:
            raise ValueError(f"不支持的分箱方法: {self.binning_method}")

        if self.output_path:
            save_pickle_file(binned_data, self.output_path)
            save_pickle_file(binner, self.output_path.replace(".pkl", "_binner.pkl"))

        logger.info("FeatureBinningHost: 特征装仓完成")
        return binned_data

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


class FeatureBinningGuest(BaseModel):
    """特征装仓访客端"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parse_params()

    def _parse_params(self):
        self.binning_method = self.common_params.get("binning_method", "equal_width")
        self.n_bins = self.common_params.get("n_bins", 10)
        self.output_path = self.common_params.get("output_path", "")

        self.data_info = self.role_params.get("data", {})

    def run(self):
        """执行特征装仓"""
        logger.info("FeatureBinningGuest: 开始特征装仓")

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

        if self.binning_method == "equal_width":
            binner = EqualWidthBinning(
                FL_type="H", role="client", channel=host_channel, n_bins=self.n_bins
            )
        elif self.binning_method == "equal_frequency":
            binner = EqualFrequencyBinning(
                FL_type="H", role="client", channel=host_channel, n_bins=self.n_bins
            )
        else:
            binner = EqualWidthBinning(
                FL_type="H", role="client", channel=host_channel, n_bins=self.n_bins
            )

        binned_data = binner.fit_transform(data)

        if self.output_path:
            save_pickle_file(binned_data, self.output_path)

        return binned_data

    def _load_data(self) -> np.ndarray:
        if self.data_info:
            data = read_data(data_info=self.data_info, selected_column=None, droped_column=None)
            return np.array(data, dtype=np.float64)
        return np.array([])
