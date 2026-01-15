"""
Federated Learning Feature Encoding Host
联邦学习特征编码主机端
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

from primihub.FL.utils.base import BaseModel
from primihub.FL.utils.net_work import MultiGrpcClients
from primihub.FL.utils.file import save_json_file, save_pickle_file
from primihub.FL.utils.dataset import read_data

from .base import FLOneHotEncoder, FLLabelEncoder, FLTargetEncoder, FLHashEncoder

logger = logging.getLogger(__name__)


class FeatureEncodingHost(BaseModel):
    """特征编码主机端"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parse_params()

    def _parse_params(self):
        """解析参数"""
        self.encoding_method = self.common_params.get("encoding_method", "onehot")
        self.categorical_columns = self.common_params.get("categorical_columns", [])
        self.output_path = self.common_params.get("output_path", "")

        # 编码器特定参数
        self.n_components = self.common_params.get("n_components", 8)
        self.smoothing = self.common_params.get("smoothing", 1.0)

        self.data_info = self.role_params.get("data", {})
        self.label_column = self.role_params.get("label_column", None)

    def run(self):
        """执行特征编码"""
        logger.info("FeatureEncodingHost: 开始特征编码")

        # 建立通信
        guest_parties = [p for p in self.roles if self.roles[p] == "guest"]
        guest_channel = None
        if guest_parties:
            guest_channel = MultiGrpcClients(
                self.node_info["self_party"],
                guest_parties,
                self.node_info,
                self.task_info,
            )

        # 加载数据
        data, labels = self._load_data()
        logger.info(f"FeatureEncodingHost: 数据 shape={data.shape}")

        # 执行编码
        if self.encoding_method == "onehot":
            encoder = FLOneHotEncoder(FL_type="H", role="host", channel=guest_channel)
        elif self.encoding_method == "label":
            encoder = FLLabelEncoder(FL_type="H", role="host", channel=guest_channel)
        elif self.encoding_method == "target":
            encoder = FLTargetEncoder(
                FL_type="H", role="host", channel=guest_channel, smoothing=self.smoothing
            )
        elif self.encoding_method == "hash":
            encoder = FLHashEncoder(
                FL_type="H", role="host", channel=guest_channel, n_components=self.n_components
            )
        else:
            raise ValueError(f"不支持的编码方法: {self.encoding_method}")

        # 拟合和转换
        if self.encoding_method == "target" and labels is not None:
            encoded_data = encoder.fit_transform(data, labels)
        else:
            encoded_data = encoder.fit_transform(data)

        logger.info(f"FeatureEncodingHost: 编码后 shape={encoded_data.shape}")

        # 保存结果
        if self.output_path:
            save_pickle_file(encoded_data, self.output_path)
            save_pickle_file(encoder, self.output_path.replace(".pkl", "_encoder.pkl"))

        logger.info("FeatureEncodingHost: 特征编码完成")
        return encoded_data

    def _load_data(self):
        """加载数据"""
        labels = None
        if self.data_info:
            data = read_data(
                data_info=self.data_info,
                selected_column=self.categorical_columns if self.categorical_columns else None,
                droped_column=None,
            )
            data = np.array(data)

            if self.label_column is not None:
                # 假设标签在最后一列
                labels = data[:, -1]
                data = data[:, :-1]

            return data, labels
        return np.array([]), None
