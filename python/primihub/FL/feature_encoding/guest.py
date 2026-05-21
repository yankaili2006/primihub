"""
Federated Learning Feature Encoding Guest
联邦学习特征编码访客端
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

from primihub.FL.utils.base import BaseModel
from primihub.FL.utils.net_work import GrpcClient
from primihub.FL.utils.file import save_pickle_file
from primihub.FL.utils.dataset import read_data

from .base import FLOneHotEncoder, FLLabelEncoder, FLTargetEncoder, FLHashEncoder

logger = logging.getLogger(__name__)


class FeatureEncodingGuest(BaseModel):
    """特征编码访客端"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parse_params()

    def _parse_params(self):
        """解析参数"""
        self.encoding_method = self.common_params.get("encoding_method", "onehot")
        self.categorical_columns = self.common_params.get("categorical_columns", [])
        self.output_path = self.common_params.get("output_path", "")
        self.n_components = self.common_params.get("n_components", 8)

        self.data_info = self.role_params.get("data", {})

    def run(self):
        """执行特征编码"""
        logger.info("FeatureEncodingGuest: 开始特征编码")

        # 建立通信
        host_party = None
        for party, role in self.roles.items():
            if role == "host":
                host_party = party
                break

        host_channel = None
        if host_party:
            host_channel = GrpcClient(
                self.node_info["self_party"],
                host_party,
                self.node_info,
                self.task_info,
            )

        # 加载数据
        data = self._load_data()
        logger.info(f"FeatureEncodingGuest: 数据 shape={data.shape}")

        # 执行编码
        if self.encoding_method == "onehot":
            encoder = FLOneHotEncoder(FL_type="H", role="guest", channel=host_channel)
        elif self.encoding_method == "label":
            encoder = FLLabelEncoder(FL_type="H", role="guest", channel=host_channel)
        elif self.encoding_method == "hash":
            encoder = FLHashEncoder(
                FL_type="H", role="guest", channel=host_channel, n_components=self.n_components
            )
        else:
            raise ValueError(f"不支持的编码方法: {self.encoding_method}")

        encoded_data = encoder.fit_transform(data)

        # 保存结果
        if self.output_path:
            save_pickle_file(encoded_data, self.output_path)

        logger.info("FeatureEncodingGuest: 特征编码完成")
        return encoded_data

    def _load_data(self) -> np.ndarray:
        """加载数据"""
        if self.data_info:
            data = read_data(
                data_info=self.data_info,
                selected_column=self.categorical_columns if self.categorical_columns else None,
                droped_column=None,
            )
            return np.array(data)
        return np.array([])
