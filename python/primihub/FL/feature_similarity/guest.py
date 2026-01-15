"""
Federated Learning Feature Similarity Guest
联邦学习特征相似度分析访客端
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

from primihub.FL.utils.base import BaseModel
from primihub.FL.utils.net_work import GrpcClient
from primihub.FL.utils.file import save_json_file
from primihub.FL.utils.dataset import read_data

from .base import FeatureSimilarityAnalyzer

logger = logging.getLogger(__name__)


class FeatureSimilarityGuest(BaseModel):
    """特征相似度分析访客端"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parse_params()

    def _parse_params(self):
        """解析参数"""
        self.methods = self.common_params.get("methods", ["cosine", "pearson"])
        self.threshold = self.common_params.get("threshold", 0.8)
        self.output_path = self.common_params.get("output_path", "")
        self.cross_party = self.common_params.get("cross_party", True)

        self.data_info = self.role_params.get("data", {})
        self.selected_columns = self.role_params.get("selected_columns", [])
        self.feature_names = self.role_params.get("feature_names", [])

    def run(self):
        """执行特征相似度分析"""
        logger.info("FeatureSimilarityGuest: 开始特征相似度分析")

        # 建立与host的通信
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
        logger.info(f"FeatureSimilarityGuest: 数据 shape={data.shape}")

        results = {}

        # 本地相似度分析
        analyzer = FeatureSimilarityAnalyzer(
            methods=self.methods,
            FL_type="V",
            role="guest",
            channel=host_channel,
        )
        local_similarity = analyzer.analyze(data)
        results["local"] = {k: v.tolist() for k, v in local_similarity.items()}

        # 跨方相似度
        if self.cross_party and host_channel:
            self._cross_party_similarity(data, host_channel)

        # 保存结果
        if self.output_path:
            save_json_file(results, self.output_path)

        logger.info("FeatureSimilarityGuest: 特征相似度分析完成")
        return results

    def _load_data(self) -> np.ndarray:
        """加载数据"""
        if self.data_info:
            data = read_data(
                data_info=self.data_info,
                selected_column=self.selected_columns if self.selected_columns else None,
                droped_column=None,
            )
            return np.array(data, dtype=np.float64)
        return np.array([])

    def _cross_party_similarity(
        self, local_data: np.ndarray, host_channel: GrpcClient
    ):
        """参与跨方相似度计算"""
        # 标准化本地数据
        local_norm = local_data - np.mean(local_data, axis=0)
        local_std = np.std(local_data, axis=0, ddof=1)

        # 接收host数据
        host_channel.recv("host_data_norm")
        host_channel.recv("host_data_std")

        # 发送本地数据
        host_channel.send("guest_data_norm", local_norm.tolist())
        host_channel.send("guest_data_std", local_std.tolist())
