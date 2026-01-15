"""
Federated Learning Feature Similarity Host
联邦学习特征相似度分析主机端
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

from primihub.FL.utils.base import BaseModel
from primihub.FL.utils.net_work import GrpcClient, MultiGrpcClients
from primihub.FL.utils.file import save_json_file
from primihub.FL.utils.dataset import read_data

from .base import FeatureSimilarityAnalyzer, CosineSimilarity, PearsonCorrelation

logger = logging.getLogger(__name__)


class FeatureSimilarityHost(BaseModel):
    """特征相似度分析主机端"""

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
        logger.info("FeatureSimilarityHost: 开始特征相似度分析")

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
        data = self._load_data()
        logger.info(f"FeatureSimilarityHost: 数据 shape={data.shape}")

        results = {}

        # 本地相似度分析
        analyzer = FeatureSimilarityAnalyzer(
            methods=self.methods,
            FL_type="V",
            role="host",
            channel=guest_channel,
        )
        local_similarity = analyzer.analyze(data)
        results["local"] = {k: v.tolist() for k, v in local_similarity.items()}

        # 跨方相似度分析
        if self.cross_party and guest_channel:
            cross_similarity = self._cross_party_similarity(data, guest_channel)
            results["cross_party"] = cross_similarity

        # 找出相似特征
        if "cosine" in local_similarity:
            similar_pairs = analyzer.find_similar_features(
                local_similarity["cosine"],
                threshold=self.threshold,
                feature_names_x=self.feature_names,
                feature_names_y=self.feature_names,
            )
            results["similar_pairs"] = similar_pairs

        # 保存结果
        if self.output_path:
            save_json_file(results, self.output_path)
            logger.info(f"FeatureSimilarityHost: 结果已保存到 {self.output_path}")

        logger.info("FeatureSimilarityHost: 特征相似度分析完成")
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
        self, local_data: np.ndarray, guest_channel: MultiGrpcClients
    ) -> Dict[str, Any]:
        """计算跨方特征相似度"""
        results = {}

        # 标准化本地数据
        local_norm = local_data - np.mean(local_data, axis=0)
        local_std = np.std(local_data, axis=0, ddof=1)

        # 发送统计量
        guest_channel.send_all("host_data_norm", local_norm.tolist())
        guest_channel.send_all("host_data_std", local_std.tolist())

        # 接收guest数据
        guest_data_norm = guest_channel.recv_all("guest_data_norm")
        guest_data_std = guest_channel.recv_all("guest_data_std")

        for party, g_norm in guest_data_norm.items():
            if g_norm is not None:
                g_norm = np.array(g_norm)
                g_std = np.array(guest_data_std.get(party, []))

                # 计算跨方皮尔逊相关
                n = local_norm.shape[0]
                cov = np.dot(local_norm.T, g_norm) / (n - 1)
                corr = cov / (np.outer(local_std, g_std) + 1e-10)

                results[party] = {
                    "pearson_correlation": corr.tolist(),
                    "max_correlation": float(np.max(np.abs(corr))),
                    "mean_correlation": float(np.mean(np.abs(corr))),
                }

        return results
