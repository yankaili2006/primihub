"""
Federated Learning Data Fusion Coordinator
联邦学习数据融合协调者

协调者负责安全数据融合的协调工作
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

from primihub.FL.utils.base import BaseModel
from primihub.FL.utils.net_work import GrpcClient, MultiGrpcClients
from primihub.FL.utils.file import save_json_file

from .base import SecureDataFusion, StatisticalDataFusion

logger = logging.getLogger(__name__)


class DataFusionCoordinator(BaseModel):
    """数据融合协调者"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parse_params()

    def _parse_params(self):
        """解析参数"""
        # 通用参数
        self.fusion_type = self.common_params.get("fusion_type", "secure")
        self.fusion_method = self.common_params.get("fusion_method", "secure_average")
        self.encryption_type = self.common_params.get("encryption_type", "paillier")
        self.output_path = self.common_params.get("output_path", "")

    def run(self):
        """执行数据融合协调"""
        logger.info("DataFusionCoordinator: 开始数据融合协调任务")

        # 获取所有参与方
        parties = list(self.roles.keys())
        other_parties = [p for p in parties if p != self.node_info["self_party"]]

        # 分离host和guest
        host_party = None
        guest_parties = []
        for party, role in self.roles.items():
            if role == "host":
                host_party = party
            elif role == "guest":
                guest_parties.append(party)

        # 建立通信通道
        if host_party:
            host_channel = GrpcClient(
                self.node_info["self_party"],
                host_party,
                self.node_info,
                self.task_info,
            )
        else:
            host_channel = None

        if guest_parties:
            guest_channel = MultiGrpcClients(
                self.node_info["self_party"],
                guest_parties,
                self.node_info,
                self.task_info,
            )
        else:
            guest_channel = None

        # 根据融合类型执行协调
        if self.fusion_type == "secure":
            result = self._secure_fusion_coordinate(host_channel, guest_channel)
        elif self.fusion_type == "aggregation":
            result = self._aggregation_coordinate(host_channel, guest_channel)
        else:
            raise ValueError(f"不支持的融合类型: {self.fusion_type}")

        # 保存结果
        self._save_results(result)

        logger.info("DataFusionCoordinator: 数据融合协调任务完成")
        return result

    def _secure_fusion_coordinate(
        self,
        host_channel: Optional[GrpcClient],
        guest_channel: Optional[MultiGrpcClients],
    ) -> Dict[str, Any]:
        """安全融合协调"""
        logger.info("DataFusionCoordinator: 执行安全融合协调")

        # 生成加密密钥
        from primihub.FL.crypto.paillier import Paillier

        public_key, private_key = Paillier.generate_keypair()

        # 分发公钥
        if host_channel:
            host_channel.send("public_key", public_key)
        if guest_channel:
            guest_channel.send_all("public_key", public_key)

        # 收集加密数据
        encrypted_data_list = []

        if host_channel:
            host_encrypted = host_channel.recv("encrypted_data")
            if host_encrypted is not None:
                encrypted_data_list.append(host_encrypted)

        if guest_channel:
            guest_encrypted = guest_channel.recv_all("encrypted_data")
            for party, data in guest_encrypted.items():
                if data is not None:
                    encrypted_data_list.append(data)

        # 执行安全融合
        fusioner = SecureDataFusion(
            fusion_method=self.fusion_method,
            encryption_type=self.encryption_type,
        )
        fusioner.set_keys(public_key, private_key)

        if encrypted_data_list:
            fused_encrypted = fusioner.fuse(encrypted_data_list)

            # 解密结果
            paillier = Paillier(public_key, private_key)
            if hasattr(fused_encrypted, '__iter__'):
                result_data = paillier.decrypt_vector(fused_encrypted)
            else:
                result_data = paillier.decrypt_scalar(fused_encrypted)
        else:
            result_data = None

        # 分发结果
        if host_channel:
            host_channel.send("fused_result", result_data)
        if guest_channel:
            guest_channel.send_all("fused_result", result_data)

        return {"fused_data": result_data, "num_parties": len(encrypted_data_list)}

    def _aggregation_coordinate(
        self,
        host_channel: Optional[GrpcClient],
        guest_channel: Optional[MultiGrpcClients],
    ) -> Dict[str, Any]:
        """聚合协调"""
        logger.info("DataFusionCoordinator: 执行聚合协调")

        # 收集统计信息
        stats_list = []
        sample_counts = []

        if host_channel:
            host_stats = host_channel.recv("local_stats")
            host_count = host_channel.recv("sample_count")
            if host_stats is not None:
                stats_list.append(host_stats)
                sample_counts.append(host_count or 0)

        if guest_channel:
            guest_stats = guest_channel.recv_all("local_stats")
            guest_counts = guest_channel.recv_all("sample_count")
            for party, stats in guest_stats.items():
                if stats is not None:
                    stats_list.append(stats)
                    sample_counts.append(guest_counts.get(party, 0))

        # 执行统计融合
        fusioner = StatisticalDataFusion(stat_type="mean")
        aggregated_stats = fusioner.fuse_statistics(stats_list, sample_counts)

        # 分发聚合结果
        if host_channel:
            host_channel.send("aggregated_stats", aggregated_stats)
        if guest_channel:
            guest_channel.send_all("aggregated_stats", aggregated_stats)

        return {
            "aggregated_stats": aggregated_stats,
            "total_samples": sum(sample_counts),
            "num_parties": len(stats_list),
        }

    def _save_results(self, result: Dict[str, Any]):
        """保存结果"""
        if self.output_path:
            # 转换numpy数组为列表
            serializable_result = {}
            for k, v in result.items():
                if hasattr(v, "tolist"):
                    serializable_result[k] = v.tolist()
                elif isinstance(v, dict):
                    serializable_result[k] = {
                        sk: sv.tolist() if hasattr(sv, "tolist") else sv
                        for sk, sv in v.items()
                    }
                else:
                    serializable_result[k] = v

            save_json_file(serializable_result, self.output_path)
            logger.info(f"DataFusionCoordinator: 结果已保存到 {self.output_path}")
