"""
Federated Learning Data Fusion Host
联邦学习数据融合主机端

主机端负责持有标签数据，协调数据融合过程
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

from primihub.FL.utils.base import BaseModel
from primihub.FL.utils.net_work import GrpcClient, MultiGrpcClients
from primihub.FL.utils.file import save_json_file, save_pickle_file
from primihub.FL.utils.dataset import read_data

from .base import (
    HorizontalDataFusion,
    VerticalDataFusion,
    SecureDataFusion,
    StatisticalDataFusion,
)

logger = logging.getLogger(__name__)


class DataFusionHost(BaseModel):
    """数据融合主机端"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parse_params()

    def _parse_params(self):
        """解析参数"""
        # 通用参数
        self.fusion_type = self.common_params.get("fusion_type", "vertical")
        self.fusion_method = self.common_params.get("fusion_method", "concat")
        self.secure = self.common_params.get("secure", False)
        self.output_path = self.common_params.get("output_path", "")

        # 角色参数
        self.data_info = self.role_params.get("data", {})
        self.selected_columns = self.role_params.get("selected_columns", [])
        self.id_column = self.role_params.get("id_column", None)
        self.label_column = self.role_params.get("label_column", None)

    def run(self):
        """执行数据融合"""
        logger.info("DataFusionHost: 开始数据融合任务")

        # 获取所有参与方
        parties = self.roles.keys()
        guest_parties = [p for p in parties if p != self.node_info["self_party"]]

        # 建立通信通道
        if len(guest_parties) > 0:
            guest_channel = MultiGrpcClients(
                self.node_info["self_party"],
                guest_parties,
                self.node_info,
                self.task_info,
            )
        else:
            guest_channel = None

        # 加载本地数据
        local_data = self._load_local_data()
        logger.info(f"DataFusionHost: 加载本地数据 shape={local_data.shape}")

        # 根据融合类型执行不同操作
        if self.fusion_type == "vertical":
            result = self._vertical_fusion(local_data, guest_channel)
        elif self.fusion_type == "horizontal":
            result = self._horizontal_fusion(local_data, guest_channel)
        elif self.fusion_type == "statistics":
            result = self._statistics_fusion(local_data, guest_channel)
        else:
            raise ValueError(f"不支持的融合类型: {self.fusion_type}")

        # 保存结果
        self._save_results(result)

        logger.info("DataFusionHost: 数据融合任务完成")
        return result

    def _load_local_data(self) -> np.ndarray:
        """加载本地数据"""
        if self.data_info:
            data = read_data(
                data_info=self.data_info,
                selected_column=self.selected_columns if self.selected_columns else None,
                droped_column=None,
            )
            return np.array(data)
        return np.array([])

    def _vertical_fusion(
        self, local_data: np.ndarray, guest_channel: Optional[MultiGrpcClients]
    ) -> np.ndarray:
        """纵向数据融合"""
        logger.info("DataFusionHost: 执行纵向数据融合")

        # 发送样本ID给guest进行对齐
        if self.id_column is not None and guest_channel:
            guest_channel.send_all("sample_ids", local_data[:, 0].tolist())

        # 接收guest数据
        data_list = [local_data]
        if guest_channel:
            guest_data_dict = guest_channel.recv_all("guest_data")
            for party, data in guest_data_dict.items():
                if data is not None:
                    data_list.append(np.array(data))
                    logger.info(f"DataFusionHost: 收到来自 {party} 的数据")

        # 执行融合
        fusioner = VerticalDataFusion(
            fusion_method=self.fusion_method,
            secure=self.secure,
        )

        result = fusioner.fuse(data_list)
        logger.info(f"DataFusionHost: 融合完成, 结果 shape={result.shape}")

        return result

    def _horizontal_fusion(
        self, local_data: np.ndarray, guest_channel: Optional[MultiGrpcClients]
    ) -> np.ndarray:
        """横向数据融合"""
        logger.info("DataFusionHost: 执行横向数据融合")

        # 收集所有数据
        data_list = [local_data]
        if guest_channel:
            guest_data_dict = guest_channel.recv_all("guest_data")
            for party, data in guest_data_dict.items():
                if data is not None:
                    data_list.append(np.array(data))

        # 执行融合
        fusioner = HorizontalDataFusion(
            fusion_method=self.fusion_method,
            secure=self.secure,
        )

        result = fusioner.fuse(data_list)
        logger.info(f"DataFusionHost: 融合完成, 结果 shape={result.shape}")

        return result

    def _statistics_fusion(
        self, local_data: np.ndarray, guest_channel: Optional[MultiGrpcClients]
    ) -> Dict[str, np.ndarray]:
        """统计融合"""
        logger.info("DataFusionHost: 执行统计融合")

        # 计算本地统计量
        local_stats = {
            "mean": np.mean(local_data, axis=0),
            "std": np.std(local_data, axis=0),
            "min": np.min(local_data, axis=0),
            "max": np.max(local_data, axis=0),
            "count": local_data.shape[0],
        }

        # 收集guest统计量
        stats_list = [local_stats]
        sample_counts = [local_data.shape[0]]

        if guest_channel:
            guest_stats = guest_channel.recv_all("guest_stats")
            guest_counts = guest_channel.recv_all("guest_sample_count")

            for party in guest_stats:
                if guest_stats[party] is not None:
                    stats_list.append(guest_stats[party])
                    sample_counts.append(guest_counts.get(party, 0))

        # 融合统计量
        fusioner = StatisticalDataFusion(stat_type="mean")
        result = fusioner.fuse_statistics(stats_list, sample_counts)

        return result

    def _save_results(self, result):
        """保存结果"""
        if self.output_path:
            if isinstance(result, dict):
                save_json_file(
                    {k: v.tolist() if hasattr(v, "tolist") else v for k, v in result.items()},
                    self.output_path,
                )
            else:
                save_pickle_file(result, self.output_path)
            logger.info(f"DataFusionHost: 结果已保存到 {self.output_path}")
